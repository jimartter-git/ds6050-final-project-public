"""One train/eval entry point for supervised training, masked pretraining and fine-tuning.

Test evaluation is explicit and checkpoint-only.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from .config import ModelConfig, load_config
from .data import (OCCLUSION_MODES, PennActionDataset, collate, ids_hash, read_catalog,
                   split_never_visible, subsample, synthetic_clips, validation_split)
from .features import model_batch, pretraining_batch
from .masking import joint_states
from .metrics import classification_metrics, reconstruction_error
from .models import build_model
from .utils import RunLogger, git_commit, seed_everything, seed_worker


def run_epoch(model, loader, device, input_mode, num_classes, optimizer=None):
    model.train(optimizer is not None)
    labels, predictions, records = [], [], []
    loss_sum, count, recon_sum, masked_count = 0., 0, 0., 0
    for raw in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
        with torch.set_grad_enabled(optimizer is not None):
            output = model(model_batch(batch, input_mode))
            logits = output['logits']
            if logits.shape != (len(batch['label']), num_classes):
                raise ValueError('model logits must be [B,C]')
            loss = torch.nn.functional.cross_entropy(logits, batch['label'])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite classification loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        predicted = logits.detach().argmax(1).cpu().tolist()
        target = batch['label'].cpu().tolist()
        labels.extend(target)
        predictions.extend(predicted)
        records.extend({'clip_id': cid, 'target': y, 'predicted': p}
                       for cid, y, p in zip(batch['clip_id'], target, predicted))
        n = len(target)
        loss_sum += loss.item() * n
        count += n
        if 'recon' in output:
            # No deliberate masks in supervised mode: returns None, never scores NAT_MISSING.
            states, _ = joint_states(batch['vis'], batch['pad_mask'])
            metric = reconstruction_error(output['recon'].detach(), batch['coords'],
                                          states, batch['pad_mask'])
            if metric['masked_joints']:
                recon_sum += metric['reconstruction_error'] * metric['masked_joints']
                masked_count += metric['masked_joints']
    if count == 0:
        raise ValueError('empty train/evaluation split')
    metrics = classification_metrics(labels, predictions, num_classes)
    metrics.update(loss=loss_sum / count, reconstruction_error=recon_sum / masked_count
                   if masked_count else None, masked_joints=masked_count)
    return metrics, records


def run_pretrain_epoch(model, loader, device, cfg, optimizer=None, generator=None):
    """Masked-joint reconstruction. Loss covers every masked slot (for MSTD this includes
    imputed invisible targets); the reported error covers originally visible ones only."""
    model.train(optimizer is not None)
    loss_sum, slots, error_sum, visible_joints = 0., 0, 0., 0
    for raw in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
        inputs = pretraining_batch(batch, cfg.train.mask_ratio, generator, cfg.train.mask_policy)
        selected = inputs['masked']
        n = int(selected.sum())
        if n == 0:
            continue
        with torch.set_grad_enabled(optimizer is not None):
            recon = model(inputs)['recon']
            loss = torch.nn.functional.mse_loss(recon[selected], batch['coords'][selected])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite reconstruction loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * n
        slots += n
        metric = reconstruction_error(recon.detach(), batch['coords'], inputs['states'],
                                      batch['pad_mask'], vis=batch['vis'])
        if metric['masked_joints']:
            error_sum += metric['reconstruction_error'] * metric['masked_joints']
            visible_joints += metric['masked_joints']
    if slots == 0:
        raise ValueError('no joints were masked; check mask_ratio')
    return {'loss': loss_sum / slots, 'masked_slots': slots,
            'reconstruction_error': error_sum / visible_joints if visible_joints else None,
            'masked_visible_joints': visible_joints}


def check_checkpoint(checkpoint, cfg, metadata):
    """Refuse a checkpoint trained on other classes, clips, model settings or preprocessing."""
    for key in ('classes', 'joint_count', 'synthetic', 'split_ids_hash'):
        if checkpoint['metadata'][key] != metadata[key]:
            raise ValueError(f'checkpoint/dataset mismatch: {key}')
    # Fields added after a checkpoint was saved count as their defaults.
    saved_model = {**asdict(ModelConfig()), **checkpoint['config']['model']}
    if saved_model != asdict(cfg.model):
        raise ValueError('checkpoint/model config mismatch')
    # Prevent evaluation with altered normalization, coordinate policy or temporal length.
    for key in ('normalization', 'hidden_policy', 'frames'):
        if checkpoint['config']['data'][key] != asdict(cfg.data)[key]:
            raise ValueError(f'checkpoint/preprocessing mismatch: {key}')


def load_pretrained(model, path, cfg, metadata, device):
    """Initialize everything except the classification head from a pretraining checkpoint."""
    pretrained = torch.load(path, map_location=device, weights_only=True)
    if pretrained['config']['train'].get('objective') != 'pretrain':
        raise ValueError(f'{path} is not a pretraining checkpoint')
    check_checkpoint(pretrained, cfg, metadata)
    state = {k: v for k, v in pretrained['model'].items() if not k.startswith('classifier.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    head = {k for k in model.state_dict() if k.startswith('classifier.')}
    if unexpected or set(missing) != head:
        raise ValueError(f'pretrained weights do not match the model: {missing}, {unexpected}')
    metadata.update(init_checkpoint=path, init_checkpoint_seed=pretrained['metadata']['seed'],
                    init_checkpoint_epoch=pretrained['epoch'],
                    init_mask_policy=pretrained['config']['train']['mask_policy'])


def seconds_since(start, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)  # GPU work runs ahead of the Python clock
    return time.perf_counter() - start


def datasets(cfg):
    if cfg.data.synthetic:
        clips = synthetic_clips(cfg.train.seed)
        classes = sorted({c.action for c in clips})
        official_train = [c for c in clips if c.split_value == 1]
        val = {c.clip_id for c in official_train[-6:]}
        train_value, test_value = 1, -1  # Fixture only, not dataset assumptions.
        joints = 4
    else:
        clips, schema = read_catalog(cfg.data.data_root, cfg.data.schema)
        classes, joints = schema['classes'], schema['joint_count']
        train_value, test_value = schema['train_value'], schema['test_value']
        val = validation_split(clips, train_value, cfg.data.val_ids)
    groups = {
        'train': [c for c in clips if c.split_value == train_value and c.clip_id not in val],
        'val': [c for c in clips if c.clip_id in val],
        'test': [c for c in clips if c.split_value == test_value],
    }
    # Never-visible clips are all zeros after imputation; val/test still fail loudly.
    groups['train'], excluded = split_never_visible(groups['train'])
    ds = {name: PennActionDataset(subsample(group, cfg.data.debug_per_class, cfg.train.seed),
                                  classes, cfg.data.frames, cfg.data.normalization,
                                  cfg.data.hidden_policy,
                                  vis_dropout=cfg.train.vis_dropout if name == 'train' else 0.0)
          for name, group in groups.items()}
    metadata = {'classes': classes, 'joint_count': joints,
                'split_ids_hash': {name: ids_hash(value.clips) for name, value in ds.items()},
                'synthetic': cfg.data.synthetic, 'excluded_train_ids': excluded}
    return ds, metadata


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif name.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use train.device=auto or cpu')
    return torch.device(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='section.key=value')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--checkpoint')
    parser.add_argument('--occlusion', type=float, default=0.0,
                        help='eval only: fraction of visible joint-frames to hide (RQ3)')
    parser.add_argument('--occlusion-seed', type=int, default=0,
                        help='keep identical across models so they see the same hidden joints')
    parser.add_argument('--occlusion-mode', choices=OCCLUSION_MODES, default='frames',
                        help='frames: single joint-frames; spans: 8-16 consecutive frames')
    args = parser.parse_args(argv)
    if args.mode == 'train' and (args.split == 'test' or args.checkpoint or args.occlusion):
        parser.error('training never evaluates test, loads a checkpoint, or occludes')
    if not 0 <= args.occlusion < 1:
        parser.error('--occlusion must be in [0, 1)')
    if args.mode == 'eval' and not args.checkpoint:
        parser.error('eval requires --checkpoint (use best.pt)')
    cfg = load_config(args.config, args.set)
    pretrain = cfg.train.objective == 'pretrain'
    if args.mode == 'eval' and pretrain:
        parser.error('eval scores classifiers; evaluate the fine-tuned checkpoint instead')
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    ds, metadata = datasets(cfg)
    model = build_model(cfg.model, metadata['joint_count'], len(metadata['classes']),
                        cfg.data.frames).to(device)
    if args.mode == 'eval':
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if checkpoint['config']['train'].get('objective', 'classify') != 'classify':
            raise ValueError('cannot evaluate a pretraining checkpoint as a classifier')
        check_checkpoint(checkpoint, cfg, metadata)
        model.load_state_dict(checkpoint['model'])
        metadata.update(checkpoint=args.checkpoint, checkpoint_seed=checkpoint['metadata']['seed'],
                        checkpoint_epoch=checkpoint['epoch'], occlusion_rate=args.occlusion,
                        occlusion_seed=args.occlusion_seed, occlusion_mode=args.occlusion_mode)
        if args.occlusion:
            d = ds[args.split]
            ds[args.split] = PennActionDataset(d.clips, d.classes, d.frames, d.normalization,
                                               d.hidden_policy, args.occlusion, args.occlusion_seed,
                                               args.occlusion_mode)
    elif cfg.train.init_checkpoint:
        load_pretrained(model, cfg.train.init_checkpoint, cfg, metadata, device)
    output = Path(cfg.train.output_dir)
    output.mkdir(parents=True, exist_ok=False)  # No silent checkpoint or metric overwrite.
    config_dict = asdict(cfg)
    metadata.update(git_commit=git_commit(), seed=cfg.train.seed, device=str(device),
                    mode=args.mode, evaluation_split=args.split if args.mode == 'eval' else 'val')
    (output / 'config.yaml').write_text(yaml.safe_dump(config_dict, sort_keys=False))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    def loader(split, shuffle=False):
        return DataLoader(ds[split], batch_size=cfg.train.batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=cfg.train.workers,
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(cfg.train.seed))

    logger = RunLogger(output)
    # Wall-clock times go to timing.json so metrics.jsonl stays identical across reruns.
    if args.mode == 'eval':
        start = time.perf_counter()
        metrics, records = run_epoch(model, loader(args.split), device, cfg.model.input_mode,
                                    len(metadata['classes']))
        timing = {'eval_seconds': seconds_since(start, device), 'clips': len(records)}
        logger.log({'split': args.split, **metrics})
        (output / 'predictions.json').write_text(json.dumps(records, indent=2))
        (output / 'timing.json').write_text(json.dumps(timing, indent=2))
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.epochs)
                 if cfg.train.scheduler == 'cosine' else None)
    best = -np.inf
    train_loader, val_loader = loader('train', True), loader('val')
    timing = {'train_seconds': [], 'val_seconds': []}
    for epoch in range(cfg.train.epochs):
        lr = optimizer.param_groups[0]['lr']
        start = time.perf_counter()
        if pretrain:
            train_metrics = run_pretrain_epoch(model, train_loader, device, cfg, optimizer)
        else:
            train_metrics, _ = run_epoch(model, train_loader, device, cfg.model.input_mode,
                                         len(metadata['classes']), optimizer)
        if scheduler is not None:
            scheduler.step()
        timing['train_seconds'].append(seconds_since(start, device))
        start = time.perf_counter()
        if pretrain:
            # Identical validation masks every epoch, so epochs are comparable.
            generator = torch.Generator(device=device).manual_seed(cfg.train.seed)
            val_metrics = run_pretrain_epoch(model, val_loader, device, cfg,
                                             generator=generator)
            if val_metrics['reconstruction_error'] is None:
                raise ValueError('validation has no masked visible joints to score')
            score = -val_metrics['reconstruction_error']
        else:
            val_metrics, records = run_epoch(model, val_loader, device, cfg.model.input_mode,
                                             len(metadata['classes']))
            score = val_metrics['macro_f1']
        timing['val_seconds'].append(seconds_since(start, device))
        logger.log({'epoch': epoch + 1, 'lr': lr, 'train': train_metrics, 'val': val_metrics})
        if score > best:
            best = score
            torch.save({'model': model.state_dict(), 'config': config_dict,
                        'metadata': metadata, 'epoch': epoch + 1}, output / 'best.pt')
            if not pretrain:
                (output / 'val_predictions.json').write_text(json.dumps(records, indent=2))
    (output / 'timing.json').write_text(json.dumps(timing, indent=2))


if __name__ == '__main__':
    main()
