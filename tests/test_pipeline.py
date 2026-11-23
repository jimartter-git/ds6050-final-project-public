import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.io import savemat
import torch
from penn_action.config import load_config
from penn_action.data import (Clip, PennActionDataset, collate, preprocess, read_clip,
                              synthetic_clips, validation_split)
from penn_action.features import model_batch
from penn_action.masking import JointState, joint_states, mask_coordinates
from penn_action.metrics import classification_metrics, reconstruction_error
from penn_action.train import main, run_epoch
from penn_action.utils import seed_everything
from torch.utils.data import DataLoader


def fixture_batch():
    clips = synthetic_clips()
    dataset = PennActionDataset(clips, ['fixture-0', 'fixture-1', 'fixture-2'], frames=8)
    return collate([dataset[0], dataset[1]])


def test_batch_keys_and_padding():
    b = fixture_batch()
    assert set(b) == {'coords', 'vis', 'pad_mask', 'label', 'clip_id'}
    assert b['coords'].shape == (2, 6, 4, 2)
    assert b['vis'].dtype == b['pad_mask'].dtype == torch.bool
    assert b['coords'].dtype == torch.float32 and b['label'].dtype == torch.int64
    assert b['pad_mask'].tolist() == [[True]*5+[False], [True]*6]
    assert not b['vis'][0, -1].any() and not b['coords'][0, -1].any()


@pytest.mark.parametrize('mode,keys', [
    ('coords_only', {'coords', 'pad_mask'}),
    ('coords_vis', {'coords', 'pad_mask', 'vis', 'states'}),
    ('coords_vis_mask', {'coords', 'pad_mask', 'vis', 'states', 'masked'}),
    ('vis_only', {'vis', 'pad_mask'})])
def test_input_whitelist(mode, keys):
    from penn_action.features import frame_features, input_channels
    b = fixture_batch()
    b['sensitive_extra'] = torch.ones(1)
    out = model_batch(b, mode)
    assert set(out) == keys
    assert frame_features(out).shape == (*b['pad_mask'].shape, 4 * input_channels(mode))
    if mode == 'coords_only':
        altered = {**b, 'vis': ~b['vis']}
        assert torch.equal(out['coords'], model_batch(altered, mode)['coords'])
        assert 'vis' not in out and 'states' not in out
    if mode == 'coords_vis_mask':
        assert not out['masked'].any()  # m=0 outside pretraining
    if mode == 'vis_only':
        assert not out['vis'][~b['pad_mask']].any()


def test_states_never_mask_invisible_or_padding():
    b = fixture_batch()
    states, masked = joint_states(b['vis'], b['pad_mask'], 1)
    assert torch.equal(masked, b['vis'] & b['pad_mask'].unsqueeze(-1))
    assert (states[masked] == JointState.MASKED).all()
    assert (states[~masked] == JointState.NAT_MISSING).all()
    filled = mask_coordinates(b["coords"], states, b["pad_mask"])
    assert not filled[masked].any()
    natural = ~b["vis"] & b["pad_mask"].unsqueeze(-1)
    assert torch.equal(filled[natural], b["coords"][natural])
    states0, masked0 = joint_states(b['vis'], b['pad_mask'], 0)
    assert not masked0.any()
    assert (states0[b['vis']] == JointState.OBSERVED).all()


def test_mask_reproducibility():
    b = fixture_batch()
    a = joint_states(b['vis'], b['pad_mask'], .5, torch.Generator().manual_seed(5))
    c = joint_states(b['vis'], b['pad_mask'], .5, torch.Generator().manual_seed(5))
    assert torch.equal(a[1], c[1])


def test_reconstruction_only_masked_real_joints():
    states = torch.tensor([[[JointState.MASKED, JointState.NAT_MISSING],
                            [JointState.OBSERVED, JointState.MASKED]]])
    target = torch.zeros(1, 2, 2, 2)
    prediction = torch.ones_like(target) * 1000
    prediction[0, 0, 0] = torch.tensor([3., 4.])
    pad = torch.tensor([[True, False]])
    result = reconstruction_error(prediction, target, states, pad)
    assert result == {'reconstruction_error': 5., 'masked_joints': 1}
    assert reconstruction_error(prediction, target, states, ~pad)['masked_joints'] == 1
    assert reconstruction_error(prediction, target, states*0, pad)['reconstruction_error'] is None


def test_classification_metrics_all_classes():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 3)
    assert m['accuracy'] == .75
    assert m['macro_f1'] == pytest.approx((2/3 + 4/5 + 0)/3)
    assert m['per_class_accuracy'] == [.5, 1., None]
    assert m['mean_per_class_accuracy'] == pytest.approx(.75)
    assert m['confusion_matrix'] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]


def test_mat_roles_and_shape_assertions(tmp_path):
    # Arbitrary field names prove loader has no dependency on remembered .mat keys.
    p = tmp_path / 'example.mat'
    fields = {'x': 'horizontal', 'y': 'vertical', 'visibility': 'seen',
              'action': 'activity', 'split': 'partition', 'nframes': 'length'}
    raw = dict(horizontal=np.ones((3, 7)), vertical=np.ones((3, 7)),
               seen=np.ones((3, 7)), activity='arbitrary_class', partition=9, length=3)
    savemat(p, {'arbitrary_container': raw})
    clip = read_clip(p, fields)
    assert clip.coords.shape == (3, 7, 2) and clip.split_value == 9
    raw['seen'][0, 0] = 2
    savemat(p, raw)
    with pytest.raises(ValueError, match='binary'):
        read_clip(p, fields)
    raw['seen'] = np.ones((7, 3))
    savemat(p, raw)
    with pytest.raises(ValueError, match='time-first'):
        read_clip(p, fields)


def test_interpolation_and_visible_normalization():
    coords = np.array([[[1., 2.]], [[999., 999.]], [[3., 4.]]], dtype=np.float32)
    vis = np.array([[True], [False], [True]])
    clip = Clip('x', coords, vis, 'a', 1)
    result, _ = preprocess(clip, 3, 'none', 'interpolate')
    assert result[1, 0].tolist() == [2., 3.]
    a, _ = preprocess(clip, 3, 'visible_clip', 'interpolate')
    clip.coords[1, 0] = -999
    b, _ = preprocess(clip, 3, 'visible_clip', 'interpolate')
    assert torch.equal(a, b)


def test_validation_ids_fixed_train_only(tmp_path):
    clips = synthetic_clips()
    p = tmp_path/'val.txt'
    ids = validation_split(clips, 1, p, create=True)
    assert ids == validation_split(clips, 1, p)
    assert not ids & {c.clip_id for c in clips if c.split_value == -1}
    with pytest.raises(FileExistsError):
        validation_split(clips, 1, p, create=True)
    p.write_text(next(c.clip_id for c in clips if c.split_value == -1))
    with pytest.raises(ValueError, match='official-train'):
        validation_split(clips, 1, p)


def test_config_rejects_typos_and_bad_values(tmp_path):
    p = tmp_path/'config.yaml'
    p.write_text('train:\n  epohs: 1\n')
    with pytest.raises(TypeError):
        load_config(p)
    p.write_text('data:\n  hidden_policy: unresolved\n')
    with pytest.raises(ValueError, match='hidden_policy'):
        load_config(p)
    p.write_text('data:\n  synthetic: true\ntrain:\n  epochs: "2"\n')
    with pytest.raises(TypeError):
        load_config(p)


def test_training_eval_smoke_and_seed_repeatability(tmp_path):
    config = Path(__file__).parents[1]/'configs/debug.yaml'
    runs = [tmp_path/'one', tmp_path/'two']
    for run in runs:
        main(['--config', str(config), '--set', f'train.output_dir={run}'])
    assert (runs[0]/'metrics.jsonl').read_text() == (runs[1]/'metrics.jsonl').read_text()
    main(['--config', str(config), '--mode', 'eval', '--split', 'test',
          '--checkpoint', str(runs[0]/'best.pt'), '--set', f'train.output_dir={tmp_path}/eval'])
    assert (tmp_path/'eval/predictions.json').exists()
    assert json.loads((tmp_path/'eval/timing.json').read_text())['clips'] > 0
    with pytest.raises(ValueError, match='preprocessing mismatch'):
        main(['--config', str(config), '--mode', 'eval', '--checkpoint', str(runs[0]/'best.pt'),
              '--set', 'data.frames=7', '--set', f'train.output_dir={tmp_path}/bad'])


def test_inspection_quantifies_hidden_coordinates():
    path = Path(__file__).parents[1]/'scripts/inspect_labels.py'
    spec = importlib.util.spec_from_file_location('inspect_labels', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    coords = np.array([[[2., 2.]], [[0., 0.]], [[4., 4.]]], dtype=np.float32)
    clip = Clip('a', coords, np.array([[True], [False], [True]]), 'fixture', 7,
                np.tile([0., 0., 5., 5.], (3, 1)), np.array([10, 10, 3]))
    test = Clip('b', coords, np.ones((3, 1), dtype=bool), 'fixture', 8)
    report = module.inspect([clip, test], 7, 8)
    h = report['hidden_coordinates']
    assert h['zero_pair_fraction'] == 1
    assert h['inside_bbox_fraction'] == 1
    assert h['outside_image_fraction'] == 0
    assert h['interpolation_error_pixels']['mean'] == pytest.approx(np.sqrt(18))
    assert report['counts_by_class_split'] == {'fixture': {'train': 1, 'test': 1}}


def test_inspect_prepare_and_real_format_loader_with_synthetic_mat_files(tmp_path, capsys):
    from penn_action.data import read_catalog
    path = Path(__file__).parents[1]/'scripts/inspect_labels.py'
    spec = importlib.util.spec_from_file_location('inspect_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path/'dataset'
    (root/'labels').mkdir(parents=True)
    for i, clip in enumerate(synthetic_clips()):
        savemat(root/'labels'/f'{clip.clip_id}.mat', {
            'horizontal': clip.coords[..., 0], 'vertical': clip.coords[..., 1],
            'visible': clip.vis, 'class_string': clip.action,
            'partition_code': 77 if clip.split_value == 1 else 88,
            'length': len(clip.coords),
        })
    module.main(['--data-root', str(root)])
    assert 'horizontal' in capsys.readouterr().out
    fields = {'x': 'horizontal', 'y': 'vertical', 'visibility': 'visible',
              'action': 'class_string', 'split': 'partition_code', 'nframes': 'length'}
    schema_path = tmp_path/'schema.json'
    module.main(['--data-root', str(root), '--fields', json.dumps(fields),
                 '--train-value', '77', '--test-value', '88',
                 '--output', str(tmp_path/'report.json'), '--schema-output', str(schema_path)])
    clips, schema = read_catalog(root, schema_path)
    assert schema['classes'] == ['fixture-0', 'fixture-1', 'fixture-2']
    assert schema['joint_count'] == 4
    val_path = tmp_path/'val.txt'
    validation_split(clips, 77, val_path, create=True)
    cfg = tmp_path/'real.yaml'
    import yaml
    cfg.write_text(yaml.safe_dump({'data': {'data_root': str(root), 'schema': str(schema_path),
                                          'val_ids': str(val_path), 'hidden_policy': 'interpolate'},
                                  'train': {'epochs': 1, 'output_dir': str(tmp_path/'run')},
                                  'model': {'input_mode': 'coords_vis', 'hidden_size': 8}}))
    main(['--config', str(cfg)])
    assert (tmp_path/'run/best.pt').exists()
    schema['classes'] = ['invented_wrong_mapping']
    schema_path.write_text(json.dumps(schema))
    with pytest.raises(ValueError, match='class mapping'):
        read_catalog(root, schema_path)


def test_pretraining_keeps_targets_and_zeroes_masks():
    from penn_action.features import pretraining_batch, frame_features
    from penn_action.masking import refill_missing
    b = fixture_batch()
    targets = b['coords'].clone()
    inputs = pretraining_batch(b, 1.0)
    visible = b['vis'] & b['pad_mask'].unsqueeze(-1)
    natural = ~b['vis'] & b['pad_mask'].unsqueeze(-1)
    assert torch.equal(inputs['masked'], visible)
    assert not inputs['coords'][visible].any()
    assert not inputs['coords'][natural].any()  # all observations masked: nothing to refill from
    assert torch.equal(b['coords'], targets)
    assert set(inputs) == {'coords', 'pad_mask', 'vis', 'states', 'masked'}
    assert frame_features(inputs).shape[-1] == 4 * targets.shape[2]
    unmasked = pretraining_batch(b, 0.0)
    assert not unmasked['masked'].any()
    assert torch.equal(unmasked['coords'][visible], targets[visible])
    none = torch.zeros_like(b['vis'])
    assert torch.equal(unmasked['coords'], refill_missing(targets, b['vis'], none, b['pad_mask']))


def test_refill_interpolates_missing_joints_from_unmasked_observations_only():
    from penn_action.masking import refill_missing
    # Frames 1 and 3 are naturally missing; preprocessing filled them from frames 0, 2 and 4.
    coords = torch.tensor([0., 5., 10., 7., 4.]).view(1, 5, 1, 1).repeat(1, 1, 1, 2)
    vis = torch.tensor([True, False, True, False, True]).view(1, 5, 1)
    masked = torch.tensor([False, False, True, False, False]).view(1, 5, 1)
    pad = torch.ones(1, 5, dtype=torch.bool)
    out = refill_missing(coords, vis, masked, pad)
    assert out[0, :, 0, 0].tolist() == [0., 1., 10., 3., 4.]  # frame 2 is zeroed later
    # Endpoints hold the nearest observation; a joint with none left becomes zero.
    coords = torch.tensor([[9., 2., 9.], [6., 6., 6.]]).T.reshape(1, 3, 2, 1).repeat(1, 1, 1, 2)
    vis = torch.tensor([[False, True], [True, False], [False, False]]).view(1, 3, 2)
    masked = torch.tensor([[False, True], [False, False], [False, False]]).view(1, 3, 2)
    out = refill_missing(coords, vis, masked, torch.ones(1, 3, dtype=torch.bool))
    assert out[0, :, 0, 0].tolist() == [2., 2., 2.]
    assert out[0, :, 1, 0].tolist() == [6., 0., 0.]


@pytest.mark.parametrize('policy', ['visible', 'all'])
def test_pretraining_inputs_do_not_depend_on_masked_targets(policy):
    from penn_action.data import fill_hidden
    from penn_action.features import pretraining_batch
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(2, 12, 3, 2)).astype(np.float32)
    vis = rng.random((2, 12, 3)) < 0.6

    def batch_from(r):  # the shared preprocessing interpolates from every visible frame
        coords = np.stack([fill_hidden(r[i], vis[i]) for i in range(len(r))])
        return {'coords': torch.from_numpy(coords), 'vis': torch.from_numpy(vis),
                'pad_mask': torch.ones(2, 12, dtype=torch.bool)}

    b = batch_from(raw)
    inputs = pretraining_batch(b, .4, torch.Generator().manual_seed(1), policy)
    changed = raw.copy()
    changed[(inputs['masked'] & b['vis']).numpy()] += 100
    b2 = batch_from(changed)
    again = pretraining_batch(b2, .4, torch.Generator().manual_seed(1), policy)
    assert not torch.equal(b2['coords'], b['coords'])  # the targets did change
    assert torch.equal(again['masked'], inputs['masked'])
    assert torch.equal(again['coords'], inputs['coords'])


def test_download_extracts_only_label_files(tmp_path):
    import io
    import tarfile
    path = Path(__file__).parents[1]/'scripts/download_data.py'
    spec = importlib.util.spec_from_file_location('download_data', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path/'fixture.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in ['Penn_Action/labels/0001.mat', 'Penn_Action/labels/0002.mat',
                     'Penn_Action/frames/0001/000001.jpg', 'Penn_Action/README']:
            info = tarfile.TarInfo(name)
            info.size = len(name)
            tar.addfile(info, io.BytesIO(name.encode()))
    root = tmp_path/'data'
    module.main(['--data-root', str(root), '--url', archive.as_uri()])
    assert sorted(p.name for p in (root/'labels').iterdir()) == ['0001.mat', '0002.mat']
    assert not (root/'labels.partial').exists()
    with pytest.raises(SystemExit):
        module.main(['--data-root', str(root), '--url', archive.as_uri()])
    module.main(['--data-root', str(root), '--url', archive.as_uri(), '--force'])


def test_never_visible_clips_are_split_out():
    from penn_action.data import split_never_visible
    clips = synthetic_clips()[:3]
    clips[1].vis[:] = False
    kept, excluded = split_never_visible(clips)
    assert [c.clip_id for c in kept] == [clips[0].clip_id, clips[2].clip_id]
    assert excluded == [clips[1].clip_id]


def test_never_visible_joint_is_zero_after_normalization():
    c = Clip('fixture', np.array([[[10., 20.], [999., 999.]],
                                  [[30., 40.], [999., 999.]]], dtype=np.float32),
             np.array([[True, False], [True, False]]), 'fixture', 1)
    for normalization in ('none', 'visible_clip'):
        coords, _ = preprocess(c, 2, normalization, 'interpolate')
        assert not coords[:, 1].any()


def test_model_factory():
    from penn_action.config import ModelConfig
    from penn_action.models import TCNClassifier, TransformerClassifier, build_model
    for mode, channels in [('coords_only', 2), ('coords_vis', 3), ('coords_vis_mask', 4),
                           ('vis_only', 1)]:
        tcn = build_model(ModelConfig(name='tcn', input_mode=mode, hidden_size=8,
                                      num_layers=2, kernel_size=3), 4, 3)
        assert isinstance(tcn, TCNClassifier)
        assert tcn.convs[0].in_channels == 4 * channels and len(tcn.convs) == 2
        tf = build_model(ModelConfig(name='transformer', input_mode=mode, hidden_size=8,
                                     num_layers=2, num_heads=2), 4, 3, max_frames=8)
        assert isinstance(tf, TransformerClassifier)
        assert tf.embed.in_features == 4 * channels and tf.position.shape == (1, 8, 8)
    with pytest.raises(ValueError, match='unknown model'):
        build_model(ModelConfig(name='typo'), 4, 3)


def test_tcn_padding_invariance_batch_independence_and_backward():
    from penn_action.models import TCNClassifier
    seed_everything(2)
    b = fixture_batch()
    model = TCNClassifier(4, 3, hidden_size=8, num_layers=3, kernel_size=3).eval()
    a = model(model_batch(b, 'coords_only'))['logits']
    assert a.shape == (2, 3)
    padded = {**b, 'coords': b['coords'].clone()}
    padded['coords'][~b['pad_mask']] = 10000
    assert torch.allclose(a, model(model_batch(padded, 'coords_only'))['logits'])
    # The shorter clip alone (no padding) must match its batched prediction.
    n = int(b['pad_mask'][0].sum())
    alone = {'coords': b['coords'][:1, :n], 'pad_mask': b['pad_mask'][:1, :n]}
    assert torch.allclose(a[:1], model(alone)['logits'], atol=1e-6)
    model.train()
    torch.nn.functional.cross_entropy(model(model_batch(b, 'coords_only'))['logits'],
                                      b['label']).backward()
    assert model.convs[0].weight.grad is not None
    with pytest.raises(ValueError, match='odd'):
        TCNClassifier(4, 3, kernel_size=4)


def test_config_rejects_invalid_tcn_settings(tmp_path):
    p = tmp_path/'config.yaml'
    for bad in ('kernel_size: 4', 'num_layers: 0', 'dropout: 1.0'):
        p.write_text(f'data:\n  synthetic: true\nmodel:\n  name: tcn\n  {bad}\n')
        with pytest.raises(ValueError):
            load_config(p)
    p.write_text('data:\n  synthetic: true\ntrain:\n  scheduler: step\n')
    with pytest.raises(ValueError, match='scheduler'):
        load_config(p)


def test_standard_mask_policy_masks_any_real_slot_never_padding():
    from penn_action.features import pretraining_batch
    b = fixture_batch()
    real = b['pad_mask'].unsqueeze(-1).expand_as(b['vis'])
    states, masked = joint_states(b['vis'], b['pad_mask'], 1, policy='all')
    assert torch.equal(masked, real)
    assert (states[real] == JointState.MASKED).all()
    inputs = pretraining_batch(b, 1.0, policy='all')
    invisible = ~b['vis'] & real
    assert inputs['masked'][invisible].all() and not inputs['vis'][invisible].any()  # v=0, m=1
    assert not inputs['coords'][real].any()
    with pytest.raises(ValueError, match='policy'):
        joint_states(b['vis'], b['pad_mask'], .3, policy='random')


def test_reconstruction_error_can_score_visible_masked_only():
    states = torch.full((1, 1, 2), int(JointState.MASKED))
    target = torch.zeros(1, 1, 2, 2)
    prediction = torch.tensor([[[[3., 4.], [100., 0.]]]])
    pad = torch.ones(1, 1, dtype=torch.bool)
    vis = torch.tensor([[[True, False]]])
    assert reconstruction_error(prediction, target, states, pad)['masked_joints'] == 2
    assert reconstruction_error(prediction, target, states, pad, vis) == {
        'reconstruction_error': 5., 'masked_joints': 1}


def test_transformer_padding_batch_independence_recon_and_backward():
    from penn_action.models import TransformerClassifier
    seed_everything(3)
    b = fixture_batch()
    model = TransformerClassifier(4, 3, 'coords_vis_mask', hidden_size=8, num_layers=2,
                                  num_heads=2, max_frames=8).eval()
    out = model(model_batch(b, 'coords_vis_mask'))
    assert out['logits'].shape == (2, 3) and out['recon'].shape == b['coords'].shape
    padded = {**b, 'coords': b['coords'].clone()}
    padded['coords'][~b['pad_mask']] = 10000
    assert torch.allclose(out['logits'], model(model_batch(padded, 'coords_vis_mask'))['logits'],
                          atol=1e-5)
    n = int(b['pad_mask'][0].sum())
    alone = {k: v[:1, :n] for k, v in model_batch(b, 'coords_vis_mask').items()}
    assert torch.allclose(out['logits'][:1], model(alone)['logits'], atol=1e-5)
    model.train()
    loss = torch.nn.functional.cross_entropy(model(model_batch(b, 'coords_vis_mask'))['logits'],
                                             b['label'])
    loss.backward()
    assert model.embed.weight.grad is not None
    with pytest.raises(ValueError, match='max_frames'):
        TransformerClassifier(4, 3, max_frames=2)(model_batch(b, 'coords_only'))
    with pytest.raises(ValueError, match='divisible'):
        TransformerClassifier(4, 3, hidden_size=10, num_heads=4)


def test_config_rejects_invalid_pretraining_settings(tmp_path):
    p = tmp_path/'config.yaml'
    base = 'data:\n  synthetic: true\nmodel:\n  name: transformer\n  input_mode: coords_vis_mask\n'
    for train in ('objective: pretrain\n  mask_ratio: 0.0', 'objective: pretrain\n  mask_policy: x',
                  'objective: pretrain\n  init_checkpoint: a.pt', 'objective: finetune'):
        p.write_text(f'{base}train:\n  {train}\n')
        with pytest.raises(ValueError):
            load_config(p)
    p.write_text('data:\n  synthetic: true\nmodel:\n  name: tcn\ntrain:\n  objective: pretrain\n')
    with pytest.raises(ValueError, match='pretraining requires'):
        load_config(p)
    p.write_text(f'{base}  hidden_size: 10\n  num_heads: 4\n')
    with pytest.raises(ValueError, match='num_heads'):
        load_config(p)


def test_every_config_file_loads():
    for path in sorted((Path(__file__).parents[1]/'configs').glob('*.yaml')):
        load_config(path)


@pytest.mark.parametrize('policy', ['visible', 'all'])
def test_pretrain_finetune_eval_pipeline(tmp_path, policy):
    config = Path(__file__).parents[1]/'configs/debug.yaml'
    model = ['--set', 'model.name=transformer', '--set', 'model.input_mode=coords_vis_mask',
             '--set', 'model.num_heads=2', '--set', 'model.num_layers=2']
    pre = tmp_path/'pretrain'
    main(['--config', str(config), *model, '--set', 'train.objective=pretrain',
          '--set', f'train.mask_policy={policy}', '--set', 'train.epochs=2',
          '--set', f'train.output_dir={pre}'])
    rows = [json.loads(line) for line in (pre/'metrics.jsonl').read_text().splitlines()]
    assert all(r['val']['reconstruction_error'] is not None for r in rows)
    assert (pre/'best.pt').exists() and not (pre/'val_predictions.json').exists()
    with pytest.raises(SystemExit):  # pretraining checkpoints are not classifiers
        main(['--config', str(config), *model, '--set', 'train.objective=pretrain',
              '--mode', 'eval', '--checkpoint', str(pre/'best.pt'),
              '--set', f'train.output_dir={tmp_path}/bad_eval'])
    fine = tmp_path/'finetune'
    main(['--config', str(config), *model, '--set', 'train.epochs=2',
          '--set', f'train.init_checkpoint={pre}/best.pt', '--set', f'train.output_dir={fine}'])
    meta = json.loads((fine/'metadata.json').read_text())
    assert meta['init_mask_policy'] == policy
    from penn_action.models import build_model
    from penn_action.train import datasets, load_pretrained
    cfg = load_config(config, [*model[1::2], f'train.init_checkpoint={pre}/best.pt'])
    _, metadata = datasets(cfg)
    net = build_model(cfg.model, metadata['joint_count'], len(metadata['classes']), cfg.data.frames)
    load_pretrained(net, cfg.train.init_checkpoint, cfg, metadata, 'cpu')
    pretrained = torch.load(pre/'best.pt', weights_only=True)['model']
    assert torch.equal(net.state_dict()['embed.weight'], pretrained['embed.weight'])
    assert not torch.equal(net.state_dict()['classifier.weight'], pretrained['classifier.weight'])
    main(['--config', str(config), *model, '--mode', 'eval', '--split', 'test',
          '--checkpoint', str(fine/'best.pt'), '--set', f'train.output_dir={tmp_path}/eval'])
    assert (tmp_path/'eval/predictions.json').exists()
    with pytest.raises(ValueError, match='model config'):
        main(['--config', str(config), *model, '--set', 'model.hidden_size=16',
              '--set', f'train.init_checkpoint={pre}/best.pt',
              '--set', f'train.output_dir={tmp_path}/mismatch'])


def test_training_time_and_reconstruction_summary(tmp_path):
    config = Path(__file__).parents[1]/'configs/debug.yaml'
    model = ['--set', 'model.name=transformer', '--set', 'model.input_mode=coords_vis_mask',
             '--set', 'model.num_heads=2', '--set', 'model.num_layers=2',
             '--set', 'data.debug_per_class=0']  # same clips for every seed
    module = load_script('summarize_training')
    runs = []
    for seed in (1, 2):
        run = tmp_path/f'pretrain{seed}'
        main(['--config', str(config), *model, '--set', 'train.objective=pretrain',
              '--set', 'train.epochs=2', '--set', f'train.seed={seed}',
              '--set', f'train.output_dir={run}'])
        timing = json.loads((run/'timing.json').read_text())
        assert len(timing['train_seconds']) == len(timing['val_seconds']) == 2
        runs.append(run)
    summary = module.summarize(runs)
    assert summary['objective'] == 'pretrain' and summary['seeds'] == [1, 2]
    assert summary['train_minutes']['mean'] > 0
    assert len(summary['val_reconstruction_error']['per_seed']) == 2
    main(['--config', str(config), '--set', 'data.debug_per_class=0',
          '--set', f'train.output_dir={tmp_path}/classify'])
    assert 'val_reconstruction_error' not in module.summarize([tmp_path/'classify'])
    with pytest.raises(ValueError, match='mix objectives'):
        module.summarize([runs[0], tmp_path/'classify'])
    with pytest.raises(ValueError, match='duplicate'):
        module.summarize([runs[0], runs[0]])


def test_seeded_cosine_runs_and_summary(tmp_path):
    config = Path(__file__).parents[1]/'configs/debug.yaml'
    path = Path(__file__).parents[1]/'scripts/summarize_runs.py'
    spec = importlib.util.spec_from_file_location('summarize_runs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evals = []
    for seed in (1, 2):
        run = tmp_path/f'train{seed}'
        main(['--config', str(config), '--set', 'model.name=tcn', '--set', 'train.scheduler=cosine', '--set', 'data.debug_per_class=0',
              '--set', 'train.epochs=3', '--set', f'train.seed={seed}',
              '--set', f'train.output_dir={run}'])
        lrs = [json.loads(line)['lr'] for line in (run/'metrics.jsonl').read_text().splitlines()]
        assert lrs[0] == pytest.approx(1e-3) and lrs[0] > lrs[1] > lrs[2]
        evals.append(tmp_path/f'eval{seed}')
        main(['--config', str(config), '--set', 'model.name=tcn', '--set', 'train.scheduler=cosine', '--set', 'data.debug_per_class=0',
              '--set', 'train.epochs=3', '--set', f'train.seed={seed}', '--mode', 'eval', '--split', 'test',
              '--checkpoint', str(run/'best.pt'), '--set', f'train.output_dir={evals[-1]}'])
    summary = module.summarize(evals)
    assert summary['split'] == 'test' and summary['seeds'] == [1, 2]
    assert len(summary['macro_f1']['per_seed']) == 2 and summary['macro_f1']['std'] is not None
    with pytest.raises(ValueError, match='duplicate'):
        module.summarize([evals[0], evals[0]])


def load_script(name):
    path = Path(__file__).parents[1]/f'scripts/{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_occlusion_is_deterministic_nested_and_reimputed():
    from penn_action.data import fill_hidden, occlude
    clip = synthetic_clips()[0]
    visible = int(clip.vis.sum())
    low, high = occlude(clip.vis, .2, 0, clip.clip_id), occlude(clip.vis, .5, 0, clip.clip_id)
    assert np.array_equal(low, occlude(clip.vis, .2, 0, clip.clip_id))
    assert not (low & ~clip.vis).any() and not (high & ~low).any()
    hidden = lambda rate: int(np.floor(rate * visible + .5))
    assert visible - low.sum() == hidden(.2) and visible - high.sum() == hidden(.5)
    assert not np.array_equal(high, occlude(clip.vis, .5, 1, clip.clip_id))
    assert np.array_equal(occlude(clip.vis, 0., 0, clip.clip_id), clip.vis)
    with pytest.raises(ValueError, match='interpolate'):
        PennActionDataset([clip], ['fixture-0'], occlusion=.3)
    occluded = PennActionDataset([clip], ['fixture-0'], frames=100, normalization='none',
                                 hidden_policy='interpolate', occlusion=.5)[0]
    assert torch.equal(occluded['vis'], torch.from_numpy(high))
    expected = fill_hidden(clip.coords, high)
    expected[:, ~high.any(0)] = 0
    assert torch.allclose(occluded['coords'], torch.from_numpy(expected))


def test_span_occlusion_hides_runs_of_consecutive_frames():
    from penn_action.data import occlude
    vis = np.ones((100, 4), dtype=bool)
    vis[40:50, 2] = False
    low, high = (occlude(vis, rate, 0, 'clip', mode='spans') for rate in (.2, .4))
    assert np.array_equal(high, occlude(vis, .4, 0, 'clip', mode='spans'))
    assert not (low & ~vis).any() and not (high & ~low).any()
    assert vis.sum() - high.sum() == int(np.floor(.4 * vis.sum() + .5))
    hidden = vis & ~high
    runs = [len(run) for j in range(4) for run in
            np.split(np.flatnonzero(hidden[:, j]), np.flatnonzero(np.diff(np.flatnonzero(hidden[:, j])) > 1) + 1)
            if len(run)]
    assert np.median(runs) >= 8
    # The default stays single random joint-frames, so earlier results are unchanged.
    assert np.array_equal(occlude(vis, .4, 0, 'clip'), occlude(vis, .4, 0, 'clip', mode='frames'))
    with pytest.raises(ValueError, match='mode'):
        occlude(vis, .4, 0, 'clip', mode='limbs')


def test_visibility_dropout_hides_training_joints_only():
    from penn_action.train import datasets
    clip = synthetic_clips()[0]
    with pytest.raises(ValueError, match='interpolate'):
        PennActionDataset([clip], ['fixture-0'], vis_dropout=.5)
    ds = PennActionDataset([clip], ['fixture-0'], frames=100, normalization='none',
                           hidden_policy='interpolate', vis_dropout=.5)
    np.random.seed(0)
    draws = [ds[0]['vis'].numpy() for _ in range(6)]
    assert all(not (d & ~clip.vis).any() for d in draws)
    assert len({d.tobytes() for d in draws}) > 1
    assert all(clip.vis.sum() - d.sum() <= np.floor(.5 * clip.vis.sum() + .5) for d in draws)
    config = Path(__file__).parents[1]/'configs/debug.yaml'
    cfg = load_config(config, ['data.hidden_policy=interpolate', 'train.vis_dropout=0.5'])
    split = datasets(cfg)[0]
    assert (split['train'].vis_dropout, split['val'].vis_dropout, split['test'].vis_dropout) == (.5, 0, 0)
    with pytest.raises(ValueError, match='vis_dropout'):
        load_config(config, ['data.hidden_policy=interpolate', 'train.vis_dropout=0.5',
                             'train.objective=pretrain', 'model.name=transformer',
                             'model.input_mode=coords_vis_mask'])
    with pytest.raises(ValueError, match='vis_dropout'):
        load_config(config, ['data.hidden_policy=raw', 'train.vis_dropout=0.5'])


def test_mcnemar_and_tertiles():
    from penn_action.analysis import mcnemar, occlusion_tertiles, tertile_metrics
    rec = lambda c, y, p: {'clip_id': c, 'target': y, 'predicted': p}
    a = [rec('1', 0, 0), rec('2', 1, 1), rec('3', 0, 0), rec('4', 1, 0)]
    b = [rec('1', 0, 1), rec('2', 1, 0), rec('3', 0, 0), rec('4', 1, 0)]
    result = mcnemar(a, b)
    assert (result['only_a_correct'], result['only_b_correct']) == (2, 0)
    assert result['p_value'] == pytest.approx(.5) and result['accuracy_a'] == .75
    assert mcnemar(a, a)['p_value'] == 1.0
    with pytest.raises(ValueError, match='identical set'):
        mcnemar(a, b[:3])
    rates = {str(i): r for i, r in enumerate([.5, .1, .3, .3, 0., .9, .2])}
    actions = {str(i): 'x' if i < 4 else 'y' for i in range(7)}
    tertile = occlusion_tertiles(rates, actions)
    assert [tertile[str(i)] for i in range(7)] == [2, 0, 0, 1, 0, 2, 1]
    metrics = tertile_metrics([rec(c, 0, 0) for c in rates], tertile, 2)
    assert [m['clips'] for m in metrics] == [3, 2, 2] and metrics[0]['accuracy'] == 1.0


def test_occlusion_sweep_curve_tertiles_and_mcnemar_scripts(tmp_path, capsys):
    config = Path(__file__).parents[1]/'configs/debug.yaml'
    common = ['--set', 'data.debug_per_class=0', '--set', 'data.hidden_policy=interpolate',
              '--set', 'train.epochs=1']
    for name in ('transformer', 'tcn'):
        for seed in (1, 2):
            args = ['--config', str(config), *common, '--set', f'model.name={name}',
                    '--set', f'train.seed={seed}']
            main([*args, '--set', f'train.output_dir={tmp_path}/{name}{seed}'])
            main([*args, '--mode', 'eval', '--split', 'test',
                  '--checkpoint', f'{tmp_path}/{name}{seed}/best.pt',
                  '--set', f'train.output_dir={tmp_path}/{name}{seed}-test'])
            load_script('occlusion_sweep').main([
                '--config', str(config), '--checkpoint', f'{tmp_path}/{name}{seed}/best.pt',
                '--seed', str(seed), '--output', f'{tmp_path}/{name}{seed}-occ',
                '--rates', '0', '0.3', *common, '--set', f'model.name={name}'])
    occ = json.loads((tmp_path/'tcn1-occ/rate_0.30/metadata.json').read_text())
    assert (occ['occlusion_rate'], occ['occlusion_seed']) == (.3, 0)
    tests = {name: [str(tmp_path/f'{name}{s}-test') for s in (1, 2)] for name in ('transformer', 'tcn')}
    sweeps = {name: [str(tmp_path/f'{name}{s}-occ') for s in (1, 2)] for name in ('transformer', 'tcn')}

    curve = load_script('occlusion_curve').analyze(list(sweeps.items()))
    assert curve['rates'] == [0.0, 0.3] and curve['models']['tcn']['0.30']['seeds'] == [1, 2]
    clean = json.loads((tmp_path/'tcn1-test/metrics.jsonl').read_text())
    assert curve['models']['tcn']['0.00']['accuracy']['per_seed'][0] == clean['accuracy']
    load_script('occlusion_curve').main(['--model', 'TF', *sweeps['transformer'], '--model', 'TCN',
                                         *sweeps['tcn'], '--csv', str(tmp_path/'curve.csv')])
    assert len((tmp_path/'curve.csv').read_text().splitlines()) == 1 + 2 * 2 * 3
    assert curve['occlusion_mode'] == 'frames'
    for seed in (1, 2):
        load_script('occlusion_sweep').main([
            '--config', str(config), '--checkpoint', f'{tmp_path}/tcn{seed}/best.pt',
            '--seed', str(seed), '--output', f'{tmp_path}/tcn{seed}-spans', '--rates', '0', '0.3',
            '--occlusion-mode', 'spans', *common, '--set', 'model.name=tcn'])
    spans = [str(tmp_path/f'tcn{s}-spans') for s in (1, 2)]
    assert json.loads((tmp_path/'tcn1-spans/rate_0.30/metadata.json').read_text())[
        'occlusion_mode'] == 'spans'
    assert load_script('occlusion_curve').analyze([('tcn', spans)])['occlusion_mode'] == 'spans'
    with pytest.raises(ValueError, match='occlusion settings'):
        load_script('occlusion_curve').analyze([('transformer', sweeps['transformer']), ('tcn', spans)])

    tertiles = load_script('occlusion_tertiles').analyze(list(tests.items()))
    assert sum(t['clips'] for t in tertiles['tertiles'].values()) == 12
    assert tertiles['tertiles']['low']['mean_rate'] <= tertiles['tertiles']['high']['mean_rate']
    assert tertiles['models']['transformer']['high']['macro_f1']['std'] is not None
    with pytest.raises(ValueError, match='without controlled occlusion'):
        load_script('occlusion_tertiles').analyze(
            [('transformer', [f'{d}/rate_0.30' for d in sweeps['transformer']])])

    mc = load_script('mcnemar')
    result = mc.compare('transformer', tests['transformer'], 'tcn', tests['tcn'])
    assert sorted(result['per_seed']) == [1, 2] and result['per_seed'][1]['clips'] == 12
    with pytest.raises(ValueError, match='occlusion settings'):
        mc.compare('transformer', tests['transformer'], 'tcn', [f'{d}/rate_0.30' for d in sweeps['tcn']])
    mc.main(['--a', 'TF', *tests['transformer'], '--b', 'TCN', *tests['tcn']])
    assert 'p' in capsys.readouterr().out
