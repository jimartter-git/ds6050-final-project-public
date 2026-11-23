"""Summarize training runs (one per seed): wall-clock training time and the kept epoch.

For pretraining runs it also reports the validation reconstruction error on masked,
originally visible joints at the kept checkpoint, the reconstruction metric for RQ2b.
Time covers the training and validation passes of every epoch.
"""
import argparse
import json
from pathlib import Path
import yaml
from penn_action.analysis import mean_std


def load_training_run(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('mode') != 'train':
        raise ValueError(f'{directory}: not a training run; pass training output directories')
    if not (directory / 'timing.json').exists():
        raise ValueError(f'{directory}: no timing.json; the run predates timing')
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    timing = json.loads((directory / 'timing.json').read_text())
    return metadata, config, rows, timing


def summarize(directories):
    runs = [load_training_run(d) for d in directories]
    objectives = {config['train']['objective'] for _, config, _, _ in runs}
    if len(objectives) != 1:
        raise ValueError(f'runs mix objectives: {sorted(objectives)}')
    if len({json.dumps(meta['split_ids_hash'], sort_keys=True) for meta, *_ in runs}) != 1:
        raise ValueError('runs use different clip splits')
    seeds = [meta['seed'] for meta, *_ in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'duplicate seeds: {seeds}')
    pretrain = objectives.pop() == 'pretrain'
    # Same rule as train.py: the first epoch with the best validation score is kept.
    if pretrain:
        kept = [min(rows, key=lambda r: r['val']['reconstruction_error']) for _, _, rows, _ in runs]
    else:
        kept = [max(rows, key=lambda r: r['val']['macro_f1']) for _, _, rows, _ in runs]
    summary = {'objective': 'pretrain' if pretrain else 'classify', 'runs': len(runs),
               'seeds': seeds, 'devices': sorted({meta['device'] for meta, *_ in runs}),
               'epochs': [len(rows) for _, _, rows, _ in runs],
               'kept_epoch': [row['epoch'] for row in kept],
               'train_minutes': mean_std([(sum(t['train_seconds']) + sum(t['val_seconds'])) / 60
                                          for *_, t in runs])}
    if pretrain:
        summary['val_reconstruction_error'] = mean_std(
            [row['val']['reconstruction_error'] for row in kept])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', help='training output directories, one per seed')
    parser.add_argument('--output', help='optional JSON file for the summary')
    args = parser.parse_args(argv)
    summary = summarize(args.runs)
    print(f"{summary['objective']}, {summary['runs']} seeds {summary['seeds']}, "
          f"device {', '.join(summary['devices'])}, kept epochs {summary['kept_epoch']}")
    names = ['train_minutes'] + (['val_reconstruction_error'] if 'val_reconstruction_error'
                                 in summary else [])
    for name in names:
        m = summary[name]
        std = f" ± {m['std']:.4f}" if m['std'] is not None else ''
        values = ', '.join(f'{v:.4f}' for v in m['per_seed'])
        print(f'{name:26} {m["mean"]:.4f}{std}   ({values})')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
