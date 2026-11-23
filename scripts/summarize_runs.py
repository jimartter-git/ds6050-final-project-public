"""Summarize eval runs (one per seed) as mean ± sample std of the headline metrics."""
import argparse
import json
from pathlib import Path
from penn_action.analysis import by_seed, check_comparable, load_eval_run, mean_std

METRICS = ('accuracy', 'macro_f1', 'mean_per_class_accuracy')


def summarize(directories):
    runs = [load_eval_run(d) for d in directories]
    split, (rate, _) = check_comparable(runs)
    seeds = list(by_seed(runs, 'runs'))
    summary = {'split': split, 'occlusion_rate': rate, 'runs': len(runs), 'seeds': seeds,
               'devices': sorted({meta['device'] for meta, _, _ in runs})}
    for name in METRICS:
        summary[name] = mean_std([metrics[name] for _, metrics, _ in runs])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', help='eval output directories, one per seed')
    parser.add_argument('--output', help='optional JSON file for the summary')
    args = parser.parse_args(argv)
    summary = summarize(args.runs)
    print(f"{summary['split']} split, {summary['runs']} seeds {summary['seeds']}, "
          f"device {', '.join(summary['devices'])}, occlusion {summary['occlusion_rate']}")
    for name in METRICS:
        m = summary[name]
        std = f" ± {m['std']:.4f}" if m['std'] is not None else ''
        values = ', '.join(f'{v:.4f}' for v in m['per_seed'])
        print(f'{name:24} {m["mean"]:.4f}{std}   ({values})')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
