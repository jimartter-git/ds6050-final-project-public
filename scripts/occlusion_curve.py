"""RQ3 controlled occlusion curves: metrics per occlusion rate, mean ± std over seeds.

Each DIR is an occlusion_sweep.py output (one per seed). Writes JSON and, with --csv, a
long-format table (model, rate, metric, mean, std) for plotting.

    python scripts/occlusion_curve.py --model MT1 outputs/mt1_s42-occlusion ... \
        --model MVA outputs/mva_s42-occlusion ... --output outputs/rq3_curve.json --csv outputs/rq3_curve.csv
"""
import argparse
import csv
import json
from pathlib import Path
from penn_action.analysis import by_seed, check_comparable, load_eval_run, mean_std

METRICS = ('accuracy', 'macro_f1', 'mean_per_class_accuracy')


def analyze(models):
    by_rate = {}
    for label, dirs in models:
        for d in dirs:
            subdirs = sorted(Path(d).glob('rate_*'))
            if not subdirs:
                raise ValueError(f'{d}: no rate_* eval directories; run occlusion_sweep.py')
            for sub in subdirs:
                run = load_eval_run(sub)
                by_rate.setdefault(run[0].get('occlusion_rate', 0.0), {}).setdefault(
                    label, []).append(run)
    seeds_used = {(meta.get('occlusion_seed', 0)) for group in by_rate.values()
                  for runs in group.values() for meta, _, _ in runs}
    if len(seeds_used) != 1:
        raise ValueError(f'models were occluded with different seeds: {sorted(seeds_used)}')
    labels = [label for label, _ in models]
    result = {'occlusion_seed': seeds_used.pop(), 'rates': sorted(by_rate), 'models': {}}
    modes = {meta.get('occlusion_mode', 'frames') for group in by_rate.values()
             for runs in group.values() for meta, _, _ in runs}
    if len(modes) != 1:
        raise ValueError(f'runs use different occlusion settings: {sorted(modes)}')
    result['occlusion_mode'] = modes.pop()
    for rate in result['rates']:
        group = by_rate[rate]
        if sorted(group) != sorted(labels):
            raise ValueError(f'rate {rate}: missing models {sorted(set(labels) - set(group))}')
        split, _ = check_comparable([r for runs in group.values() for r in runs])
        result['split'] = split
        for label in labels:
            runs = by_seed(group[label], f'{label} at rate {rate}')
            result['models'].setdefault(label, {})[f'{rate:.2f}'] = {
                'seeds': sorted(runs),
                **{m: mean_std([runs[s][1][m] for s in sorted(runs)]) for m in METRICS}}
    seed_sets = {json.dumps(v['seeds']) for model in result['models'].values()
                 for v in model.values()}
    if len(seed_sets) != 1:
        raise ValueError('every model and rate needs the same training seeds')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', nargs='+', action='append', required=True,
                        metavar=('LABEL', 'DIR'), help='label followed by sweep dirs (one per seed)')
    parser.add_argument('--output', help='optional JSON file for the results')
    parser.add_argument('--csv', help='optional long-format CSV for plotting')
    args = parser.parse_args(argv)
    if any(len(m) < 2 for m in args.model):
        parser.error('each --model takes a label followed by sweep directories')
    result = analyze([(m[0], m[1:]) for m in args.model])
    for metric in ('accuracy', 'macro_f1'):
        print(f"\n{metric} by occlusion rate ({result['split']}, mean ± std over seeds)")
        print(f"  {'model':8} " + ' '.join(f'{r:>13.2f}' for r in result['rates']))
        for label, model in result['models'].items():
            cells = [f"{v[metric]['mean']:.3f}" + (f"±{v[metric]['std']:.3f}"
                                                   if v[metric]['std'] is not None else '')
                     for v in model.values()]
            print(f'  {label:8} ' + ' '.join(f'{c:>13}' for c in cells))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['model', 'rate', 'metric', 'mean', 'std'])
            for label, model in result['models'].items():
                for rate, values in model.items():
                    for metric in METRICS:
                        writer.writerow([label, rate, metric, values[metric]['mean'],
                                         values[metric]['std']])


if __name__ == '__main__':
    main()
