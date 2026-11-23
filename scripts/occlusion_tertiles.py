"""RQ3 natural occlusion: metrics by within-class occlusion tertile, mean ± std over seeds.

A clip's natural occlusion rate is the fraction of annotated-invisible joint slots in the
frames the model sees (after the shared temporal resampling). Within each action, clips
are ranked by that rate (ties by clip ID) and split into low/medium/high thirds.

    python scripts/occlusion_tertiles.py --model MT1 outputs/mt1_s42-test ... \
        --model MVA outputs/mva_s42-test ... --output outputs/rq3_tertiles.json

Gaps are reported relative to the first --model.
"""
import argparse
import json
from pathlib import Path
from penn_action.analysis import (by_seed, check_comparable, load_eval_run, mean_std,
                                  occlusion_tertiles, tertile_metrics)
from penn_action.config import load_config
from penn_action.train import datasets

NAMES = ('low', 'medium', 'high')
METRICS = ('accuracy', 'macro_f1', 'mean_per_class_accuracy')


def natural_rates(run_dir, split):
    """Per-clip natural occlusion rate and action, rebuilt from the run's saved config."""
    cfg = load_config(Path(run_dir) / 'config.yaml')
    ds, _ = datasets(cfg)
    dataset = ds[split]
    rates, actions = {}, {}
    for i, clip in enumerate(dataset.clips):
        vis = dataset[i]['vis']
        rates[clip.clip_id] = 1 - vis.float().mean().item()
        actions[clip.clip_id] = clip.action
    return rates, actions, len(dataset.classes)


def analyze(models):
    runs = {label: [load_eval_run(d) for d in dirs] for label, dirs in models}
    split, (rate, _) = check_comparable([r for group in runs.values() for r in group])
    if rate:
        raise ValueError('natural-occlusion tertiles need runs without controlled occlusion')
    frames = {load_config(Path(d) / 'config.yaml').data.frames for _, dirs in models for d in dirs}
    if len(frames) != 1:
        raise ValueError(f'runs resample to different frame counts: {sorted(frames)}')
    rates, actions, num_classes = natural_rates(models[0][1][0], split)
    tertile = occlusion_tertiles(rates, actions)
    result = {'split': split, 'tertiles': {}, 'models': {}}
    for t, name in enumerate(NAMES):
        values = [rates[c] for c in tertile if tertile[c] == t]
        result['tertiles'][name] = {'clips': len(values), 'min_rate': min(values),
                                    'max_rate': max(values),
                                    'mean_rate': sum(values) / len(values)}
    for label, group in runs.items():
        per_seed = {seed: tertile_metrics(records, tertile, num_classes)
                    for seed, (_, _, records) in by_seed(group, label).items()}
        if any(r['clip_id'] not in tertile for _, _, records in group for r in records):
            raise ValueError(f'{label}: predictions include clips outside the analyzed split')
        result['models'][label] = {
            'seeds': sorted(per_seed),
            **{name: {m: mean_std([per_seed[s][t][m] for s in sorted(per_seed)])
                      for m in METRICS} for t, name in enumerate(NAMES)}}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', nargs='+', action='append', required=True,
                        metavar=('LABEL', 'DIR'), help='label followed by eval dirs (one per seed)')
    parser.add_argument('--output', help='optional JSON file for the results')
    args = parser.parse_args(argv)
    if any(len(m) < 2 for m in args.model):
        parser.error('each --model takes a label followed by eval directories')
    result = analyze([(m[0], m[1:]) for m in args.model])
    print(f"{result['split']} split; natural occlusion rate per tertile:")
    for name, t in result['tertiles'].items():
        print(f"  {name:7} {t['clips']:5} clips  rate {t['min_rate']:.3f}-{t['max_rate']:.3f} "
              f"(mean {t['mean_rate']:.3f})")
    reference = args.model[0][0]
    for metric in ('accuracy', 'macro_f1'):
        print(f'\n{metric} (mean ± std over seeds; gap vs {reference})')
        for label, model in result['models'].items():
            cells = []
            for name in NAMES:
                m, ref = model[name][metric], result['models'][reference][name][metric]
                std = f"±{m['std']:.3f}" if m['std'] is not None else ''
                gap = f" ({m['mean'] - ref['mean']:+.3f})" if label != reference else ''
                cells.append(f"{m['mean']:.3f}{std}{gap}")
            print(f'  {label:8} ' + '   '.join(f'{n}: {c}' for n, c in zip(NAMES, cells)))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
