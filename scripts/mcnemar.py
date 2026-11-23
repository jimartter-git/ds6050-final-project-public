"""Exact McNemar test between two models, one test per matched training seed.

Seeds are never pooled: each seed's pair of runs scores the same clips once.

    python scripts/mcnemar.py --a MT0 outputs/mt0_s42-test outputs/mt0_s43-test ... \
        --b MT1 outputs/mt1_s42-test outputs/mt1_s43-test ...
"""
import argparse
import json
from pathlib import Path
from penn_action.analysis import by_seed, check_comparable, load_eval_run, mcnemar


def compare(label_a, dirs_a, label_b, dirs_b):
    runs_a = [load_eval_run(d) for d in dirs_a]
    runs_b = [load_eval_run(d) for d in dirs_b]
    split, (rate, _) = check_comparable(runs_a + runs_b)
    a, b = by_seed(runs_a, label_a), by_seed(runs_b, label_b)
    if a.keys() != b.keys():
        raise ValueError(f'seeds differ: {label_a} {sorted(a)} vs {label_b} {sorted(b)}')
    return {'a': label_a, 'b': label_b, 'split': split, 'occlusion_rate': rate,
            'per_seed': {seed: mcnemar(a[seed][2], b[seed][2]) for seed in sorted(a)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', nargs='+', required=True, metavar=('LABEL', 'DIR'))
    parser.add_argument('--b', nargs='+', required=True, metavar=('LABEL', 'DIR'))
    parser.add_argument('--output', help='optional JSON file for the results')
    args = parser.parse_args(argv)
    if len(args.a) < 2 or len(args.b) < 2:
        parser.error('--a and --b take a label followed by eval directories')
    result = compare(args.a[0], args.a[1:], args.b[0], args.b[1:])
    print(f"{result['a']} vs {result['b']}, {result['split']} split, "
          f"occlusion {result['occlusion_rate']}")
    print(f"{'seed':>6} {'acc ' + result['a']:>12} {'acc ' + result['b']:>12} "
          f"{'only ' + result['a']:>12} {'only ' + result['b']:>12} {'p':>10}")
    for seed, r in result['per_seed'].items():
        print(f"{seed:>6} {r['accuracy_a']:>12.4f} {r['accuracy_b']:>12.4f} "
              f"{r['only_a_correct']:>12} {r['only_b_correct']:>12} {r['p_value']:>10.4g}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
