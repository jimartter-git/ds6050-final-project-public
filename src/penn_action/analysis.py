"""Post-hoc analysis of eval runs: seed summaries, McNemar's test, occlusion tertiles."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from .metrics import classification_metrics


def load_eval_run(directory):
    """Return (metadata, metrics, predictions) for one eval output directory."""
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('mode') != 'eval':
        raise ValueError(f'{directory}: not an eval run; pass eval output directories')
    lines = (directory / 'metrics.jsonl').read_text().splitlines()
    if len(lines) != 1:
        raise ValueError(f'{directory}: expected one eval metrics line')
    records = json.loads((directory / 'predictions.json').read_text())
    return metadata, json.loads(lines[0]), records


def check_comparable(runs):
    """Runs must score the same split, clips and occlusion setting, each from its own seed."""
    splits = {metrics['split'] for _, metrics, _ in runs}
    if len(splits) != 1:
        raise ValueError(f'runs evaluate different splits: {sorted(splits)}')
    if len({json.dumps(meta['split_ids_hash'], sort_keys=True) for meta, _, _ in runs}) != 1:
        raise ValueError('runs use different clip splits')
    occlusion = {(meta.get('occlusion_rate', 0.0), meta.get('occlusion_seed', 0))
                 for meta, _, _ in runs}
    modes = {meta.get('occlusion_mode', 'frames') for meta, _, _ in runs}
    if len(occlusion) != 1 or len(modes) != 1:
        raise ValueError(f'runs use different occlusion settings: {sorted(occlusion)}, '
                         f'{sorted(modes)}')
    return splits.pop(), occlusion.pop()


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) > 1 else None,
            'per_seed': values.tolist()}


def by_seed(runs, label):
    seeds = [meta['checkpoint_seed'] for meta, _, _ in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'{label}: duplicate checkpoint seeds {seeds}')
    return dict(zip(seeds, runs))


def mcnemar(records_a, records_b):
    """Exact two-sided McNemar test on paired per-clip top-1 correctness."""
    a = {r['clip_id']: r for r in records_a}
    b = {r['clip_id']: r for r in records_b}
    if a.keys() != b.keys() or len(a) != len(records_a):
        raise ValueError('McNemar needs predictions for the identical set of clips')
    if any(a[c]['target'] != b[c]['target'] for c in a):
        raise ValueError('runs disagree on clip labels')
    right_a = {c: a[c]['predicted'] == a[c]['target'] for c in a}
    right_b = {c: b[c]['predicted'] == b[c]['target'] for c in a}
    only_a = sum(right_a[c] and not right_b[c] for c in a)
    only_b = sum(right_b[c] and not right_a[c] for c in a)
    discordant = only_a + only_b
    p = binomtest(only_a, discordant, 0.5).pvalue if discordant else 1.0
    return {'clips': len(a), 'accuracy_a': sum(right_a.values()) / len(a),
            'accuracy_b': sum(right_b.values()) / len(a),
            'only_a_correct': only_a, 'only_b_correct': only_b, 'p_value': float(p)}


def occlusion_tertiles(rates, actions):
    """Within each action, rank clips by natural occlusion rate (ties by clip ID) and split
    the ranking into low/medium/high thirds of near-equal size. Returns {clip_id: 0|1|2}."""
    tertile = {}
    for action in sorted(set(actions.values())):
        ids = sorted((c for c in actions if actions[c] == action), key=lambda c: (rates[c], c))
        if len(ids) < 3:
            raise ValueError(f'{action}: need at least 3 clips to form tertiles')
        for t, part in enumerate(np.array_split(np.array(ids, dtype=object), 3)):
            tertile.update({c: t for c in part})
    return tertile


def tertile_metrics(records, tertile, num_classes):
    """Accuracy, macro-F1 and mean per-class accuracy within each tertile."""
    result = []
    for t in range(3):
        subset = [r for r in records if tertile[r['clip_id']] == t]
        metrics = classification_metrics([r['target'] for r in subset],
                                         [r['predicted'] for r in subset], num_classes)
        result.append({k: metrics[k] for k in
                       ('accuracy', 'macro_f1', 'mean_per_class_accuracy')} | {'clips': len(subset)})
    return result
