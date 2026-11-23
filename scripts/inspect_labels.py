"""Inspect every annotation; discovery first, then explicit schema and split semantics."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from penn_action.data import label_paths, mat_fields, read_clip


def fraction(num, den):
    return float(num / den) if den else None


def summary(values):
    values = np.asarray(values)
    return {'count': int(values.size), 'min': float(values.min()), 'max': float(values.max()),
            'mean': float(values.mean()),
            'quantiles_0_25_50_75_100': np.quantile(values, [0, .25, .5, .75, 1]).tolist()}


def inspect(clips, train_value, test_value, examples=3):
    if train_value == test_value:
        raise ValueError('train/test values must differ')
    if {c.split_value for c in clips} != {train_value, test_value}:
        raise ValueError('observed splits do not match supplied train/test values')
    joints = {c.coords.shape[1] for c in clips}
    if len(joints) != 1:
        raise ValueError('inconsistent joint counts')
    counts = defaultdict(Counter)
    rates = defaultdict(lambda: [0, 0])
    joint_visible = np.zeros(next(iter(joints)), dtype=np.int64)
    joint_total = np.zeros_like(joint_visible)
    hidden_pairs, interpolated_errors, examples_out = [], [], []
    outside = inside_bbox = hidden_total = image_den = bbox_den = 0
    lengths, invisible_fractions = [], []
    for clip in clips:
        split = 'train' if clip.split_value == train_value else 'test'
        counts[clip.action][split] += 1
        lengths.append(len(clip.coords))
        v, total = int(clip.vis.sum()), clip.vis.size
        invisible_fractions.append(1 - v / total)
        for name in ('overall', f'class:{clip.action}', f'split:{split}'):
            rates[name][0] += v
            rates[name][1] += total
        joint_visible += clip.vis.sum(0)
        joint_total += len(clip.vis)
        hidden = ~clip.vis
        values = clip.coords[hidden]
        hidden_pairs.extend(map(tuple, values.tolist()))
        hidden_total += len(values)
        if clip.dimensions is not None:
            h, w, _ = clip.dimensions
            outside += int(((values[:, 0] < 0) | (values[:, 0] > w) |
                            (values[:, 1] < 0) | (values[:, 1] > h)).sum())
            image_den += len(values)
        if clip.bbox is not None:
            # Explicit xyxy convention; validated before this report is enabled.
            boxes = np.broadcast_to(clip.bbox[:, None, :], (*clip.vis.shape, 4))[hidden]
            inside_bbox += int(((values[:, 0] >= boxes[:, 0]) & (values[:, 0] <= boxes[:, 2]) &
                                (values[:, 1] >= boxes[:, 1]) & (values[:, 1] <= boxes[:, 3])).sum())
            bbox_den += len(values)
        for j in range(clip.coords.shape[1]):
            observed = np.flatnonzero(clip.vis[:, j])
            for t in np.flatnonzero(hidden[:, j]):
                left, right = observed[observed < t], observed[observed > t]
                if len(left) and len(right):
                    a, b = left[-1], right[0]
                    predicted = clip.coords[a, j] + (t - a) / (b - a) * (clip.coords[b, j] - clip.coords[a, j])
                    interpolated_errors.append(float(np.linalg.norm(clip.coords[t, j] - predicted)))
        if len(examples_out) < examples and hidden.any():
            t = int(np.flatnonzero(hidden.any(1))[0])
            examples_out.append({'clip_id': clip.clip_id, 'frame_index': t,
                                 'coords': clip.coords[t].tolist(), 'visibility': clip.vis[t].tolist()})
    pair_counts = Counter(hidden_pairs)
    common = pair_counts.most_common(10)
    histogram, edges = np.histogram(invisible_fractions, bins=np.linspace(0, 1, 11))
    return {
        'clips': len(clips), 'classes': sorted(counts), 'joint_count': next(iter(joints)),
        'counts_by_class_split': {k: dict(v) for k, v in sorted(counts.items())},
        'clip_lengths': summary(lengths),
        'visibility_rates': {k: fraction(*v) for k, v in sorted(rates.items())},
        'visibility_rate_by_joint_index': (joint_visible / joint_total).tolist(),
        'invisible_fraction_per_clip': {c.clip_id: f for c, f in zip(clips, invisible_fractions)},
        'invisible_fraction_histogram': {'edges': edges.tolist(), 'counts': histogram.tolist()},
        'hidden_coordinates': {
            'count': hidden_total, 'zero_pair_fraction': fraction(pair_counts[(0., 0.)], hidden_total),
            'most_common_exact_pairs': [{'xy': list(xy), 'count': n,
                                         'fraction': fraction(n, hidden_total)} for xy, n in common],
            'outside_image_fraction': fraction(outside, image_den),
            'outside_image_denominator': image_den,
            'inside_bbox_fraction': fraction(inside_bbox, bbox_den),
            'inside_bbox_denominator': bbox_den,
            'interpolation_error_pixels': summary(interpolated_errors) if interpolated_errors else None,
        }, 'examples': examples_out,
        'notes': ['No automatic plausible/placeholder verdict. Review this evidence before choosing hidden_policy.',
                  'Interpolation comparisons include only hidden joints bracketed by visible frames.',
                  'Joint names are not inferred from coordinates; rates use file column indices.'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--fields', help='JSON mapping roles to file fields; see README')
    parser.add_argument('--train-value', type=int)
    parser.add_argument('--test-value', type=int)
    parser.add_argument('--bbox-format', choices=['xyxy'], help='explicitly confirm bbox convention')
    parser.add_argument('--dimensions-order', choices=['hwt'], help='explicitly confirm height,width,time order')
    parser.add_argument('--output', default='outputs/label_inspection.json')
    parser.add_argument('--schema-output', default='metadata/penn_action_schema.json')
    args = parser.parse_args(argv)
    paths = label_paths(args.data_root)
    if args.fields is None:
        # Discovery intentionally reads every file, without assuming semantic field names.
        layouts = Counter()
        for path in paths:
            fields = mat_fields(path)
            layout = {k: {'shape': list(np.asarray(v).shape), 'dtype': str(np.asarray(v).dtype)}
                      for k, v in fields.items()}
            layouts[json.dumps(layout, sort_keys=True)] += 1
        print(json.dumps({'files': len(paths), 'layouts': [{'count': n, 'fields': json.loads(k)}
                                                        for k, n in layouts.items()]}, indent=2))
        print('Map actual field roles with --fields and confirm official split semantics before full inspection.')
        return
    if args.train_value is None or args.test_value is None:
        parser.error('full inspection requires explicit --train-value and --test-value')
    fields = json.loads(args.fields)
    if 'bbox' in fields and not args.bbox_format:
        parser.error('bbox mapping requires --bbox-format xyxy after verifying its convention')
    if 'dimensions' in fields and not args.dimensions_order:
        parser.error('dimensions mapping requires --dimensions-order hwt after verifying its convention')
    clips = [read_clip(p, fields, args.dimensions_order) for p in paths]
    if 'bbox' in fields:
        for clip in clips:
            if (clip.bbox[:, 2:] < clip.bbox[:, :2]).any():
                raise ValueError(f'{clip.clip_id}: bbox does not satisfy xyxy ordering')
    report = inspect(clips, args.train_value, args.test_value)
    schema = {'fields': fields, 'train_value': args.train_value, 'test_value': args.test_value,
              'classes': report['classes'], 'joint_count': report['joint_count'],
              'bbox_format': args.bbox_format, 'dimensions_order': args.dimensions_order}
    for filename, value in ((args.output, report), (args.schema_output, schema)):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    printable = {k: v for k, v in report.items() if k != 'invisible_fraction_per_clip'}
    print(json.dumps(printable, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
