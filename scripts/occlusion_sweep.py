"""RQ3 controlled occlusion: evaluate one checkpoint at several test-time occlusion rates.

Writes one eval directory per rate (<output>/rate_0.00, rate_0.10, ...). Use the same
--occlusion-seed for every model so all arms see identical hidden joints.

    python scripts/occlusion_sweep.py --config configs/mva_finetune.yaml \
        --checkpoint outputs/mva_s42/best.pt --seed 42 --output outputs/mva_s42-occlusion
"""
import argparse
from pathlib import Path
from penn_action.train import main as run

DEFAULT_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seed', type=int, required=True, help='the checkpoint training seed')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--rates', type=float, nargs='+', default=DEFAULT_RATES)
    parser.add_argument('--occlusion-seed', type=int, default=0)
    parser.add_argument('--occlusion-mode', choices=['frames', 'spans'], default='frames',
                        help='frames: single joint-frames; spans: 8-16 consecutive frames')
    parser.add_argument('--set', action='append', default=[], metavar='section.key=value',
                        help='the same config overrides used for training')
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error(f'{args.output} already exists')
    for rate in args.rates:
        run(['--config', args.config, '--mode', 'eval', '--split', args.split,
             '--checkpoint', args.checkpoint, '--occlusion', str(rate),
             '--occlusion-seed', str(args.occlusion_seed), '--occlusion-mode', args.occlusion_mode,
             *[a for s in args.set for a in ('--set', s)],
             '--set', f'train.seed={args.seed}',
             '--set', f'train.output_dir={args.output}/rate_{rate:.2f}'])


if __name__ == '__main__':
    main()
