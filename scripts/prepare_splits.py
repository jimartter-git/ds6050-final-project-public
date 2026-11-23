"""Persist a class-stratified validation split from official train; refuses overwrite."""
import argparse
from penn_action.data import read_catalog, validation_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--output', default='splits/val_ids.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fraction', type=float, default=0.2)
    args = parser.parse_args()
    clips, schema = read_catalog(args.data_root, args.schema)
    ids = validation_split(clips, schema['train_value'], args.output,
                           args.seed, args.fraction, create=True)
    print(f'Saved {len(ids)} validation IDs to {args.output}; review and commit once.')


if __name__ == '__main__':
    main()
