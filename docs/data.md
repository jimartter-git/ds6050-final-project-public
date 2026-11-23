# Data

## Download

The labels come from the [official Penn Action page](https://dreamdragon.github.io/PennAction/).
`scripts/download_data.py` streams the 3.2 GB archive and writes only the label files, so the
archive and RGB frames never reach disk:

```bash
python scripts/download_data.py              # or: bash docker.sh python scripts/download_data.py
```

The result is `data/Penn_Action/labels/*.mat` (2,326 files). The script refuses to overwrite
existing labels without `--force` and warns if the count differs. To keep the data elsewhere,
set `data.data_root`. Data, archives, outputs and checkpoints are never committed; check the
dataset's terms before sharing it.

## What is committed

- `metadata/penn_action_schema.json`: field names, train/test codes, class order and joint
  count, checked against the label files on every run. Findings from inspecting the labels
  are in [metadata/README.md](../metadata/README.md).
- `splits/val_ids.txt`: 253 validation clips, about 20% of each class's official training
  clips, drawn once with seed 42 and shared by every model and training seed.

Clip 1516 has no visible joint in any frame, so training leaves it out and records it in each
run's `metadata.json`. That gives 1,004 training, 253 validation and 1,068 test clips. The
test clips are the dataset's official test split and are only used for reported numbers.

These files were produced once with the commands below; there is no need to rerun them.

```bash
python scripts/inspect_labels.py --data-root data/Penn_Action \
  --fields '{"x":"x","y":"y","visibility":"visibility","action":"action","split":"train","nframes":"nframes","dimensions":"dimensions"}' \
  --train-value 1 --test-value -1 --dimensions-order hwt
python scripts/prepare_splits.py --data-root data/Penn_Action \
  --schema metadata/penn_action_schema.json --seed 42 --fraction 0.2
```

Training checks that every class appears in both training and validation and that every
validation clip comes from the official training split.

## Preprocessing

Every model gets the same preprocessing (`data.preprocess`):

1. Each invisible joint is filled by linear interpolation between the nearest visible frames
   of the same joint, holding the first or last observed position at the ends of the clip.
   A joint that is never visible in a clip is set to zero.
2. Each clip is centred on the mean position of its visible joints and scaled by their
   root-mean-square distance from that mean.
3. At most 32 evenly spaced frames are kept; shorter clips are padded when batched.

`hidden_policy: raw` and `normalization: none` exist for diagnostics only. Even after
interpolation, coordinates can still hint at which joints were missing.

The validation split is stratified by class but not grouped by source video, because the
labels carry no video or performer ID.
