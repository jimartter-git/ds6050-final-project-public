# Visibility-aware skeleton action recognition on Penn Action

DS 6050 Deep Learning (UVA, Fall 2026), team G02: John Baker, James Martter, Bryson Matsuda,
Brian Seals.

Skeleton action recognition models usually treat every joint coordinate as observed. In
Penn Action, 22% of joint-frames are annotated as invisible and get filled in by
interpolation, so a model cannot tell measured joints from estimated ones. We test whether
giving a Transformer a per-joint visibility flag helps, whether masked-joint pretraining
that masks and reconstructs only visible joints helps further, and how these models hold
up when joints are hidden on purpose at test time.

| Model | What it is | Input per joint |
|---|---|---|
| MB0 | temporal CNN baseline | x, y |
| MT0 | Transformer | x, y |
| MT1 | Transformer with the visibility flag | x, y, v, m |
| MT1-dropout | MT1 trained with random visibility dropout | x, y, v, m |
| MSTD | MT1 pretrained with standard random masking, then fine-tuned | x, y, v, m |
| MVA | MT1 pretrained masking only visible joints, then fine-tuned | x, y, v, m |
| V1 | visibility-only diagnostic | v |

Measured results are in [docs/results.md](docs/results.md); what each comparison tests is in
[docs/experiments.md](docs/experiments.md).

## Setup

Everything runs from the repo root. The easiest route is the Docker launcher, which builds
the pinned environment on first use and then prints whether it will use CUDA or CPU:

```bash
git clone https://github.com/jimartter-git/ds6050-final-project-public.git
cd ds6050-final-project-public
bash docker.sh python -m pytest -q
```

It needs Docker Desktop (or Docker Engine on Linux, plus the NVIDIA Container Toolkit for
GPUs). On Windows use `.\docker.ps1` in place of `bash docker.sh`, with forward slashes in
paths and each command on one line.

Without Docker, use Python 3.11:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

and drop the `bash docker.sh` prefix from the commands below. PyTorch from PyPI is CPU-only
on Windows; the Docker image uses the CUDA 12.8 build.

On UVA's Rivanna cluster, follow [docs/rivanna.md](docs/rivanna.md). Its batch scripts run
every experiment and analysis below.

## Data

```bash
bash docker.sh python scripts/download_data.py
```

This streams the 3.2 GB Penn Action archive and keeps only the 2,326 label files in
`data/Penn_Action/labels/`. Data is never committed. The dataset schema and the fixed
253-clip validation split are committed (`metadata/`, `splits/val_ids.txt`), so everyone
uses the same train/validation/test split; the test set is the dataset's official one.
`scripts/inspect_labels.py` and `scripts/prepare_splits.py` produced those files and do not
need to be rerun. See [docs/data.md](docs/data.md).

## Running experiments

Train, then evaluate the best checkpoint (chosen by validation macro-F1):

```bash
bash docker.sh python -m penn_action.train --config configs/mb0_tcn.yaml --set train.output_dir=outputs/mb0_s42
bash docker.sh python -m penn_action.train --config configs/mb0_tcn.yaml --mode eval --split test \
  --checkpoint outputs/mb0_s42/best.pt --set train.output_dir=outputs/mb0_s42-test
```

A training folder holds `config.yaml`, `metadata.json` (seed, Git commit, device, split
hashes), per-epoch `metrics.jsonl`, `timing.json`, `val_predictions.json` and `best.pt`. An
eval folder holds the metrics and per-clip `predictions.json`. For a quick check on real
data, add `--set data.debug_per_class=2 --set train.epochs=2`.

Results are reported over seeds 42, 43 and 44 (`--set train.seed=43`, using the same seed
for training and evaluation), summarized as mean ± sample standard deviation:

```bash
bash docker.sh python scripts/summarize_runs.py outputs/mb0_s42-test outputs/mb0_s43-test outputs/mb0_s44-test
```

MSTD and MVA pretrain first, then fine-tune from the pretraining checkpoint of the same seed:

```bash
bash docker.sh python -m penn_action.train --config configs/mva_pretrain.yaml --set train.output_dir=outputs/mva_pretrain_s42
bash docker.sh python -m penn_action.train --config configs/mva_finetune.yaml \
  --set train.init_checkpoint=outputs/mva_pretrain_s42/best.pt --set train.output_dir=outputs/mva_s42
```

`scripts/summarize_training.py` reports training time over seeds and, for pretraining runs,
the validation reconstruction error.

Comparisons read the eval folders; each `--model`, `--a` or `--b` takes a label followed by
one folder per seed:

```bash
bash docker.sh python scripts/mcnemar.py --a MT0 outputs/mt0_s42-test outputs/mt0_s43-test outputs/mt0_s44-test \
  --b MT1 outputs/mt1_s42-test outputs/mt1_s43-test outputs/mt1_s44-test
bash docker.sh python scripts/occlusion_tertiles.py --model MB0 outputs/mb0_s42-test outputs/mb0_s43-test outputs/mb0_s44-test
bash docker.sh python scripts/occlusion_sweep.py --config configs/mb0_tcn.yaml --checkpoint outputs/mb0_s42/best.pt \
  --seed 42 --output outputs/mb0_s42-occlusion
bash docker.sh python scripts/occlusion_curve.py --model MB0 outputs/mb0_s42-occlusion outputs/mb0_s43-occlusion outputs/mb0_s44-occlusion
```

The occlusion sweep hides 0–50% of visible joint-frames at test time. Add
`--occlusion-mode spans` to hide one joint for 8–16 consecutive frames instead of single
frames. `scripts/plot_results.py` draws the training-curve and occlusion figures (it needs
matplotlib, which is not in the training environment).

A few rules the code enforces: output folders must be new; evaluation refuses a checkpoint
whose classes, split, model or preprocessing differ from the config; and the analysis
scripts refuse to mix splits, seeds or occlusion settings. Use the test split only for
reported numbers, never to choose settings. Runs repeat exactly with the same config, seed
and hardware.

## Layout

| Path | Contents |
|---|---|
| `src/penn_action/` | data loading and preprocessing, model inputs, masking, training, metrics, models |
| `configs/` | one config per model; `debug.yaml` runs on invented data |
| `scripts/` | data download, split preparation, summaries, McNemar, occlusion analyses, figures |
| `scripts/rivanna/` | Slurm batch scripts for the full run |
| `metadata/`, `splits/` | dataset schema and findings, validation clip IDs |
| `docs/` | experiment plan, results, code interfaces, data notes, Rivanna guide |
| `tests/` | tests on invented data; no dataset needed |
| `docker/` | Dockerfile and compose files used by `docker.sh` / `docker.ps1` |

## Team

- **John Baker**: literature review, MB0 review and training analysis, Method and
  Preliminary Experiments drafts.
- **James Martter**: project proposal and outline, shared repository and pipeline,
  Transformer and pretraining experiments, figures.
- **Bryson Matsuda**: MB0 temporal CNN, training procedure and evaluation metrics.
- **Brian Seals**: literature review, Milestone I proposal, paper drafting and team
  coordination.
