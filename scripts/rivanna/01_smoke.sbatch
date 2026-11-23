#!/bin/bash
#SBATCH --job-name=ds6050-smoke
#SBATCH --account=shakeri_ds6050
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/ds6050/logs/%x-%j.out
# Tests plus two-epoch runs of every model on a few clips, to check the GPU setup first.
set -euo pipefail
PY=/scratch/$USER/ds6050/env/bin/python
cd "$SLURM_SUBMIT_DIR"
git rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$PY" -m pytest -q
OUT=outputs/smoke_$SLURM_JOB_ID
D=(--set data.debug_per_class=2 --set train.epochs=2)
for arm in mb0_tcn mt0_transformer mt1_transformer_vis mt1_vis_dropout v1_vis_only; do
  "$PY" -m penn_action.train --config configs/$arm.yaml "${D[@]}" --set train.output_dir=$OUT/$arm
done
for arm in mstd mva; do
  "$PY" -m penn_action.train --config configs/${arm}_pretrain.yaml "${D[@]}" --set train.output_dir=$OUT/${arm}_pretrain
  "$PY" -m penn_action.train --config configs/${arm}_finetune.yaml "${D[@]}" --set train.init_checkpoint=$OUT/${arm}_pretrain/best.pt --set train.output_dir=$OUT/$arm
done
echo SMOKE_OK
