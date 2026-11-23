#!/bin/bash
#SBATCH --job-name=ds6050-arm
#SBATCH --account=shakeri_ds6050
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-20%6
#SBATCH --output=/scratch/%u/ds6050/logs/%x-%A_%a.out
# One job per (model, seed): train, evaluate val and test, then sweep both occlusion modes.
# Index = model*3 + seed position. OUT must be a new folder; train.py refuses to overwrite.
set -euo pipefail
PY=/scratch/$USER/ds6050/env/bin/python
cd "$SLURM_SUBMIT_DIR"
ARMS=(mb0_tcn mt0_transformer mt1_transformer_vis mt1_vis_dropout v1_vis_only mstd mva)
SEEDS=(42 43 44)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
OUT=${OUT:-outputs/rivanna}
RUN=$OUT/${ARM}_s$SEED
echo "arm=$ARM seed=$SEED commit=$(git rev-parse HEAD) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ $ARM == mstd || $ARM == mva ]]; then
  CONFIG=configs/${ARM}_finetune.yaml
  "$PY" -m penn_action.train --config configs/${ARM}_pretrain.yaml --set train.seed=$SEED --set train.output_dir=$OUT/${ARM}_pretrain_s$SEED
  "$PY" -m penn_action.train --config "$CONFIG" --set train.seed=$SEED --set train.init_checkpoint=$OUT/${ARM}_pretrain_s$SEED/best.pt --set train.output_dir=$RUN
else
  CONFIG=configs/$ARM.yaml
  "$PY" -m penn_action.train --config "$CONFIG" --set train.seed=$SEED --set train.output_dir=$RUN
fi
for SPLIT in val test; do
  "$PY" -m penn_action.train --config "$CONFIG" --set train.seed=$SEED --mode eval --split $SPLIT --checkpoint $RUN/best.pt --set train.output_dir=$RUN-$SPLIT
done
"$PY" scripts/occlusion_sweep.py --config "$CONFIG" --checkpoint $RUN/best.pt --seed $SEED --output $RUN-occlusion
"$PY" scripts/occlusion_sweep.py --config "$CONFIG" --checkpoint $RUN/best.pt --seed $SEED --occlusion-mode spans --output $RUN-spans
echo ARM_OK
