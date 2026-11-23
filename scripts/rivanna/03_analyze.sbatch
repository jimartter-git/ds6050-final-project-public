#!/bin/bash
#SBATCH --job-name=ds6050-analyze
#SBATCH --account=shakeri_ds6050
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/ds6050/logs/%x-%j.out
# Summaries and comparisons over all runs in OUT. Each step runs even if an earlier one
# fails; failures are counted at the end. Writes a tarball without checkpoints.
set -uo pipefail
ROOT=/scratch/$USER/ds6050
PY="$ROOT/env/bin/python"
cd "$SLURM_SUBMIT_DIR"
OUT=${OUT:-outputs/rivanna}
A=$OUT/analysis
mkdir -p $A
FAILED=0
step() { echo "=== $*"; "$@" || { echo "STEP_FAILED: $*"; FAILED=$((FAILED+1)); }; }
dirs() { for s in 42 43 44; do echo "$OUT/$1_s$s${2:-}"; done; }
ARMS=(mb0_tcn mt0_transformer mt1_transformer_vis mt1_vis_dropout v1_vis_only mstd mva)
LABELS=(MB0 MT0 MT1 MT1-dropout V1 MSTD MVA)
for arm in "${ARMS[@]}"; do
  for split in val test; do
    step "$PY" scripts/summarize_runs.py $(dirs $arm -$split) --output $A/${arm}_${split}_summary.json
  done
  step "$PY" scripts/summarize_training.py $(dirs $arm) --output $A/${arm}_training_summary.json
done
for arm in mstd mva; do
  step "$PY" scripts/summarize_training.py $(dirs ${arm}_pretrain) --output $A/${arm}_pretrain_summary.json
done
step "$PY" scripts/mcnemar.py --a MT0 $(dirs mt0_transformer -test) --b MT1 $(dirs mt1_transformer_vis -test) --output $A/rq1_mt0_mt1_mcnemar.json
step "$PY" scripts/mcnemar.py --a MT1 $(dirs mt1_transformer_vis -test) --b MVA $(dirs mva -test) --output $A/rq2a_mt1_mva_mcnemar.json
step "$PY" scripts/mcnemar.py --a MSTD $(dirs mstd -test) --b MVA $(dirs mva -test) --output $A/rq2b_mstd_mva_mcnemar.json
models() { for i in "${!ARMS[@]}"; do echo --model "${LABELS[$i]}" $(dirs "${ARMS[$i]}" "$1"); done; }
step "$PY" scripts/occlusion_tertiles.py $(models -test) --output $A/occlusion_tertiles.json
step "$PY" scripts/occlusion_curve.py $(models -occlusion) --output $A/occlusion_frames.json --csv $A/occlusion_frames.csv
step "$PY" scripts/occlusion_curve.py $(models -spans) --output $A/occlusion_spans.json --csv $A/occlusion_spans.csv
tar --exclude='best.pt' -czf "$ROOT/results-$SLURM_JOB_ID.tgz" -C "$(dirname $OUT)" "$(basename $OUT)"
ls -la "$ROOT/results-$SLURM_JOB_ID.tgz"
echo "ANALYZE_DONE failed_steps=$FAILED"
[[ $FAILED -eq 0 ]]
