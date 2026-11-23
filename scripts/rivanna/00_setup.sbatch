#!/bin/bash
#SBATCH --job-name=ds6050-setup
#SBATCH --account=shakeri_ds6050
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/ds6050/logs/%x-%j.out
# One-time setup, submitted from the repo root: Python env in /scratch/$USER/ds6050/env and
# the Penn Action labels in data/. Safe to rerun; an existing env is reused.
set -euo pipefail
ROOT=/scratch/$USER/ds6050
cd "$SLURM_SUBMIT_DIR"
test -f configs/mb0_tcn.yaml || { echo "submit from the repo root"; exit 1; }
module load miniforge
if [[ ! -x $ROOT/env/bin/python ]]; then
  conda create -y -p "$ROOT/env" python=3.11
fi
PY="$ROOT/env/bin/python"
"$PY" -m pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -q -r requirements.txt
"$PY" -m pip install -q -e . --no-deps
[[ -d data/Penn_Action/labels ]] || "$PY" scripts/download_data.py --data-root data/Penn_Action
ls data/Penn_Action/labels | wc -l
git rev-parse HEAD
echo SETUP_OK
