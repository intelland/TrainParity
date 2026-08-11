#!/bin/bash
#SBATCH --job-name=trainparity_gate7i_compat
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --account=mp25
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-2%1
#SBATCH --output=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%A_%a.out
#SBATCH --error=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%A_%a.err

set -euo pipefail

VERSIONS=(2.7.0 2.10.0 2.13.0)
TORCH_VERSION="${VERSIONS[$SLURM_ARRAY_TASK_ID]}"
PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
REPO="${TRAINPARITY_REPO:-$PROJECT_ROOT/repo_gate7i_53e9ddc}"
BASE_PYTHON="$PROJECT_ROOT/envs/gate3/bin/python"
RUN_ROOT="$PROJECT_ROOT/tmp/gate7i-compat-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
VENV="$RUN_ROOT/venv"
WHEELS="$RUN_ROOT/wheels"
OUTSIDE="$RUN_ROOT/outside-repository"
OUTPUT="$PROJECT_ROOT/outputs/gate7i/compatibility-$TORCH_VERSION.json"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"

mkdir -p "$RUN_ROOT" "$WHEELS" "$OUTSIDE" "$(dirname "$OUTPUT")" "$PIP_CACHE_DIR"
cd "$REPO"
"$BASE_PYTHON" -m build --wheel --outdir "$WHEELS"
"$BASE_PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" -m pip install "$WHEELS"/*.whl
cd "$OUTSIDE"
"$VENV/bin/python" "$REPO/scripts/compatibility_check.py" \
  --expected-torch "$TORCH_VERSION" \
  --repository "$REPO" \
  --output "$OUTPUT"
case "$RUN_ROOT" in
  "$PROJECT_ROOT"/tmp/gate7i-compat-*) ;;
  *) echo "refusing to clean unexpected run root: $RUN_ROOT" >&2; exit 2 ;;
esac
rm -rf -- "$RUN_ROOT"
