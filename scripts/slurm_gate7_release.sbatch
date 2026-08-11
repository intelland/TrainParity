#!/bin/bash
#SBATCH --job-name=trainparity_gate7_release
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --account=mp25
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.out
#SBATCH --error=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.err

set -euo pipefail

if [[ "${1:-}" != "--gate" || "${2:-}" != "7" ]]; then
    echo "usage: sbatch scripts/slurm_gate7_release.sbatch --gate 7" >&2
    exit 2
fi

PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
REPO="$PROJECT_ROOT/repo_gate7"
PYTHON="$PROJECT_ROOT/envs/gate3/bin/python"
export TMPDIR="$PROJECT_ROOT/tmp/gate7-$SLURM_JOB_ID"
export TRAINPARITY_RELEASE_TEMP_ROOT="$TMPDIR/release"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export PYTHONPATH="$REPO/src:$REPO"

mkdir -p "$PROJECT_ROOT/outputs/output_log" "$TMPDIR" "$PIP_CACHE_DIR"
cd "$REPO"

echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)"
make release-check PYTHON="$PYTHON"
echo "finish=$(date --iso-8601=seconds)"
