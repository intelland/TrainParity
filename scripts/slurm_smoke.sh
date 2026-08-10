#!/bin/bash
#SBATCH --job-name=trainparity_smoke
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --account=mp25
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.out
#SBATCH --error=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.err

set -euo pipefail

export PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
export TMPDIR="$PROJECT_ROOT/tmp"

mkdir -p "$PROJECT_ROOT/outputs/output_log" "$TMPDIR"
cd "$PROJECT_ROOT/repo"

test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"

echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)"
echo "project_root=$PROJECT_ROOT code_dir=$(pwd -P)"
echo "trainparity_smoke_ok"
