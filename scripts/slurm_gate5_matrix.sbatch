#!/bin/bash
#SBATCH --job-name=trainparity_gate5
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=mp25
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.out
#SBATCH --error=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.err

set -euo pipefail

if [[ "${1:-}" != "--gate" || "${2:-}" != "5" ]]; then
    echo "usage: sbatch scripts/slurm_gate5_matrix.sbatch --gate 5" >&2
    exit 2
fi

export PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
export TMPDIR="$PROJECT_ROOT/tmp"
PYTHON="$PROJECT_ROOT/envs/gate3/bin/python"
REPO="$PROJECT_ROOT/repo_gate5"
OUTPUT="$PROJECT_ROOT/outputs/gate5/job_$SLURM_JOB_ID/gpu_matrix.json"

mkdir -p "$PROJECT_ROOT/outputs/output_log" "$TMPDIR" "$(dirname "$OUTPUT")"
cd "$REPO"
export PYTHONPATH="$REPO/src"

echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
"$PYTHON" -m experiments.gate5.run_matrix --output "$OUTPUT" --device cuda --repeats 1
