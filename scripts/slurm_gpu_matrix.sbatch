#!/bin/bash
#SBATCH --job-name=trainparity_gate3_gpu
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=mp25
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.out
#SBATCH --error=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.err

set -euo pipefail

if [[ "${1:-}" != "--gate" || "${2:-}" != "3" ]]; then
    echo "usage: sbatch scripts/slurm_gpu_matrix.sbatch --gate 3" >&2
    exit 2
fi

export PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
export TMPDIR="$PROJECT_ROOT/tmp"
export TRAINPARITY_DEVICE=cuda
PYTHON="$PROJECT_ROOT/envs/gate3/bin/python"

mkdir -p "$PROJECT_ROOT/outputs/output_log" "$PROJECT_ROOT/outputs/gate3" "$TMPDIR"
cd "$PROJECT_ROOT/repo"

echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
"$PYTHON" -m experiments.gate3.run_gpu_matrix \
    --output "$PROJECT_ROOT/outputs/gate3/gpu_matrix.json"
