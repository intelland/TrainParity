#!/bin/bash
#SBATCH --job-name=trainparity_gate4
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=mp25
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.out
#SBATCH --error=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.err

set -euo pipefail

if [[ "${1:-}" != "--gate" || "${2:-}" != "4" ]]; then
    echo "usage: sbatch scripts/slurm_gate4_matrix.sbatch --gate 4 [--project NAME]" >&2
    exit 2
fi
shift 2

export PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
export TMPDIR="$PROJECT_ROOT/tmp"
export TRAINPARITY_GATE4_DEVICE=cuda
export TRAINPARITY_GATE4_SITE="$PROJECT_ROOT/envs/gate4-site"
PYTHON="$PROJECT_ROOT/envs/gate3/bin/python"
OUTPUT="${TRAINPARITY_GATE4_OUTPUT:-$PROJECT_ROOT/outputs/gate4/matrix.json}"

mkdir -p "$PROJECT_ROOT/outputs/output_log" "$TMPDIR"
cd "$PROJECT_ROOT/repo"

echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
"$PYTHON" -m experiments.gate4.run_matrix \
    --external-root "$PROJECT_ROOT/external/gate4" \
    --output "$OUTPUT" \
    "$@"
