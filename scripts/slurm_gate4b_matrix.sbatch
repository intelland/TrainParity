#!/bin/bash
#SBATCH --job-name=trainparity_gate4b
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=mp25
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.out
#SBATCH --error=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.err

set -euo pipefail

if [[ "${1:-}" != "--gate" || "${2:-}" != "4b" ]]; then
    echo "usage: sbatch scripts/slurm_gate4b_matrix.sbatch --gate 4b" >&2
    exit 2
fi

export PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
export TMPDIR="$PROJECT_ROOT/tmp"
export TRAINPARITY_GATE4_DEVICE=cuda
export TRAINPARITY_GATE4_SITE="$PROJECT_ROOT/envs/gate4-site"
PYTHON="$PROJECT_ROOT/envs/gate3/bin/python"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/gate4b/job_$SLURM_JOB_ID"

mkdir -p "$PROJECT_ROOT/outputs/output_log" "$TMPDIR"
cd "$PROJECT_ROOT/repo"

echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
"$PYTHON" -m experiments.gate4b.run_matrix \
    --output-root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/matrix.json" \
    --profile-pre "$PROJECT_ROOT/outputs/gate4b/profile_pre.json" \
    --profile-post "$PROJECT_ROOT/outputs/gate4b/profile_post.json"
