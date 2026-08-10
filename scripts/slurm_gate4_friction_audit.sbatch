#!/bin/bash
#SBATCH --job-name=trainparity_g4_audit
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

if [[ "${1:-}" != "--gate" || "${2:-}" != "4" || "${3:-}" != "--friction-audit" ]]; then
    echo "usage: sbatch scripts/slurm_gate4_friction_audit.sbatch --gate 4 --friction-audit" >&2
    exit 2
fi

export PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
export TMPDIR="$PROJECT_ROOT/tmp"
export TRAINPARITY_GATE4_DEVICE=cuda
export TRAINPARITY_GATE4_SITE="$PROJECT_ROOT/envs/gate4-site"
PYTHON="$PROJECT_ROOT/envs/gate3/bin/python"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/gate4_friction_audit/job_$SLURM_JOB_ID"
USER_DOCUMENT_SHA256="${TRAINPARITY_USER_DOCUMENT_SHA256:?set the local document SHA-256}"

mkdir -p "$PROJECT_ROOT/outputs/output_log" "$TMPDIR"
cd "$PROJECT_ROOT/repo"

echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
"$PYTHON" -m experiments.gate4.friction.run_audit \
    --output-root "$OUTPUT_ROOT" \
    --report-json "$PROJECT_ROOT/repo/artifacts/gate_reports/gate_4_friction_audit.json" \
    --report-markdown "$PROJECT_ROOT/repo/artifacts/gate_reports/gate_4_friction_audit.md" \
    --user-document-sha256 "$USER_DOCUMENT_SHA256"
"$PYTHON" scripts/verify_gate4_friction_audit.py
