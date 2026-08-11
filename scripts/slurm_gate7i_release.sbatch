#!/bin/bash
#SBATCH --job-name=trainparity_gate7i_release
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --account=mp25
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.out
#SBATCH --error=/scratch/mp25/jwuu0254/zxh/TrainParity/outputs/output_log/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
REPO="${TRAINPARITY_REPO:?submit with TRAINPARITY_REPO set to the pinned Gate 7I checkout}"
BASE_PYTHON="$PROJECT_ROOT/envs/gate3/bin/python"
RUN_ROOT="$PROJECT_ROOT/tmp/gate7i-release-$SLURM_JOB_ID"
VENV="$RUN_ROOT/venv"
OUTSIDE="$RUN_ROOT/outside-repository"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export TRAINPARITY_RELEASE_TEMP_ROOT="$RUN_ROOT/release-smoke"
unset PYTHONPATH

mkdir -p "$RUN_ROOT" "$OUTSIDE" "$PIP_CACHE_DIR" "$TRAINPARITY_RELEASE_TEMP_ROOT"
"$BASE_PYTHON" -m venv "$VENV"
PYTHON="$VENV/bin/python"
"$PYTHON" -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
"$PYTHON" -m pip install -e "$REPO[dev]"

cd "$REPO"
make PYTHON="$PYTHON" lint
make PYTHON="$PYTHON" typecheck
make PYTHON="$PYTHON" test \
  PYTEST_ARGS="--junitxml=experiments/gate7i/recorded/junit.xml"
"$PYTHON" scripts/record_test_summary.py \
  --junit experiments/gate7i/recorded/junit.xml \
  --coverage coverage.json \
  --job-id "$SLURM_JOB_ID" \
  --output experiments/gate7i/recorded/test_summary.json
"$PYTHON" -m pytest -q --no-cov examples/test_readme_case.py
"$PYTHON" -m trainparity.quickstarts.resume
"$PYTHON" -m trainparity.quickstarts.accumulation
"$PYTHON" -m trainparity.quickstarts.sample_coverage
make PYTHON="$PYTHON" release-check

for gate in 0 1 2 3 4; do
  COPY="$RUN_ROOT/verify-gate-$gate"
  cp -a "$REPO" "$COPY"
  cd "$COPY"
  "$PYTHON" scripts/verify_gate.py "$gate"
done

cd "$REPO"
"$PYTHON" scripts/verify_gate4_friction_audit.py
"$PYTHON" scripts/verify_gate4b.py
"$PYTHON" scripts/verify_gate5.py --allow-pending-ci
"$PYTHON" scripts/verify_gate6.py --allow-pending-ci
"$PYTHON" scripts/verify_gate7.py --allow-pending-ci
"$PYTHON" -m experiments.gate7i.build_report
"$PYTHON" -m scripts.build_gate7i_bundle
"$PYTHON" scripts/verify_gate7i.py
