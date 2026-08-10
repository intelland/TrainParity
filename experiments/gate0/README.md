# Gate 0 experiments

This directory contains deterministic, CPU-only experiments for validating the
proposed TrainParity product boundary against OrderLab/TrainCheck. TrainCheck is
used strictly as a black-box CLI dependency; no TrainCheck source is copied.

## M3 environment

Run only on M3. Keep the environment and all generated data under the MP25
project boundary:

```bash
export PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
export CONDA_PKGS_DIRS="$PROJECT_ROOT/caches/conda/pkgs"
export PIP_CACHE_DIR="$PROJECT_ROOT/caches/pip"
export TMPDIR="$PROJECT_ROOT/tmp"

module load miniforge3/24.3.0-0
conda create --solver libmamba \
  -p "$PROJECT_ROOT/envs/gate0" python=3.11 pip -y
"$PROJECT_ROOT/envs/gate0/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.13.0+cpu
"$PROJECT_ROOT/envs/gate0/bin/python" -m pip install traincheck==0.1.2
```

The recorded run used Python 3.11.15, PyTorch 2.13.0+cpu, and TrainCheck 0.1.2.

## Run

From the M3 repository checkout:

```bash
export PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
export PYTHONNOUSERSITE=1
cd "$PROJECT_ROOT/repo"
PY="$PROJECT_ROOT/envs/gate0/bin/python"

"$PY" -m experiments.gate0.run_fault_matrix \
  --output "$PROJECT_ROOT/outputs/gate0/recorded/fault_matrix.json"

"$PY" -m experiments.gate0.run_traincheck_matrix \
  --runtime-root "$PROJECT_ROOT/outputs/gate0/traincheck" \
  --output "$PROJECT_ROOT/outputs/gate0/recorded/traincheck_summary.json"

"$PY" scripts/verify_gate.py 0
```

Raw TrainCheck traces and phase logs remain under
`$PROJECT_ROOT/outputs/gate0/traincheck`. The committed files in `recorded/`
are compact experiment records with environment metadata, exact commands,
durations, return codes, log tails, clean-control results, and structured
failure evidence.

The runner copies each competitor entry into its case runtime directory before
instrumentation and runs every TrainCheck phase from there. TrainCheck-generated
Python files and tool logs therefore remain outside the Git checkout.

## Interpretation rule

Each TrainCheck case has a reference run, a second clean control run, and a
fault run. A fault counts as detected only when the fault checker emits a
violation signature absent from the clean control. A matching count of noisy
violations in clean and fault runs does not count as detection.
