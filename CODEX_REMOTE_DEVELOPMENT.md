# Codex Remote Development Workflow For TrainParity

This file is the operating guide for developing TrainParity from the local
Windows workspace while executing compute work on the M3 cluster.

## Canonical Locations

```text
Local repository:
D:\STUDY\MASTER\Research\04.code\code\frameworks\TrainParity

M3 project root:
/scratch/mp25/jwuu0254/zxh/TrainParity

Shared Slurm guide:
/scratch/mp25/jwuu0254/zxh/CODEX_SLURM_RESOURCE_WORKFLOW.md
```

MP25 is the only remote storage location for TrainParity. All source checkouts,
environments, models, datasets, caches, temporary files, logs, generated outputs,
and other project artifacts must remain somewhere under the M3 project root.
The repository decides its own internal layout; this guide does not prescribe
subdirectory names. Do not create TrainParity content on RA25, DV90, DV94, or in
`/home`.

## SSH Identity Separation

Two different keys have different scopes:

- From the local machine, connect to M3 with `~/.ssh/id_mac2`.
- On M3, access the project owner's GitHub repositories with
  `~/.ssh/id_rsa_zxh`.

The M3 account is borrowed/shared. Never change global SSH configuration,
`~/.ssh/config`, or global Git settings to select the project GitHub key. Set the
key only in the TrainParity repository:

```bash
git config --local core.sshCommand \
  'ssh -i ~/.ssh/id_rsa_zxh -o IdentitiesOnly=yes'
```

For the initial clone, apply the key only to that command:

```bash
git -c core.sshCommand='ssh -i ~/.ssh/id_rsa_zxh -o IdentitiesOnly=yes' \
  clone git@github.com:<github-owner>/TrainParity.git \
  <chosen-path-under-/scratch/mp25/jwuu0254/zxh/TrainParity>
```

Do not print, copy, commit, or otherwise expose either private key.

## Repository Bootstrap

After the GitHub repository exists, initialize or connect the local repository:

```powershell
Set-Location D:\STUDY\MASTER\Research\04.code\code\frameworks\TrainParity
git init
git remote add origin git@github.com:<github-owner>/TrainParity.git
git add .
git commit -m "Initialize TrainParity workflow"
git branch -M main
git push -u origin main
```

Then connect to M3 and clone it using the repository-scoped key configuration:

```powershell
ssh -i $HOME/.ssh/id_mac2 m3
```

```bash
mkdir -p /scratch/mp25/jwuu0254/zxh/TrainParity
git -c core.sshCommand='ssh -i ~/.ssh/id_rsa_zxh -o IdentitiesOnly=yes' \
  clone git@github.com:<github-owner>/TrainParity.git \
  <chosen-path-under-/scratch/mp25/jwuu0254/zxh/TrainParity>
cd <chosen-path-under-/scratch/mp25/jwuu0254/zxh/TrainParity>
git config --local core.sshCommand \
  'ssh -i ~/.ssh/id_rsa_zxh -o IdentitiesOnly=yes'
```

## Normal Development Loop

The local repository is the normal editing workspace. Use Git, not ad hoc source
copies, to synchronize code:

1. Inspect the local worktree and read existing code before editing.
2. Make scoped local changes and run all feasible local tests.
3. Review `git diff` and commit only the intended files.
4. Push the branch to GitHub.
5. On M3, verify the remote worktree is clean, then pull with `--ff-only`.
6. Run smoke tests or submit Slurm jobs from the M3 checkout.
7. Keep generated outputs under the MP25 TrainParity project root.
8. Bring back only compact summaries or selected ignored visualizations. Commit
   code and reproducibility metadata, not large artifacts.

Typical synchronization commands:

```powershell
git status --short
git diff --check
git push
ssh -i $HOME/.ssh/id_mac2 m3 `
  'cd <chosen-TrainParity-repository-path-on-MP25> && git status --short && git pull --ff-only'
```

If the M3 worktree has uncommitted changes, inspect them and preserve them. Never
reset or overwrite remote changes merely to make a pull succeed.

## Cluster Execution

Before choosing a partition or QOS, read and follow:

```text
/scratch/mp25/jwuu0254/zxh/CODEX_SLURM_RESOURCE_WORKFLOW.md
```

In particular, run `show_cluster`, determine which accessible QOS entries have
appropriate idle GPUs, check the user's existing allocation with
`squeue -u "$USER"`, and only then submit. Lightweight inference may use A40 or
L40-class GPUs; use high-memory accelerators when the workload requires them.

Every TrainParity Slurm script should define the project storage boundary:

```bash
export PROJECT_ROOT=/scratch/mp25/jwuu0254/zxh/TrainParity
```

Any code, environment, model, dataset, cache, temporary, log, and output path
used by the job must resolve under `$PROJECT_ROOT`. Choose concrete subdirectory
names from the repository's actual needs and configuration rather than assuming
a fixed layout from this guide.

After `sbatch`, always run `squeue -u "$USER"`. Continue independent coding while
jobs wait or run, monitor long-running work every one to two hours, and verify the
terminal state with `sacct` plus output logs.

## Git And Artifact Rules

Add at least these patterns to `.gitignore` when the repository is initialized:

```gitignore
.env
*.key
*.pem
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
*.out
*.err
*.ckpt
*.pt
*.pth
*.safetensors
```

Also ignore whichever project-specific directories are chosen for environments,
models, datasets, caches, temporary data, logs, and generated outputs. Their
names are not prescribed here.

Never commit credentials, Hugging Face tokens, SSH keys, model weights, datasets,
generated images, checkpoints, caches, or cluster logs. Commit Slurm scripts and
configs with placeholders or environment variables instead of secrets.

## Codex Start Checklist

At the beginning of a TrainParity task, Codex should:

1. Confirm the newest user request and current local worktree state.
2. Read this document and the shared Slurm guide.
3. Inspect existing local and remote changes without reverting either.
4. Confirm M3 access with local `id_mac2` when remote work is required.
5. Confirm the M3 repository has local `core.sshCommand` pointing only to
   `~/.ssh/id_rsa_zxh` before GitHub operations.
6. Check MP25 capacity before large writes.
7. Implement, test, synchronize through Git, submit appropriately sized jobs,
   monitor them, validate artifacts, and report exact paths and job IDs.
