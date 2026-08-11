# Security

## Security model

TrainParity is not a sandbox. A user adapter and its training commands execute
with the permissions of the caller. The fresh-process boundary tests resume
behavior; it does not isolate hostile code.

- Execute only training code and external repositories you trust.
- Load only trusted checkpoints. PyTorch deserialization and project-specific
  loaders must be treated as code/data trust boundaries.
- Use dedicated low-privilege environments for experiments.
- Inspect explicit child-process environment propagation. Environment values
  are not recorded in TrainParity reports by default; only key names are
  retained where the contract requires evidence.
- Do not put credentials, tokens, private keys, or confidential data in case
  names, paths, observations, exception messages, or artifacts.

TrainParity does not contact an LLM or service at runtime. It also does not
download repositories or checkpoints as part of the public runtime API.

## Reporting a vulnerability

Do not disclose a vulnerability through a public issue. Use GitHub's private
security-advisory reporting for `intelland/TrainParity` when available. Include
the affected version, a minimal reproduction, impact, and any suggested
mitigation. Do not include live credentials or private training data.

No response-time or remediation-time guarantee is made for this pre-release
project.

## Release permissions

Pull-request CI and scheduled validation use an explicit read-only
`GITHUB_TOKEN`, persist no checkout credential, and receive no repository
secrets or OIDC permission. The manual release workflow is restricted to the
default branch and references the `pypi` environment. Before enabling it, the
repository owner must configure that environment with required human reviewers
and set `PYPI_ENVIRONMENT_PROTECTED=true`; otherwise the publish job is skipped.
PyPI publication uses Trusted Publishing with a job-scoped OIDC token and does
not consume artifacts from pull-request workflows.
