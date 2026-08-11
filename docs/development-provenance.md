# Development provenance

TrainParity was implemented with assistance from OpenAI Codex under
human-authored specifications and explicit Gate-by-Gate acceptance. The human
reviewer defined scope, required fault controls, evidence standards, product
friction thresholds, and the decision to include the sample-coverage module.
Codex helped inspect, implement, test, document, and prepare the repository.

This provenance does not extend into the product runtime. TrainParity makes
deterministic local decisions from user-declared observations, explicit
comparison policies, and versioned Python code. It does not call Codex, another
LLM, or an agent service while checking training.

Development plans, acceptance criteria, status, decisions, and Codex goals are
preserved in `docs/development/`. Accepted Gate reports and recorded experiment
outputs remain in the repository for auditability but are excluded from wheel
and source-distribution payloads.

The presence of generated or assisted implementation does not replace review.
Users should evaluate the public API, source, tests, validation limits, and
security model for their own environment.
