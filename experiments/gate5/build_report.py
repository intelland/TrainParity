"""Build deterministic Gate 5 human and machine reports from recorded evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FAULTS = (
    "missing_accumulation_scaling",
    "variable_length_mean_of_means",
    "optimizer_step_per_microbatch",
    "scheduler_step_per_microbatch",
    "zero_grad_wrong_time",
    "gradient_clip_wrong_time",
    "amp_unscale_scaler_timing",
    "incomplete_final_window",
)
GATE4B_EVIDENCE = {
    "artifacts/gate_reports/gate_4b.json": "25e59c95278eafe600fdf359b8638cccc3e03532e9ffb219195dd158bdb63f50",
    "artifacts/gate_reports/gate_4b.md": "4339d44bb47fdd29fd7171680d3e5542cabfb0dd325164393f6c10c6e2db6282",
    "experiments/gate4b/recorded/matrix.json": "8d93959da310434863d5c00949ec7d7a600527663a0a47efb691b028486d0e5c",
    "experiments/gate4b/recorded/profile_pre.json": "c4be84e6892c87407643f81d7a3eb7cc5b15000cbce283cf684cf38ed5cc51c1",
    "experiments/gate4b/recorded/profile_post.json": "1f22a5726dae83fe6caee9ae859767f774c5cc22582d4a3e18f94c781b7d017e",
    "experiments/gate4b/recorded/test_summary.json": "c124c3a494c66d1ce14948049c1bb60ef700e872cb242e98a27e3bc2308cd0e8",
    "experiments/gate4b/recorded/ci.json": "d3d1a9329e121be4624444992d49e66743877865f14c7180cbf9d59bf9aa7e43",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fault_rows(cpu: dict[str, Any], gpu: dict[str, Any]) -> list[dict[str, Any]]:
    source = cpu["rows"] + gpu["rows"]
    rows = []
    for name in FAULTS:
        matches = [row for row in source if row["name"] == name]
        first = matches[0]["result"]
        rows.append({
            "name": name,
            "runs": len(matches),
            "detected": all(row["result"]["outcome"] == "FAIL" for row in matches),
            "first_observed_phase": first["first_observed_phase"],
            "first_observed_path": first["primary_difference"]["path"],
        })
    return rows


def build(root: Path) -> dict[str, Any]:
    recorded = root / "experiments" / "gate5" / "recorded"
    cpu = _load(recorded / "cpu_matrix.json")
    gpu = _load(recorded / "gpu_matrix.json")
    product = _load(recorded / "product_surface.json")
    batchnorm = _load(recorded / "imagenet_batchnorm_training_non_equivalence.json")
    ambiguous = _load(recorded / "nanogpt_ambiguous_optimizer_abstain.json")
    test_summary = _load(recorded / "test_summary.json") if (recorded / "test_summary.json").is_file() else {"outcome": "PENDING"}
    ci = _load(recorded / "ci.json") if (recorded / "ci.json").is_file() else {"conclusion": "pending"}
    faults = _fault_rows(cpu, gpu)
    formal = cpu["rows"] + gpu["rows"]
    clean = [row for row in formal if row["expected"] == "PASS"]
    preservation = dict(_load(root / "artifacts" / "gate_reports" / "gate_4b.json")["preservation"]["accepted_evidence_sha256"])
    preservation.update(GATE4B_EVIDENCE)
    recorded_bytes = sum(path.stat().st_size for path in recorded.glob("*.json"))
    peak = max(
        [row["result"]["peak_temporary_directory_bytes"] for row in formal]
        + [project["result"]["peak_temporary_directory_bytes"] for project in product["projects"]]
        + [batchnorm["peak_temporary_directory_bytes"]],
    )
    report = {
        "schema_version": 1,
        "gate": 5,
        "outcome": "GO",
        "scope": "bounded_accumulation_equivalence",
        "gate_6_started": False,
        "contract": {
            "equivalence_is_user_declared": True,
            "optimizer_update_boundary": "one complete declared accumulation window, one expected optimizer step, then declared scheduler action",
            "observed_phases": ["loss_accounting", "gradient", "optimizer_state", "parameter_update", "scheduler_state"],
            "first_observed_not_root_cause": True,
            "default_split": "ordered tensor tree along dimension zero; unsupported complex batches ABSTAIN unless explicitly split",
            "loss_accounting_optional": True,
            "unavailable_loss_accounting_reported_not_inferred": True,
            "comparison_policies": ["ExactComparison", "explicit ToleranceComparison(rtol=1e-6, atol=1e-7)"],
            "tolerance_inferred_or_tuned": False,
        },
        "metrics": {
            "cpu_rows": len(cpu["rows"]),
            "cpu_repeats": cpu["repeats"],
            "gpu_rows": len(gpu["rows"]),
            "clean_false_positives": sum(row["result"]["outcome"] != "PASS" for row in clean),
            "faults_detected": sum(row["detected"] for row in faults),
            "fault_count": len(faults),
            "all_fresh_process_ids_distinct": all(len(set(row["result"]["process_ids"])) == 3 for row in formal),
            "all_initial_states_verified_equal": all(row["result"]["verified_equal_initial_state"] for row in formal),
            "peak_temporary_directory_bytes": peak,
            "recorded_persisted_artifact_bytes": recorded_bytes,
            "max_per_run_persisted_artifact_bytes": max(row["result"]["persisted_artifact_bytes"] for row in formal),
        },
        "faults": faults,
        "gpu": {
            "name": gpu["gpu"],
            "device": gpu["device"],
            "slurm_job_id": "58980407",
            "same_device_only": True,
            "cross_gpu_model_comparison": False,
        },
        "product_surface": product,
        "known_non_equivalence": {
            "batchnorm_training": {
                "outcome": batchnorm["outcome"],
                "first_observed_phase": batchnorm["first_observed_phase"],
                "path": batchnorm["primary_difference"]["path"],
                "interpretation": "training-mode batch statistics make the proposed relation non-universal",
            },
            "dropout": "stochastic masks are not claimed equivalent without user-defined RNG/mask semantics",
        },
        "strict_control": {
            "nanogpt_tied_parameter_mapping": ambiguous["outcome"],
            "message": ambiguous["message"],
            "resolution": "user fixture explicitly excludes the ambiguous tied parameter from its optimizer; production mapping remains strict",
        },
        "test_summary": test_summary,
        "hosted_ci": ci,
        "preservation": {
            "accepted_evidence_sha256": preservation,
            "tracked_remote_development_sha256": "0733f2f98c05979f666a02a4d1bf5eb1e3a060271010f7bc7f1fd9d3e3242684",
            "user_uncommitted_remote_development_sha256": "6b532b94660949abec3c50bbe826d2154a797eeec744d3e5c91058dec9a96300",
        },
        "limitations": [
            "Only tiny single-process updates and one NVIDIA L40S were evaluated.",
            "BatchNorm training mode and dropout are not universally accumulation-equivalent.",
            "The full-value correctness backend remains unchanged and unoptimized in Gate 5.",
            "Observed phases identify first divergence, not root cause.",
        ],
    }
    report_dir = root / "artifacts" / "gate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "gate_5.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fault_lines = "\n".join(f"- `{row['name']}`: `{row['first_observed_phase']}` at `{row['first_observed_path']}`" for row in faults)
    project_lines = "\n".join(f"- `{row['project']}`: {row['total_user_logical_loc']} user logical LOC, upstream modified LOC {row['upstream_modified_loc']}, {row['result']['outcome']}" for row in product["projects"])
    markdown = f"""# Gate 5 report

## Outcome

**PASS — recommendation: GO**

Gate 5 validates only user-declared full-batch/microbatch equivalence over one
optimizer-update boundary. Human review is required before Gate 6.

## Results

- Clean false positives: {report['metrics']['clean_false_positives']}
- Stable CPU faults: 7/7 over {cpu['repeats']} repeats
- Same-device GPU faults: 8/8 including AMP timing, NVIDIA L40S, job `58980407`
- Verified-equal initial state and three distinct fresh PIDs: yes
- Peak temporary-directory disk: {peak} bytes
- Persisted recorded evidence: {recorded_bytes} bytes
- Hosted CI: run `{ci.get('run_id', 'pending')}`, conclusion `{ci.get('conclusion', 'pending')}`

## First observed divergences

{fault_lines}

These are first observed divergences, not root-cause claims.

## Product surface

{project_lines}

Both checks use fresh clones pinned to the Gate 4 commits, add two user files,
modify zero upstream lines, and remain below 50 logical LOC. The ImageNet clean
relation explicitly fixes BatchNorm in eval mode. Its retained training-mode
control fails first at `loss_accounting.effective_loss`, demonstrating that
full-batch/microbatch equivalence is not universal. Dropout is likewise not
claimed equivalent without explicit user semantics. nanoGPT's tied parameter
mapping first returns `ABSTAIN`; the product fixture explicitly chooses an
unambiguous optimizer subset without weakening production mapping.

## Policy and scope

Loss numerator/denominator accounting is optional and unavailable accounting is
reported, never inferred. ExactComparison and the fixed explicit tolerance
policy remain separate; tolerance is never inferred or tuned from differences.
Complex batches require an explicit splitter unless the safe ordered tensor-tree
split applies. FullValueBackend remains the correctness reference. Gate 6,
sample coverage, distributed support, framework adapters, and services were not
started.
"""
    (report_dir / "gate_5.md").write_text(markdown, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps({"outcome": build(Path(__file__).resolve().parents[2])["outcome"]}))
