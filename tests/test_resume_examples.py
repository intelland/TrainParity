from __future__ import annotations

from experiments.gate1.run_adapter_evaluation import run_resume_probe
from trainparity.examples.resume_cases import (
    CorrectResumeCase,
    MissingSchedulerStateCase,
    make_resume_callbacks,
)


def test_correct_resume_case_passes_example_probe() -> None:
    result = run_resume_probe(CorrectResumeCase())
    assert result["outcome"] == "PASS"
    assert result["first_observed_divergence"] is None


def test_faulty_resume_case_exposes_first_observed_difference() -> None:
    result = run_resume_probe(MissingSchedulerStateCase())
    assert result["outcome"] == "FAIL"
    assert result["first_observed_divergence"] == "optimizer.param_groups.0.lr"


def test_factory_callback_prototype_can_express_the_clean_case() -> None:
    result = run_resume_probe(make_resume_callbacks())
    assert result["outcome"] == "PASS"

