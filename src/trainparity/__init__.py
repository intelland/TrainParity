"""TrainParity v0.1 public API and compatibility imports."""

from trainparity.api import (
    AccumulationCase,
    AccumulationExecutionPlan,
    AccumulationResult,
    AtLeastOnce,
    ComparisonPolicy,
    Difference,
    ExactComparison,
    ExactlyOnce,
    ExpectedPadding,
    ExternalProcessEvidence,
    LossAccounting,
    MACHINE_REPORT_SCHEMA_VERSION,
    NoCrossRankOverlap,
    Outcome,
    PACKAGE_VERSION,
    ProcessExecutionPlan,
    ProcessResumeCase,
    ProcessResumeResult,
    SampleAnomaly,
    SampleCoverageAuditor,
    SampleCoverageResult,
    SampleObservation,
    SampleViolation,
    ToleranceComparison,
    TrainingState,
    audit_rank_iterables,
    audit_sample_coverage,
    check_accumulation,
    check_resume,
)
from trainparity.api import __all__ as _PUBLIC_API

__version__ = PACKAGE_VERSION
__all__ = [*_PUBLIC_API, "__version__"]

# Compatibility imports retained so accepted Gate evidence remains replayable.
# They are intentionally absent from ``__all__`` and the frozen v0.1 API.
from trainparity.accumulation import (  # noqa: E402
    AccumulationRunner,
    UnsafeBatchSplit,
    split_tensor_tree,
)
from trainparity.assertions import assert_resume_equivalent  # noqa: E402
from trainparity.comparison import ComparisonResult  # noqa: E402
from trainparity.importing import (  # noqa: E402
    CaseImportError,
    load_accumulation_case,
    load_case,
    load_process_case,
)
from trainparity.process_resume import ProcessResumeRunner  # noqa: E402
from trainparity.protocols import (  # noqa: E402
    ResumeCase,
    ResumeExecutionCase,
    StepObservation,
)
from trainparity.results import ProcessEvidence, ResumeResult  # noqa: E402
from trainparity.runner import ResumeRunner  # noqa: E402
from trainparity.snapshot import CaptureResult, Snapshot, capture_snapshot  # noqa: E402
