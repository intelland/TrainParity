"""TrainParity v0.1 public API and compatibility imports."""

from trainparity.api import (  # noqa: F401
    MACHINE_REPORT_SCHEMA_VERSION,
    PACKAGE_VERSION,
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
    NoCrossRankOverlap,
    Outcome,
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
from trainparity.accumulation import (  # noqa: F401
    AccumulationRunner,
    UnsafeBatchSplit,
    split_tensor_tree,
)
from trainparity.assertions import assert_resume_equivalent  # noqa: F401
from trainparity.comparison import ComparisonResult  # noqa: F401
from trainparity.importing import (  # noqa: F401
    CaseImportError,
    load_accumulation_case,
    load_case,
    load_process_case,
)
from trainparity.process_resume import ProcessResumeRunner  # noqa: F401
from trainparity.protocols import (  # noqa: F401
    ResumeCase,
    ResumeExecutionCase,
    StepObservation,
)
from trainparity.results import ProcessEvidence, ResumeResult  # noqa: F401
from trainparity.runner import ResumeRunner  # noqa: F401
from trainparity.snapshot import CaptureResult, Snapshot, capture_snapshot  # noqa: F401
