"""Package and machine-report versions for the current source tree."""

PACKAGE_VERSION = "0.1.1.dev0"
MACHINE_REPORT_SCHEMA_VERSION = 2


def add_report_metadata(payload: dict[str, object]) -> dict[str, object]:
    """Add stable schema and producer versions to a public machine report."""
    return {
        "schema_version": MACHINE_REPORT_SCHEMA_VERSION,
        "trainparity_version": PACKAGE_VERSION,
        **payload,
    }
