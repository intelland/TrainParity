"""Package and machine-report versions for the v0.1 release candidate."""

PACKAGE_VERSION = "0.1.0rc2"
MACHINE_REPORT_SCHEMA_VERSION = 1


def add_report_metadata(payload: dict[str, object]) -> dict[str, object]:
    """Add stable schema and producer versions to a public machine report."""
    return {
        "schema_version": MACHINE_REPORT_SCHEMA_VERSION,
        "trainparity_version": PACKAGE_VERSION,
        **payload,
    }

