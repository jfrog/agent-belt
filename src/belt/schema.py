# (c) JFrog Ltd. (2026)

"""Schema version checking for output artifacts.

All persistent artifacts (turn_output.json, score.json, run_meta.json,
results.json) carry a ``schema_version`` field. Readers call
``check_schema_version`` to detect version mismatches early - before
silent data misinterpretation causes wrong scores or broken reports.
"""

from __future__ import annotations

from loguru import logger

from belt.constants import SCHEMA_VERSION


def check_schema_version(found: str | None, artifact_label: str) -> None:
    """Warn when an artifact's schema version doesn't match the current reader.

    Args:
        found: The ``schema_version`` value from the artifact, or None if
            the writer omitted the field (the v1 contract permits it).
        artifact_label: Human-readable label for log messages (e.g. file path).
    """
    if found is None:
        logger.warning("{}: missing schema_version (assuming v{})", artifact_label, SCHEMA_VERSION)
        return
    if found == SCHEMA_VERSION:
        return
    logger.warning(
        "{}: schema version mismatch - artifact is v{}, reader expects v{}",
        artifact_label,
        found,
        SCHEMA_VERSION,
    )
