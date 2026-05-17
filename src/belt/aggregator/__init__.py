# (c) JFrog Ltd. (2026)

"""Aggregator helpers shared across terminal and markdown renderers."""

from __future__ import annotations

from belt.entities import ScenarioScore

# Tag attached to schema-coverage scenarios that don't run cleanly against
# a generic CLI agent: either they reference fields no agent surfaces (cost
# reporting, multi-agent handoffs, ``has_thinking`` …) or they are sensitive
# to cross-agent naming drift (strict ``tools_invoked`` lists). Filtering
# with ``--tags real-runnable`` is the documented escape hatch; the renderer
# footnote tells first-time users about it.
DRY_RUN_ONLY_TAG = "dry-run-only"


def dry_run_only_failure_count(scores: list[ScenarioScore]) -> int:
    """Count failed scenarios whose effective tags include ``dry-run-only``.

    Used by both the terminal and markdown renderers to emit a single
    "you probably want ``--tags real-runnable``" footnote when the failure
    set is dominated by showcase examples that need an opt-in flag.
    """
    return sum(1 for s in scores if not s.overall_pass and DRY_RUN_ONLY_TAG in s.tags)
