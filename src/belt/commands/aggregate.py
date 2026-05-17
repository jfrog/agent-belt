#!/usr/bin/env python3
# (c) JFrog Ltd. (2026)

"""
Aggregate ``score.json`` files into a summary.

Terminal: compact box + failure details + threshold enforcement.
GitHub: markdown table written to ``$GITHUB_STEP_SUMMARY``.
Disk: ``results.json`` in the run dir.

This module is the thin CLI entry point for ``belt aggregate``. Library
logic lives in:

- ``belt.aggregator.thresholds`` - threshold parsing, validation,
  per-dimension enforcement, ``ThresholdEnforcer``, ``ThresholdCheck``
- ``belt.aggregator.stats`` - statistics, cost/timing, reliability
- ``belt.aggregator.render_terminal`` - Rich/terminal renderer
- ``belt.aggregator.render_markdown`` - GitHub step-summary renderer

Default (no ``--threshold`` flags): report only, always exit 0.
With ``--threshold``: enforce per-dimension failure budgets, exit 1 if any exceeded.

Usage:
    belt aggregate
    belt aggregate --threshold rules/execution:0
    belt aggregate --threshold rules/execution:0 --threshold rules/trajectory:10
    belt aggregate --threshold llm/execution:0 --llm-fail-on low
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from belt._io import write_json
from belt._ui import eprint
from belt.aggregator.render_markdown import build_markdown
from belt.aggregator.render_terminal import (
    SCORE_EMOJI,
    _evidence_paths,
    _extract_error_lines,
    _failure_context,
    build_result_table,
    print_terminal,
)
from belt.aggregator.stats import (
    aggregate_cost_timing,
    build_bottom_line,
    build_stats,
    collect_agent_errors,
    collect_judge_errors,
)
from belt.aggregator.stats import compute_partial_score
from belt.aggregator.stats import compute_partial_score as _compute_partial_score
from belt.aggregator.stats import compute_reliability
from belt.aggregator.stats import extract_scorer_usage
from belt.aggregator.stats import extract_scorer_usage as _extract_scorer_usage
from belt.aggregator.thresholds import (
    VALID_LLM_FAIL_ON,
    ThresholdCheck,
    ThresholdEnforcer,
    count_llm_failures,
    count_rules_failures,
    discover_llm_dimensions,
    enforce_thresholds,
    parse_threshold,
    validate_thresholds,
)
from belt.constants import EVAL_DIR, LOG_FILE, RESULTS_FILE, SCHEMA_VERSION, SCORE_FILE
from belt.entities import AggregatedResults, ScenarioScore
from belt.schema import check_schema_version
from belt.scorer.entities import DEFAULT_LLM_FAIL_ON_STR

# Re-export library symbols so callers that imported them from this module
# keep working. Tests and forks rely on these names.
__all__ = [
    "SCORE_EMOJI",
    "VALID_LLM_FAIL_ON",
    "ThresholdCheck",
    "ThresholdEnforcer",
    "_compute_partial_score",
    "_evidence_paths",
    "_extract_error_lines",
    "_extract_scorer_usage",
    "_failure_context",
    "aggregate_cost_timing",
    "build_bottom_line",
    "build_markdown",
    "build_result_table",
    "build_stats",
    "collect_agent_errors",
    "compute_partial_score",
    "compute_reliability",
    "count_llm_failures",
    "count_rules_failures",
    "discover_llm_dimensions",
    "discover_scores",
    "enforce_thresholds",
    "extract_scorer_usage",
    "main",
    "parse_threshold",
    "print_terminal",
    "validate_thresholds",
]


def discover_scores(root: Path) -> list[ScenarioScore]:
    """Walk ``root`` for ``score.json`` files and parse each one."""
    scores = []
    for p in sorted(root.rglob(SCORE_FILE)):
        try:
            score = ScenarioScore.model_validate_json(p.read_text())
            check_schema_version(score.schema_version, str(p))
            scores.append(score)
        except Exception as e:
            logger.warning("Failed to parse {}: {}", p, e)
    return scores


def _read_setup_errors(outcomes_root: Path) -> list[dict[str, Any]]:
    """Read ``setup_errors.json`` from the run dir.

    Written by ``runner/phases/setup_groups`` whenever a group fails its
    setup gate (external ``working_dir``, ``workspace_isolation: none``,
    ``fixture_repo``/``working_dir`` mutex, fixture clone failure, or
    agent ``setup_group`` raising). Missing means "no group setup
    failed"; an unreadable file falls back to an empty list so the
    aggregate still completes.
    """
    sidecar = outcomes_root / "setup_errors.json"
    if not sidecar.exists():
        return []
    try:
        raw = json.loads(sidecar.read_text())
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and "group" in item and "error" in item:
            out.append(
                {
                    "group": str(item["group"]),
                    "scenarios": list(item.get("scenarios") or []),
                    "error": str(item["error"]),
                }
            )
    return out


def _read_scenarios_skipped(outcomes_root: Path) -> int:
    """Read ``scenarios_skipped`` from ``run_meta.json`` for this run.

    The run phase persists the count produced by ``parse_and_filter``;
    we surface it on ``AggregatedResults`` so external tooling sees the
    full input fleet, not just what reached the scorer. Best-effort: a
    missing or unreadable ``run_meta.json`` (older runs, partial writes)
    falls back to ``0`` so the aggregate still completes.
    """
    from belt.constants import RUN_META_FILE

    meta_path = outcomes_root / RUN_META_FILE
    if not meta_path.exists():
        return 0
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return 0
    if not isinstance(meta, dict):
        return 0
    raw = meta.get("scenarios_skipped", 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _resolve_run_agent_name(outcomes_root: Path) -> str | None:
    """Resolve the agent name used for this run, for remediation lookup.

    Read order:

    1. ``run_meta.json``'s top-level ``agent`` field. ``commands/run.py``
       writes this at run-init from ``--agent`` (when set) or from the
       group configs (when uniform). A string means a single-agent run;
       a list means a heterogeneous run and no single remediation
       applies.
    2. Per-scenario ``_runtime_info.json`` sidecars (defensive fallback
       for runs where the ``run_meta.json`` enrichment pass failed and
       left the field absent). When every sidecar names the same agent,
       return it; otherwise return ``None`` so the headline falls back
       to the agent-agnostic hint.

    Returns ``None`` when the agent cannot be resolved or the run is
    heterogeneous - the caller treats that the same as "no hint".
    """
    import json

    from belt.constants import RUN_META_FILE, RUNTIME_INFO_FILE

    meta_path = outcomes_root / RUN_META_FILE
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        agent_field = meta.get("agent") if isinstance(meta, dict) else None
        if isinstance(agent_field, str) and agent_field:
            return agent_field
        # A list signals a heterogeneous run; no per-agent hint applies.
        if isinstance(agent_field, list):
            return None

    # Defensive fallback: synthesise the agent name from per-scenario
    # ``_runtime_info.json`` sidecars when ``run_meta.json`` enrichment
    # failed and the top-level ``agent`` field is absent.
    adapter_names: set[str] = set()
    for p in outcomes_root.rglob(RUNTIME_INFO_FILE):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        agent_section = data.get("agent") if isinstance(data, dict) else None
        if isinstance(agent_section, dict):
            name = agent_section.get("name")
            if isinstance(name, str) and name:
                adapter_names.add(name)
    if len(adapter_names) == 1:
        return next(iter(adapter_names))
    return None


def main(argv: list[str] | None = None) -> int:
    # Terminal logging is configured here (idempotent) and the file
    # handler is attached additively below once the outcomes directory
    # is known. The previous ``logger.remove()`` indiscriminately
    # detached the terminal handler installed by ``belt eval``, which
    # broke ``-v`` / ``BELT_LOG_LEVEL`` for nested invocations.
    from belt._logging import configure_terminal_logging

    configure_terminal_logging(0)

    parser = argparse.ArgumentParser(
        prog="belt aggregate",
        description="Aggregate evaluation scores. Default (no --threshold): report only, always exit 0.",
        allow_abbrev=False,
    )
    # ``add_argument`` calls below are alphabetised by long flag name.
    # Enforced by ``tests/test_cli_order.py``.
    parser.add_argument(
        "--export",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help=(
            "Run an exporter after aggregation, writing to PATH. Repeatable. "
            "NAME is any name shown by `belt doctor` under Exporters. "
            "Example: --export csv:results.csv --export junit:report.xml."
        ),
    )
    parser.add_argument(
        "--export-config",
        metavar="PATH",
        help="YAML file describing exporter entries (name + path + options).",
    )
    parser.add_argument(
        "--llm-fail-on",
        default=DEFAULT_LLM_FAIL_ON_STR,
        help=(
            f"Comma-separated LLM verdicts that count as failures. "
            f"Default ``{DEFAULT_LLM_FAIL_ON_STR}`` covers ternary 'low', "
            f"binary 'fail', and inconclusive verdicts on either scale. Pass "
            f"e.g. ``low,fail`` to treat inconclusive as informational only."
        ),
    )
    parser.add_argument("--run-dir", help="Outcomes run directory (default: latest)")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="MODE/DIM:PCT",
        help=(
            "Per-dimension failure threshold as mode/dimension:max_percent. "
            "Repeatable. Example: --threshold rules/execution:0 --threshold rules/trajectory:10. "
            "No --threshold flags = report only (exit 0)."
        ),
    )
    args = parser.parse_args(argv)

    thresholds: dict[str, dict[str, float]] = {}
    for raw in args.threshold:
        try:
            mode, dimension, pct = parse_threshold(raw)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))
        thresholds.setdefault(mode, {})[dimension] = pct

    if thresholds:
        validate_thresholds(thresholds)

    llm_fail_on = {s.strip() for s in args.llm_fail_on.split(",")}
    invalid_levels = llm_fail_on - VALID_LLM_FAIL_ON
    if invalid_levels:
        eprint(f"  ❌ Invalid --llm-fail-on values: {invalid_levels} (valid: {VALID_LLM_FAIL_ON})")
        return 1

    if args.run_dir:
        outcomes_root = Path(args.run_dir)
    else:
        from belt.manifest import Manifest

        latest = Manifest().latest_run
        if not latest:
            eprint("No previous run found - run runner first or pass --run-dir")
            return 1
        outcomes_root = Path(latest)

    if not outcomes_root.is_dir():
        eprint(f"Outcomes directory does not exist: {outcomes_root}")
        return 1

    logger.add(outcomes_root / LOG_FILE, level="DEBUG", rotation=None)

    scores = discover_scores(outcomes_root)
    if not scores:
        eprint(f"No {SCORE_FILE} files found under {outcomes_root}")
        return 1

    threshold_lines: list[str] | None = None
    threshold_checks: list[ThresholdCheck] | None = None
    thresholds_passed = True
    if thresholds:
        enforcer = ThresholdEnforcer(scores, llm_fail_on=llm_fail_on)
        threshold_lines, thresholds_passed, threshold_checks = enforcer.enforce(thresholds)

    run_label = str(outcomes_root.relative_to(EVAL_DIR)) if EVAL_DIR in outcomes_root.parents else str(outcomes_root)
    cost_timing = aggregate_cost_timing(outcomes_root, scores)
    reliability = compute_reliability(scores)
    # Best-effort agent-name resolution from the run_meta sidecar so the
    # bottom-line headline can carry an agent-specific remediation hint
    # (e.g. ``run `claude login```). Falls back to ``None`` when the
    # sidecar is missing or the agent set is heterogeneous - the
    # headline stays correct, just without the agent-specific suffix.
    agent_name = _resolve_run_agent_name(outcomes_root)
    judge_errors = collect_judge_errors(scores)
    agent_errors = collect_agent_errors(outcomes_root, scores, agent_name=agent_name, judge_errors=judge_errors)
    bottom_line = build_bottom_line(scores, agent_errors=agent_errors, judge_errors=judge_errors)

    try:
        results_path = outcomes_root / RESULTS_FILE
    except Exception:
        results_path = None

    # ``run_label`` is the run directory (relative to cwd when possible),
    # not ``results.json``. The renderer's footer points the user at
    # ``belt view <run_label>``, which only accepts a run directory.
    print_terminal(
        scores,
        run_label,
        threshold_lines,
        outcomes_root=outcomes_root,
        cost_timing=cost_timing,
        reliability=reliability,
        agent_errors=agent_errors,
        judge_errors=judge_errors,
    )

    failed = [s for s in scores if not s.overall_pass]
    setup_errors = _read_setup_errors(outcomes_root)
    # A run with group-setup failures is not "overall_pass" even when
    # every scenario that *did* run came back green - the user asked for
    # N scenarios and only N - M ran, so the bottom-line verdict must
    # reflect the missing work. Threshold enforcement still trumps this
    # (it can already flip the headline either way), so we only ratchet
    # the bool down, never up.
    overall_pass = not bool(failed) and not setup_errors
    aggregated = AggregatedResults(
        schema_version=SCHEMA_VERSION,
        total=len(scores),
        passed=len(scores) - len(failed),
        failed=len(failed),
        overall_pass=overall_pass,
        scenarios_skipped=_read_scenarios_skipped(outcomes_root),
        stats=build_stats(scores),
        cost_timing=cost_timing,
        reliability=reliability,
        agent_errors=agent_errors,
        judge_errors=judge_errors,
        bottom_line=bottom_line,
        thresholds_passed=thresholds_passed if thresholds else None,
        scenarios=[s.model_dump(mode="json") for s in scores],
        setup_errors=setup_errors,
    )
    # Disk shape stays a plain dict (sort_keys=False preserves a stable
    # field ordering for diff-friendly results.json across runs).
    results_json = aggregated.model_dump(mode="json", exclude_none=False)
    if results_path and not write_json(results_path, results_json, sort_keys=False):
        logger.error("Failed to write {}", RESULTS_FILE)

    # Build and emit the benchmark card after ``results.json`` is on disk
    # so card consumers (and the GITHUB_STEP_SUMMARY append below) see a
    # consistent view of the run. The card builder is best-effort:
    # missing inputs (older runs without ``run_meta.json`` provenance
    # blocks, partial sidecars) yield a card with empty optional fields
    # rather than aborting the aggregate.
    try:
        from belt.benchmark_card import build_card, render_markdown, write_card

        card = build_card(outcomes_root, results_json)
        # ``write_card`` returns the json + md paths but we don't print
        # them: the aggregator's single ``→ belt view <run_dir>`` footer
        # is the canonical artifact pointer. Printing each card path on
        # its own line duplicates the run directory three times for the
        # same actionable answer ("open the run dir").
        write_card(card, outcomes_root)
    except Exception as e:
        logger.warning("benchmark card emission failed: {}", e)
        card = None

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            markdown = build_markdown(scores, threshold_checks, thresholds_passed, judge_errors=judge_errors)
            with open(summary_path, "a") as f:
                f.write(markdown)
                # Append the human-readable card under the failures block
                # so a single click on the GitHub step summary reveals
                # full provenance alongside the pass/fail story.
                if card is not None:
                    f.write("\n")
                    f.write(render_markdown(card))
        except Exception as e:
            logger.error("Failed to write GitHub step summary: {}", e)

    export_rc = 0
    if args.export or args.export_config:
        from belt.commands.export import run_exporters

        export_rc = run_exporters(
            run_dir=outcomes_root,
            results=aggregated,
            scores=scores,
            to_specs=list(args.export or []),
            config_path=args.export_config,
        )

    if thresholds and not thresholds_passed:
        return 1

    # A judge-infra failure means at least one scenario's verdicts never
    # arrived; the rules-side pass cannot be trusted as a quality signal.
    # Surface this as a non-zero exit so CI never marks such a run green,
    # mirroring the bottom_line "do not treat passing rules as a green"
    # message that downstream readers already see.
    if judge_errors is not None:
        return 1

    return export_rc


if __name__ == "__main__":
    sys.exit(main())
