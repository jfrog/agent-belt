# (c) JFrog Ltd. (2026)

"""Aggregate-time card builder.

Reads the static-provenance artifacts left by the run phase
(``run_meta.json``, per-scenario ``_runtime_info.json`` sidecars,
``run_fixtures.json``, ``score.json``) and merges them with the
in-memory ``results`` dict that the aggregator just produced. The
result is a fully-populated :class:`BenchmarkCard`.

The builder is best-effort: a missing input degrades to a safe default
(``"unknown"`` strings, empty collections) rather than raising. This
lets a card be reconstructed from older run directories that pre-date
some provenance fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from belt.constants import LOG_FILE, RESULTS_FILE, RUN_META_FILE

from .collect import collect_judges, collect_runtime_info_sidecars
from .entities import (
    AgentErrorsSummary,
    BeltProvenance,
    BenchmarkCard,
    CostTimingSummary,
    FixtureProvenance,
    HostProvenance,
    Invocation,
    JudgeErrorsSummary,
    RuntimeConfig,
    ScenarioSelection,
    ScoreSummary,
    ScoringConfig,
    TaskQualitySplit,
)
from .io import iso_utc, read_json, started_at_from_run_dir

_M = TypeVar("_M", bound=BaseModel)


def _validate_or_default(model: Type[_M], data: dict[str, Any] | None, default: _M) -> _M:
    """Validate ``data`` against ``model``, or return ``default`` if absent.

    Centralises the "use the run_meta block if present, otherwise fall
    back to a safe minimal instance" pattern that ``build_card`` applies
    to every section of the card. Removes six near-identical
    conditional blocks at the call site.
    """
    if not data:
        return default
    return model.model_validate(data)


def _build_agent_errors(results: dict[str, Any]) -> AgentErrorsSummary | None:
    """Project the aggregator's ``agent_errors`` block into the card's typed shape.

    The aggregator's block carries a ``per_scenario`` list with reply
    text snippets - useful for the terminal renderer, but verbose for a
    reproducibility manifest. The card keeps the rolled-up counts
    (scenarios with errors, vacuous passes, by-type tally, remediation
    hint) and discards the per-scenario detail.

    The ``task_quality`` partition copies all three axis counters
    (``env_failed_agent``, ``env_failed_judge``, ``task_failed``)
    verbatim from the aggregator: dropping either axis would force
    downstream tooling to re-derive it from raw turn outputs to
    distinguish "agent CLI rate-limited" from "judge backend
    rate-limited".
    """
    block = results.get("agent_errors")
    if not isinstance(block, dict):
        return None
    split_dict = block.get("task_quality")
    task_quality: TaskQualitySplit | None = None
    if isinstance(split_dict, dict):
        pct_raw = split_dict.get("pct")
        task_quality = TaskQualitySplit(
            attempted=int(split_dict.get("attempted", 0) or 0),
            env_failed=int(split_dict.get("env_failed", 0) or 0),
            env_failed_agent=int(split_dict.get("env_failed_agent", 0) or 0),
            env_failed_judge=int(split_dict.get("env_failed_judge", 0) or 0),
            completed=int(split_dict.get("completed", 0) or 0),
            passed=int(split_dict.get("passed", 0) or 0),
            task_failed=int(split_dict.get("task_failed", 0) or 0),
            pct=float(pct_raw) if isinstance(pct_raw, (int, float)) else None,
        )
    return AgentErrorsSummary(
        scenarios_with_errors=int(block.get("scenarios_with_errors", 0) or 0),
        scenarios_total=int(block.get("scenarios_total", 0) or 0),
        vacuous_passes=int(block.get("vacuous_passes", 0) or 0),
        by_error_type={str(k): int(v) for k, v in (block.get("by_error_type") or {}).items()},
        remediation=block.get("remediation"),
        task_quality=task_quality,
    )


def _build_judge_errors(results: dict[str, Any]) -> JudgeErrorsSummary | None:
    """Project the aggregator's ``judge_errors`` block into the card's typed shape.

    Structurally parallel to :func:`_build_agent_errors`: the aggregator's
    block carries a ``per_scenario`` list with vendor-specific error
    messages that we discard for the card; the rolled-up tally of
    judge-axis environmental failures is what a reproducibility manifest
    needs.
    """
    block = results.get("judge_errors")
    if not isinstance(block, dict):
        return None
    return JudgeErrorsSummary(
        scenarios_with_errors=int(block.get("scenarios_with_errors", 0) or 0),
        scenarios_total=int(block.get("scenarios_total", 0) or 0),
        by_error_type={str(k): int(v) for k, v in (block.get("by_error_type") or {}).items()},
    )


def _build_summary(results: dict[str, Any]) -> ScoreSummary:
    total = int(results.get("total", 0) or 0)
    passed = int(results.get("passed", 0) or 0)
    failed = int(results.get("failed", 0) or 0)
    return ScoreSummary(
        total=total,
        passed=passed,
        failed=failed,
        overall_pass=bool(results.get("overall_pass", False)),
        pass_rate=(passed / total) if total else 0.0,
        thresholds_passed=results.get("thresholds_passed"),
    )


def _build_cost_timing(results: dict[str, Any]) -> CostTimingSummary:
    """Read the aggregator's already-computed totals from ``results["cost_timing"]``."""
    cost_timing = results.get("cost_timing") or {}

    def _opt_float(key: str) -> float | None:
        value = cost_timing.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return CostTimingSummary(
        agent_cost_usd=_opt_float("agent_cost_usd"),
        judge_cost_usd=_opt_float("judge_cost_usd"),
        total_cost_usd=_opt_float("total_cost_usd"),
        total_seconds=cost_timing.get("total_seconds"),
        mean_seconds=cost_timing.get("mean_seconds"),
    )


def _link_path(run_dir: Path, name: str) -> str | None:
    p = run_dir / name
    return str(p) if p.exists() else None


def build_card(run_dir: Path, results: dict[str, Any]) -> BenchmarkCard:
    """Assemble a :class:`BenchmarkCard` from on-disk run artifacts.

    Inputs read from ``run_dir``:

    - ``run_meta.json`` (required): static run-time provenance written
      by ``commands/run.py``. Missing fields fall back to safe defaults
      so a run produced by an older belt version still yields a
      card.
    - ``<group>/<scenario>/_runtime_info.json`` (zero or more):
      per-scenario agent runtime info; deduplicated per group.
    - ``<group>/<scenario>/score.json`` (zero or more): used to
      discover the set of LLM judge backends actually consulted.

    ``results`` is the in-memory aggregator output (rather than the
    on-disk ``results.json``) to avoid a re-read race when ``aggregate``
    is the caller. External tooling can use
    :func:`belt.benchmark_card.load_results_for_card` to reproduce
    the same dict from ``results.json``.
    """
    run_meta = read_json(run_dir / RUN_META_FILE) or {}

    # ``run_fixtures.json`` is written by
    # ``runner/phases/setup_groups.py`` after every group finishes
    # setup, while branches and SHAs are still stable. Prefer it over
    # any inline ``fixtures`` block in ``run_meta.json`` (the latter is
    # a fallback for older runs).
    fixtures_block = run_meta.get("fixtures") or []
    run_fixtures = read_json(run_dir / "run_fixtures.json")
    if isinstance(run_fixtures, list) and run_fixtures:
        fixtures_block = run_fixtures

    belt = _validate_or_default(
        BeltProvenance,
        run_meta.get("belt"),
        BeltProvenance(version="unknown", install_kind="unknown"),
    )
    host = _validate_or_default(
        HostProvenance,
        run_meta.get("host"),
        HostProvenance(
            os="unknown",
            machine="unknown",
            python_version="unknown",
            python_implementation="unknown",
        ),
    )
    invocation = _validate_or_default(Invocation, run_meta.get("invocation"), Invocation())
    scenarios = _validate_or_default(
        ScenarioSelection,
        run_meta.get("scenarios"),
        ScenarioSelection(scenarios_root=run_meta.get("scenarios_root", "unknown")),
    )
    runtime = _validate_or_default(RuntimeConfig, run_meta.get("runtime"), RuntimeConfig())
    fixtures = [FixtureProvenance.model_validate(f) for f in fixtures_block if isinstance(f, dict)]

    agents = collect_runtime_info_sidecars(run_dir)
    judges = collect_judges(run_dir)

    scoring_block = run_meta.get("scoring") or {}
    scoring = ScoringConfig(
        modes=list(scoring_block.get("modes") or []),
        consensus=scoring_block.get("consensus"),
        thresholds=dict(scoring_block.get("thresholds") or {}),
        judges=judges,
    )

    links: dict[str, str] = {}
    for name in (RESULTS_FILE, RUN_META_FILE, LOG_FILE):
        path = _link_path(run_dir, name)
        if path:
            links[name] = path

    return BenchmarkCard(
        run_id=run_dir.name,
        started_at=run_meta.get("started_at") or started_at_from_run_dir(run_dir),
        ended_at=iso_utc(),
        belt=belt,
        host=host,
        invocation=invocation,
        scenarios=scenarios,
        fixtures=fixtures,
        agents=agents,
        scoring=scoring,
        runtime=runtime,
        cost_timing=_build_cost_timing(results),
        summary=_build_summary(results),
        agent_errors=_build_agent_errors(results),
        judge_errors=_build_judge_errors(results),
        links=links,
    )
