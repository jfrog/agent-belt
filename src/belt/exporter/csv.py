# (c) JFrog Ltd. (2026)

"""CSV exporter - one row per scenario.

Default columns mirror the per-scenario summary in the terminal renderer so a
spreadsheet view of an eval run is familiar at a glance:

    group, scenario, passed, tags, trials, trials_passed, agent_cost_usd,
    judge_cost_usd, total_cost_usd, total_seconds, failed_rule_checks,
    llm_low_dimensions

Optional columns are emitted only when the source data populates them so the
file stays clean for users who don't run with cost tracking or LLM scoring.

Untrusted text reaches CSV cells from scenario tags, LLM reasoning snippets,
and rule-check ``details`` strings. CSV formula-injection hardening lives in
:func:`belt._safe.csv_safe` (OWASP guidance: prefix any cell whose first
character is one of ``=``, ``+``, ``-``, ``@``, ``\\t``, ``\\r`` with a single
quote so spreadsheet apps render the literal value).

Options:
    delimiter: column delimiter (default ``,``).
    granularity: ``"scenario"`` (default) emits one row per scenario;
        ``"trial"`` emits one row per trial entry (mirrors the on-disk
        ``__trial_N`` outcome dirs under ``--trials N``).

Stdlib-only: no optional dependencies.
"""

from __future__ import annotations

import csv as _stdlib_csv
from pathlib import Path
from typing import Any, Iterable

from belt._safe import csv_safe
from belt.entities import ScenarioScore
from belt.exporter.base import BaseExporter
from belt.exporter.entities import ExportContext
from belt.exporter.helpers import collapse_trials
from belt.scorer.payloads import PerTurnLLMPayload, iter_dimension_feedback, iter_llm_payloads

_BASE_FIELDS: tuple[str, ...] = (
    "group",
    "scenario",
    "passed",
    "tags",
    "trials",
    "trials_passed",
    "agent_cost_usd",
    "judge_cost_usd",
    "total_cost_usd",
    "total_seconds",
    "failed_rule_checks",
    "llm_low_dimensions",
)

_PER_TURN_FIELDS: tuple[str, ...] = (
    "group",
    "scenario",
    "scorer_key",
    "turn_idx",
    "dimension",
    "score",
    "reasoning",
    "judge_errored",
)
"""Sidecar CSV columns for per-turn LLM judging.

Written to ``<output>.per_turn<ext>`` (e.g. ``results.per_turn.csv``)
alongside the main per-scenario CSV when at least one
:class:`PerTurnLLMPayload` is present in ``ctx.scores``. Sidecar
inherits the same trust boundary as the user-supplied ``output``
path; no new path-traversal surface.

Every cell flows through :func:`belt._safe.csv_safe` for OWASP
formula-injection hardening.
"""


def _row_total_cost(agent: Any, judge: Any) -> float | None:
    """Return ``agent + judge``, or ``None`` when both components are missing."""
    a = float(agent) if isinstance(agent, (int, float)) else None
    j = float(judge) if isinstance(judge, (int, float)) else None
    if a is None and j is None:
        return None
    return (a or 0.0) + (j or 0.0)


def _scenario_cost_timing(ctx: ExportContext, group: str, scenario: str) -> dict[str, Any]:
    """Look up the per-scenario cost/timing entry the aggregator already wrote.

    The aggregator's ``cost_timing.scenarios`` array carries one record per
    scored scenario; reusing that data avoids re-walking turn output files.
    """
    key = f"{group}/{scenario}"
    for entry in ctx.results.cost_timing.get("scenarios", []) or []:
        if entry.get("scenario") == key:
            return entry
    return {}


class CsvExporter(BaseExporter):
    """One row per scenario (default) or per trial."""

    @property
    def name(self) -> str:
        return "csv"

    def export(self, ctx: ExportContext, output: Path, options: dict[str, Any]) -> None:
        delimiter = options.get("delimiter", ",")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            delimiter = ","
        granularity = options.get("granularity", "scenario")
        if granularity not in ("scenario", "trial"):
            granularity = "scenario"

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = _stdlib_csv.writer(f, delimiter=delimiter, quoting=_stdlib_csv.QUOTE_MINIMAL)
            writer.writerow(_BASE_FIELDS)
            if granularity == "trial":
                for s in ctx.scores:
                    writer.writerow(self._row(ctx, s.group, s.scenario_name, [s]))
            else:
                for key, group_scores in collapse_trials(ctx.scores).items():
                    group, scenario = key.split("/", 1)
                    writer.writerow(self._row(ctx, group, scenario, group_scores))

        # Sidecar: only when at least one scenario has a PerTurnLLMPayload.
        # ``output.with_suffix(".per_turn" + output.suffix)`` keeps the
        # extension intact so ``results.csv`` → ``results.per_turn.csv``
        # and ``results.tsv`` → ``results.per_turn.tsv``.
        if any(isinstance(payload, PerTurnLLMPayload) for s in ctx.scores for _name, payload in iter_llm_payloads(s)):
            sidecar = output.with_suffix(".per_turn" + output.suffix)
            with sidecar.open("w", newline="", encoding="utf-8") as f:
                writer = _stdlib_csv.writer(f, delimiter=delimiter, quoting=_stdlib_csv.QUOTE_MINIMAL)
                writer.writerow(_PER_TURN_FIELDS)
                for s in ctx.scores:
                    for scorer_name, payload in iter_llm_payloads(s):
                        if not isinstance(payload, PerTurnLLMPayload):
                            continue
                        for tv in payload.turns:
                            judge_errored = "true" if tv.judge_errored else "false"
                            if not tv.dimensions and tv.judge_errored:
                                writer.writerow(
                                    [
                                        csv_safe(s.group),
                                        csv_safe(s.scenario_name),
                                        csv_safe(scorer_name),
                                        str(tv.turn_idx),
                                        "",
                                        csv_safe(tv.judge_error_type or "other"),
                                        "",
                                        judge_errored,
                                    ]
                                )
                                continue
                            for dim, vd in tv.dimensions.items():
                                writer.writerow(
                                    [
                                        csv_safe(s.group),
                                        csv_safe(s.scenario_name),
                                        csv_safe(scorer_name),
                                        str(tv.turn_idx),
                                        csv_safe(dim),
                                        csv_safe(vd.score),
                                        csv_safe(vd.reasoning),
                                        judge_errored,
                                    ]
                                )

    def _row(
        self,
        ctx: ExportContext,
        group: str,
        scenario: str,
        trial_scores: Iterable[ScenarioScore],
    ) -> list[str]:
        trials = list(trial_scores)
        passed_count = sum(1 for s in trials if s.overall_pass)
        first = trials[0]
        cost_timing = _scenario_cost_timing(ctx, group, first.scenario_name)
        # Tags are a property of the scenario, not the trial; first is fine.
        tags = ",".join(sorted(first.tags))
        # Reach into rules.checks (per-check granularity) but go through
        # iter_dimension_feedback for the LLM ``low`` summary - rules need
        # the per-check name in the cell, LLM only needs the dimension.
        from belt.scorer.payloads import RulesPayload

        rule_failures: list[str] = []
        llm_lows: list[str] = []
        for s in trials:
            rules = s.scores.get("rules")
            if isinstance(rules, RulesPayload):
                for c in rules.checks:
                    if c.passed is False:
                        rule_failures.append(f"{c.dimension}/{c.check}")
            # Walk every LLM-shaped payload (multi-judge non-consensus
            # and per-turn included). For per-turn the registered
            # iterator emits worst-of-turns rows with ``raw["worst_score"]``;
            # for scenario-level it emits ``raw["score"]``. Both shapes
            # surface ``"low"`` here so the headline column captures
            # every judge's downgrades.
            for fb in iter_dimension_feedback(s):
                token = fb.raw.get("score") or fb.raw.get("worst_score")
                if token == "low":
                    label = fb.dimension if fb.scorer_name == "llm" else f"{fb.scorer_name}/{fb.dimension}"
                    llm_lows.append(label)
        agent_cost = cost_timing.get("agent_cost_usd")
        judge_cost = cost_timing.get("judge_cost_usd")
        total_cost = _row_total_cost(agent_cost, judge_cost)
        raw = [
            group,
            scenario,
            "true" if (passed_count == len(trials) and trials) else "false",
            tags,
            str(len(trials)),
            str(passed_count),
            _fmt_float(agent_cost),
            _fmt_float(judge_cost),
            _fmt_float(total_cost),
            _fmt_float(cost_timing.get("total_seconds")),
            ";".join(sorted(set(rule_failures))),
            ";".join(sorted(set(llm_lows))),
        ]
        return [csv_safe(cell) for cell in raw]


def _fmt_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return ""
