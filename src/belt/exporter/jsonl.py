# (c) JFrog Ltd. (2026)

"""JSONL exporter - one JSON object per scenario, line-delimited.

Stream-friendly: append-mode pipelines (BigQuery, DuckDB, jq, fluentd) ingest
the file without parsing the whole document. Each row carries the full
:class:`ScenarioScore` model_dump plus the matching cost/timing record from
``results.cost_timing.scenarios`` so a downstream tool needs zero joins.

Options:
    granularity: ``"scenario"`` (default, one row per trial entry on disk -
        matches the natural shape of ``ExportContext.scores``) or
        ``"summary"`` (one row per base scenario, with ``trials`` /
        ``trials_passed`` aggregates).

Stdlib-only: no optional dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from belt.exporter.base import BaseExporter
from belt.exporter.entities import ExportContext
from belt.exporter.helpers import collapse_trials


def _scenario_cost_timing(ctx: ExportContext, key: str) -> dict[str, Any]:
    for entry in ctx.results.cost_timing.get("scenarios", []) or []:
        if entry.get("scenario") == key:
            return entry
    return {}


class JsonlExporter(BaseExporter):
    """One JSON object per row, UTF-8, ``\\n``-delimited."""

    @property
    def name(self) -> str:
        return "jsonl"

    def export(self, ctx: ExportContext, output: Path, options: dict[str, Any]) -> None:
        granularity = options.get("granularity", "scenario")
        if granularity not in ("scenario", "summary"):
            granularity = "scenario"

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            if granularity == "summary":
                for key, trial_scores in collapse_trials(ctx.scores).items():
                    record = {
                        "scenario": key,
                        "trials": len(trial_scores),
                        "trials_passed": sum(1 for s in trial_scores if s.overall_pass),
                        "overall_pass": all(s.overall_pass for s in trial_scores),
                        "tags": sorted({t for s in trial_scores for t in s.tags}),
                        "cost_timing": _scenario_cost_timing(ctx, key),
                    }
                    f.write(json.dumps(record, sort_keys=True) + "\n")
            else:
                for s in ctx.scores:
                    record = s.model_dump(mode="json")
                    record["cost_timing"] = _scenario_cost_timing(ctx, f"{s.group}/{s.scenario_name}")
                    f.write(json.dumps(record, sort_keys=True) + "\n")
