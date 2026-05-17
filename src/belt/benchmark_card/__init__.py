# (c) JFrog Ltd. (2026)

"""Reproducibility manifest (benchmark card) for an evaluation run.

A benchmark card is the canonical, sharable artifact for "what exactly
was evaluated, and what was the answer". Two runs with the same
scenario names can disagree because of hidden variables - agent
version, model version, local auth source, scenario file SHA, fixture
branch, scoring config, parallelism, streaming mode, provider
behaviour. This package captures those variables in a single
Pydantic-validated record so downstream tooling, CI artifacts, and PR
comments have a single answer to point at.

Two artifacts are written per ``belt eval`` run, both inside the
run directory:

- ``benchmark-card.json`` - the full machine-readable card (see
  :class:`BenchmarkCard`).
- ``benchmark-card.md`` - a human-readable rendering, suitable for
  posting to ``$GITHUB_STEP_SUMMARY`` or pasting into an issue.

Inputs are assembled from three on-disk sources written earlier in the
pipeline; see :func:`build_card` for the full list. Secret hygiene is
delegated to :mod:`belt._redact` - no field on the card admits a
free-form value that bypasses the redactor.

Module layout (internal):

- :mod:`.entities` - Pydantic data contracts.
- :mod:`.collect` - run-phase static-provenance collectors.
- :mod:`.build` - aggregate-phase card assembler.
- :mod:`.render` - Markdown renderer for ``$GITHUB_STEP_SUMMARY``.
- :mod:`.io` - filesystem helpers (read/write/iso_utc).

Public surface is the union of names re-exported below.
"""

from __future__ import annotations

from .build import build_card
from .collect import collect_belt_provenance, collect_host_provenance, collect_invocation, hash_scenario_files
from .entities import (
    AgentIdentity,
    AgentProvenance,
    BeltProvenance,
    BenchmarkCard,
    CliIdentity,
    CostTimingSummary,
    FixtureProvenance,
    HostProvenance,
    Invocation,
    JudgeProvenance,
    RuntimeConfig,
    ScenarioFile,
    ScenarioSelection,
    ScoreSummary,
    ScoringConfig,
)
from .io import iso_utc as _iso_utc  # used by commands/run.py
from .io import load_results_for_card, write_card
from .render import render_markdown

__all__ = [
    "AgentIdentity",
    "AgentProvenance",
    "BeltProvenance",
    "BenchmarkCard",
    "CliIdentity",
    "CostTimingSummary",
    "FixtureProvenance",
    "HostProvenance",
    "Invocation",
    "JudgeProvenance",
    "RuntimeConfig",
    "ScenarioFile",
    "ScenarioSelection",
    "ScoreSummary",
    "ScoringConfig",
    "build_card",
    "collect_belt_provenance",
    "collect_host_provenance",
    "collect_invocation",
    "hash_scenario_files",
    "load_results_for_card",
    "render_markdown",
    "write_card",
]
