# (c) JFrog Ltd. (2026)

"""Data contracts for the export phase.

Pydantic models, one per concern:

* :class:`ExportContext` - read-only snapshot of an aggregated run handed to
  :meth:`belt.exporter.base.BaseExporter.export`.
* :class:`ExporterEntry` - a single ``--export-config`` YAML entry: name,
  output path, and free-form options.
* :class:`ExporterFile` - the parsed top-level ``--export-config`` document
  (a list of :class:`ExporterEntry`).

Per Design Principle 2, entities carry data but no business logic. Helpers
that *act* on these shapes (trial collapsing, output path resolution) live in
:mod:`belt.exporter.helpers`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from belt.entities import AggregatedResults, ScenarioScore


class ExportContext(BaseModel):
    """Read-only view of one aggregated run, fed to every exporter.

    The CLI driver builds one :class:`ExportContext` per ``belt export``
    invocation and re-uses it across every requested exporter. Exporters MUST
    treat the fields as read-only - the framework does not deep-copy because
    that would re-serialise large per-scenario score blocks for every
    additional exporter.

    ``scores`` is the *trial-expanded* list (one entry per
    ``<scenario>__trial_N`` outcome dir under ``--trials N``). Reliability
    summaries live on ``results.reliability``; the helper
    :func:`belt.exporter.helpers.collapse_trials` re-groups by base
    scenario name when the exporter needs the collapsed view (e.g. for
    JUnit, where one ``<testcase>`` per scenario is the convention).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_dir: Path
    results: AggregatedResults
    scores: list[ScenarioScore] = Field(default_factory=list)
    benchmark_card: Optional[dict[str, Any]] = None


class ExporterEntry(BaseModel):
    """One entry in an ``--export-config`` YAML document.

    The YAML maps directly:

    .. code-block:: yaml

        exporters:
          - name: junit
            path: report.xml
            options:
              suite_name: my-matrix-build
              max_body_bytes: 16384
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    options: dict[str, Any] = Field(default_factory=dict)


class ExporterFile(BaseModel):
    """Top-level shape of the YAML document accepted by ``--export-config``.

    The single key ``exporters`` is a list of :class:`ExporterEntry`. Pydantic
    rejects unknown keys (``extra='forbid'``) so a typo in the doc surfaces at
    parse time rather than as a silent no-op at run time.
    """

    model_config = ConfigDict(extra="forbid")

    exporters: list[ExporterEntry] = Field(default_factory=list)
