# (c) JFrog Ltd. (2026)

"""Export phase - post-aggregation result writers.

Reads ``results.json`` (typed as :class:`AggregatedResults`) plus per-scenario
``score.json`` files from a completed run directory and emits the data to one
or more configured destinations. The set of destinations is open: built-ins
ship in core, vendor plugins register via the ``belt.exporters``
entry-point group. The current built-in roster is whatever
:func:`belt.exporter.registry.available_exporters` reports - prose here
intentionally avoids enumerating it so docs do not drift behind the registry.

Architecturally, ``exporter/`` is a phase that runs after ``aggregator/`` and
sits parallel to ``runner/`` and ``scorer/``. Like its peer phases, it
communicates only via files on disk (Design Principle 1): it does not import
from runner, scorer, or aggregator implementation modules - only the
cross-phase entities in :mod:`belt.entities`.

Public entry points:

* :class:`belt.exporter.base.BaseExporter` - the closed-to-modification
  base class plugin authors subclass.
* :class:`belt.exporter.entities.ExportContext` - the read-only snapshot
  of an aggregated run handed to :meth:`BaseExporter.export`.
* :func:`belt.exporter.registry.get_exporter_class` - name-to-class
  resolver, mirroring the agent and scorer registries.

The CLI surface ``belt export`` lives in :mod:`belt.commands.export`;
``--export``/``--export-config`` chain flags on ``belt eval`` and
``belt aggregate`` proxy through that subcommand.
"""

from __future__ import annotations

from belt.exporter.base import BaseExporter
from belt.exporter.entities import ExportContext, ExporterEntry, ExporterFile
from belt.exporter.registry import available_exporters, get_exporter_class

__all__ = [
    "BaseExporter",
    "ExportContext",
    "ExporterEntry",
    "ExporterFile",
    "available_exporters",
    "get_exporter_class",
]
