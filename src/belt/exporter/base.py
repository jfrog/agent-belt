# (c) JFrog Ltd. (2026)

"""Closed-to-modification base class for exporters (Design Principle 3).

Plugin authors subclass :class:`BaseExporter` and declare:

1. :attr:`name` - registry key used by ``--export <name>:<path>``.
2. :meth:`is_available` - readiness probe (``doctor``, preflight). Built-ins
   always return ``True``; vendor plugins probe credentials / SDK installs.
3. :meth:`export` - write the run to ``output``, given an
   :class:`ExportContext` and a free-form ``options`` dict (merged from
   ``--export-config`` YAML + per-call defaults).

The signature is frozen by Design Principle 3: adding parameters here would
break every downstream exporter. Per-exporter tuning rides on the ``options``
dict instead.

A failing exporter must not abort other exporters or the run. The CLI driver
(:mod:`belt.commands.export`) wraps each :meth:`export` call in its own
try/except and continues on failure, surfacing each as a typed
:class:`belt.errors.BeltError` per Design Principle 7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from belt.exporter.entities import ExportContext


class BaseExporter(ABC):
    """ABC for post-aggregation result exporters.

    Built-in subclasses live as siblings to this module under
    ``belt.exporter.<name>`` (mirroring the layout of concrete agents
    under ``belt.agent.<name>``); third-party subclasses live in their
    own packages and register via the ``belt.exporters`` entry-point
    group. The contract is intentionally minimal: hand the exporter a typed
    snapshot of a run + a destination path + an options dict; the exporter
    is responsible for everything else (formatting, network I/O, retries).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable registry key (e.g. ``"csv"``, ``"junit"``, ``"markdown"``)."""

    def is_available(self) -> bool:  # pragma: no cover - default trivially true
        """Return True iff the exporter can run in the current environment.

        Built-in exporters always return True (filesystem-only, stdlib-only).
        Vendor plugins should probe optional dependencies and credentials so
        ``belt doctor`` can render a green/red row at the same fidelity as
        the agent and LLM-provider sections. Default ``True`` keeps the path
        minimal for plugins that have no readiness signal.
        """
        return True

    @abstractmethod
    def export(self, ctx: ExportContext, output: Path, options: dict[str, Any]) -> None:
        """Write the run snapshot ``ctx`` to ``output``.

        Args:
            ctx: Read-only :class:`ExportContext` containing
                :class:`belt.entities.AggregatedResults`, the
                trial-expanded list of :class:`belt.entities.ScenarioScore`,
                the run directory, and (when present) the parsed
                ``benchmark-card.json``.
            output: Destination path. Plugins that don't write files (e.g.
                vendor APIs) should ignore the value but still receive a
                non-empty placeholder so the CLI surface stays uniform.
            options: Merged exporter-specific options from
                ``--export-config`` YAML. Empty dict when none are supplied.
        """
