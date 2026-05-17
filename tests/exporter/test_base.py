# (c) JFrog Ltd. (2026)

"""Contract checks on :class:`BaseExporter`.

These pin Design Principle 3 ("base interface is closed to modification")
mechanically: the script ``scripts/check_design.py`` enforces signature
stability for ``BaseAgentAdapter``; this file does the equivalent for
``BaseExporter`` so a future contributor cannot quietly grow ``export()``
to ``export(ctx, output, options, *, run_id)`` and break every plugin.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from belt.exporter.base import BaseExporter
from belt.exporter.entities import ExportContext


class TestSignatureStability:
    def test_export_signature_is_frozen(self):
        # Mirrors the spirit of ``_BASE_SIGNATURES`` in scripts/check_design.py
        # for BaseAgentAdapter. A change here is a deliberate breaking change
        # to every plugin in the wild and must be reviewed accordingly.
        sig = inspect.signature(BaseExporter.export)
        assert tuple(sig.parameters) == ("self", "ctx", "output", "options")

    def test_name_property_is_abstract(self):
        # Both ``name`` and ``export`` MUST be abstract on the base; the
        # default :meth:`is_available` is concrete so plugins that have no
        # readiness signal opt in for free.
        assert getattr(BaseExporter.name.fget, "__isabstractmethod__", False)
        assert getattr(BaseExporter.export, "__isabstractmethod__", False)
        assert not getattr(BaseExporter.is_available, "__isabstractmethod__", False)


class TestUninstantiable:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseExporter()  # type: ignore[abstract]


class _MinimalExporter(BaseExporter):
    """Smallest possible concrete subclass - the plugin contract floor."""

    @property
    def name(self) -> str:
        return "_minimal"

    def export(self, ctx: ExportContext, output: Path, options: dict[str, Any]) -> None:
        output.write_text("minimal")


class TestMinimalSubclass:
    def test_default_is_available_returns_true(self, tmp_path: Path):
        # Plugins that don't override ``is_available`` must still satisfy
        # the readiness probe used by ``belt doctor``.
        assert _MinimalExporter().is_available() is True

    def test_minimal_subclass_runs(self, export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "out.txt"
        _MinimalExporter().export(export_context, out, {})
        assert out.read_text() == "minimal"
