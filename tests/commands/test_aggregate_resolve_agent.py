# (c) JFrog Ltd. (2026)

"""Tests for ``_resolve_run_agent_name``.

The resolver maps a run directory to a single agent name so the
aggregator headline can include an agent-specific remediation hint.
The source-of-truth is the top-level ``agent`` field in ``run_meta.json``;
runs where the enrichment pass failed and left the field absent fall back
to per-scenario ``_runtime_info.json`` sidecars.
"""

from __future__ import annotations

import json
from pathlib import Path

from belt.commands.aggregate import _resolve_run_agent_name
from belt.constants import RUN_META_FILE, RUNTIME_INFO_FILE


def _write_meta(run_dir: Path, body: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_META_FILE).write_text(json.dumps(body))


def _write_sidecar(scenario_dir: Path, body: dict) -> None:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / RUNTIME_INFO_FILE).write_text(json.dumps(body))


class TestResolveFromRunMeta:
    """Primary path: read the agent name directly from run_meta.json."""

    def test_string_agent_field(self, tmp_path: Path) -> None:
        # Single-agent runs (``--agent`` set, or uniform group configs)
        # write a string. The resolver returns it without scanning any
        # sidecars.
        _write_meta(tmp_path, {"agent": "claude-code"})
        assert _resolve_run_agent_name(tmp_path) == "claude-code"

    def test_list_agent_field_returns_none(self, tmp_path: Path) -> None:
        # Heterogeneous runs (different agents per group) write a list.
        # The aggregator headline has no single remediation that fits, so
        # the resolver returns ``None`` and the headline falls back to
        # the agent-agnostic hint.
        _write_meta(tmp_path, {"agent": ["claude-code", "codex"]})
        assert _resolve_run_agent_name(tmp_path) is None

    def test_null_agent_falls_through_to_sidecars(self, tmp_path: Path) -> None:
        # ``run_meta.json`` exists but has no useful ``agent`` field
        # (older belt, or enrichment failed). The resolver should
        # still find the agent via the sidecar fallback.
        _write_meta(tmp_path, {"schema_version": "1"})
        _write_sidecar(tmp_path / "g" / "s", {"agent": {"name": "claude-code"}})
        assert _resolve_run_agent_name(tmp_path) == "claude-code"

    def test_run_meta_takes_precedence_over_sidecars(self, tmp_path: Path) -> None:
        # If both sources are present, ``run_meta.json`` wins. The
        # sidecars may legitimately disagree if a third-party plugin
        # writes a non-canonical class name; ``run_meta.json`` is the
        # authoritative record of what the user invoked.
        _write_meta(tmp_path, {"agent": "claude-code"})
        _write_sidecar(tmp_path / "g" / "s", {"agent": {"name": "codex"}})
        assert _resolve_run_agent_name(tmp_path) == "claude-code"


class TestSidecarFallback:
    """Defensive path: no run_meta.json, derive from per-scenario sidecars."""

    def test_uniform_sidecars(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path / "g" / "a", {"agent": {"name": "claude-code"}})
        _write_sidecar(tmp_path / "g" / "b", {"agent": {"name": "claude-code"}})
        assert _resolve_run_agent_name(tmp_path) == "claude-code"

    def test_heterogeneous_sidecars_return_none(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path / "g" / "a", {"agent": {"name": "claude-code"}})
        _write_sidecar(tmp_path / "g" / "b", {"agent": {"name": "codex"}})
        assert _resolve_run_agent_name(tmp_path) is None

    def test_no_inputs(self, tmp_path: Path) -> None:
        # Empty run dir (or runs that pre-date sidecar emission) must
        # not crash the aggregator.
        assert _resolve_run_agent_name(tmp_path) is None

    def test_malformed_meta_falls_through(self, tmp_path: Path) -> None:
        # Corrupt JSON in ``run_meta.json`` should fall through to the
        # sidecar path rather than aborting the aggregator.
        (tmp_path / RUN_META_FILE).write_text("{not json")
        _write_sidecar(tmp_path / "g" / "s", {"agent": {"name": "claude-code"}})
        assert _resolve_run_agent_name(tmp_path) == "claude-code"
