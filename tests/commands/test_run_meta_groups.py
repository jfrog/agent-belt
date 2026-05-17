# (c) JFrog Ltd. (2026)

"""Tests for per-group provenance in ``run_meta.json``.

``initialize_run_dir`` snapshots every matched group's effective
config under ``scenarios.matched_groups``. Downstream tooling (results
viewer, dashboards, CI gates) must be able to tell from artifacts
alone whether a group ran with ``workspace_isolation: "none"`` or
under a ``working_dir`` / ``fixture_repo``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from belt.commands import run as runner_cli
from belt.scenario import GroupConfig


def _bare_args() -> SimpleNamespace:
    """Minimal args namespace ``initialize_run_dir`` reads via ``getattr``."""
    return SimpleNamespace(
        progress="plain",
        agent=None,
        groups=None,
        tags=None,
        exclude_tags=None,
        scenarios=None,
        workers=1,
        trials=1,
        no_stream=False,
        scenario_delay=0.0,
        thresholds=None,
        threshold=None,
    )


def _matched_group(name: str, config: GroupConfig) -> SimpleNamespace:
    """Stub ``MatchedGroup`` with the attribute surface the runner reads."""
    return SimpleNamespace(
        name=name,
        config=config,
        group_dir=Path("/tmp/parity-stub"),
        scenarios=[],
    )


def test_run_meta_records_per_group_workspace_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_meta.json`` carries ``workspace_isolation`` for every matched group.

    Without this snapshot, a downstream reader inspecting the run
    directory cannot tell that ``group_inplace`` ran without
    per-scenario worktrees - even if ``allow_inplace=true`` is in
    ``invocation``, the per-group value is the load-bearing one.
    """
    for name in list(os.environ):
        if name.startswith("BELT_"):
            monkeypatch.delenv(name, raising=False)

    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)

    ctx = SimpleNamespace(
        run_dir=run_dir,
        scenarios_root=tmp_path,
        workspace=tmp_path,
        args=_bare_args(),
        scenarios_skipped=0,
        matched_groups=[
            _matched_group("group_isolated", GroupConfig(agent="claude-code")),
            _matched_group(
                "group_inplace",
                GroupConfig(agent="claude-code", workspace_isolation="none"),
            ),
            _matched_group(
                "group_with_fixture",
                GroupConfig(
                    agent="claude-code",
                    fixture_repo="https://example.test/repo.git",
                    fixture_ref="main",
                ),
            ),
        ],
    )

    monkeypatch.setattr(runner_cli, "_configure_logging", lambda *a, **k: None)
    rc = runner_cli.initialize_run_dir(ctx)
    assert rc is None

    meta = json.loads((run_dir / "run_meta.json").read_text())
    matched = meta["scenarios"]["matched_groups"]

    by_name = {entry["name"]: entry for entry in matched}
    assert set(by_name) == {"group_isolated", "group_inplace", "group_with_fixture"}

    assert by_name["group_isolated"]["workspace_isolation"] == "git-worktree"
    assert by_name["group_isolated"]["agent"] == "claude-code"

    assert by_name["group_inplace"]["workspace_isolation"] == "none"

    fixture_entry = by_name["group_with_fixture"]
    assert fixture_entry["fixture_repo"] == "https://example.test/repo.git"
    assert fixture_entry["fixture_ref"] == "main"


def test_run_meta_omits_optional_fields_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional config fields stay out of the snapshot when not set.

    Keeps ``run_meta.json`` from filling with ``null`` placeholders
    that would force every reader to handle the ``None`` case.
    """
    for name in list(os.environ):
        if name.startswith("BELT_"):
            monkeypatch.delenv(name, raising=False)

    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)

    ctx = SimpleNamespace(
        run_dir=run_dir,
        scenarios_root=tmp_path,
        workspace=tmp_path,
        args=_bare_args(),
        scenarios_skipped=0,
        matched_groups=[_matched_group("g", GroupConfig(agent="claude-code"))],
    )

    monkeypatch.setattr(runner_cli, "_configure_logging", lambda *a, **k: None)
    runner_cli.initialize_run_dir(ctx)

    entry = json.loads((run_dir / "run_meta.json").read_text())["scenarios"]["matched_groups"][0]
    assert "working_dir" not in entry
    assert "fixture_repo" not in entry
    assert "fixture_ref" not in entry
