# (c) JFrog Ltd. (2026)

"""Tests for ``runner.phases.parse_filter`` - the always-on summary line,
the ``scenarios_skipped`` count, and the ``--strict`` gate that turns
malformed scenarios into a non-zero exit instead of a silent log line.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from belt import _internal_envvars
from belt.runner.phases.parse_filter import parse_and_filter


@pytest.fixture(autouse=True)
def _restore_scenarios_root_env() -> None:
    """``parse_and_filter`` writes the resolved scenarios root into
    ``_BELT_SCENARIOS_ROOT`` so downstream phases (and the scorer's
    scenarios-map lookup) can find the tree without a CLI plumb. The env
    var is process-global; if a test pointed it at ``tmp_path`` and we
    didn't restore the original value, unrelated scorer tests would
    resolve scenario paths against the dead tmp dir and fail in
    confusing ways.
    """
    original = os.environ.get(_internal_envvars.SCENARIOS_ROOT)
    yield
    if original is None:
        os.environ.pop(_internal_envvars.SCENARIOS_ROOT, None)
    else:
        os.environ[_internal_envvars.SCENARIOS_ROOT] = original


def _make_args(path: Path, *, strict: bool = False, strict_config: bool = False) -> argparse.Namespace:
    """Build the minimal argparse.Namespace ``parse_and_filter`` consumes.

    Only the fields actually read by the function are populated - keeps
    the test surface narrow so a future arg addition does not silently
    break these tests.
    """
    return argparse.Namespace(
        path=str(path),
        tags=None,
        exclude_tags=None,
        scenarios=None,
        agent=None,
        strict=strict,
        strict_config=strict_config,
    )


def _write_group(group_dir: Path, *, agent: str = "stub") -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "_config.json").write_text(json.dumps({"agent": agent}))


def _write_scenario(group_dir: Path, name: str, *, message: str = "hi") -> None:
    (group_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "description": "test",
                "turns": [{"message": message}],
            }
        )
    )


def _write_malformed(group_dir: Path, name: str) -> None:
    """Write a scenario JSON Pydantic will reject (unknown top-level key)."""
    (group_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "description": "test",
                "turns": [{"message": "hi"}],
                # ``Scenario`` declares ``extra="forbid"`` - a typo here
                # is the exact failure mode the loader silently dropped
                # before scenarios_skipped surfaced it.
                "this_field_does_not_exist": True,
            }
        )
    )


# ── Summary line ──


class TestSummaryLine:
    def test_all_valid_scenarios_summary_omits_skipped_clause(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        group = tmp_path / "g"
        _write_group(group)
        _write_scenario(group, "a")
        _write_scenario(group, "b")

        result = parse_and_filter(_make_args(tmp_path))

        assert isinstance(result, tuple)
        scenarios_root, matched, skipped = result
        assert scenarios_root == tmp_path.resolve()
        assert len(matched) == 1
        assert sum(len(mg.scenarios) for mg in matched) == 2
        assert skipped == 0

        out = capsys.readouterr().err
        assert "Loaded 2 scenarios across 1 groups" in out
        # No "(M malformed skipped)" parenthetical when nothing was skipped.
        assert "malformed skipped" not in out

    def test_one_valid_one_malformed_summary_reports_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        group = tmp_path / "g"
        _write_group(group)
        _write_scenario(group, "valid")
        _write_malformed(group, "broken")

        result = parse_and_filter(_make_args(tmp_path))

        assert isinstance(result, tuple)
        _, matched, skipped = result
        # The valid scenario survives; the malformed one is dropped.
        assert sum(len(mg.scenarios) for mg in matched) == 1
        assert skipped == 1

        out = capsys.readouterr().err
        assert "Loaded 1 scenarios across 1 groups (1 malformed skipped)" in out

    def test_summary_aggregates_across_multiple_groups(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        for name in ("g1", "g2"):
            group = tmp_path / name
            _write_group(group)
            _write_scenario(group, "a")
            _write_malformed(group, "broken")

        result = parse_and_filter(_make_args(tmp_path))

        assert isinstance(result, tuple)
        _, matched, skipped = result
        assert len(matched) == 2
        assert skipped == 2

        out = capsys.readouterr().err
        assert "Loaded 2 scenarios across 2 groups (2 malformed skipped)" in out


# ── --strict gate ──


class TestStrictMode:
    def test_strict_with_malformed_returns_exit_1_and_lists_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        group = tmp_path / "g"
        _write_group(group)
        _write_scenario(group, "valid")
        _write_malformed(group, "broken")

        result = parse_and_filter(_make_args(tmp_path, strict=True))

        assert result == 1
        out = capsys.readouterr().err
        # Summary still prints (operator sees the count even on abort).
        assert "(1 malformed skipped)" in out
        # And the explicit error list with the offending file name.
        assert "--strict: refusing to run with malformed scenarios" in out
        assert "broken.json" in out

    def test_strict_with_all_valid_scenarios_proceeds_normally(self, tmp_path: Path) -> None:
        group = tmp_path / "g"
        _write_group(group)
        _write_scenario(group, "a")

        result = parse_and_filter(_make_args(tmp_path, strict=True))

        assert isinstance(result, tuple)
        _, matched, skipped = result
        assert sum(len(mg.scenarios) for mg in matched) == 1
        assert skipped == 0

    def test_non_strict_with_malformed_continues_with_survivors(self, tmp_path: Path) -> None:
        """Default behaviour (no ``--strict``) is unchanged: malformed
        files are logged + dropped and the run proceeds. Regression
        guard: the new code must not change this path."""
        group = tmp_path / "g"
        _write_group(group)
        _write_scenario(group, "valid")
        _write_malformed(group, "broken")

        result = parse_and_filter(_make_args(tmp_path, strict=False))

        assert isinstance(result, tuple)
        _, matched, skipped = result
        assert sum(len(mg.scenarios) for mg in matched) == 1
        assert skipped == 1


# ── --strict-config gate ──


def _write_silent_typo_scenario(group_dir: Path, name: str) -> None:
    """Write a scenario that Pydantic accepts but ``--strict-config`` rejects.

    ``tools_invoke`` (no ``d``) lands in ``TurnExpectation.model_extra``
    today (extra="allow") and produces zero coverage at runtime. The
    file loads cleanly; only the strict validator catches it.
    """
    (group_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "description": "test",
                "turns": [{"message": "hi", "expect": {"tools_invoke": ["foo"]}}],
            }
        )
    )


class TestStrictConfigMode:
    def test_strict_config_rejects_silent_typo(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        group = tmp_path / "g"
        _write_group(group)
        _write_silent_typo_scenario(group, "typo")

        result = parse_and_filter(_make_args(tmp_path, strict_config=True))

        assert result == 1
        out = capsys.readouterr().err
        # The flag's name appears in the error so an operator can grep
        # logs and find which gate fired.
        assert "--strict-config: refusing to run with malformed scenarios" in out
        assert "typo.json" in out
        assert "tools_invoke" in out
        # Did-you-mean is part of the rejection message.
        assert "Did you mean 'tools_invoked'" in out

    def test_strict_config_aborts_independently_of_strict(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The two flags are orthogonal; ``--strict-config`` must abort on
        # config errors even when ``--strict`` is off.
        group = tmp_path / "g"
        _write_group(group)
        _write_silent_typo_scenario(group, "typo")

        result = parse_and_filter(_make_args(tmp_path, strict=False, strict_config=True))

        assert result == 1
        out = capsys.readouterr().err
        assert "--strict-config" in out

    def test_strict_config_default_off_keeps_silent_typo_passing(self, tmp_path: Path) -> None:
        # Regression guard: without the flag, today's permissive
        # behaviour stays untouched - a silent typo loads cleanly and
        # the run proceeds.
        group = tmp_path / "g"
        _write_group(group)
        _write_silent_typo_scenario(group, "typo")

        result = parse_and_filter(_make_args(tmp_path))

        assert isinstance(result, tuple)
        _, matched, skipped = result
        assert sum(len(mg.scenarios) for mg in matched) == 1
        assert skipped == 0

    def test_strict_config_clean_scenarios_pass(self, tmp_path: Path) -> None:
        group = tmp_path / "g"
        _write_group(group)
        _write_scenario(group, "ok")

        result = parse_and_filter(_make_args(tmp_path, strict_config=True))

        assert isinstance(result, tuple)
        _, matched, skipped = result
        assert sum(len(mg.scenarios) for mg in matched) == 1
        assert skipped == 0

    def test_strict_config_rejects_group_config_typo(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # Group-config errors are unconditionally fatal (no scenarios
        # survive a misconfigured group).
        group = tmp_path / "g"
        group.mkdir()
        (group / "_config.json").write_text(json.dumps({"agent": "stub", "agnet": "stub"}))  # typo
        _write_scenario(group, "ok")

        result = parse_and_filter(_make_args(tmp_path, strict_config=True))

        assert result == 1
        out = capsys.readouterr().err
        assert "--strict-config: group config validation failed" in out
        assert "agnet" in out
