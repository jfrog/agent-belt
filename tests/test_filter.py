# (c) JFrog Ltd. (2026)

"""Tests for belt.filter - ScenarioFilter shared filtering logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from belt.entities import GroupConfig, Scenario
from belt.errors import ScenarioError
from belt.filter import ScenarioFilter

# ── Helpers ──


def _make_scenario(name: str = "test", tags: list[str] | None = None) -> Scenario:
    return Scenario(name=name, description="d", tags=tags or [], turns=[])


def _make_group_config(agent: str = "stub", default_tags: list[str] | None = None) -> GroupConfig:
    return GroupConfig(agent=agent, default_tags=default_tags or [])


def _write_group(base: Path, group_name: str, scenarios: list[str] | None = None) -> Path:
    """Create a group directory with _config.json and optional scenario files."""
    group_dir = base / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    config = {"agent": "stub"}
    (group_dir / "_config.json").write_text(json.dumps(config))
    for s in scenarios or []:
        data = {"name": s, "description": "test", "turns": [{"message": "hi", "expect": {"has_reply": True}}]}
        (group_dir / f"{s}.json").write_text(json.dumps(data))
    return group_dir


# ── Construction ──


class TestFromCliArgs:
    def test_no_filters(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path)
        assert sf.is_empty
        assert not sf.has_tag_filter
        assert not sf.has_path_filter

    def test_tags_only(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, tags="smoke,fast")
        assert sf.tags == frozenset({"smoke", "fast"})
        assert sf.has_tag_filter
        assert not sf.has_path_filter
        assert not sf.is_empty

    def test_tags_whitespace_handling(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, tags=" smoke , fast , ")
        assert sf.tags == frozenset({"smoke", "fast"})

    def test_tags_empty_string(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, tags="")
        assert sf.tags == frozenset()
        assert not sf.has_tag_filter

    def test_scenarios_group_path(self, tmp_path: Path):
        _write_group(tmp_path, "claude-code")
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="claude-code")
        assert sf.has_path_filter
        assert sf.parsed_paths is not None
        assert len(sf.parsed_paths) == 1
        assert sf.parsed_paths[0][1] is None  # whole group

    def test_scenarios_specific_scenario(self, tmp_path: Path):
        _write_group(tmp_path, "claude-code", ["simple_math"])
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="claude-code/simple_math")
        assert sf.parsed_paths is not None
        assert len(sf.parsed_paths) == 1
        assert sf.parsed_paths[0][1] == "simple_math"

    def test_scenarios_strips_json_extension(self, tmp_path: Path):
        _write_group(tmp_path, "claude-code", ["simple_math"])
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="claude-code/simple_math.json")
        assert sf.parsed_paths is not None
        assert sf.parsed_paths[0][1] == "simple_math"

    def test_scenarios_multiple_comma_separated(self, tmp_path: Path):
        _write_group(tmp_path, "claude-code", ["a", "b"])
        _write_group(tmp_path, "gemini", ["c"])
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="claude-code/a,gemini")
        assert sf.parsed_paths is not None
        assert len(sf.parsed_paths) == 2

    def test_nonexistent_group_raises(self, tmp_path: Path):
        with pytest.raises(ScenarioError, match="No such group directory"):
            ScenarioFilter.from_cli_args(tmp_path, scenarios="nonexistent/test")

    def test_both_tags_and_scenarios(self, tmp_path: Path):
        _write_group(tmp_path, "group1")
        sf = ScenarioFilter.from_cli_args(tmp_path, tags="smoke", scenarios="group1")
        assert sf.has_tag_filter
        assert sf.has_path_filter
        assert not sf.is_empty

    def test_empty_scenarios_string(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="")
        assert not sf.has_path_filter

    def test_only_commas_no_filter(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios=",,,")
        assert not sf.has_path_filter

    def test_whitespace_commas_no_filter(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios=" , , ")
        assert not sf.has_path_filter

    def test_immutable(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, tags="smoke")
        with pytest.raises(AttributeError):
            sf.tags = frozenset({"other"})  # type: ignore[misc]


# ── Group matching ──


class TestMatchesGroup:
    def test_no_filter_matches_all(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path)
        group = _write_group(tmp_path, "any-group")
        matched, allowed = sf.matches_group(group)
        assert matched is True
        assert allowed == frozenset()

    def test_group_filter_matches(self, tmp_path: Path):
        _write_group(tmp_path, "target")
        _write_group(tmp_path, "other")
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="target")
        matched, _ = sf.matches_group(tmp_path / "target")
        assert matched is True
        other_matched, _ = sf.matches_group(tmp_path / "other")
        assert other_matched is False

    def test_scenario_name_filter_returns_allowed(self, tmp_path: Path):
        _write_group(tmp_path, "group1", ["a", "b", "c"])
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="group1/a")
        matched, allowed = sf.matches_group(tmp_path / "group1")
        assert matched is True
        assert allowed == frozenset({"a"})

    def test_multiple_scenario_names(self, tmp_path: Path):
        _write_group(tmp_path, "g", ["a", "b", "c"])
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="g/a,g/b")
        matched, allowed = sf.matches_group(tmp_path / "g")
        assert matched is True
        assert allowed == frozenset({"a", "b"})

    def test_whole_group_overrides_specific_scenarios(self, tmp_path: Path):
        _write_group(tmp_path, "g", ["a", "b"])
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="g,g/a")
        matched, allowed = sf.matches_group(tmp_path / "g")
        assert matched is True
        assert allowed == frozenset()  # whole group = all scenarios

    def test_nested_group_matches_parent_filter(self, tmp_path: Path):
        parent = _write_group(tmp_path, "parent")
        child = parent / "child"
        child.mkdir()
        (child / "_config.json").write_text('{"agent": "stub"}')
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="parent")
        matched, _ = sf.matches_group(child)
        assert matched is True


# ── Scenario matching (tags) ──


class TestMatchesScenario:
    def test_no_tags_matches_all(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path)
        assert sf.matches_scenario(_make_scenario(tags=[]), _make_group_config()) is True

    def test_matching_tag(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, tags="smoke")
        assert sf.matches_scenario(_make_scenario(tags=["smoke", "fast"]), _make_group_config()) is True

    def test_missing_tag(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, tags="smoke")
        assert sf.matches_scenario(_make_scenario(tags=["fast"]), _make_group_config()) is False

    def test_multiple_required_tags_and_logic(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, tags="smoke,fast")
        assert sf.matches_scenario(_make_scenario(tags=["smoke", "fast"]), _make_group_config()) is True
        assert sf.matches_scenario(_make_scenario(tags=["smoke"]), _make_group_config()) is False

    def test_group_default_tags_merged(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, tags="smoke,v2")
        scenario = _make_scenario(tags=["smoke"])
        config = _make_group_config(default_tags=["v2"])
        assert sf.matches_scenario(scenario, config) is True

    def test_group_default_tags_alone(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, tags="v2")
        scenario = _make_scenario(tags=[])
        config = _make_group_config(default_tags=["v2"])
        assert sf.matches_scenario(scenario, config) is True

    def test_superset_of_tags_still_matches(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, tags="smoke")
        scenario = _make_scenario(tags=["smoke", "fast", "nightly"])
        assert sf.matches_scenario(scenario, _make_group_config()) is True


# ── Integration: end-to-end filtering with runner dry-run ──


class TestRunnerDryRunIntegration:
    """Verify that ScenarioFilter works correctly through the runner CLI."""

    def test_dry_run_with_tag_filter(self, tmp_path: Path):
        _write_group(tmp_path, "g1", ["a"])
        from belt.commands.run import main as run_main

        rc = run_main([str(tmp_path), "--dry-run", "--tags", "nonexistent-tag"])
        assert rc == 1  # no match = 1 (even in dry-run)

    def test_dry_run_with_scenario_filter(self, tmp_path: Path):
        _write_group(tmp_path, "g1", ["a", "b"])
        from belt.commands.run import main as run_main

        rc = run_main([str(tmp_path), "--dry-run", "--scenarios", "g1/a"])
        assert rc == 0

    def test_dry_run_nonexistent_scenario_filter(self, tmp_path: Path):
        _write_group(tmp_path, "g1", ["a"])
        from belt.commands.run import main as run_main

        rc = run_main([str(tmp_path), "--dry-run", "--scenarios", "g1/nonexistent"])
        assert rc == 1  # unknown scenario name → error

    def test_dry_run_nonexistent_group_filter(self, tmp_path: Path):
        _write_group(tmp_path, "g1", ["a"])
        from belt.commands.run import main as run_main

        rc = run_main([str(tmp_path), "--dry-run", "--scenarios", "badgroup/x"])
        assert rc == 1
