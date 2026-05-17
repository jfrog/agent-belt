# (c) JFrog Ltd. (2026)

"""Tests for scenario filter parsing and matching - migrated to ScenarioFilter API.

These tests exercise real filesystem paths and edge cases from the original
runner/cli.py private functions.  Core unit tests live in tests/test_filter.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from belt.errors import ScenarioError
from belt.filter import ScenarioFilter


class TestPathFilterParsing:
    def test_none_scenarios_means_match_all(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios=None)
        assert sf.parsed_paths is None
        assert not sf.has_path_filter

    def test_empty_string_means_match_all(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="")
        assert not sf.has_path_filter

    def test_only_commas_means_match_all(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios=",,,")
        assert not sf.has_path_filter

    def test_whitespace_commas_means_match_all(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios=" , , ")
        assert not sf.has_path_filter

    def test_group_directory(self, tmp_path: Path):
        group = tmp_path / "production" / "web_conversation_v1"
        group.mkdir(parents=True)
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="production/web_conversation_v1")
        assert sf.parsed_paths is not None
        assert len(sf.parsed_paths) == 1
        assert sf.parsed_paths[0] == (group.resolve(), None)

    def test_scenario_path(self, tmp_path: Path):
        group = tmp_path / "production" / "web_conversation_v1"
        group.mkdir(parents=True)
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="production/web_conversation_v1/deployment_investigation")
        assert sf.parsed_paths is not None
        assert len(sf.parsed_paths) == 1
        assert sf.parsed_paths[0] == (group.resolve(), "deployment_investigation")

    def test_scenario_path_with_json_extension(self, tmp_path: Path):
        group = tmp_path / "production" / "web_conversation_v1"
        group.mkdir(parents=True)
        sf = ScenarioFilter.from_cli_args(
            tmp_path, scenarios="production/web_conversation_v1/deployment_investigation.json"
        )
        assert sf.parsed_paths is not None
        assert sf.parsed_paths[0] == (group.resolve(), "deployment_investigation")

    def test_nonexistent_group_shows_root_and_available(self, tmp_path: Path):
        (tmp_path / "claude-code").mkdir()
        (tmp_path / "gemini").mkdir()
        with pytest.raises(ScenarioError, match=r"(?s)Your scenarios root is:.*Available groups:.*claude-code.*gemini"):
            ScenarioFilter.from_cli_args(tmp_path, scenarios="nonexistent_group/some_scenario")

    def test_deeply_nested_bad_path_raises(self, tmp_path: Path):
        (tmp_path / "claude-code").mkdir()
        with pytest.raises(ScenarioError, match="No such group directory"):
            ScenarioFilter.from_cli_args(tmp_path, scenarios="claude-code/sub/deep/nonexistent.json")

    def test_bare_unknown_name_raises(self, tmp_path: Path):
        """``--scenarios <name>`` with no slash must fail loud when the name doesn't match a group.

        Regression: the old logic silently fell through to ``(scenarios_root, name)``
        and ran zero scenarios - common pitfall after a layout change where
        ``claude-code`` moves to ``agents/claude-code``.
        """
        (tmp_path / "agents" / "claude-code").mkdir(parents=True)
        (tmp_path / "agents" / "claude-code" / "_config.json").write_text('{"agent": "claude-code"}')
        with pytest.raises(ScenarioError, match=r"(?s)No such group directory.*agents/claude-code"):
            ScenarioFilter.from_cli_args(tmp_path, scenarios="claude-code")

    def test_available_groups_lists_nested_paths(self, tmp_path: Path):
        """The error suggestion walks the tree and lists every ``_config.json``-bearing dir.

        Without this, after the ``agents/`` + ``experience/`` reshape a user
        typing ``--scenarios claude-code`` would see only ``agents, experience``
        - not actionable.
        """
        for rel in ["agents/claude-code", "agents/codex", "experience/bookstore"]:
            d = tmp_path / rel
            d.mkdir(parents=True)
            (d / "_config.json").write_text('{"agent": "claude-code"}')
        with pytest.raises(ScenarioError) as exc:
            ScenarioFilter.from_cli_args(tmp_path, scenarios="claude-code")
        msg = str(exc.value)
        assert "agents/claude-code" in msg
        assert "agents/codex" in msg
        assert "experience/bookstore" in msg

    def test_root_is_group_bare_scenario_name(self, tmp_path: Path):
        """When scenarios_root IS itself a group, ``--scenarios <name>`` resolves to that scenario."""
        (tmp_path / "_config.json").write_text('{"agent": "claude-code"}')
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="validation_21")
        assert sf.parsed_paths is not None
        assert sf.parsed_paths[0] == (tmp_path.resolve(), "validation_21")

    def test_root_is_group_bare_scenario_name_with_json_extension(self, tmp_path: Path):
        (tmp_path / "_config.json").write_text('{"agent": "claude-code"}')
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="validation_21.json")
        assert sf.parsed_paths is not None
        assert sf.parsed_paths[0] == (tmp_path.resolve(), "validation_21")

    def test_root_is_group_strips_redundant_root_basename_prefix(self, tmp_path: Path):
        """``--scenarios <root_name>/<scenario>`` is accepted when root IS the group named <root_name>.

        Mirrors the natural mental model of "I want scenario X in group Y" even
        when the user has already pointed the runner at group Y as the root.
        """
        group = tmp_path / "ask-github-claude-code"
        group.mkdir()
        (group / "_config.json").write_text('{"agent": "claude-code"}')
        sf = ScenarioFilter.from_cli_args(group, scenarios="ask-github-claude-code/validation_21")
        assert sf.parsed_paths is not None
        assert sf.parsed_paths[0] == (group.resolve(), "validation_21")

    def test_root_is_group_error_explains_situation(self, tmp_path: Path):
        """Error for an unresolvable filter must call out the root-is-group case explicitly."""
        group = tmp_path / "ask-github-claude-code"
        group.mkdir()
        (group / "_config.json").write_text('{"agent": "claude-code"}')
        with pytest.raises(ScenarioError) as exc:
            ScenarioFilter.from_cli_args(group, scenarios="some-other-group/scenario_x")
        msg = str(exc.value)
        assert "This root IS itself a group" in msg
        assert "ask-github-claude-code" in msg

    def test_root_is_group_available_groups_self_describing(self, tmp_path: Path):
        """``Available groups: .`` is replaced by a self-describing label."""
        (tmp_path / "_config.json").write_text('{"agent": "claude-code"}')
        with pytest.raises(ScenarioError) as exc:
            ScenarioFilter.from_cli_args(tmp_path, scenarios="some-other-group/scenario_x")
        msg = str(exc.value)
        assert f"this directory itself: {tmp_path.name}" in msg
        assert ": ." not in msg.split("Available groups:")[1]

    def test_comma_separated_mixed(self, tmp_path: Path):
        group_a = tmp_path / "production" / "web_conversation_v1"
        group_a.mkdir(parents=True)
        group_b = tmp_path / "synthetic" / "review_interrupt_v2_depth_0"
        group_b.mkdir(parents=True)
        sf = ScenarioFilter.from_cli_args(
            tmp_path,
            scenarios="production/web_conversation_v1/deployment_investigation,"
            "synthetic/review_interrupt_v2_depth_0",
        )
        assert sf.parsed_paths is not None
        assert len(sf.parsed_paths) == 2
        assert sf.parsed_paths[0] == (group_a.resolve(), "deployment_investigation")
        assert sf.parsed_paths[1] == (group_b.resolve(), None)


class TestGroupMatching:
    def test_no_filter_matches_everything(self, tmp_path: Path):
        sf = ScenarioFilter.from_cli_args(tmp_path)
        matched, allowed = sf.matches_group(tmp_path / "any")
        assert matched is True
        assert allowed == frozenset()

    def test_group_filter_matches_group(self, tmp_path: Path):
        group = tmp_path / "web_v1"
        group.mkdir()
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="web_v1")
        matched, allowed = sf.matches_group(group)
        assert matched is True
        assert allowed == frozenset()

    def test_group_filter_no_match(self, tmp_path: Path):
        (tmp_path / "web_v1").mkdir()
        (tmp_path / "other").mkdir()
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="other")
        matched, _ = sf.matches_group(tmp_path / "web_v1")
        assert matched is False

    def test_scenario_filter_matches_group_with_name(self, tmp_path: Path):
        group = tmp_path / "web_v1"
        group.mkdir()
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="web_v1/deploy")
        matched, allowed = sf.matches_group(group)
        assert matched is True
        assert allowed == frozenset({"deploy"})

    def test_multiple_scenarios_same_group(self, tmp_path: Path):
        group = tmp_path / "web_v1"
        group.mkdir()
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="web_v1/deploy,web_v1/onboard")
        matched, allowed = sf.matches_group(group)
        assert matched is True
        assert allowed == frozenset({"deploy", "onboard"})

    def test_group_filter_overrides_scenario_filter(self, tmp_path: Path):
        group = tmp_path / "web_v1"
        group.mkdir()
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="web_v1/deploy,web_v1")
        matched, allowed = sf.matches_group(group)
        assert matched is True
        assert allowed == frozenset()  # whole group = all

    def test_parent_prefix_matches(self, tmp_path: Path):
        parent = tmp_path / "production"
        child = parent / "web_v1"
        child.mkdir(parents=True)
        sf = ScenarioFilter.from_cli_args(tmp_path, scenarios="production")
        matched, allowed = sf.matches_group(child)
        assert matched is True
        assert allowed == frozenset()
