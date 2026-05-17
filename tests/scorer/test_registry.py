# (c) JFrog Ltd. (2026)

"""Tests for scorer registry - discovery, resolution, and error handling."""

from __future__ import annotations

import pytest

from belt.scorer.registry import available_scorers, get_scorer_class


class TestAvailableScorers:
    def test_returns_builtin_names(self) -> None:
        names = available_scorers()
        assert "rules" in names
        assert "llm" in names

    def test_returns_sorted(self) -> None:
        names = available_scorers()
        assert names == sorted(names)


class TestGetScorerClass:
    def test_resolves_rules(self) -> None:
        from belt.scorer.rules import RuleBasedScorer

        cls = get_scorer_class("rules")
        assert cls is RuleBasedScorer

    def test_resolves_llm(self) -> None:
        from belt.scorer.llm import LLMScorer

        cls = get_scorer_class("llm")
        assert cls is LLMScorer

    def test_unknown_name_raises(self) -> None:
        from belt.errors import ConfigError

        with pytest.raises(ConfigError, match="Unknown scorer 'nonexistent'"):
            get_scorer_class("nonexistent")

    def test_invalid_dotted_path_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from belt.errors import ConfigError

        monkeypatch.setenv("BELT_ALLOW_ARBITRARY_SCORER", "1")
        with pytest.raises(ConfigError, match="Failed to load scorer"):
            get_scorer_class("totally.bogus.Module")

    def test_non_scorer_class_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from belt.errors import ConfigError

        monkeypatch.setenv("BELT_ALLOW_ARBITRARY_SCORER", "1")
        with pytest.raises(ConfigError, match="not a BaseScorer subclass"):
            get_scorer_class("belt.entities:ToolCall")

    def test_dotted_import_with_colon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BELT_ALLOW_ARBITRARY_SCORER", "1")
        cls = get_scorer_class("belt.scorer.rules:RuleBasedScorer")
        from belt.scorer.rules import RuleBasedScorer

        assert cls is RuleBasedScorer

    def test_dotted_import_with_dot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BELT_ALLOW_ARBITRARY_SCORER", "1")
        cls = get_scorer_class("belt.scorer.rules.RuleBasedScorer")
        from belt.scorer.rules import RuleBasedScorer

        assert cls is RuleBasedScorer

    def test_dotted_path_blocked_without_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The dotted-import escape hatch is off by default."""
        from belt.errors import ConfigError

        monkeypatch.delenv("BELT_ALLOW_ARBITRARY_SCORER", raising=False)
        with pytest.raises(ConfigError, match="Unknown scorer 'totally.bogus.Module'"):
            get_scorer_class("totally.bogus.Module")


class TestPluginModeInBuildScorers:
    """Verify that _build_scorers accepts plugin scorer names."""

    def test_builtin_modes_still_work(self) -> None:
        from belt.commands.score import _build_scorers

        scorers, llm_scorers = _build_scorers("rules", {}, skip_availability=True)
        assert len(scorers) == 1
        assert scorers[0].name == "rules"

    def test_unknown_plugin_mode_raises(self) -> None:
        from belt.commands.score import _build_scorers
        from belt.errors import ConfigError

        with pytest.raises(ConfigError, match="Unknown scorer"):
            _build_scorers("rules,nonexistent", {}, skip_availability=True)

    def test_plugin_mode_via_dotted_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from belt.commands.score import _build_scorers

        monkeypatch.setenv("BELT_ALLOW_ARBITRARY_SCORER", "1")
        scorers, _ = _build_scorers(
            "rules,belt.scorer.rules:RuleBasedScorer",
            {},
            skip_availability=True,
        )
        assert len(scorers) == 2
        assert all(s.name == "rules" for s in scorers)
