# (c) JFrog Ltd. (2026)

"""Regression tests for three OSS-readiness bugs surfaced by hands-on
verification of the showcase pipeline:

- ``run_meta.json`` recorded ``selected_tags`` as ``list(args.tags)``,
  which iterates a comma-separated string char-by-char and corrupts the
  reproducibility manifest (``"a,b" -> ["a",",","b"]``).
- ``run_fixtures.json`` resolved ``working_dir`` against the process CWD
  instead of the group directory; the recorded path could point at a
  nonexistent location while the runtime used the correct one.
- The scoring pipeline silently dropped a scorer that raised (e.g. an
  LLM judge auth failure), letting CI gates green-light a broken run.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from belt import _internal_envvars
from belt.commands.run import _split_csv
from belt.entities import GroupConfig
from belt.runner.phases.setup_groups import _capture_fixture_provenance_for_group
from belt.scorer.payloads import RulesPayload
from belt.scorer.pipeline import score_scenario


class TestSplitCsv:
    def test_comma_separated_string_is_split(self) -> None:
        assert _split_csv("real-runnable,smoke") == ["real-runnable", "smoke"]

    def test_single_value_string_returns_singleton_list(self) -> None:
        # Iterating a string char-by-char would have produced
        # ``['r','e','a','l','-','r','u','n','n','a','b','l','e']``.
        assert _split_csv("real-runnable") == ["real-runnable"]

    def test_none_returns_empty(self) -> None:
        assert _split_csv(None) == []

    def test_list_passthrough_strips_blanks(self) -> None:
        assert _split_csv(["a", " b ", ""]) == ["a", "b"]

    def test_whitespace_around_commas_is_trimmed(self) -> None:
        assert _split_csv("a, b , c") == ["a", "b", "c"]


class TestFixtureProvenanceWorkingDir:
    """The runtime in ``run_scenarios.py`` resolves ``working_dir``
    relative to the group directory; provenance capture must use the
    same anchor or the manifest disagrees with what actually ran.
    """

    def test_relative_working_dir_resolves_against_group_dir(self, tmp_path: Path) -> None:
        sibling = tmp_path / "fixtures" / "sample-project"
        sibling.mkdir(parents=True)
        scenarios_root = tmp_path / "scenarios"
        group_dir = scenarios_root / "editing"
        group_dir.mkdir(parents=True)

        gc = GroupConfig(agent="claude-code", working_dir="../../fixtures/sample-project")
        info = _capture_fixture_provenance_for_group(group_dir, gc)
        # The recorded path must match the actual filesystem location,
        # which is reachable by walking up from ``group_dir``.
        assert info["working_dir"] == str(sibling.resolve())
        assert Path(info["working_dir"]).exists()

    def test_absolute_working_dir_is_preserved(self, tmp_path: Path) -> None:
        abs_path = (tmp_path / "abs-fixture").resolve()
        abs_path.mkdir()
        gc = GroupConfig(agent="claude-code", working_dir=str(abs_path))
        info = _capture_fixture_provenance_for_group(tmp_path / "group", gc)
        assert info["working_dir"] == str(abs_path)

    def test_no_working_dir_yields_none(self, tmp_path: Path) -> None:
        gc = GroupConfig(agent="claude-code")
        info = _capture_fixture_provenance_for_group(tmp_path / "group", gc)
        assert info["working_dir"] is None


class TestScorerFailureSurfacing:
    """A scorer that raises must not be silently dropped: the scenario
    has to fail and the failure must show up in the rules payload so
    downstream consumers (results.json, exporters, threshold gating) see
    it.

    Tests build a minimal on-disk outcome (a single ``turn_0_cli.txt``
    plus a stub group ``_config.json``) so ``score_scenario`` can run its
    real disk-loading path; only the scorer instances are mocked.
    """

    @pytest.fixture(autouse=True)
    def _isolate_scenarios_root_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``scenarios_root()`` consults ``_BELT_SCENARIOS_ROOT`` before
        # ``run_meta.json``; another test in the suite may have set it
        # and not restored it. Clearing it forces resolution through the
        # ``run_meta.json`` we write in ``_make_outcome_dir``.
        monkeypatch.delenv(_internal_envvars.SCENARIOS_ROOT, raising=False)

    @staticmethod
    def _make_outcome_dir(tmp_path: Path) -> tuple[Path, Path]:
        scenarios_root = tmp_path / "scenarios"
        group_dir = scenarios_root / "g"
        group_dir.mkdir(parents=True)
        # Minimal valid scenario + group config so ``map_to_scenario`` and
        # ``ScenarioLoader.load_scenario`` succeed without scaffolding the
        # whole runner stack.
        (group_dir / "_config.json").write_text(json.dumps({"agent": "claude-code"}))
        (group_dir / "s.json").write_text(
            json.dumps(
                {
                    "name": "s",
                    "description": "regression scenario for silent-scorer-failure fix",
                    "turns": [{"message": "hi", "expect": {"has_reply": True}}],
                }
            )
        )

        outcomes_root = tmp_path / "outcomes"
        run_dir = outcomes_root / "20260508-000000-test"
        outcome_dir = run_dir / "g" / "s"
        outcome_dir.mkdir(parents=True)
        # ``map_to_scenario`` reads ``scenarios_root`` from
        # ``run_meta.json`` at the run directory; pin it so the loader
        # resolves to our fake scenario tree above.
        (run_dir / "run_meta.json").write_text(json.dumps({"scenarios_root": str(scenarios_root)}))
        (outcome_dir / "turn_0_cli.txt").write_text("{}\n")
        return run_dir, outcome_dir

    @staticmethod
    def _passing_rules_scorer() -> Mock:
        scorer = Mock()
        scorer.name = "rules"
        result = Mock()
        result.data = RulesPayload(checks=[], passed=True, has_error=None)
        result.passed = True
        scorer.score = Mock(return_value=result)
        return scorer

    @staticmethod
    def _raising_scorer(name: str) -> Mock:
        scorer = Mock()
        scorer.name = name
        scorer.score = Mock(side_effect=RuntimeError("LLM judge fatal HTTP 401"))
        return scorer

    def test_failing_scorer_flips_overall_pass(self, tmp_path: Path) -> None:
        outcomes_root, outcome_dir = self._make_outcome_dir(tmp_path)
        score = score_scenario(
            outcome_dir=outcome_dir,
            outcomes_root=outcomes_root,
            scorers=[self._passing_rules_scorer(), self._raising_scorer("llm")],
        )
        assert score.overall_pass is False, (
            "A scorer that raised must not silently leave overall_pass=True - "
            "an LLM judge HTTP 401 would otherwise let CI gates pass on a broken run."
        )

    def test_failing_scorer_records_synthetic_check(self, tmp_path: Path) -> None:
        outcomes_root, outcome_dir = self._make_outcome_dir(tmp_path)
        score = score_scenario(
            outcome_dir=outcome_dir,
            outcomes_root=outcomes_root,
            scorers=[self._passing_rules_scorer(), self._raising_scorer("llm")],
        )
        rules = score.scores.get("rules")
        assert isinstance(rules, RulesPayload)
        synthetic = [c for c in rules.checks if c.check == "llm_scorer_ran"]
        assert synthetic, "Expected a synthetic 'llm_scorer_ran' check carrying the failure"
        assert synthetic[0].passed is False
        assert "LLM judge fatal HTTP 401" in synthetic[0].details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
