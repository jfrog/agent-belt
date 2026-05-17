# (c) JFrog Ltd. (2026)

"""Tests for schema versioning of output artifacts.

Covers:
- schema_version field on TurnOutput and ScenarioScore
- Round-trip serialization preserves the version
- The field is optional in the v1 contract: artifacts that omit it
  parse and surface ``schema_version is None``
- check_schema_version warns on mismatch and missing version
- Writers (orchestrator, scorer, runner, aggregator) stamp the version
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from loguru import logger

from belt.constants import SCHEMA_VERSION
from belt.entities import ScenarioScore, TurnOutput
from belt.schema import check_schema_version


@pytest.fixture()
def log_sink():
    """Capture loguru messages into a list for assertions."""
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


# ── Entity defaults ──


def test_turn_output_schema_version_default_none() -> None:
    to = TurnOutput(raw_cli="hello")
    assert to.schema_version is None


def test_scenario_score_schema_version_default_none() -> None:
    ss = ScenarioScore(scenario_name="s", group="g", overall_pass=True)
    assert ss.schema_version is None


def test_turn_output_schema_version_explicit() -> None:
    to = TurnOutput(raw_cli="hello", schema_version=SCHEMA_VERSION)
    assert to.schema_version == SCHEMA_VERSION


def test_scenario_score_schema_version_explicit() -> None:
    ss = ScenarioScore(scenario_name="s", group="g", overall_pass=True, schema_version=SCHEMA_VERSION)
    assert ss.schema_version == SCHEMA_VERSION


# ── Round-trip serialization ──


def test_turn_output_roundtrip_with_version() -> None:
    to = TurnOutput(raw_cli="output", schema_version=SCHEMA_VERSION)
    restored = TurnOutput.model_validate_json(to.model_dump_json())
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.raw_cli == "output"


def test_scenario_score_roundtrip_with_version() -> None:
    ss = ScenarioScore(scenario_name="s", group="g", overall_pass=False, schema_version=SCHEMA_VERSION)
    restored = ScenarioScore.model_validate_json(ss.model_dump_json())
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.overall_pass is False


# ── Optional schema_version field (default None) ──


def test_turn_output_without_schema_version_loads_as_none() -> None:
    """``turn_output.json`` that omits ``schema_version`` parses cleanly
    and surfaces ``schema_version is None``."""
    blob = json.dumps({"raw_cli": "hello", "has_reply": True})
    to = TurnOutput.model_validate_json(blob)
    assert to.raw_cli == "hello"
    assert to.schema_version is None


def test_scenario_score_without_schema_version_loads_as_none() -> None:
    """``score.json`` that omits ``schema_version`` parses cleanly and
    surfaces ``schema_version is None``."""
    blob = json.dumps({"scenario_name": "s1", "group": "g1", "scores": {}, "overall_pass": True})
    ss = ScenarioScore.model_validate_json(blob)
    assert ss.overall_pass is True
    assert ss.schema_version is None


# ── check_schema_version ──


def test_check_matching_version_no_warning(log_sink: list[str]) -> None:
    check_schema_version(SCHEMA_VERSION, "test.json")
    assert not log_sink


def test_check_missing_version_warns(log_sink: list[str]) -> None:
    check_schema_version(None, "score.json")
    assert len(log_sink) == 1
    assert "missing schema_version" in log_sink[0]
    assert "score.json" in log_sink[0]


def test_check_mismatched_version_warns(log_sink: list[str]) -> None:
    check_schema_version("999", "future.json")
    assert len(log_sink) == 1
    assert "mismatch" in log_sink[0]
    assert "future.json" in log_sink[0]


# ── Orchestrator stamps schema_version on TurnOutput ──


def test_orchestrator_stamps_version(tmp_path: Path) -> None:
    from belt.runner.orchestrator import _write_turn_artifacts

    to = TurnOutput(raw_cli="test output")
    assert to.schema_version is None

    _write_turn_artifacts(tmp_path, 0, to)

    assert to.schema_version == SCHEMA_VERSION

    output_file = tmp_path / "turn_0_output.json"
    data = json.loads(output_file.read_text())
    assert data["schema_version"] == SCHEMA_VERSION


# ── Runner stamps schema_version on run_meta.json ──


def test_run_meta_has_version(tmp_path: Path) -> None:
    """run_meta.json should include schema_version."""
    from belt.constants import SCHEMA_VERSION as SV

    run_meta = {"schema_version": SV, "scenarios_root": "/some/path", "workspace": "/ws"}
    meta_path = tmp_path / "run_meta.json"
    meta_path.write_text(json.dumps(run_meta) + "\n")

    data = json.loads(meta_path.read_text())
    assert data["schema_version"] == SV


# ── Scorer stamps schema_version on ScenarioScore ──


def test_scorer_score_has_version() -> None:
    """All ScenarioScore constructions in scorer should include schema_version."""
    ss = ScenarioScore(schema_version=SCHEMA_VERSION, scenario_name="s", group="g", overall_pass=True)
    data = json.loads(ss.model_dump_json())
    assert data["schema_version"] == SCHEMA_VERSION


# ── Aggregator stamps schema_version on results.json ──


def test_results_json_has_version(tmp_path: Path) -> None:
    results = {
        "schema_version": SCHEMA_VERSION,
        "total": 1,
        "passed": 1,
        "failed": 0,
        "overall_pass": True,
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    data = json.loads(path.read_text())
    assert data["schema_version"] == SCHEMA_VERSION


# ── Aggregator discover_scores checks version ──


def test_discover_scores_checks_version(tmp_path: Path, log_sink: list[str]) -> None:
    from belt.commands.aggregate import discover_scores

    score_dir = tmp_path / "group" / "scenario"
    score_dir.mkdir(parents=True)

    ss = ScenarioScore(schema_version=SCHEMA_VERSION, scenario_name="s", group="g", overall_pass=True)
    (score_dir / "score.json").write_text(ss.model_dump_json())

    scores = discover_scores(tmp_path)
    assert len(scores) == 1
    assert not log_sink


def test_discover_scores_warns_when_schema_version_missing(tmp_path: Path, log_sink: list[str]) -> None:
    from belt.commands.aggregate import discover_scores

    score_dir = tmp_path / "group" / "scenario"
    score_dir.mkdir(parents=True)

    blob = json.dumps({"scenario_name": "s", "group": "g", "scores": {}, "overall_pass": True})
    (score_dir / "score.json").write_text(blob)

    scores = discover_scores(tmp_path)
    assert len(scores) == 1
    assert any("missing schema_version" in m for m in log_sink)


def test_discover_scores_warns_future_version(tmp_path: Path, log_sink: list[str]) -> None:
    from belt.commands.aggregate import discover_scores

    score_dir = tmp_path / "group" / "scenario"
    score_dir.mkdir(parents=True)

    ss = ScenarioScore(schema_version="99", scenario_name="s", group="g", overall_pass=True)
    (score_dir / "score.json").write_text(ss.model_dump_json())

    scores = discover_scores(tmp_path)
    assert len(scores) == 1
    assert any("mismatch" in m for m in log_sink)


# ── End-to-end: orchestrator run produces versioned artifacts ──


def test_e2e_run_produces_versioned_artifacts(tmp_path: Path) -> None:
    """run_scenario_turns writes turn_output.json with schema_version stamped."""
    from belt.agent.base import BaseAgentAdapter
    from belt.entities import AgentConfig, GroupConfig, Scenario, Turn
    from belt.runner.orchestrator import run_scenario_turns

    class _Stub(BaseAgentAdapter):
        def setup(self, config):
            pass

        def execute(self, message, flags=None):
            return "raw output"

        def fetch_results(self, raw_output):
            return TurnOutput(raw_cli=raw_output, reply_text="hi")

        def teardown(self):
            pass

    scenario = Scenario(name="hello", description="test", turns=[Turn(message="hi")])
    gc = GroupConfig(agent="stub")
    config = AgentConfig(group_config=gc, scenario_name="hello")

    outcome_dir = tmp_path / "group" / "hello"
    run_scenario_turns(_Stub(), scenario, outcome_dir, config, stream=False)

    output_file = outcome_dir / "turn_0_output.json"
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
