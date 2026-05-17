# (c) JFrog Ltd. (2026)

"""Behavior matrix for ``ScoringStrategy`` resolution at score time.

Three sources can contribute LLM judge dimensions when an outcome is scored:

1. The group ``_config.json`` ``llm_dimensions`` field.
2. The agent class's ``scoring_strategy()`` override (custom agents only).
3. The judge's build-time strategy, populated from ``--scorer-config`` YAML
   in ``pipeline.build_scorers()``.

Plus the framework default (``GENERIC_DIMENSIONS``) when none of the above
applies.

Precedence: group ``llm_dimensions`` > agent override > judge build-time
strategy > framework default. The tests below pin one combination of those
inputs each, so any future change to ``resolve_scoring_strategy()`` or the
scorer rebuild path produces a visible diff.

Every test asserts the dimension list that actually reaches the judge call
payload, computed by replaying the same rebuild logic ``commands/score.py``
runs for every outcome and reading the result via ``LLMScorer.dry_run()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from belt.agent.base import BaseAgentAdapter
from belt.agent.scoring import GENERIC_DIMENSIONS, DimensionDef, ScoringStrategy
from belt.entities import TurnOutput
from belt.scenario import Scenario, Turn
from belt.scorer import ConsensusScorer, LLMScorer
from belt.scorer.entities import JudgeConfig
from belt.scorer.llm.backend import OpenAIBackend
from belt.scorer.scenario_map import resolve_scoring_strategy

_YAML_DIM = DimensionDef(name="custom_yaml_dim", description="from --scorer-config")
_AGENT_DIM = DimensionDef(name="custom_agent_dim", description="from agent override")
_GROUP_DIM_DICT = {"name": "custom_group_dim", "description": "from group _config.json"}


class _OverrideAgent(BaseAgentAdapter):
    """Test-only agent that overrides ``scoring_strategy``.

    Any concrete ``BaseAgentAdapter`` subclass that redefines
    ``scoring_strategy`` produces the same effect; this minimal stub is
    sufficient because the resolver only ever calls
    ``instance.scoring_strategy()`` on the resolved class.
    """

    def setup(self, config):  # pragma: no cover - never invoked
        return None

    def execute(self, message, flags):  # pragma: no cover - never invoked
        return ""

    def fetch_results(self, raw):  # pragma: no cover - never invoked
        return TurnOutput(raw_cli=raw)

    def teardown(self):  # pragma: no cover - never invoked
        return None

    def scoring_strategy(self) -> ScoringStrategy:
        return ScoringStrategy(dimensions=[_AGENT_DIM])


def _write_group(
    base: Path,
    *,
    agent: str,
    llm_dimensions: list | None = None,
) -> tuple[Path, Path]:
    """Build a ``scenarios/<group>/`` + ``outcomes/<group>/<scenario>/`` pair.

    Returns ``(outcomes_root, outcome_dir)``. The structure mirrors what
    the runner writes during a real ``belt eval`` run, so
    ``resolve_scoring_strategy()`` exercises its production code path.
    """
    scenarios_root = base / "scenarios"
    group_dir = scenarios_root / "g"
    group_dir.mkdir(parents=True)
    config: dict = {"agent": agent}
    if llm_dimensions is not None:
        config["llm_dimensions"] = llm_dimensions
    (group_dir / "_config.json").write_text(json.dumps(config))

    outcomes_root = base / "outcomes"
    outcome_dir = outcomes_root / "g" / "demo"
    outcome_dir.mkdir(parents=True)
    (outcomes_root / "run_meta.json").write_text(json.dumps({"scenarios_root": str(scenarios_root)}))
    return outcomes_root, outcome_dir


def _write_root_group(
    base: Path,
    *,
    agent: str,
    llm_dimensions: list | None = None,
) -> tuple[Path, Path]:
    """Build the layout produced by ``belt eval <single-group-path>``.

    When the user points ``belt eval`` at one group directory directly,
    ``scenarios_root`` IS that group dir and outcomes land at
    ``<run>/<scenario>/`` (no group sub-directory). The ``_config.json``
    sits at the scenarios-root itself.
    """
    scenarios_root = base / "scenarios" / "g"
    scenarios_root.mkdir(parents=True)
    config: dict = {"agent": agent}
    if llm_dimensions is not None:
        config["llm_dimensions"] = llm_dimensions
    (scenarios_root / "_config.json").write_text(json.dumps(config))

    outcomes_root = base / "outcomes"
    outcome_dir = outcomes_root / "demo"
    outcome_dir.mkdir(parents=True)
    (outcomes_root / "run_meta.json").write_text(json.dumps({"scenarios_root": str(scenarios_root)}))
    return outcomes_root, outcome_dir


def _build_judge(strategy: ScoringStrategy | None) -> LLMScorer:
    """Construct an ``LLMScorer`` with a stubbed backend.

    ``skip_availability=True`` lets ``dry_run`` build the judge payload
    without provider credentials or network access - sufficient because
    the dry-run path never calls the backend.
    """
    return LLMScorer(
        config=JudgeConfig(model="openai/gpt-4.1"),
        backend=OpenAIBackend(),
        skip_availability=True,
        strategy=strategy,
    )


def _rebuild_like_score_one(judge: LLMScorer, strategy: ScoringStrategy | None) -> LLMScorer:
    """Replay the per-outcome rebuild step from ``commands/score.py``.

    When ``resolve_scoring_strategy`` returns a non-``None`` strategy,
    the score command rebuilds every judge with that strategy. When it
    returns ``None``, the original judge is reused unchanged (so the
    build-time strategy survives).
    """
    if strategy is None:
        return judge
    return LLMScorer(
        config=judge.config,
        max_retries=judge.max_retries,
        backend=OpenAIBackend(),
        skip_availability=True,
        strategy=strategy,
    )


def _payload_dimensions(judge: LLMScorer) -> list[str]:
    """Return the dimension names the judge would actually send."""
    scenario = Scenario(
        name="demo",
        description="d",
        turns=[Turn(message="hi")],
    )
    return judge.dry_run(scenario, [TurnOutput(raw_cli="x")])["dimensions"]


@pytest.fixture
def _patched_override_agent():
    """Make ``get_agent_class("test-override-agent")`` return ``_OverrideAgent``.

    ``resolve_scoring_strategy`` looks up the agent class via
    ``belt.agent.registry.get_agent_class``. Patching the function
    in the registry module is enough because every caller - including the
    ``__import__`` indirection inside the resolver - reads the symbol
    from that module's namespace.
    """
    from belt.agent.registry import get_agent_class as real_get_agent_class

    def _get(name: str):
        if name == "test-override-agent":
            return _OverrideAgent
        return real_get_agent_class(name)

    with patch("belt.agent.registry.get_agent_class", side_effect=_get):
        yield


_GENERIC_NAMES = [d.name for d in GENERIC_DIMENSIONS]


class TestNoOverrides:
    """Nothing custom configured anywhere - generic defaults reach the judge."""

    def test_falls_back_to_generic_dimensions(self, tmp_path: Path) -> None:
        outcomes_root, outcome_dir = _write_group(tmp_path, agent="claude-code")

        strategy = resolve_scoring_strategy(outcome_dir, outcomes_root)
        judge = _build_judge(strategy=None)
        effective = _rebuild_like_score_one(judge, strategy)

        assert _payload_dimensions(effective) == _GENERIC_NAMES


class TestScorerConfigOnly:
    """``--scorer-config`` YAML is the sole source of dimensions.

    Protects the ``--scorer-config`` contract: a YAML-defined dimension
    list must reach the judge when the group has no ``llm_dimensions``
    and the agent does not override ``scoring_strategy``.
    """

    def test_yaml_dimensions_reach_judge(self, tmp_path: Path) -> None:
        outcomes_root, outcome_dir = _write_group(tmp_path, agent="claude-code")
        yaml_strategy = ScoringStrategy(dimensions=[_YAML_DIM])

        strategy = resolve_scoring_strategy(outcome_dir, outcomes_root)
        judge = _build_judge(strategy=yaml_strategy)
        effective = _rebuild_like_score_one(judge, strategy)

        assert _payload_dimensions(effective) == [_YAML_DIM.name]


class TestAgentOverrideOnly:
    """Only the agent class overrides ``scoring_strategy``."""

    def test_agent_override_reaches_judge(self, tmp_path: Path, _patched_override_agent) -> None:
        outcomes_root, outcome_dir = _write_group(tmp_path, agent="test-override-agent")

        strategy = resolve_scoring_strategy(outcome_dir, outcomes_root)
        judge = _build_judge(strategy=None)
        effective = _rebuild_like_score_one(judge, strategy)

        assert _payload_dimensions(effective) == [_AGENT_DIM.name]


class TestAgentOverrideBeatsScorerConfig:
    """Agent override + YAML both set; the agent override wins."""

    def test_agent_override_takes_precedence_over_yaml(self, tmp_path: Path, _patched_override_agent) -> None:
        outcomes_root, outcome_dir = _write_group(tmp_path, agent="test-override-agent")
        yaml_strategy = ScoringStrategy(dimensions=[_YAML_DIM])

        strategy = resolve_scoring_strategy(outcome_dir, outcomes_root)
        judge = _build_judge(strategy=yaml_strategy)
        effective = _rebuild_like_score_one(judge, strategy)

        assert _payload_dimensions(effective) == [_AGENT_DIM.name]


class TestGroupOnly:
    """Group ``_config.json`` is the sole source of dimensions."""

    def test_group_dimensions_reach_judge(self, tmp_path: Path) -> None:
        outcomes_root, outcome_dir = _write_group(tmp_path, agent="claude-code", llm_dimensions=[_GROUP_DIM_DICT])

        strategy = resolve_scoring_strategy(outcome_dir, outcomes_root)
        judge = _build_judge(strategy=None)
        effective = _rebuild_like_score_one(judge, strategy)

        assert _payload_dimensions(effective) == [_GROUP_DIM_DICT["name"]]


class TestGroupBeatsScorerConfig:
    """Group ``_config.json`` + YAML both set; the group wins."""

    def test_group_takes_precedence_over_yaml(self, tmp_path: Path) -> None:
        outcomes_root, outcome_dir = _write_group(tmp_path, agent="claude-code", llm_dimensions=[_GROUP_DIM_DICT])
        yaml_strategy = ScoringStrategy(dimensions=[_YAML_DIM])

        strategy = resolve_scoring_strategy(outcome_dir, outcomes_root)
        judge = _build_judge(strategy=yaml_strategy)
        effective = _rebuild_like_score_one(judge, strategy)

        assert _payload_dimensions(effective) == [_GROUP_DIM_DICT["name"]]


class TestRootGroup:
    """Group dir IS the scenarios root (``belt eval <group-path>`` directly).

    The natural CLI invocation points at one group: ``belt eval
    examples/scenarios/showcase/verdict-scales``. Outcomes land at
    ``<run>/<scenario>/`` with no group sub-directory, so the resolver
    must look up ``_config.json`` at the scenarios-root itself - not
    under an empty ``Path(*group_parts)``.
    """

    def test_root_group_dimensions_reach_judge(self, tmp_path: Path) -> None:
        outcomes_root, outcome_dir = _write_root_group(tmp_path, agent="claude-code", llm_dimensions=[_GROUP_DIM_DICT])

        strategy = resolve_scoring_strategy(outcome_dir, outcomes_root)
        judge = _build_judge(strategy=None)
        effective = _rebuild_like_score_one(judge, strategy)

        assert _payload_dimensions(effective) == [_GROUP_DIM_DICT["name"]]


class TestConsensusJudgeRebuild:
    """``ConsensusScorer`` participates in the same rebuild path.

    When ``resolve_scoring_strategy`` returns ``None``, every judge inside
    a ``ConsensusScorer`` keeps its build-time strategy. When it returns
    a non-``None`` strategy, every judge inside the consensus is rebuilt
    with that strategy. This pins the no-override case so YAML
    dimensions on individual consensus judges are not dropped.
    """

    def test_yaml_dimensions_survive_inside_consensus(self, tmp_path: Path) -> None:
        outcomes_root, outcome_dir = _write_group(tmp_path, agent="claude-code")
        yaml_strategy = ScoringStrategy(dimensions=[_YAML_DIM])

        judge_a = _build_judge(strategy=yaml_strategy)
        judge_a.judge_name = "a"
        judge_b = _build_judge(strategy=yaml_strategy)
        judge_b.judge_name = "b"
        consensus = ConsensusScorer([judge_a, judge_b], strategy="majority")

        strategy = resolve_scoring_strategy(outcome_dir, outcomes_root)

        if strategy is None:
            effective_judges = consensus.judges
        else:
            effective_judges = [_rebuild_like_score_one(j, strategy) for j in consensus.judges]

        for j in effective_judges:
            assert _payload_dimensions(j) == [_YAML_DIM.name]
