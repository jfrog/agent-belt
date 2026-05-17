# (c) JFrog Ltd. (2026)

"""End-to-end test for the agent-errors signal.

Drives the full pipeline (orchestrator -> scorer -> aggregator ->
benchmark card) using an in-process mock agent that always returns
``has_error=True`` with an auth-failure ``reply_text``. Using a mock
adapter rather than a standalone CLI binary keeps the test fast and
hermetic while exercising the same code paths as a real adapter.

Assertions cover the full propagation chain:

1. ``ScenarioResult.agent_errors`` is populated by the orchestrator.
2. ``collect_agent_errors`` counts vacuous passes correctly.
3. ``build_bottom_line`` emits the agent-error headline.
4. ``AggregatedResults.agent_errors`` carries the typed block.
5. The benchmark card carries the typed ``agent_errors`` block and
   the markdown rendering surfaces a dedicated section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from belt.agent.base import BaseAgentAdapter
from belt.aggregator.stats import aggregate_cost_timing, build_bottom_line, collect_agent_errors, compute_reliability
from belt.benchmark_card import build_card, render_markdown
from belt.commands.score import score_scenario
from belt.constants import SCHEMA_VERSION
from belt.entities import (
    AUTHENTICATION_FAILED,
    AgentConfig,
    AggregatedResults,
    GroupConfig,
    Scenario,
    Turn,
    TurnExpectation,
    TurnOutput,
)
from belt.runner.orchestrator import build_agent_config, run_scenario_turns
from belt.scorer.rules import RuleBasedScorer

# ── Mock-error adapter ──


class MockErrorAgentAdapter(BaseAgentAdapter):
    """Always returns the canonical Claude Code logged-out shape.

    Lives in the test module rather than the registry so the public
    agent list isn't polluted. The orchestrator path doesn't care about
    discovery - it accepts any ``BaseAgentAdapter`` instance directly,
    same as :class:`StubAgentAdapter` in
    :mod:`tests.test_integration_flow`.
    """

    REPLY_TEXT = "Not logged in \u00b7 Please run /login"

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def setup(self, config: AgentConfig) -> None:
        pass

    def execute(self, message: str, flags: list[str]) -> str:
        return '{"type":"result","is_error":true}\n'

    def fetch_results(self, raw_output: str) -> TurnOutput:
        # Hand-crafted to match what the real Claude Code adapter
        # produces from a logged-out session: has_error=true,
        # error_type="authentication_failed", reply_text carries the
        # CLI's own login message.
        return TurnOutput(
            raw_cli=raw_output,
            reply_text=self.REPLY_TEXT,
            has_reply=True,
            has_error=True,
            error_type=AUTHENTICATION_FAILED,
        )

    def teardown(self) -> None:
        pass

    def metadata(self) -> dict[str, Any] | None:
        return None


# ── Helpers ──


def _scenario(name: str, *, expect_contains: list[str] | None = None) -> Scenario:
    """Build a one-turn scenario.

    ``expect_contains`` controls vacuousness: omit it (or pass an
    empty list) and the rules will pass on any response, masking the
    agent error - the canonical "vacuous pass" shape.
    """
    expect = TurnExpectation(no_errors=True, has_reply=True, contains=expect_contains)
    return Scenario(
        name=name,
        description=f"E2E mock-error scenario: {name}",
        turns=[Turn(message="Hello", expect=expect)],
    )


def _run_pipeline(
    tmp_path: Path,
    scenarios: list[Scenario],
) -> tuple[AggregatedResults, dict | None]:
    """Drive scenarios through orchestrator -> scorer -> aggregator.

    Mirrors the production flow in :mod:`belt.commands.aggregate`:
    each scenario is executed, scored, then the aggregator collects
    agent_errors and builds the headline.
    """
    group = GroupConfig(agent="mock-error")
    outcomes_root = tmp_path / "outcomes"
    scenarios_dir = tmp_path / "scenarios" / "g"
    scenarios_dir.mkdir(parents=True)
    (scenarios_dir / "_config.json").write_text(group.model_dump_json(indent=2))

    scores = []
    for s in scenarios:
        agent = MockErrorAgentAdapter()
        config = build_agent_config(group, s, shared_state=None)
        outcome_dir = outcomes_root / "g" / s.name
        run_scenario_turns(agent, s, outcome_dir, config)
        (scenarios_dir / f"{s.name}.json").write_text(s.model_dump_json(indent=2))
        with patch("belt.scorer.scenario_map.SCENARIOS_DIR", scenarios_dir.parent):
            scores.append(score_scenario(outcome_dir, outcomes_root, [RuleBasedScorer()]))

    cost_timing = aggregate_cost_timing(outcomes_root, scores)
    reliability = compute_reliability(scores)
    agent_errors = collect_agent_errors(outcomes_root, scores, agent_name="claude-code")
    bottom_line = build_bottom_line(scores, agent_errors=agent_errors)

    failed = [s for s in scores if not s.overall_pass]
    aggregated = AggregatedResults(
        schema_version=SCHEMA_VERSION,
        total=len(scores),
        passed=len(scores) - len(failed),
        failed=len(failed),
        overall_pass=not bool(failed),
        cost_timing=cost_timing,
        reliability=reliability,
        agent_errors=agent_errors,
        bottom_line=bottom_line,
        scenarios=[s.model_dump(mode="json") for s in scores],
    )
    return aggregated, agent_errors


# ── Tests ──


class TestAuthFailureScenario:
    """One scenario, real auth-failure shape, hard rule expectations."""

    def test_orchestrator_records_agent_errors(self, tmp_path: Path) -> None:
        agent = MockErrorAgentAdapter()
        scenario = _scenario("auth_failure_basic", expect_contains=["impossible_response"])
        config = build_agent_config(GroupConfig(agent="mock-error"), scenario, shared_state=None)
        outcome_dir = tmp_path / "g" / "auth_failure_basic"
        result = run_scenario_turns(agent, scenario, outcome_dir, config)

        # Per-scenario record carries the auth-failed token.
        assert result.agent_errors == [AUTHENTICATION_FAILED]
        assert result.error is None  # subprocess didn't crash

    def test_aggregator_surfaces_agent_error_headline(self, tmp_path: Path) -> None:
        scenarios = [_scenario("auth_failure_basic", expect_contains=["impossible_response"])]
        aggregated, agent_errors = _run_pipeline(tmp_path, scenarios)

        assert agent_errors is not None
        assert agent_errors["scenarios_with_errors"] == 1
        assert agent_errors["by_error_type"] == {AUTHENTICATION_FAILED: 1}
        # Agent-specific remediation is included when ``agent_name`` is
        # passed (the aggregator resolves this from ``_runtime_info``
        # sidecars in production).
        assert "claude login" in agent_errors["remediation"]

        # bottom_line leads with the agent-error headline.
        joined = "\n".join(aggregated.bottom_line)
        assert "Agent error in 1/1" in joined
        assert AUTHENTICATION_FAILED in joined


class TestVacuousPassScenario:
    """Rules pass while the agent errored - the misleading 5/7 case."""

    def test_vacuous_pass_counted_and_warned(self, tmp_path: Path) -> None:
        # The default ``TurnExpectation.no_errors=True`` adds the
        # ``execution/no_errors`` rule, which would catch the
        # ``has_error=True`` and fail the scenario. To construct a
        # genuinely vacuous pass we explicitly disable it - the shape
        # produced by a scenario whose author cared about
        # ``has_reply`` only and forgot to assert ``no_errors``.
        scenario = Scenario(
            name="vacuous_pass",
            description="Rules pass even when the agent errored",
            turns=[Turn(message="Hello", expect=TurnExpectation(no_errors=False, has_reply=True))],
        )
        aggregated, agent_errors = _run_pipeline(tmp_path, [scenario])

        # Vacuous-pass annotation surfaces in both the typed block and
        # the rendered headline.
        assert agent_errors is not None
        assert agent_errors["vacuous_passes"] == 1
        joined = "\n".join(aggregated.bottom_line)
        assert "vacuous" in joined.lower()
        assert "rules" in joined.lower()


class TestBenchmarkCardCarriesBlock:
    def test_card_block_and_markdown_section(self, tmp_path: Path) -> None:
        scenarios = [_scenario("auth_failure_basic", expect_contains=["impossible_response"])]
        aggregated, _ = _run_pipeline(tmp_path, scenarios)

        # ``build_card`` reads ``run_meta.json``; create a minimal one
        # so the best-effort builder succeeds.
        outcomes_root = tmp_path / "outcomes"
        (outcomes_root / "run_meta.json").write_text("{}")

        # The card builder consumes the same dict that ``aggregate``
        # serialises to ``results.json``.
        results_dict = aggregated.model_dump(mode="json", exclude_none=False)
        card = build_card(outcomes_root, results_dict)

        # Typed block on the card.
        assert card.agent_errors is not None
        assert card.agent_errors.scenarios_with_errors == 1
        assert card.agent_errors.by_error_type == {AUTHENTICATION_FAILED: 1}
        assert "claude login" in card.agent_errors.remediation

        md = render_markdown(card)
        assert "## Agent errors" in md
        # Underscores escape via md_safe in the rendered table.
        assert r"authentication\_failed" in md


class TestPipelineSummary:
    """Smoke test - the full chain holds for a single run."""

    def test_full_chain_satisfied(self, tmp_path: Path) -> None:
        scenarios = [_scenario("auth_failure_basic", expect_contains=["impossible_response"])]
        aggregated, agent_errors = _run_pipeline(tmp_path, scenarios)

        # Orchestrator-level signal lands in the aggregator's view.
        assert agent_errors is not None and agent_errors["scenarios_with_errors"] == 1
        # Bottom-line headline references the run.
        assert any("Agent error in 1/1" in line for line in aggregated.bottom_line)
        # Typed block on AggregatedResults survives serialisation.
        assert aggregated.agent_errors is not None
        # Per-scenario detail includes the literal reply.
        per = aggregated.agent_errors["per_scenario"]
        assert per and per[0]["first_reply_text"].startswith("Not logged in")
        # Scenario actually failed (rules detected has_error).
        assert aggregated.failed == 1
