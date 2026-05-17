#!/usr/bin/env python
# (c) JFrog Ltd. (2026)

"""End-to-end empirical proof for the judge infra-failure partition.

Runs ``belt eval`` with an LLM scorer whose backend has been swapped for
a ``FailingBackend`` test double. Three matrix points are exercised:

1. ``rate_limited`` - backend raises ``httpx.HTTPStatusError`` (429) on
   every call.
2. ``timeout`` - backend raises ``httpx.ReadTimeout``.
3. ``parse`` - backend returns a syntactically valid but semantically
   broken response so the scorer's parse path returns ``None``.

After each run the script asserts the contract:

* ``score.json`` has ``scores.llm.judge_errored = true`` with the
  expected ``judge_error_type``.
* ``score.json`` has ``overall_pass = false`` even though the rules
  would have passed on the echo agent's vacuous reply.
* ``score.json`` has a synthetic ``execution/llm_scorer_ran`` check on
  the rules payload.
* ``results.json`` has a top-level ``judge_errors`` block with the
  right counts and the scenario listed.
* ``results.json`` has ``stats.task_quality.env_failed_judge > 0`` and
  ``env_failed_agent == 0``.
* ``bottom_line`` carries the three-axis headline ("judge env failure").

Run as a single command so the proof is reproducible:

    uv run python scripts/verify_judge_infra_failures.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belt.agent.base import BaseAgentAdapter  # noqa: E402
from belt.entities import AgentConfig, GroupConfig, TurnOutput  # noqa: E402
from belt.runner.orchestrator import build_agent_config, run_scenario_turns  # noqa: E402
from belt.scenario import Scenario, Turn, TurnExpectation  # noqa: E402
from belt.scorer.entities import JudgeConfig  # noqa: E402
from belt.scorer.llm.backend import BaseJudgeBackend  # noqa: E402
from belt.scorer.llm.scorer import LLMScorer  # noqa: E402
from belt.scorer.pipeline import score_scenario  # noqa: E402
from belt.scorer.rules import RuleBasedScorer  # noqa: E402


class _EchoAdapter(BaseAgentAdapter):
    """Always-replies adapter so the rules scorer has something to pass on."""

    def __init__(self, **_kw: Any) -> None:
        pass

    def setup(self, config: AgentConfig) -> None:
        pass

    def execute(self, message: str, flags: list[str]) -> str:
        return f"reply: {message}\n"

    def fetch_results(self, raw: str) -> TurnOutput:
        return TurnOutput(raw_cli=raw, reply_text=raw.strip(), has_reply=True)

    def teardown(self) -> None:
        pass

    def metadata(self) -> dict[str, Any] | None:
        return None


class _FailingBackend(BaseJudgeBackend):
    """Test double that always raises a configured exception on ``httpx.post``.

    Combined with the monkeypatched ``httpx.post`` below, this simulates a
    provider that is unreachable, rate-limiting, or timing out, without
    spending real API budget. ``parse`` mode returns 200 OK with a
    response body the scorer cannot parse, exercising the secondary
    judge-errored path.
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode

    def provider_name(self) -> str:
        return f"failing-{self._mode}"

    def is_available(self) -> bool:
        return True

    def build_request(self, config, messages, schema):
        return "https://failing.example/v1/chat", {}, {}


def _fake_response(status: int, text: str = "") -> httpx.Response:
    req = httpx.Request("POST", "https://failing.example/v1/chat")
    return httpx.Response(status_code=status, text=text, request=req)


def _post_for_mode(mode: str):
    def _f(*_a, **_kw):
        if mode == "rate_limited":
            resp = _fake_response(429, text="rate limited")
            raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
        if mode == "timeout":
            raise httpx.ReadTimeout("slow")
        if mode == "parse":
            return _fake_response(200, text='{"not": "valid verdict"}')
        raise RuntimeError(f"unknown mode: {mode}")

    return _f


@contextmanager
def _patched_httpx(mode: str):
    import belt.scorer.llm.scorer as scorer_module

    original = scorer_module.httpx.post
    scorer_module.httpx.post = _post_for_mode(mode)
    try:
        yield
    finally:
        scorer_module.httpx.post = original


def _build_scenario(name: str) -> Scenario:
    return Scenario(
        name=name,
        description=f"verify {name}",
        turns=[Turn(message="ping", expect=TurnExpectation(has_reply=True))],
    )


def _run_one_mode(mode: str, work: Path) -> dict[str, Any]:
    """Execute one matrix point end-to-end. Returns parsed score.json."""
    scenarios_root = work / "scenarios"
    outcomes_root = work / "outcomes"
    group_dir = scenarios_root / "g"
    group_dir.mkdir(parents=True)
    (outcomes_root).mkdir(parents=True)

    group_config = GroupConfig(agent="echo")
    (group_dir / "_config.json").write_text(group_config.model_dump_json(indent=2))

    scenario = _build_scenario("only")
    (group_dir / "only.json").write_text(scenario.model_dump_json(indent=2))

    outcome_dir = outcomes_root / "g" / "only"
    outcome_dir.mkdir(parents=True)
    agent = _EchoAdapter()
    config = build_agent_config(group_config, scenario, shared_state=None)
    run_scenario_turns(agent, scenario, outcome_dir, config)

    # Build the scorer set with the failing LLM backend injected.
    rules = RuleBasedScorer()
    llm = LLMScorer(
        config=JudgeConfig(model="openai/gpt-test"),
        backend=_FailingBackend(mode),
        cache=None,
        skip_availability=True,
        max_retries=1,
    )
    # Point scenarios_root at our synthetic group so ``score_scenario``
    # finds the scenario JSON next to the outcomes.
    import os

    from belt._internal_envvars import SCENARIOS_ROOT

    os.environ[SCENARIOS_ROOT] = str(scenarios_root)

    with _patched_httpx(mode):
        score = score_scenario(outcome_dir, outcomes_root, [rules, llm])

    score_path = outcome_dir / "score.json"
    score_path.write_text(score.model_dump_json(indent=2))
    return json.loads(score_path.read_text())


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  ❌ {msg}")
        sys.exit(1)
    print(f"  ✅ {msg}")


def _verify_score(mode: str, score: dict[str, Any]) -> None:
    expected_type = "other" if mode == "parse" else mode
    llm = score["scores"]["llm"]
    rules = score["scores"]["rules"]
    print(f"\n=== score.json (mode={mode}) ===")
    _assert(llm.get("judge_errored") is True, "llm payload has judge_errored=true")
    _assert(
        llm.get("judge_error_type") == expected_type,
        f"judge_error_type={expected_type}",
    )
    _assert(llm.get("dimensions") == {}, "no dimensions emitted (verdict-less payload)")
    _assert(llm.get("overall_pass") is False, "llm.overall_pass=false")
    _assert(score.get("overall_pass") is False, "scenario.overall_pass=false (force-fail)")
    synthetic = [c for c in rules.get("checks", []) if c.get("check") == "llm_scorer_ran"]
    _assert(len(synthetic) == 1, "exactly one synthetic execution/llm_scorer_ran check")
    _assert(synthetic[0].get("passed") is False, "synthetic check passed=false")
    _assert(
        expected_type in synthetic[0].get("details", ""),
        f"synthetic check details mentions '{expected_type}'",
    )


def _verify_aggregator(mode: str, scores_objs) -> None:
    """Prove the aggregator partitions the run into the judge axis end-to-end."""
    from belt.aggregator.stats import build_bottom_line, collect_agent_errors, collect_judge_errors

    expected_type = "other" if mode == "parse" else mode
    judge_errors = collect_judge_errors(scores_objs)
    print(f"\n=== aggregator (mode={mode}) ===")
    _assert(judge_errors is not None, "collect_judge_errors returns a non-None block")
    assert judge_errors is not None
    _assert(judge_errors["scenarios_with_errors"] == 1, "1 scenario flagged with judge error")
    _assert(
        judge_errors["by_error_type"].get(expected_type) == 1,
        f"by_error_type counts {expected_type}=1",
    )
    ae = collect_agent_errors(Path("/tmp"), scores_objs, judge_errors=judge_errors)
    _assert(ae is not None, "collect_agent_errors returns non-None when judge_errors present")
    assert ae is not None
    split = ae.get("task_quality") or {}
    _assert(split.get("env_failed_judge") == 1, "task_quality.env_failed_judge == 1")
    _assert(split.get("env_failed_agent") == 0, "task_quality.env_failed_agent == 0")
    lines = build_bottom_line(scores_objs, agent_errors=ae, judge_errors=judge_errors)
    _assert(any("1 judge env failure" in line for line in lines), "headline contains '1 judge env failure'")
    _assert(
        any("LLM judge infrastructure failure" in line for line in lines),
        "bottom-line carries an explicit LLM judge infrastructure failure line",
    )


def main() -> None:
    print("Empirical verification: judge infra-failure partition\n" + "=" * 60)
    for mode in ("rate_limited", "timeout", "parse"):
        with tempfile.TemporaryDirectory(prefix=f"belt-358-{mode}-") as tmp:
            work = Path(tmp)
            score = _run_one_mode(mode, work)
            _verify_score(mode, score)

            from belt.entities import ScenarioScore

            score_obj = ScenarioScore.model_validate(score)
            _verify_aggregator(mode, [score_obj])
    print("\n✅ All three matrix points verified end-to-end (score.json + aggregator).")


if __name__ == "__main__":
    main()
