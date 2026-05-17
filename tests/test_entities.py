# (c) JFrog Ltd. (2026)

"""Tests for pure entity models."""

from __future__ import annotations

from belt.entities import (
    AgentConfig,
    GroupConfig,
    ScenarioResult,
    ScenarioScore,
    ToolCall,
    TurnExpectation,
    TurnOutput,
    TurnTiming,
)


def test_group_config_with_default_tags() -> None:
    gc = GroupConfig(agent="test", default_tags=["production", "v1", "web"])
    assert gc.default_tags == ["production", "v1", "web"]


def test_group_config_ignores_unknown_keys() -> None:
    """Plugin-specific keys in ``_config.json`` are silently dropped (Pydantic
    default ``extra='ignore'``). This is how the schema boundary works: core
    declares only what core reads; plugins add their own keys without
    coordinating with core. See docs/glossary/PLUGGABILITY.md#schema-boundary.
    """
    import json

    gc = GroupConfig.model_validate_json(
        json.dumps({"agent": "test", "plugin_foo": "x.json", "plugin_bar": "refs.json"})
    )
    assert gc.agent == "test"
    assert not hasattr(gc, "plugin_foo")
    assert not hasattr(gc, "plugin_bar")


def test_group_config_missing_agent_raises_validation_error() -> None:
    """Constructor must receive ``agent``; Pydantic reports the field as ``agent``."""
    import pytest
    from pydantic import ValidationError

    for kwargs in ({}, {"default_tags": ["smoke"]}):
        with pytest.raises(ValidationError) as exc_info:
            GroupConfig(**kwargs)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("agent",)


def test_group_config_missing_agent_model_validate_dict() -> None:
    """Parsed dicts (e.g. from JSON/YAML) must include ``agent``."""
    import pytest
    from pydantic import ValidationError

    for payload in ({}, {"default_tags": ["smoke"]}):
        with pytest.raises(ValidationError) as exc_info:
            GroupConfig.model_validate(payload)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("agent",)


def test_group_config_requires_agent_in_json() -> None:
    """``_config.json`` must set ``agent``; Pydantic reports the missing field by name."""
    import json

    import pytest
    from pydantic import ValidationError

    for payload in ({}, {"default_tags": ["smoke"]}):
        with pytest.raises(ValidationError) as exc_info:
            GroupConfig.model_validate_json(json.dumps(payload))
        assert "agent" in str(exc_info.value)


def test_group_config_minimal() -> None:
    gc = GroupConfig(agent="claude-code")
    assert gc.agent == "claude-code"
    assert gc.default_tags == []


def test_turn_expectation_defaults() -> None:
    e = TurnExpectation()
    assert e.no_errors is True
    assert e.has_reply is True
    assert e.tools_invoked == []
    assert e.has_thinking is None


def test_turn_expectation_accepts_plugin_extras() -> None:
    # extra="allow" lets adapter plugins declare their own scoring keys in
    # scenario JSON without core schema changes (consumed via expect.model_extra).
    e = TurnExpectation.model_validate({"plugin_specific_check": True, "max_handoffs": 3})
    assert e.model_extra == {"plugin_specific_check": True, "max_handoffs": 3}


# ── TurnOutput ──


def test_turn_output_minimal() -> None:
    output = TurnOutput(raw_cli="some output")
    assert output.raw_cli == "some output"
    assert output.raw_state is None
    assert output.reply_text == ""
    assert output.tool_calls == []
    assert output.has_reply is False
    assert output.has_error is None
    assert output.timing is None


def test_turn_output_full() -> None:
    output = TurnOutput(
        raw_cli="│ 1.0s 🧠 reply_to_user",
        raw_state='{"values": {}}',
        reply_text="Here are your results",
        tool_calls=[ToolCall(name="search", call_id="call_1", args={"query": "test"})],
        has_reply=True,
        has_error=False,
        timing=TurnTiming(ttft=1.0, total=2.0),
    )
    assert output.has_reply is True
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].name == "search"
    assert output.timing.total == 2.0


def test_turn_output_serialization_roundtrip() -> None:
    output = TurnOutput(
        raw_cli="cli text",
        reply_text="hello",
        tool_calls=[ToolCall(name="tool_a", call_id="c1")],
        has_reply=True,
    )
    json_str = output.model_dump_json()
    restored = TurnOutput.model_validate_json(json_str)
    assert restored.reply_text == "hello"
    assert restored.tool_calls[0].name == "tool_a"


# ── AgentConfig ──


def test_agent_config_with_shared_state() -> None:
    gc = GroupConfig(agent="test")
    config = AgentConfig(group_config=gc, scenario_name="test_scenario", shared_state={"key": "value"})
    assert config.shared_state["key"] == "value"
    assert config.scenario_name == "test_scenario"


def test_agent_config_no_shared_state() -> None:
    gc = GroupConfig(agent="claude-code")
    config = AgentConfig(group_config=gc, scenario_name="read_file")
    assert config.shared_state is None


def test_agent_config_scenario_options() -> None:
    gc = GroupConfig(agent="test")
    config = AgentConfig(group_config=gc, scenario_name="test", scenario_options={"custom_key": "custom_value"})
    assert config.scenario_options["custom_key"] == "custom_value"


def test_agent_config_scenario_options_default_empty() -> None:
    gc = GroupConfig(agent="test")
    config = AgentConfig(group_config=gc, scenario_name="test")
    assert config.scenario_options == {}


# ── ScenarioResult ──


def test_scenario_result_with_agent_metadata() -> None:
    result = ScenarioResult(
        scenario_name="test",
        group_path="production/web",
        turns_completed=3,
        agent_metadata={"k1": "v1", "k2": "v2"},
    )
    assert result.agent_metadata["k1"] == "v1"
    assert result.agent_metadata["k2"] == "v2"


def test_scenario_result_no_metadata() -> None:
    result = ScenarioResult(scenario_name="test", group_path="group")
    assert result.agent_metadata is None
    assert result.turns_completed == 0


# ── JudgeVerdict (dynamic dimensions) ──


def test_judge_verdict_dynamic_dimensions() -> None:
    """JudgeVerdict accepts arbitrary dimension fields via extra='allow'."""
    from belt.entities import JudgeVerdict

    raw = {
        "execution": {"reasoning": "clean", "score": "high"},
        "custom_dim": {"reasoning": "good", "score": "medium"},
        "overall_pass": True,
    }
    verdict = JudgeVerdict(**raw)
    assert verdict.overall_pass is True
    assert verdict.model_extra["execution"]["score"] == "high"
    assert verdict.model_extra["custom_dim"]["score"] == "medium"


def test_judge_verdict_dimension_scores_property() -> None:
    from belt.entities import JudgeVerdict

    raw = {
        "execution": {"reasoning": "ok", "score": "high"},
        "trajectory": {"reasoning": "fine", "score": "medium"},
        "overall_pass": False,
    }
    verdict = JudgeVerdict(**raw)
    dims = verdict.dimension_scores
    assert "execution" in dims
    assert "trajectory" in dims
    assert dims["execution"].score.value == "high"


def test_judge_verdict_model_validate_json() -> None:
    import json

    from belt.entities import JudgeVerdict

    data = {
        "execution": {"reasoning": "good", "score": "high"},
        "response_quality": {"reasoning": "accurate", "score": "high"},
        "overall_pass": True,
    }
    verdict = JudgeVerdict.model_validate_json(json.dumps(data))
    assert verdict.overall_pass is True
    assert len(verdict.dimension_scores) == 2


def test_judge_verdict_model_dump() -> None:
    from belt.entities import JudgeVerdict

    raw = {
        "execution": {"reasoning": "clean", "score": "high"},
        "overall_pass": True,
    }
    verdict = JudgeVerdict(**raw)
    dumped = verdict.model_dump(mode="json")
    assert dumped["overall_pass"] is True
    assert dumped["execution"]["score"] == "high"


# ── ScenarioScore ──


def test_scenario_score_default_cost_fields() -> None:
    score = ScenarioScore(scenario_name="s1", group="g1", overall_pass=True)
    assert score.judge_cost_usd is None
    assert score.scorer_prompt_tokens == 0
    assert score.scorer_completion_tokens == 0


def test_scenario_score_with_cost() -> None:
    score = ScenarioScore(
        scenario_name="s1",
        group="g1",
        overall_pass=True,
        judge_cost_usd=0.0042,
        scorer_prompt_tokens=1500,
        scorer_completion_tokens=300,
    )
    assert score.judge_cost_usd == 0.0042
    assert score.scorer_prompt_tokens == 1500


def test_scenario_score_serialization_roundtrip() -> None:
    score = ScenarioScore(
        scenario_name="s1",
        group="g1",
        overall_pass=False,
        judge_cost_usd=0.0123,
        scorer_prompt_tokens=2000,
        scorer_completion_tokens=500,
    )
    json_str = score.model_dump_json()
    restored = ScenarioScore.model_validate_json(json_str)
    assert restored.judge_cost_usd == 0.0123
    assert restored.scorer_prompt_tokens == 2000
    assert restored.scorer_completion_tokens == 500


def test_scenario_score_cost_fields_default_when_absent() -> None:
    """Cost fields are optional in the v1 contract: ``judge_cost_usd``
    defaults to ``None`` and ``scorer_prompt_tokens`` defaults to ``0``
    when callers don't supply them."""
    import json

    payload = {
        "scenario_name": "s1",
        "group": "g1",
        "scores": {},
        "overall_pass": True,
    }
    score = ScenarioScore.model_validate_json(json.dumps(payload))
    assert score.judge_cost_usd is None
    assert score.scorer_prompt_tokens == 0


def test_scenario_score_tags_default_to_empty_list() -> None:
    """The ``tags`` field is optional in the v1 contract and defaults to
    an empty list when omitted by the caller."""
    import json

    payload = {
        "scenario_name": "s1",
        "group": "g1",
        "scores": {},
        "overall_pass": True,
    }
    score = ScenarioScore.model_validate_json(json.dumps(payload))
    assert score.tags == []


def test_scenario_score_tags_round_trip() -> None:
    """Tags survive a JSON round-trip and are preserved verbatim."""
    score = ScenarioScore(
        scenario_name="s1",
        group="g1",
        overall_pass=True,
        tags=["showcase", "real-runnable", "single-turn"],
    )
    restored = ScenarioScore.model_validate_json(score.model_dump_json())
    assert restored.tags == ["showcase", "real-runnable", "single-turn"]


def test_group_config_uses_agent_key() -> None:
    """_config.json files use the 'agent' key."""
    import json

    from belt.entities import GroupConfig

    gc = GroupConfig.model_validate_json(json.dumps({"agent": "claude-code"}))
    assert gc.agent == "claude-code"


def test_group_config_rejects_legacy_adapter_key() -> None:
    """The legacy 'adapter' key is no longer accepted (clean break)."""
    import json

    import pytest
    from pydantic import ValidationError

    from belt.entities import GroupConfig

    with pytest.raises(ValidationError):
        GroupConfig.model_validate_json(json.dumps({"adapter": "cursor"}))


# ── AggregatedResults.scenarios_skipped ──


def test_aggregated_results_scenarios_skipped_default_zero() -> None:
    """Optional v1 field - construction sites that don't pass
    ``scenarios_skipped`` get the documented default of ``0``."""
    from belt.entities import AggregatedResults

    ar = AggregatedResults()
    assert ar.scenarios_skipped == 0


def test_aggregated_results_scenarios_skipped_round_trip() -> None:
    """Round-trip through JSON - the count survives ``write_json`` →
    re-parse, which is the path ``results.json`` consumers depend on."""
    import json

    from belt.entities import AggregatedResults

    original = AggregatedResults(
        schema_version="1",
        total=10,
        passed=8,
        failed=2,
        scenarios_skipped=3,
        overall_pass=False,
    )
    blob = original.model_dump_json()
    parsed = AggregatedResults.model_validate(json.loads(blob))
    assert parsed.scenarios_skipped == 3


def test_aggregated_results_parses_when_scenarios_skipped_absent() -> None:
    """``scenarios_skipped`` is optional in the v1 contract: a
    ``results.json`` that omits it parses cleanly and surfaces ``0``."""
    import json

    from belt.entities import AggregatedResults

    blob = json.dumps(
        {
            "schema_version": "1",
            "total": 5,
            "passed": 5,
            "failed": 0,
            "overall_pass": True,
        }
    )
    parsed = AggregatedResults.model_validate_json(blob)
    assert parsed.scenarios_skipped == 0
