# (c) JFrog Ltd. (2026)

"""Pydantic schema for ``--scorer-config`` YAML.

Pinned guarantees:

1. ``ScorerConfigFile`` and ``JudgeDef`` reject unknown keys
   (``extra="forbid"``) so a typo (``resolutionn:``) hard-fails at
   YAML load rather than silently downgrading to defaults.
2. ``JudgeDef.resolution`` and ``JudgeDef.evidence_scope`` default to
   the safe values (``"scenario"`` and ``"isolated"``).
3. Reserved scorer names (``rules``, ``llm`` without ``consensus:``)
   collide with built-in keys and reject at validation time.
4. Unknown enum values for ``resolution`` and ``evidence_scope``
   surface the offending field name in the error message.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from belt.scorer.config_schema import JudgeDef, ScorerConfigFile


class TestExtraForbid:
    def test_unknown_judge_def_key_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            JudgeDef.model_validate({"name": "ok", "resolutionn": "turn"})
        assert "resolutionn" in str(ei.value)

    def test_unknown_top_level_key_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            ScorerConfigFile.model_validate({"judges": {"a": {"name": "a"}}, "consensuss": "majority"})
        assert "consensuss" in str(ei.value)


class TestDefaults:
    def test_resolution_defaults_to_scenario(self) -> None:
        j = JudgeDef(name="ok")
        assert j.resolution == "scenario"

    def test_evidence_scope_defaults_to_isolated(self) -> None:
        j = JudgeDef(name="ok")
        assert j.evidence_scope == "isolated"


class TestReservedNames:
    """Built-in scorer keys (``rules``, ``llm``) must not be reused.

    The validator lives on ``JudgeDef`` so both direct construction
    (``JudgeDef(name="rules")``) and the round-trip flatten
    (``ScorerConfigFile(judges={...}).to_judge_defs()``) catch the
    collision. Reusing either name would silently overwrite the
    built-in contribution in ``ScenarioScore.scores`` - the validator
    rejects loudly instead.
    """

    @pytest.mark.parametrize("reserved", ["rules", "llm"])
    def test_judge_def_rejects_reserved_name(self, reserved: str) -> None:
        with pytest.raises(ValidationError) as ei:
            JudgeDef(name=reserved)
        assert reserved in str(ei.value)

    @pytest.mark.parametrize("reserved", ["rules", "llm"])
    def test_scorer_config_flatten_rejects_reserved_name(self, reserved: str) -> None:
        cfg = ScorerConfigFile(judges={reserved: {}})
        with pytest.raises(ValidationError) as ei:
            cfg.to_judge_defs()
        assert reserved in str(ei.value)

    def test_arbitrary_judge_names_allowed(self) -> None:
        # A typical consensus bundle uses arbitrary names like
        # ``sonnet``, ``gpt`` - these must round-trip cleanly.
        cfg = ScorerConfigFile(
            judges={"sonnet": {"model": "anthropic/claude-sonnet-4-5"}, "gpt": {"model": "openai/gpt-5.4-mini"}},
            consensus="majority",
        )
        defs = cfg.to_judge_defs()
        assert {d.name for d in defs} == {"sonnet", "gpt"}
        assert cfg.consensus == "majority"


class TestEnums:
    def test_unknown_resolution_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            JudgeDef.model_validate({"name": "ok", "resolution": "tunr"})
        assert "resolution" in str(ei.value)

    def test_unknown_evidence_scope_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            JudgeDef.model_validate({"name": "ok", "evidence_scope": "isloated"})
        assert "evidence_scope" in str(ei.value)

    @pytest.mark.parametrize("res", ["scenario", "turn"])
    def test_resolution_accepts_known(self, res: str) -> None:
        j = JudgeDef(name="ok", resolution=res)
        assert j.resolution == res

    @pytest.mark.parametrize("scope", ["isolated", "cumulative"])
    def test_evidence_scope_accepts_known(self, scope: str) -> None:
        j = JudgeDef(name="ok", evidence_scope=scope)
        assert j.evidence_scope == scope


class TestOutputMode:
    def test_output_mode_defaults_to_none(self) -> None:
        assert JudgeDef.model_validate({"name": "j"}).output_mode is None

    def test_json_schema_output_mode_accepted(self) -> None:
        assert JudgeDef.model_validate({"name": "j", "output_mode": "json_schema"}).output_mode == "json_schema"

    def test_unknown_output_mode_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            JudgeDef.model_validate({"name": "j", "output_mode": "xml"})
        assert "output_mode" in str(ei.value)


class TestScorerConfigReachesJudgeConfig:
    """``output_mode`` and an explicit ``temperature: null`` reach the judge's ``JudgeConfig``."""

    def _build(self, tmp_path, yaml_text: str):
        from belt.scorer.pipeline import build_scorers

        path = tmp_path / "scorer.yaml"
        path.write_text(yaml_text)
        _, llm_scorers = build_scorers("llm", {}, str(path), skip_availability=True)
        return llm_scorers[0].config

    def test_output_mode_and_null_temperature(self, tmp_path) -> None:
        cfg = self._build(
            tmp_path,
            "judges:\n  j:\n    model: anthropic/claude-sonnet-5-5\n"
            "    output_mode: json_schema\n    temperature: null\n",
        )
        assert cfg.output_mode == "json_schema"
        assert cfg.temperature is None

    def test_omitted_keys_keep_defaults(self, tmp_path) -> None:
        cfg = self._build(tmp_path, "judges:\n  j:\n    model: anthropic/claude-sonnet-4-6\n")
        assert cfg.output_mode == "tool"
        assert cfg.temperature == 0.0
