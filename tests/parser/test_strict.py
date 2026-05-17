# (c) JFrog Ltd. (2026)

"""Tests for ``belt.parser.strict`` and the ``--strict-config`` loader path.

Covers:

- ``validate_strict`` recursion and did-you-mean suggestions across
  every nested model (Scenario → Turn → TurnExpectation /
  StateExpectation, GroupConfig → Resource).
- ``register_plugin_scenario_key`` happy path, idempotency,
  reserved-key shadowing rejection, key-shape validation.
- ``ScenarioLoader.{load_scenario, load_group_config,
  load_group_scenarios}`` strict_config plumbing.
- Public API: ``register_plugin_scenario_key`` is reachable from the
  top-level ``belt`` package.

Permissive default is held by every other ``ScenarioLoader`` test in
this directory, so this file focuses exclusively on strict-mode
behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from belt.parser.scenario import ScenarioLoader
from belt.parser.strict import (
    StrictConfigError,
    register_plugin_scenario_key,
    registered_plugin_scenario_keys,
    validate_strict,
)
from belt.scenario import GroupConfig, Resource, Scenario, StateExpectation, Turn, TurnExpectation


# Each test that mutates the registry must clean up after itself so
# parallel test runs and ordering changes don't leak keys.
@pytest.fixture(autouse=True)
def _clean_plugin_registry():
    from belt.parser.strict import _PLUGIN_KEYS

    snapshot = {k: set(v) for k, v in _PLUGIN_KEYS.items()}
    try:
        yield
    finally:
        _PLUGIN_KEYS.clear()
        _PLUGIN_KEYS.update(snapshot)


# ── validate_strict: top-level coverage ──


class TestValidateStrictTopLevel:
    def test_scenario_clean_doc_passes(self) -> None:
        raw = {
            "name": "ok",
            "description": "fine",
            "turns": [{"message": "hi"}],
        }
        assert validate_strict(raw, Scenario, source="x.json") == []

    def test_scenario_unknown_top_level_key_reported(self) -> None:
        raw = {
            "name": "typo",
            "description": "x",
            "turns": [{"message": "hi"}],
            "tags": [],
            "tag": "real-runnable",
        }
        errors = validate_strict(raw, Scenario, source="s.json")
        assert errors == ["s.json: unknown key 'tag'. Did you mean 'tags'?"]

    def test_scenario_unknown_with_no_close_match_omits_hint(self) -> None:
        raw = {
            "name": "x",
            "description": "x",
            "turns": [],
            "completely_unrelated_field_xyz": True,
        }
        errors = validate_strict(raw, Scenario, source="s.json")
        assert len(errors) == 1
        assert "completely_unrelated_field_xyz" in errors[0]
        assert "Did you mean" not in errors[0]

    def test_non_dict_root_returns_empty(self) -> None:
        # Pydantic produces the better error in this case; we don't
        # double up.
        assert validate_strict([1, 2, 3], Scenario, source="x.json") == []
        assert validate_strict("not a doc", Scenario, source="x.json") == []
        assert validate_strict(None, Scenario, source="x.json") == []


# ── validate_strict: nested recursion ──


class TestValidateStrictRecursion:
    def test_turn_expect_typo_caught(self) -> None:
        # ``tools_invoke`` -> ``tools_invoked``: today this would silently
        # land in ``model_extra`` (TurnExpectation has extra="allow") and
        # produce zero coverage. Strict mode must reject it.
        raw = {
            "name": "x",
            "description": "x",
            "turns": [
                {
                    "message": "hi",
                    "expect": {"tools_invoke": ["foo"]},
                }
            ],
        }
        errors = validate_strict(raw, Scenario, source="s.json")
        assert len(errors) == 1
        assert "turns[0].expect.tools_invoke" in errors[0]
        assert "Did you mean 'tools_invoked'" in errors[0]

    def test_state_expect_typo_caught(self) -> None:
        raw = {
            "name": "x",
            "description": "x",
            "turns": [
                {
                    "message": "hi",
                    "state_expect": {"file_exists": ["a.txt"]},
                }
            ],
        }
        errors = validate_strict(raw, Scenario, source="s.json")
        assert len(errors) == 1
        assert "turns[0].state_expect.file_exists" in errors[0]
        assert "Did you mean 'files_exist'" in errors[0]

    def test_multiple_turns_each_reports_own_path(self) -> None:
        raw = {
            "name": "x",
            "description": "x",
            "turns": [
                {"message": "a", "expect": {"tools_invoke": ["x"]}},
                {"message": "b", "expect": {"tool_invoke": ["y"]}},
            ],
        }
        errors = validate_strict(raw, Scenario, source="s.json")
        assert len(errors) == 2
        assert any("turns[0].expect.tools_invoke" in e for e in errors)
        assert any("turns[1].expect.tool_invoke" in e for e in errors)

    def test_multiple_unknowns_each_get_their_own_message(self) -> None:
        raw = {
            "name": "x",
            "description": "x",
            "turns": [{"message": "hi"}],
            "tag": "x",
            "descriptions": "x",
        }
        errors = validate_strict(raw, Scenario, source="s.json")
        # Order is dict insertion order; no flakiness because we built it.
        assert len(errors) == 2

    def test_resource_unknown_key_caught(self) -> None:
        # GroupConfig.resources is list[Resource]; recursion must hop
        # into each element.
        raw = {
            "agent": "x",
            "resources": [
                {"kind": "file", "source": "a", "dest": "b", "verison": "1.0"},
            ],
        }
        errors = validate_strict(raw, GroupConfig, source="cfg.json")
        assert len(errors) == 1
        assert "resources[0].verison" in errors[0]
        assert "Did you mean 'version'" in errors[0]


# ── validate_strict: GroupConfig (extra="ignore") ──


class TestValidateStrictGroupConfig:
    def test_unknown_key_caught_even_though_pydantic_would_drop_it(self) -> None:
        raw = {"agent": "claude-code", "agnet": "claude-code"}
        errors = validate_strict(raw, GroupConfig, source="cfg.json")
        assert errors
        assert "agnet" in errors[0]
        assert "Did you mean 'agent'" in errors[0]

    def test_known_keys_pass(self) -> None:
        raw = {
            "agent": "claude-code",
            "default_tags": ["real-runnable"],
            "working_dir": ".",
            "workspace_isolation": "git-worktree",
        }
        assert validate_strict(raw, GroupConfig, source="cfg.json") == []


# ── Plugin registration ──


class TestRegisterPluginScenarioKey:
    def test_registered_key_passes_validator(self) -> None:
        register_plugin_scenario_key(TurnExpectation, "max_handoffs")
        raw = {
            "name": "x",
            "description": "x",
            "turns": [{"message": "hi", "expect": {"max_handoffs": 3}}],
        }
        assert validate_strict(raw, Scenario, source="s.json") == []

    def test_registered_key_appears_in_did_you_mean(self) -> None:
        register_plugin_scenario_key(TurnExpectation, "max_handoffs")
        raw = {
            "name": "x",
            "description": "x",
            "turns": [{"message": "hi", "expect": {"max_handoff": 3}}],
        }
        errors = validate_strict(raw, Scenario, source="s.json")
        assert len(errors) == 1
        assert "Did you mean 'max_handoffs'" in errors[0]

    def test_registration_is_idempotent(self) -> None:
        register_plugin_scenario_key(TurnExpectation, "max_handoffs")
        register_plugin_scenario_key(TurnExpectation, "max_handoffs")
        assert registered_plugin_scenario_keys(TurnExpectation) == {"max_handoffs"}

    def test_per_model_isolation(self) -> None:
        # A key registered on TurnExpectation must NOT leak to GroupConfig.
        register_plugin_scenario_key(TurnExpectation, "my_key")
        assert "my_key" in registered_plugin_scenario_keys(TurnExpectation)
        assert "my_key" not in registered_plugin_scenario_keys(GroupConfig)
        raw = {"agent": "x", "my_key": "v"}
        errors = validate_strict(raw, GroupConfig, source="cfg.json")
        assert any("my_key" in e for e in errors)

    def test_reserved_key_rejected(self) -> None:
        # Plugins cannot shadow framework names.
        for reserved in ("name", "description", "tags", "turns", "agent", "schema_version"):
            with pytest.raises(ValueError, match="reserved"):
                register_plugin_scenario_key(TurnExpectation, reserved)

    @pytest.mark.parametrize(
        "bad_key",
        [
            "",  # empty
            "MyKey",  # uppercase
            "1key",  # leading digit
            "key with space",  # space
            "key/path",  # slash
            "key!",  # punctuation
            "a" * 65,  # over the length cap
        ],
    )
    def test_bad_shape_rejected(self, bad_key: str) -> None:
        with pytest.raises(ValueError, match="unsupported shape"):
            register_plugin_scenario_key(TurnExpectation, bad_key)

    @pytest.mark.parametrize(
        "good_key",
        [
            "my_key",
            "my-key",
            "myplugin.feature",
            "abc_123",
            "a",
        ],
    )
    def test_good_shape_accepted(self, good_key: str) -> None:
        register_plugin_scenario_key(TurnExpectation, good_key)
        assert good_key in registered_plugin_scenario_keys(TurnExpectation)

    def test_non_basemodel_class_rejected(self) -> None:
        class NotAModel:
            pass

        with pytest.raises(TypeError, match="BaseModel"):
            register_plugin_scenario_key(NotAModel, "x")  # type: ignore[arg-type]

    def test_non_string_key_rejected(self) -> None:
        with pytest.raises(TypeError, match="key must be str"):
            register_plugin_scenario_key(TurnExpectation, 42)  # type: ignore[arg-type]


# ── ScenarioLoader plumbing ──


class TestScenarioLoaderStrict:
    def _write(self, tmp_path: Path, name: str, data: dict) -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(data))
        return p

    def test_load_scenario_default_off_keeps_legacy_behaviour(self, tmp_path: Path) -> None:
        # The same typo that strict mode rejects must load fine in
        # default mode - no behaviour change for existing users.
        p = self._write(
            tmp_path,
            "s.json",
            {
                "name": "x",
                "description": "x",
                "turns": [{"message": "hi", "expect": {"tools_invoke": ["x"]}}],
            },
        )
        # No exception.
        ScenarioLoader.load_scenario(p)

    def test_load_scenario_strict_rejects_unknown_key(self, tmp_path: Path) -> None:
        p = self._write(
            tmp_path,
            "s.json",
            {
                "name": "x",
                "description": "x",
                "turns": [{"message": "hi", "expect": {"tools_invoke": ["x"]}}],
            },
        )
        with pytest.raises(StrictConfigError) as exc:
            ScenarioLoader.load_scenario(p, strict_config=True)
        assert "tools_invoke" in str(exc.value)
        assert "Did you mean 'tools_invoked'" in str(exc.value)

    def test_load_group_config_strict_rejects_typo(self, tmp_path: Path) -> None:
        cfg = tmp_path / "_config.json"
        cfg.write_text(json.dumps({"agent": "claude-code", "agnet": "claude-code"}))
        with pytest.raises(StrictConfigError) as exc:
            ScenarioLoader.load_group_config(tmp_path, strict_config=True)
        assert "agnet" in str(exc.value)

    def test_load_group_scenarios_collects_strict_errors(self, tmp_path: Path) -> None:
        # ``_config.json`` is required to look like a group; create it.
        (tmp_path / "_config.json").write_text(json.dumps({"agent": "x"}))
        self._write(
            tmp_path,
            "good.json",
            {"name": "good", "description": "g", "turns": [{"message": "hi"}]},
        )
        self._write(
            tmp_path,
            "bad.json",
            {
                "name": "bad",
                "description": "b",
                "turns": [{"message": "hi", "expect": {"tools_invoke": ["x"]}}],
            },
        )
        scenarios, errors = ScenarioLoader.load_group_scenarios(tmp_path, strict_config=True)
        # Good one loaded, bad one rejected.
        assert [s.name for s in scenarios] == ["good"]
        assert len(errors) == 1
        assert "bad.json" in errors[0]
        assert "tools_invoke" in errors[0]

    def test_strict_includes_full_json_path_for_nested_typo(self, tmp_path: Path) -> None:
        p = self._write(
            tmp_path,
            "s.json",
            {
                "name": "x",
                "description": "x",
                "turns": [
                    {"message": "a"},
                    {"message": "b", "expect": {"tool_invoke": ["x"]}},
                ],
            },
        )
        with pytest.raises(StrictConfigError) as exc:
            ScenarioLoader.load_scenario(p, strict_config=True)
        # The qualified path tells the author exactly which turn to fix.
        assert "turns[1].expect.tool_invoke" in str(exc.value)


# ── Public API exposure ──


class TestPublicAPI:
    def test_register_plugin_scenario_key_importable_from_top_level(self) -> None:
        # Plugins MUST be able to register without reaching into private
        # modules. The top-level package must re-export the public name.
        import belt

        assert hasattr(belt, "register_plugin_scenario_key")
        assert hasattr(belt, "registered_plugin_scenario_keys")
        # Functional smoke test: a top-level import works end-to-end.
        belt.register_plugin_scenario_key(TurnExpectation, "smoke_key")
        try:
            assert "smoke_key" in belt.registered_plugin_scenario_keys(TurnExpectation)
        finally:
            from belt.parser.strict import _PLUGIN_KEYS

            _PLUGIN_KEYS.get(TurnExpectation, set()).discard("smoke_key")


# ── Submodel resolution edge cases ──


class TestSubmodelResolution:
    """The recursive walker has to navigate Optional/list/dict containers.

    This class pins the corner cases so a future refactor of
    ``_resolve_submodel`` doesn't silently stop recursing.
    """

    def test_optional_submodel_recurses(self) -> None:
        # Turn.state_expect: Optional[StateExpectation] today is just
        # StateExpectation with a default_factory, but we still want to
        # be sure recursion hops into it.
        raw = {
            "name": "x",
            "description": "x",
            "turns": [{"message": "hi", "state_expect": {"bogus": True}}],
        }
        errors = validate_strict(raw, Scenario, source="s.json")
        assert any("turns[0].state_expect.bogus" in e for e in errors)

    def test_list_of_submodels_recurses(self) -> None:
        raw = {
            "agent": "x",
            "resources": [
                {"kind": "file", "source": "a", "dest": "b"},  # ok
                {"kind": "file", "source": "a", "dest": "b", "extra": 1},  # bad
            ],
        }
        errors = validate_strict(raw, GroupConfig, source="cfg.json")
        assert len(errors) == 1
        assert "resources[1].extra" in errors[0]

    def test_resource_validates_directly(self) -> None:
        raw = {"kind": "file", "source": "a", "dest": "b", "wrongkey": 1}
        errors = validate_strict(raw, Resource, source="r.json")
        assert errors and "wrongkey" in errors[0]

    def test_state_expectation_validates_directly(self) -> None:
        raw = {"files_exist": ["x"], "bogus": True}
        errors = validate_strict(raw, StateExpectation, source="s.json")
        assert errors and "bogus" in errors[0]

    def test_turn_validates_directly(self) -> None:
        raw = {"message": "hi", "flgs": []}
        errors = validate_strict(raw, Turn, source="t.json")
        assert errors and "flgs" in errors[0]
        assert "flags" in errors[0]
