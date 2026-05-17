# (c) JFrog Ltd. (2026)

"""Tests for ScenarioLoader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from belt.parser.scenario import ScenarioLoader
from belt.scenario import GroupConfig


def test_load_scenario_rejects_directory_in_files_not_modified(tmp_path: Path) -> None:
    # Directory-shaped paths would silently pass against the flat modified-files
    # list. The loader rejects them with a clear message that points authors at
    # specific file paths - fail fast at load time, never let a false-green sit
    # in CI.
    data = {
        "name": "no_src_changes",
        "description": "agent must not touch src/",
        "turns": [{"message": "noop", "expect": {"files_not_modified": ["src/"]}}],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    with pytest.raises(Exception) as exc_info:
        ScenarioLoader.load_scenario(p)
    msg = str(exc_info.value)
    assert "directory-shaped paths are not supported" in msg
    assert "src/" in msg


def test_load_scenario_rejects_directory_in_files_modified_any(tmp_path: Path) -> None:
    # Same trap exists on the modified_any/exact siblings - directory paths
    # never match a concrete file, so the assertion would always fail or pass
    # in misleading ways. Validate the loader rejects them too.
    data = {
        "name": "any_src",
        "description": "expects any change in src/",
        "turns": [{"message": "noop", "expect": {"files_modified_any": ["src/"]}}],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    with pytest.raises(Exception) as exc_info:
        ScenarioLoader.load_scenario(p)
    assert "directory-shaped paths are not supported" in str(exc_info.value)


def test_load_scenario_minimal(tmp_path: Path) -> None:
    data = {"name": "test", "description": "A test", "turns": [{"message": "hello"}]}
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data))
    s = ScenarioLoader.load_scenario(p)
    assert s.name == "test"
    assert len(s.turns) == 1
    assert s.turns[0].expect.no_errors is True
    assert s.turns[0].expect.has_reply is True


def test_load_scenario_with_expectations(tmp_path: Path) -> None:
    data = {
        "name": "test",
        "description": "Full",
        "tags": ["production", "v1"],
        "turns": [
            {
                "message": "do something",
                "flags": ["-rd", "approve"],
                "expect": {
                    "tools_invoked": ["my_tool"],
                    "has_reply": False,
                    "no_errors": True,
                    "contains": ["expected_text"],
                },
            }
        ],
    }
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data))
    s = ScenarioLoader.load_scenario(p)
    assert s.tags == ["production", "v1"]
    assert s.turns[0].flags == ["-rd", "approve"]
    assert s.turns[0].expect.tools_invoked == ["my_tool"]
    assert s.turns[0].expect.has_reply is False


def test_load_scenario_rejects_bad_reply_pattern_regex(tmp_path: Path) -> None:
    """Bad ``reply_pattern`` regex aborts at scenario-load, not at score-time.

    ``reply_pattern`` and ``tool_result_pattern`` share one fail-fast
    policy: every entry compiles through ``belt._regex_policy`` at
    parse time, so the runtime never sees a malformed pattern.
    """
    data = {
        "name": "bad_reply",
        "description": "exercises the policy gate",
        "turns": [{"message": "hi", "expect": {"reply_pattern": [r"[unclosed", r"good", r"(?P<x"]}}],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    with pytest.raises(Exception) as exc_info:
        ScenarioLoader.load_scenario(p)
    msg = str(exc_info.value)
    # All offending entries surface in one error report so authors fix
    # them in one pass instead of one-at-a-time.
    assert "[unclosed" in msg
    assert "(?P<x" in msg
    assert "invalid regex" in msg


def test_load_scenario_rejects_bad_tool_result_pattern_regex(tmp_path: Path) -> None:
    """``tool_result_pattern`` shares the fail-fast load-time gate.

    A typo in any entry surfaces as a Pydantic validation error before
    the scenario reaches the runner - never as a quietly-failing
    scorer assertion in CI.
    """
    data = {
        "name": "bad_tool",
        "description": "exercises the consolidated policy gate",
        "turns": [
            {
                "message": "hi",
                "expect": {
                    "tool_result_pattern": {
                        "Read": r"[unclosed",
                        "Write": r"^OK$",  # this one is fine
                    }
                },
            }
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    with pytest.raises(Exception) as exc_info:
        ScenarioLoader.load_scenario(p)
    msg = str(exc_info.value)
    assert "[unclosed" in msg
    assert "Read" in msg
    assert "invalid regex" in msg


def test_load_scenario_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not json")
    with pytest.raises(Exception):
        ScenarioLoader.load_scenario(p)


def test_load_scenario_missing_required(tmp_path: Path) -> None:
    p = tmp_path / "missing.json"
    p.write_text(json.dumps({"name": "test"}))
    with pytest.raises(Exception):
        ScenarioLoader.load_scenario(p)


def test_load_group_scenarios_skips_config(tmp_path: Path) -> None:
    good = {"name": "s1", "description": "ok", "turns": [{"message": "hi"}]}
    (tmp_path / "s1.json").write_text(json.dumps(good))
    (tmp_path / "_config.json").write_text(json.dumps({"agent": "test"}))
    scenarios, errors = ScenarioLoader.load_group_scenarios(tmp_path)
    assert len(scenarios) == 1
    assert errors == []


def test_load_scenario_with_llm_scorer_instruction(tmp_path: Path) -> None:
    data = {
        "name": "test",
        "description": "With instruction",
        "llm_scorer_instruction": "Focus on whether the agent respected type boundaries.",
        "turns": [{"message": "hello"}],
    }
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data))
    s = ScenarioLoader.load_scenario(p)
    assert s.llm_scorer_instruction == "Focus on whether the agent respected type boundaries."


def test_load_scenario_without_llm_scorer_instruction(tmp_path: Path) -> None:
    data = {"name": "test", "description": "No instruction", "turns": [{"message": "hello"}]}
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data))
    s = ScenarioLoader.load_scenario(p)
    assert s.llm_scorer_instruction == ""


# ── workspace_isolation: schema is a closed Literal ──
#
# These tests pin the *parse-time* half of the two-layer guardrail (the
# *runtime* half - the --allow-inplace gate - is covered by
# tests/runner/phases/test_setup_groups.py). Together they ensure the
# only path to disabled isolation is the exact string "none" *and* a
# conscious opt-in. A typo like "None" or "git-wortree" must fail at
# parse time, not silently fall through to "no isolation".


class TestWorkspaceIsolationLiteral:
    """``GroupConfig.workspace_isolation`` is a closed Literal: only
    ``"git-worktree"`` and ``"none"`` parse; everything else is a
    Pydantic ``ValidationError`` listing the valid options."""

    def test_default_value_is_git_worktree(self) -> None:
        gc = GroupConfig(agent="claude-code")
        assert gc.workspace_isolation == "git-worktree"

    def test_git_worktree_accepted(self) -> None:
        gc = GroupConfig(agent="claude-code", workspace_isolation="git-worktree")
        assert gc.workspace_isolation == "git-worktree"

    def test_none_accepted(self) -> None:
        # ``"none"`` is valid at the schema layer; the runtime gate
        # (--allow-inplace) decides whether to actually run such a group.
        gc = GroupConfig(agent="claude-code", workspace_isolation="none")
        assert gc.workspace_isolation == "none"

    def test_capitalised_typo_rejected(self) -> None:
        # A common-typo case the schema lock must reject loudly with a
        # message naming the valid options.
        with pytest.raises(Exception) as exc_info:
            GroupConfig(agent="claude-code", workspace_isolation="None")
        msg = str(exc_info.value)
        assert "git-worktree" in msg and "none" in msg

    def test_typo_rejected(self) -> None:
        with pytest.raises(Exception) as exc_info:
            GroupConfig(agent="claude-code", workspace_isolation="git-wortree")
        msg = str(exc_info.value)
        assert "git-worktree" in msg

    def test_invented_value_rejected(self) -> None:
        # Plausible-sounding values invented by an author also get caught.
        with pytest.raises(Exception):
            GroupConfig(agent="claude-code", workspace_isolation="off")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(Exception):
            GroupConfig(agent="claude-code", workspace_isolation="")

    def test_load_group_config_rejects_typo(self, tmp_path: Path) -> None:
        # The loader path must surface the same Pydantic error so a
        # misconfigured ``_config.json`` fails at scenario load time, not
        # at run time when half the workers may have already started.
        cfg = {"agent": "claude-code", "workspace_isolation": "Git-Worktree"}
        (tmp_path / "_config.json").write_text(json.dumps(cfg))
        with pytest.raises(Exception) as exc_info:
            ScenarioLoader.load_group_config(tmp_path)
        msg = str(exc_info.value)
        assert "git-worktree" in msg

    def test_load_group_config_accepts_none(self, tmp_path: Path) -> None:
        cfg = {"agent": "claude-code", "workspace_isolation": "none"}
        (tmp_path / "_config.json").write_text(json.dumps(cfg))
        gc = ScenarioLoader.load_group_config(tmp_path)
        assert gc.workspace_isolation == "none"


class TestSandboxProfileLiteral:
    """``GroupConfig.sandbox.provider`` is a closed Literal: only ``"host"``
    and ``"docker"`` parse; everything else is rejected at load time. The
    pin matches the workspace_isolation pattern -- a typo like ``"dckr"``
    must fail at parse time rather than silently degrade to ``"host"``.

    Wildcards in ``allowed_hosts`` and ``env_passthrough`` are rejected
    because both surfaces feed network / secret allow-lists; a wildcard
    is a foot-gun whose only legitimate use is "no policy", and we want
    that intent to be explicit.
    """

    def test_default_sandbox_is_host_and_empty(self) -> None:
        gc = GroupConfig(agent="claude-code")
        assert gc.sandbox.provider == "host"
        assert gc.sandbox.image is None
        assert gc.sandbox.allowed_hosts == []
        assert gc.sandbox.writable_paths == []
        assert gc.sandbox.env_passthrough == []
        # Default network policy must be ``open`` so existing scenarios
        # (which pre-date this field) keep working unchanged. A silent
        # default of ``none`` would break every LLM-using scenario.
        assert gc.sandbox.network_policy == "open"

    def test_network_policy_open_and_none_accepted(self) -> None:
        for policy in ("open", "none"):
            gc = GroupConfig(agent="claude-code", sandbox={"provider": "host", "network_policy": policy})
            assert gc.sandbox.network_policy == policy

    def test_network_policy_typo_rejected(self) -> None:
        # ``"closed"`` is the obvious wrong synonym; ``"off"`` is the
        # second. Closed Literal must reject both at parse time so a
        # scenario author who meant ``none`` doesn't silently get the
        # default ``open`` (which would be a security regression).
        for bad in ("closed", "off", "blocked", "disabled", ""):
            with pytest.raises(Exception) as exc_info:
                GroupConfig(
                    agent="claude-code",
                    sandbox={"provider": "host", "network_policy": bad},
                )
            msg = str(exc_info.value)
            assert (
                "open" in msg and "none" in msg
            ), f"diagnostic for network_policy={bad!r} must list valid options: {msg}"

    def test_network_policy_none_with_provider_host_is_accepted_at_parse_time(self) -> None:
        # The parser accepts the combination because an operator can toggle
        # the provider at runtime via ``--sandbox docker`` (or the
        # BELT_SANDBOX_PROVIDER env var) without re-authoring the scenario
        # _config.json -- coupling the two fields at parse time would
        # block that legitimate workflow. The runtime guarantee that the
        # combination cannot SILENTLY downgrade to "no isolation" lives in
        # the provider layer instead: ``HostSandboxProvider.validate_profile``
        # rejects this combination at scenario start, before any subprocess
        # spawns.
        gc = GroupConfig(
            agent="claude-code",
            sandbox={"provider": "host", "network_policy": "none"},
        )
        assert gc.sandbox.provider == "host"
        assert gc.sandbox.network_policy == "none"

    def test_provider_host_and_docker_accepted(self) -> None:
        for name in ("host", "docker"):
            gc = GroupConfig(agent="claude-code", sandbox={"provider": name})
            assert gc.sandbox.provider == name

    def test_provider_typo_rejected_with_valid_options(self) -> None:
        with pytest.raises(Exception) as exc_info:
            GroupConfig(agent="claude-code", sandbox={"provider": "dckr"})
        msg = str(exc_info.value)
        assert "host" in msg and "docker" in msg

    def test_provider_capitalised_typo_rejected(self) -> None:
        with pytest.raises(Exception):
            GroupConfig(agent="claude-code", sandbox={"provider": "Docker"})

    def test_legacy_local_name_rejected(self) -> None:
        # ``"local"`` was the v0 name for the host provider; pinning the
        # rename here makes any accidental revert (or stale config copied
        # from an old branch) fail loudly at parse time.
        with pytest.raises(Exception) as exc_info:
            GroupConfig(agent="claude-code", sandbox={"provider": "local"})
        msg = str(exc_info.value)
        assert "host" in msg and "docker" in msg

    def test_allowed_hosts_wildcard_rejected(self) -> None:
        with pytest.raises(Exception) as exc_info:
            GroupConfig(
                agent="claude-code",
                sandbox={"provider": "docker", "image": "x", "allowed_hosts": ["*.evil.com"]},
            )
        # Reason names the offending value so the author knows what to fix.
        assert "*" in str(exc_info.value) or "wildcard" in str(exc_info.value).lower()

    def test_env_passthrough_wildcard_rejected(self) -> None:
        with pytest.raises(Exception) as exc_info:
            GroupConfig(
                agent="claude-code",
                sandbox={"provider": "docker", "image": "x", "env_passthrough": ["AWS_*"]},
            )
        assert "*" in str(exc_info.value) or "wildcard" in str(exc_info.value).lower()

    def test_env_passthrough_shell_expansion_rejected(self) -> None:
        with pytest.raises(Exception):
            GroupConfig(
                agent="claude-code",
                sandbox={"provider": "docker", "image": "x", "env_passthrough": ["${HOME}"]},
            )

    def test_extra_keys_rejected(self) -> None:
        # ``SandboxProfile`` uses ``extra='forbid'`` so a typo like
        # ``allowed_host`` (singular) fails loudly instead of being
        # silently dropped by ``extra='ignore'``.
        with pytest.raises(Exception):
            GroupConfig(
                agent="claude-code",
                sandbox={"provider": "host", "allowed_host": ["api.x"]},
            )

    def test_load_group_config_rejects_provider_typo(self, tmp_path: Path) -> None:
        cfg = {"agent": "claude-code", "sandbox": {"provider": "dckr"}}
        (tmp_path / "_config.json").write_text(json.dumps(cfg))
        with pytest.raises(Exception) as exc_info:
            ScenarioLoader.load_group_config(tmp_path)
        msg = str(exc_info.value)
        assert "host" in msg and "docker" in msg

    def test_load_group_config_accepts_full_profile(self, tmp_path: Path) -> None:
        cfg = {
            "agent": "claude-code",
            "sandbox": {
                "provider": "docker",
                "image": "agent-belt-sandbox-cursor:dev",
                "network_policy": "open",
                "allowed_hosts": ["api.cursor.sh"],
                "writable_paths": ["/var/cache/agent"],
                "env_passthrough": ["CURSOR_API_KEY"],
            },
        }
        (tmp_path / "_config.json").write_text(json.dumps(cfg))
        gc = ScenarioLoader.load_group_config(tmp_path)
        assert gc.sandbox.provider == "docker"
        assert gc.sandbox.image == "agent-belt-sandbox-cursor:dev"
        assert gc.sandbox.network_policy == "open"
        assert gc.sandbox.allowed_hosts == ["api.cursor.sh"]
        assert gc.sandbox.writable_paths == ["/var/cache/agent"]
        assert gc.sandbox.env_passthrough == ["CURSOR_API_KEY"]

    def test_provider_empty_string_rejected(self) -> None:
        # Closing a literal hole: ``""`` is a foot-gun (parses as
        # falsy in many places, treated as None elsewhere). Must
        # reject loudly with the same diagnostic as a typo.
        with pytest.raises(Exception) as exc_info:
            GroupConfig(agent="claude-code", sandbox={"provider": ""})
        msg = str(exc_info.value)
        assert "host" in msg and "docker" in msg

    def test_provider_none_falls_back_to_default_host(self) -> None:
        # ``provider`` is optional; omitting the whole sandbox block
        # already defaults to host (covered above). Explicit
        # ``provider: null`` should be a parse error rather than
        # silently picking host -- "I forgot to set this" and
        # "I want the default" are different intents.
        with pytest.raises(Exception):
            GroupConfig(agent="claude-code", sandbox={"provider": None})

    def test_writable_paths_must_be_strings(self) -> None:
        # Type discipline: a list of paths is a list of strings.
        with pytest.raises(Exception):
            GroupConfig(
                agent="claude-code",
                sandbox={"provider": "docker", "image": "x", "writable_paths": [123]},
            )

    def test_allowed_hosts_must_be_strings(self) -> None:
        with pytest.raises(Exception):
            GroupConfig(
                agent="claude-code",
                sandbox={"provider": "docker", "image": "x", "allowed_hosts": [42]},
            )

    def test_env_passthrough_must_be_strings(self) -> None:
        with pytest.raises(Exception):
            GroupConfig(
                agent="claude-code",
                sandbox={"provider": "docker", "image": "x", "env_passthrough": [True]},
            )

    def test_image_can_be_explicit_null_when_provider_is_host(self) -> None:
        # ``image`` is only required when provider != host. The host
        # provider explicitly accepts ``image: null`` so a scenario
        # author can document "no isolation" without leaving readers
        # wondering whether image was forgotten.
        gc = GroupConfig(
            agent="claude-code",
            sandbox={"provider": "host", "image": None},
        )
        assert gc.sandbox.provider == "host"
        assert gc.sandbox.image is None


def test_load_group_scenarios_collects_errors(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("not json")
    scenarios, errors = ScenarioLoader.load_group_scenarios(tmp_path)
    assert scenarios == []
    assert len(errors) == 1
    assert "bad.json" in errors[0]


def test_load_scenario_sets_source_dir(tmp_path: Path) -> None:
    """The loader must stash the scenario JSON's parent on ``_source_dir`` so
    the LLM scorer can resolve ``llm_scorer_evidence_files`` against it."""
    data = {"name": "s", "description": "d", "turns": [{"message": "hi"}]}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    s = ScenarioLoader.load_scenario(p)
    assert s._source_dir == tmp_path.resolve()


def test_load_scenario_with_evidence_files_round_trips(tmp_path: Path) -> None:
    data = {
        "name": "s",
        "description": "d",
        "llm_scorer_evidence_files": ["rubric.md", "subdir/expected.json"],
        "turns": [{"message": "hi"}],
    }
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    s = ScenarioLoader.load_scenario(p)
    assert s.llm_scorer_evidence_files == ["rubric.md", "subdir/expected.json"]
    # PrivateAttrs must not leak into the public JSON dump.
    dumped = json.loads(s.model_dump_json())
    assert "_source_dir" not in dumped
    assert dumped["llm_scorer_evidence_files"] == ["rubric.md", "subdir/expected.json"]


def test_load_scenario_evidence_files_default_empty(tmp_path: Path) -> None:
    data = {"name": "s", "description": "d", "turns": [{"message": "hi"}]}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    s = ScenarioLoader.load_scenario(p)
    assert s.llm_scorer_evidence_files == []


def test_load_scenario_rejects_too_many_evidence_files(tmp_path: Path) -> None:
    data = {
        "name": "s",
        "description": "d",
        "llm_scorer_evidence_files": [f"f{i}.md" for i in range(21)],
        "turns": [{"message": "hi"}],
    }
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    with pytest.raises(Exception):
        ScenarioLoader.load_scenario(p)
