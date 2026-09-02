# (c) JFrog Ltd. (2026)

"""Tests for the deterministic ``verify`` exec-test grader.

Covers the schema (``VerifySpec``), the scorer (``check_verify`` + rules
integration), the runner helper (``_run_verify_command``), and the setup
gate (``_check_verify_gate``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from belt.entities import TurnOutput, VerifyResult
from belt.scenario import GroupConfig, Scenario, Turn, VerifySpec
from belt.scorer.rules.scorer import RuleBasedScorer
from belt.scorer.rules.verify import check_verify, has_verify

# ── Schema ──────────────────────────────────────────────────────────────────


class TestVerifySpecSchema:
    def test_minimal_valid(self):
        spec = VerifySpec(cmd=["python", "-m", "pytest"])
        assert spec.exit_code == 0
        assert spec.output_contains == []
        assert spec.timeout > 0

    def test_rejects_empty_cmd(self):
        with pytest.raises(ValueError):
            VerifySpec(cmd=[])

    def test_rejects_empty_string_argv(self):
        with pytest.raises(ValueError):
            VerifySpec(cmd=["python", ""])

    def test_rejects_unknown_field(self):
        with pytest.raises(ValueError):
            VerifySpec(cmd=["x"], shell=True)  # type: ignore[call-arg]


# ── check_verify ────────────────────────────────────────────────────────────


class TestCheckVerify:
    def test_has_verify(self):
        assert has_verify(VerifySpec(cmd=["x"])) is True
        assert has_verify(None) is False

    def test_pass_exit_and_substring(self):
        spec = VerifySpec(cmd=["mytool", "--run"], exit_code=0, output_contains=["passed"])
        result = VerifyResult(exit_code=0, stdout="3 passed in 0.1s")
        checks = check_verify(spec, result, turn_idx=0)
        assert all(c.passed for c in checks)
        assert all(c.dimension == "verify" for c in checks)
        # Check names self-describe (no stutter) and carry the command.
        assert any(c.check == "exit_code==0 (mytool --run)" for c in checks)
        assert any(c.check == "stdout contains 'passed'" for c in checks)
        assert not any(c.check.startswith("verify ") for c in checks)

    def test_fail_on_exit_mismatch(self):
        spec = VerifySpec(cmd=["x"], exit_code=0)
        result = VerifyResult(exit_code=1, stdout="boom")
        checks = check_verify(spec, result, turn_idx=0)
        assert checks[0].passed is False
        assert "got exit_code 1" in checks[0].details

    def test_failure_details_include_sanitized_stdout_tail(self):
        spec = VerifySpec(cmd=["x"], exit_code=0)
        # Multi-line output with an embedded ANSI escape; the failure details
        # must surface the error tail as a single sanitized line (no ESC, no
        # newlines) so it is safe in a one-line table cell.
        result = VerifyResult(exit_code=1, stdout="setup\nFAILED tests/test_x.py::test_y\n\x1b[31mboom\x1b[0m")
        details = check_verify(spec, result, turn_idx=0)[0].details
        assert "got exit_code 1" in details
        assert "FAILED tests/test_x.py::test_y" in details
        assert "\x1b" not in details
        assert "\n" not in details

    def test_fail_on_missing_substring(self):
        spec = VerifySpec(cmd=["x"], output_contains=["passed"])
        result = VerifyResult(exit_code=0, stdout="0 selected")
        checks = check_verify(spec, result, turn_idx=0)
        substr_check = [c for c in checks if "contains" in c.check][0]
        assert substr_check.passed is False

    def test_skip_when_result_missing(self):
        spec = VerifySpec(cmd=["x"])
        checks = check_verify(spec, None, turn_idx=2)
        assert len(checks) == 1
        assert checks[0].passed is None  # tri-state skip, not a failure
        assert checks[0].turn_idx == 2

    def test_scenario_level_is_turn_less(self):
        spec = VerifySpec(cmd=["x"])
        checks = check_verify(spec, VerifyResult(exit_code=0), turn_idx=None)
        assert all(c.turn_idx is None for c in checks)


# ── Rules scorer integration ────────────────────────────────────────────────


class TestRulesScorerVerify:
    def test_per_turn_verify_emits_check(self):
        scenario = Scenario(
            name="s",
            description="d",
            turns=[Turn(message="m", verify=VerifySpec(cmd=["true"], output_contains=["ok"]))],
        )
        output = TurnOutput(raw_cli="", verify_result=VerifyResult(exit_code=0, stdout="ok"))
        result = RuleBasedScorer().score(scenario, [output])
        verify_checks = [c for c in result.data.checks if c.dimension == "verify"]
        assert verify_checks and all(c.passed for c in verify_checks)

    def test_per_scenario_verify_reads_final_turn(self):
        scenario = Scenario(
            name="s",
            description="d",
            turns=[Turn(message="a"), Turn(message="b")],
            verify=VerifySpec(cmd=["true"]),
        )
        outputs = [
            TurnOutput(raw_cli=""),
            TurnOutput(raw_cli="", scenario_verify_result=VerifyResult(exit_code=0)),
        ]
        result = RuleBasedScorer().score(scenario, outputs)
        verify_checks = [c for c in result.data.checks if c.dimension == "verify"]
        assert len(verify_checks) == 1
        assert verify_checks[0].turn_idx is None
        assert verify_checks[0].passed is True

    def test_per_scenario_verify_skipped_when_no_turns(self):
        scenario = Scenario(name="s", description="d", turns=[Turn(message="a")], verify=VerifySpec(cmd=["true"]))
        # No turn output captured a scenario_verify_result -> skipped, not failed.
        result = RuleBasedScorer().score(scenario, [TurnOutput(raw_cli="")])
        verify_checks = [c for c in result.data.checks if c.dimension == "verify"]
        assert verify_checks[0].passed is None


# ── Runner helper ───────────────────────────────────────────────────────────


class TestRunVerifyCommand:
    def test_success(self, tmp_path):
        from belt.runner.orchestrator import _run_verify_command
        from belt.runner.process.spawner import LocalSpawner

        spec = VerifySpec(cmd=["python", "-c", "print('hello-verify')"])
        result = _run_verify_command(spec, LocalSpawner(), tmp_path)
        assert result.exit_code == 0
        assert "hello-verify" in result.stdout
        assert result.cmd == spec.cmd  # self-describing artifact

    def test_nonzero_exit(self, tmp_path):
        from belt.runner.orchestrator import _run_verify_command
        from belt.runner.process.spawner import LocalSpawner

        spec = VerifySpec(cmd=["python", "-c", "import sys; sys.exit(3)"])
        result = _run_verify_command(spec, LocalSpawner(), tmp_path)
        assert result.exit_code == 3

    def test_timeout_is_killed(self, tmp_path):
        from belt.runner.orchestrator import _run_verify_command
        from belt.runner.process.spawner import LocalSpawner

        spec = VerifySpec(cmd=["python", "-c", "import time; time.sleep(30)"], timeout=1)
        result = _run_verify_command(spec, LocalSpawner(), tmp_path)
        assert result.exit_code == 124
        assert "timed out" in result.stdout

    def test_ansi_stripped_at_capture(self, tmp_path):
        from belt.runner.orchestrator import _run_verify_command
        from belt.runner.process.spawner import LocalSpawner

        # Command emits an ANSI colour sequence; it must be stripped before the
        # captured stdout is stored, so no escape can reach a future renderer.
        spec = VerifySpec(cmd=["python", "-c", r"print('\x1b[31mRED\x1b[0m done')"])
        result = _run_verify_command(spec, LocalSpawner(), tmp_path)
        assert result.exit_code == 0
        assert "\x1b" not in result.stdout
        assert "RED done" in result.stdout


# ── Setup gate ──────────────────────────────────────────────────────────────


def _fake_ctx(*, allow: bool, group_config: GroupConfig, scenario: Scenario):
    mg = SimpleNamespace(name="g", scenarios=[scenario], config=group_config)
    return SimpleNamespace(
        args=SimpleNamespace(allow_verify_exec=allow),
        matched_groups=[mg],
        failed_groups=set(),
        setup_errors={},
        progress=SimpleNamespace(console=SimpleNamespace(print=lambda *a, **k: None)),
    )


class TestVerifyGate:
    def _scn_with_verify(self) -> Scenario:
        return Scenario(name="s", description="d", turns=[Turn(message="m", verify=VerifySpec(cmd=["true"]))])

    def test_refused_when_gate_off(self, monkeypatch):
        from belt.runner.phases.setup_groups import _check_verify_gate

        monkeypatch.delenv("BELT_ALLOW_VERIFY_EXEC", raising=False)
        gc = GroupConfig(agent="cursor", working_dir="../x", workspace_isolation="git-worktree")
        ctx = _fake_ctx(allow=False, group_config=gc, scenario=self._scn_with_verify())
        _check_verify_gate(ctx)
        assert "g" in ctx.failed_groups
        assert "allow-verify-exec" in ctx.setup_errors["g"]

    def test_allowed_with_worktree(self, monkeypatch):
        from belt.runner.phases.setup_groups import _check_verify_gate

        monkeypatch.delenv("BELT_ALLOW_VERIFY_EXEC", raising=False)
        gc = GroupConfig(agent="cursor", working_dir="../x", workspace_isolation="git-worktree")
        ctx = _fake_ctx(allow=True, group_config=gc, scenario=self._scn_with_verify())
        _check_verify_gate(ctx)
        assert ctx.failed_groups == set()

    def test_refused_without_worktree(self, monkeypatch):
        from belt.runner.phases.setup_groups import _check_verify_gate

        monkeypatch.delenv("BELT_ALLOW_VERIFY_EXEC", raising=False)
        gc = GroupConfig(agent="cursor")  # no working_dir / fixture_repo
        ctx = _fake_ctx(allow=True, group_config=gc, scenario=self._scn_with_verify())
        _check_verify_gate(ctx)
        assert "g" in ctx.failed_groups
        assert "isolated worktree" in ctx.setup_errors["g"]

    def test_no_verify_is_untouched(self, monkeypatch):
        from belt.runner.phases.setup_groups import _check_verify_gate

        monkeypatch.delenv("BELT_ALLOW_VERIFY_EXEC", raising=False)
        gc = GroupConfig(agent="cursor", working_dir="../x", workspace_isolation="git-worktree")
        plain = Scenario(name="s", description="d", turns=[Turn(message="m")])
        ctx = _fake_ctx(allow=False, group_config=gc, scenario=plain)
        _check_verify_gate(ctx)
        assert ctx.failed_groups == set()
