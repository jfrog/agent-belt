# (c) JFrog Ltd. (2026)

"""Tests for ``runner.phases.setup_groups`` - the workspace-isolation
opt-in gate that refuses ``workspace_isolation: "none"`` unless the
user explicitly opts in via ``--allow-inplace`` or ``BELT_ALLOW_INPLACE=1``.

The gate is the *runtime* half of the two-layer guardrail; the *schema*
half (Literal-validated enum on ``GroupConfig.workspace_isolation``) is
covered by ``tests/parser/test_strict.py``. Together they ensure the
only path to disabled isolation is the exact string ``"none"`` plus a
conscious opt-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from belt import envvars
from belt.runner.context import MatchedGroup, RunContext
from belt.runner.phases import setup_groups as setup_groups_module
from belt.runner.phases.setup_groups import (
    _check_external_working_dir_gate,
    _check_inplace_gate,
    _external_working_dir_message,
    setup_groups,
)
from belt.scenario import GroupConfig


def _make_ctx(
    *,
    workspace_isolation: str = "git-worktree",
    allow_inplace: bool = False,
    n_groups: int = 1,
) -> RunContext:
    """Minimal RunContext with one or more matched groups using a given isolation mode."""
    matched: list[MatchedGroup] = []
    for i in range(n_groups):
        gc = GroupConfig(agent="claude-code", workspace_isolation=workspace_isolation)
        matched.append(
            MatchedGroup(
                group_dir=Path(f"/tmp/g{i}"),
                config=gc,
                scenarios=[],
                name=f"g{i}",
            )
        )
    args = MagicMock()
    args.allow_inplace = allow_inplace
    progress = MagicMock()
    return RunContext(
        args=args,
        scenarios_root=Path("/tmp"),
        matched_groups=matched,
        agent_args={},
        outcomes_root=Path("/tmp/out"),
        run_dir=Path("/tmp/out/run"),
        workspace=Path("/tmp"),
        progress=progress,
    )


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test runs against a clean ``BELT_ALLOW_INPLACE`` env."""
    monkeypatch.delenv(envvars.ALLOW_INPLACE, raising=False)


class TestInplaceGateDefaultDeny:
    """Without a flag or env var, ``workspace_isolation: "none"`` is rejected."""

    def test_rejects_none_group(self) -> None:
        ctx = _make_ctx(workspace_isolation="none")
        _check_inplace_gate(ctx)
        assert "g0" in ctx.failed_groups

    def test_message_names_both_opt_ins(self) -> None:
        # The error must surface both opt-in routes (CLI + env). Users
        # discovering the gate at run time should not have to grep docs
        # to find the env equivalent.
        ctx = _make_ctx(workspace_isolation="none")
        _check_inplace_gate(ctx)
        printed = "".join(call.args[0] for call in ctx.progress.console.print.call_args_list)
        assert "--allow-inplace" in printed
        assert envvars.ALLOW_INPLACE in printed
        # No literal markdown-style backticks - the convention in
        # ``setup_groups._external_working_dir_message`` is plain
        # quoted strings, kept consistent here.
        assert "``" not in printed

    def test_rejects_every_none_group(self) -> None:
        # All matched groups with ``"none"`` are rejected; the gate is
        # not a single-group bail-out.
        ctx = _make_ctx(workspace_isolation="none", n_groups=3)
        _check_inplace_gate(ctx)
        assert ctx.failed_groups == {"g0", "g1", "g2"}


class TestInplaceGateOptIns:
    """Either the CLI flag or the env var permits ``workspace_isolation: "none"``."""

    def test_cli_flag_permits(self) -> None:
        ctx = _make_ctx(workspace_isolation="none", allow_inplace=True)
        _check_inplace_gate(ctx)
        assert ctx.failed_groups == set()
        # Opt-in is granted but the disabled-isolation status is surfaced
        # per group so it appears in stdout AND ``eval.log``. Asserting
        # the call here documents that the warning is intentional.
        printed = "".join(call.args[0] for call in ctx.progress.console.print.call_args_list)
        assert "g0" in printed
        assert "workspace_isolation: 'none'" in printed
        assert "[yellow]" in printed

    def test_env_var_permits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envvars.ALLOW_INPLACE, "1")
        ctx = _make_ctx(workspace_isolation="none", allow_inplace=False)
        _check_inplace_gate(ctx)
        assert ctx.failed_groups == set()

    def test_env_var_truthy_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mirrors the documented truthy set ("1", "true", "yes") via
        # envvars.is_truthy.
        monkeypatch.setenv(envvars.ALLOW_INPLACE, "true")
        ctx = _make_ctx(workspace_isolation="none")
        _check_inplace_gate(ctx)
        assert ctx.failed_groups == set()

    def test_env_var_falsy_zero_does_not_permit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``"0"`` is falsy in is_truthy; gate must still fire.
        monkeypatch.setenv(envvars.ALLOW_INPLACE, "0")
        ctx = _make_ctx(workspace_isolation="none")
        _check_inplace_gate(ctx)
        assert "g0" in ctx.failed_groups


class TestInplaceGateOptInWarning:
    """When opt-in is active, every ``"none"`` group surfaces a warning.

    Without the warning, a downstream reader of ``eval.log`` could not
    tell that isolation was off for a given group - the load-time gate
    fires only on rejection. Writing the warning per group also makes
    the artifact grep-friendly: a CI gate can refuse runs whose log
    contains ``workspace_isolation: 'none' is active``.
    """

    def test_warning_fires_per_none_group_with_cli_flag(self) -> None:
        ctx = _make_ctx(workspace_isolation="none", allow_inplace=True, n_groups=2)
        _check_inplace_gate(ctx)
        printed = "".join(call.args[0] for call in ctx.progress.console.print.call_args_list)
        assert "g0" in printed
        assert "g1" in printed
        # Once per group, not once total.
        assert printed.count("workspace_isolation: 'none' is active") == 2

    def test_warning_fires_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envvars.ALLOW_INPLACE, "1")
        ctx = _make_ctx(workspace_isolation="none", allow_inplace=False)
        _check_inplace_gate(ctx)
        printed = "".join(call.args[0] for call in ctx.progress.console.print.call_args_list)
        assert "is active (--allow-inplace)" in printed

    def test_warning_only_for_none_groups_in_mixed_run(self) -> None:
        gc_worktree = GroupConfig(agent="claude-code", workspace_isolation="git-worktree")
        gc_none = GroupConfig(agent="claude-code", workspace_isolation="none")
        matched = [
            MatchedGroup(group_dir=Path("/tmp/keep"), config=gc_worktree, scenarios=[], name="keep"),
            MatchedGroup(group_dir=Path("/tmp/inplace"), config=gc_none, scenarios=[], name="inplace"),
        ]
        args = MagicMock()
        args.allow_inplace = True
        ctx = RunContext(
            args=args,
            scenarios_root=Path("/tmp"),
            matched_groups=matched,
            agent_args={},
            outcomes_root=Path("/tmp/out"),
            run_dir=Path("/tmp/out/run"),
            workspace=Path("/tmp"),
            progress=MagicMock(),
        )
        _check_inplace_gate(ctx)
        printed = "".join(call.args[0] for call in ctx.progress.console.print.call_args_list)
        assert "inplace" in printed
        assert "keep" not in printed


class TestInplaceGateNoOps:
    """Default ``"git-worktree"`` is never gated, regardless of opt-ins."""

    def test_git_worktree_default_passes(self) -> None:
        ctx = _make_ctx(workspace_isolation="git-worktree")
        _check_inplace_gate(ctx)
        assert ctx.failed_groups == set()
        ctx.progress.console.print.assert_not_called()

    def test_flag_without_none_groups_is_no_op(self) -> None:
        # ``--allow-inplace`` against a fully ``git-worktree`` scenarios
        # root must not fail anything and must not noisily log.
        ctx = _make_ctx(workspace_isolation="git-worktree", allow_inplace=True)
        _check_inplace_gate(ctx)
        assert ctx.failed_groups == set()
        ctx.progress.console.print.assert_not_called()

    def test_mixed_groups_only_none_is_gated(self) -> None:
        # In a mixed scenarios root, the gate fires per-group: a
        # ``"none"`` group fails, a ``"git-worktree"`` sibling does not.
        gc_worktree = GroupConfig(agent="claude-code", workspace_isolation="git-worktree")
        gc_none = GroupConfig(agent="claude-code", workspace_isolation="none")
        matched = [
            MatchedGroup(group_dir=Path("/tmp/keep"), config=gc_worktree, scenarios=[], name="keep"),
            MatchedGroup(group_dir=Path("/tmp/drop"), config=gc_none, scenarios=[], name="drop"),
        ]
        args = MagicMock()
        args.allow_inplace = False
        ctx = RunContext(
            args=args,
            scenarios_root=Path("/tmp"),
            matched_groups=matched,
            agent_args={},
            outcomes_root=Path("/tmp/out"),
            run_dir=Path("/tmp/out/run"),
            workspace=Path("/tmp"),
            progress=MagicMock(),
        )
        _check_inplace_gate(ctx)
        assert ctx.failed_groups == {"drop"}


class TestInplaceGateMissingArgsAttr:
    """Commands that don't declare ``--allow-inplace`` (e.g. score) must
    not crash the gate."""

    def test_missing_attribute_treated_as_false(self) -> None:
        # ``args`` is a plain object with no ``allow_inplace`` attribute;
        # the gate must still gate.
        gc = GroupConfig(agent="claude-code", workspace_isolation="none")
        mg = MatchedGroup(group_dir=Path("/tmp/g0"), config=gc, scenarios=[], name="g0")

        class _ArgsNoAttr:
            pass

        ctx = RunContext(
            args=_ArgsNoAttr(),  # type: ignore[arg-type]
            scenarios_root=Path("/tmp"),
            matched_groups=[mg],
            agent_args={},
            outcomes_root=Path("/tmp/out"),
            run_dir=Path("/tmp/out/run"),
            workspace=Path("/tmp"),
            progress=MagicMock(),
        )
        # No raise expected; gate fires.
        _check_inplace_gate(ctx)
        assert "g0" in ctx.failed_groups


class TestInplaceGateIsWiredIntoSetupGroups:
    """Pin that ``setup_groups()`` actually calls ``_check_inplace_gate``.

    The unit tests above verify the helper's behaviour in isolation; this
    test pins the orchestrator-side invariant that ``setup_groups``
    invokes the helper before any other phase work, catching accidental
    deletion of the call site during refactors.
    """

    def test_setup_groups_calls_check_inplace_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Replace ``_check_inplace_gate`` with a sentinel-raising stub.
        # Reaching the raise proves the orchestrator actually invokes the
        # helper; the exception type is unique so we can distinguish it
        # from any incidental failure further along the call chain.
        class _GateCalled(Exception):
            pass

        def fake_check(ctx: RunContext) -> None:
            raise _GateCalled

        monkeypatch.setattr(setup_groups_module, "_check_inplace_gate", fake_check)

        gc = GroupConfig(agent="claude-code")
        ctx = RunContext(
            args=MagicMock(allow_inplace=False),
            scenarios_root=Path("/tmp"),
            matched_groups=[
                MatchedGroup(group_dir=Path("/tmp/g0"), config=gc, scenarios=[], name="g0"),
            ],
            agent_args={},
            outcomes_root=Path("/tmp/out"),
            run_dir=Path("/tmp/out/run"),
            workspace=Path("/tmp"),
            progress=MagicMock(),
        )
        with pytest.raises(_GateCalled):
            setup_groups(ctx)


def _capture_call_arg(mock_call: Any) -> str:
    """Helper: extract the first positional arg from a Mock call.

    Matches the call shape in ``setup_groups._check_inplace_gate`` which
    calls ``ctx.progress.console.print(message)`` with one string arg.
    """
    return mock_call.args[0]


# ── External-working-dir gate ─────────────────────────────────────────────
#
# This gate replaces the per-scenario raise that lived in
# ``runner.phases.run_scenarios``. The check is rooted on
# ``GroupConfig.working_dir`` (a group-level field by definition), so the
# correct invariant is one-banner-per-misconfigured-group, not one-per-
# scenario. Tests below pin that invariant plus both opt-in paths.


def _make_ext_ctx(
    *,
    scenarios_root: Path,
    group_dir: Path,
    working_dir: str | None,
    fixture_repo: str | None = None,
    workspace_isolation: str = "git-worktree",
    allow_external: bool = False,
    n_scenarios: int = 1,
) -> RunContext:
    """RunContext for the external-working-dir gate.

    ``working_dir`` is left raw so callers can test both absolute paths
    (``/abs/external``) and relative paths (``../sibling``); the gate
    resolves against ``group_dir``.
    """
    gc = GroupConfig(
        agent="claude-code",
        workspace_isolation=workspace_isolation,
        working_dir=working_dir,
        fixture_repo=fixture_repo,
    )
    scenarios = [MagicMock(name=f"s{i}") for i in range(n_scenarios)]
    mg = MatchedGroup(group_dir=group_dir, config=gc, scenarios=scenarios, name="g0")
    args = MagicMock()
    args.allow_external_working_dir = allow_external
    args.allow_inplace = False
    return RunContext(
        args=args,
        scenarios_root=scenarios_root,
        matched_groups=[mg],
        agent_args={},
        outcomes_root=scenarios_root / "out",
        run_dir=scenarios_root / "out" / "run",
        workspace=scenarios_root,
        progress=MagicMock(),
    )


@pytest.fixture(autouse=True)
def _clear_external_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test runs against a clean ``BELT_ALLOW_EXTERNAL_WORKING_DIR``."""
    monkeypatch.delenv(envvars.ALLOW_EXTERNAL_WORKING_DIR, raising=False)


class TestExternalWorkingDirGateDefaultDeny:
    """Without a flag or env var, a working_dir outside the scenarios root
    is rejected at the group level - not once per scenario.
    """

    def test_rejects_external_working_dir(self, tmp_path: Path) -> None:
        scenarios_root = tmp_path / "scenarios"
        scenarios_root.mkdir()
        (tmp_path / "external").mkdir()
        ctx = _make_ext_ctx(
            scenarios_root=scenarios_root,
            group_dir=scenarios_root / "g",
            working_dir=str(tmp_path / "external"),
        )
        _check_external_working_dir_gate(ctx)
        assert "g0" in ctx.failed_groups

    def test_one_banner_per_misconfigured_group(self, tmp_path: Path) -> None:
        # The whole point of moving the check here: ten scenarios in one
        # misconfigured group must produce one banner, not ten copies of
        # the same WorkspaceError.
        scenarios_root = tmp_path / "scenarios"
        scenarios_root.mkdir()
        (tmp_path / "external").mkdir()
        ctx = _make_ext_ctx(
            scenarios_root=scenarios_root,
            group_dir=scenarios_root / "g",
            working_dir=str(tmp_path / "external"),
            n_scenarios=10,
        )
        _check_external_working_dir_gate(ctx)
        assert ctx.progress.console.print.call_count == 1

    def test_relative_working_dir_inside_root_passes(self, tmp_path: Path) -> None:
        scenarios_root = tmp_path / "scenarios"
        (scenarios_root / "g" / "wd").mkdir(parents=True)
        ctx = _make_ext_ctx(
            scenarios_root=scenarios_root,
            group_dir=scenarios_root / "g",
            working_dir="wd",
        )
        _check_external_working_dir_gate(ctx)
        assert ctx.failed_groups == set()

    def test_message_names_both_escape_hatches(self, tmp_path: Path) -> None:
        msg = _external_working_dir_message(tmp_path / "external", tmp_path / "scenarios")
        assert "--allow-external-working-dir" in msg
        assert "--scenarios" in msg

    def test_message_includes_paths(self) -> None:
        msg = _external_working_dir_message(Path("/abs/external"), Path("/abs/scenarios"))
        assert "/abs/external" in msg
        assert "/abs/scenarios" in msg

    def test_message_explains_security_intent(self) -> None:
        msg = _external_working_dir_message(Path("/x"), Path("/y"))
        assert "security guardrail" in msg or "auto-init" in msg.lower()


class TestExternalWorkingDirGateOptIns:
    """Either ``--allow-external-working-dir`` or the env var permits the
    group through; a single ``WARNING`` is logged so ``eval.log`` records
    the relaxation."""

    def test_cli_flag_permits(self, tmp_path: Path) -> None:
        scenarios_root = tmp_path / "scenarios"
        scenarios_root.mkdir()
        (tmp_path / "external").mkdir()
        ctx = _make_ext_ctx(
            scenarios_root=scenarios_root,
            group_dir=scenarios_root / "g",
            working_dir=str(tmp_path / "external"),
            allow_external=True,
        )
        _check_external_working_dir_gate(ctx)
        assert ctx.failed_groups == set()

    def test_env_var_permits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envvars.ALLOW_EXTERNAL_WORKING_DIR, "1")
        scenarios_root = tmp_path / "scenarios"
        scenarios_root.mkdir()
        (tmp_path / "external").mkdir()
        ctx = _make_ext_ctx(
            scenarios_root=scenarios_root,
            group_dir=scenarios_root / "g",
            working_dir=str(tmp_path / "external"),
            allow_external=False,
        )
        _check_external_working_dir_gate(ctx)
        assert ctx.failed_groups == set()


class TestExternalWorkingDirGateScope:
    """The gate only fires on groups it actually applies to: it ignores
    ``fixture_repo`` groups (their clone lives under ``<run_dir>``) and
    ``workspace_isolation="none"`` groups (the inplace gate is the
    relevant guardrail there)."""

    def test_fixture_repo_group_is_ignored(self, tmp_path: Path) -> None:
        scenarios_root = tmp_path / "scenarios"
        scenarios_root.mkdir()
        ctx = _make_ext_ctx(
            scenarios_root=scenarios_root,
            group_dir=scenarios_root / "g",
            working_dir=None,
            fixture_repo="https://example.com/repo.git",
        )
        _check_external_working_dir_gate(ctx)
        assert ctx.failed_groups == set()

    def test_inplace_group_is_ignored(self, tmp_path: Path) -> None:
        scenarios_root = tmp_path / "scenarios"
        scenarios_root.mkdir()
        (tmp_path / "external").mkdir()
        ctx = _make_ext_ctx(
            scenarios_root=scenarios_root,
            group_dir=scenarios_root / "g",
            working_dir=str(tmp_path / "external"),
            workspace_isolation="none",
        )
        _check_external_working_dir_gate(ctx)
        # ``--allow-inplace`` covers this case.
        assert ctx.failed_groups == set()


class TestExternalWorkingDirGateWiring:
    """Pin the orchestrator-side invariant that ``setup_groups`` invokes
    ``_check_external_working_dir_gate``. The unit tests above verify the
    helper in isolation; this catches accidental deletion of the call site.
    """

    def test_setup_groups_calls_check_external_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _GateCalled(Exception):
            pass

        def fake_check(ctx: RunContext) -> None:
            raise _GateCalled

        monkeypatch.setattr(setup_groups_module, "_check_external_working_dir_gate", fake_check)

        gc = GroupConfig(agent="claude-code")
        ctx = RunContext(
            args=MagicMock(allow_inplace=False, allow_external_working_dir=False),
            scenarios_root=Path("/tmp"),
            matched_groups=[MatchedGroup(group_dir=Path("/tmp/g0"), config=gc, scenarios=[], name="g0")],
            agent_args={},
            outcomes_root=Path("/tmp/out"),
            run_dir=Path("/tmp/out/run"),
            workspace=Path("/tmp"),
            progress=MagicMock(),
        )
        with pytest.raises(_GateCalled):
            setup_groups(ctx)
