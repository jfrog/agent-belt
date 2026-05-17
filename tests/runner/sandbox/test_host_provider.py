# (c) JFrog Ltd. (2026)

"""HostSandboxProvider is the no-op default; the test pins that contract.

If HostSandboxProvider ever rewrites argv, cwd, or env, every existing
agent's behaviour shifts silently because every agent gets a LocalSpawner
by default. The pin here makes any such change land as a test failure
rather than a production regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from belt.runner.sandbox.base import SandboxConfigError, SandboxContext
from belt.runner.sandbox.host import HostSandboxProvider
from belt.scenario import SandboxProfile


def test_host_provider_name_is_host() -> None:
    assert HostSandboxProvider.name() == "host"


def test_host_wrap_returns_inputs_unchanged() -> None:
    provider = HostSandboxProvider()
    handle = provider.setup(SandboxProfile(), SandboxContext(workspace_dir=Path("/tmp/x")))
    cmd_in = ["agent", "-p", "say hi"]
    cwd_in = "/tmp/x"
    env_in = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-..."}

    cmd, cwd, env = provider.wrap(handle, cmd=cmd_in, cwd=cwd_in, env=env_in)

    # Pass-through: identity, not just equality. A future provider that
    # returned copies would still satisfy ``==`` but break callers that
    # rely on the no-op being literally a no-op.
    assert cmd is cmd_in
    assert cwd is cwd_in
    assert env is env_in


def test_host_teardown_is_idempotent() -> None:
    provider = HostSandboxProvider()
    handle = provider.setup(SandboxProfile(), SandboxContext(workspace_dir=Path("/tmp/x")))
    provider.teardown(handle)
    provider.teardown(handle)


def test_host_validate_accepts_default_profile() -> None:
    # The default profile is provider='host', network_policy='open', no
    # extra writable paths, no allowed hosts. The host provider can honour
    # all of those (they are the no-isolation defaults), so validation
    # passes silently. This pins the contract that authoring an unsandboxed
    # scenario is not gated behind a positive opt-in.
    provider = HostSandboxProvider()
    provider.validate_profile(SandboxProfile(), SandboxContext(workspace_dir=Path("/tmp/x"), scenario_name="s"))


def test_host_validate_accepts_explicit_open_network_policy() -> None:
    # Explicit ``network_policy='open'`` is the same as the default and is
    # accepted by the host provider -- the host has full network and the
    # profile is asking for full network. The check rejects only policies
    # the host cannot enforce.
    provider = HostSandboxProvider()
    profile = SandboxProfile(provider="host", network_policy="open")
    provider.validate_profile(profile, SandboxContext(workspace_dir=Path("/tmp/x"), scenario_name="s"))


def test_host_validate_rejects_network_policy_none() -> None:
    # The host provider has no mechanism to enforce a network namespace
    # (it spawns the agent as the invoking user on the host kernel with
    # the host network), so a profile asking for ``network_policy='none'``
    # under provider='host' raises SandboxConfigError at validation time.
    # This is the gate that prevents an unenforceable profile from
    # silently running with the host's open network.
    provider = HostSandboxProvider()
    profile = SandboxProfile(provider="host", network_policy="none")
    ctx = SandboxContext(workspace_dir=Path("/tmp/x"), scenario_name="my_scenario")
    with pytest.raises(SandboxConfigError) as exc:
        provider.validate_profile(profile, ctx)
    msg = str(exc.value)
    # Error must name the scenario (so multi-scenario runs surface which
    # one is broken), the offending value, the chosen provider, and the
    # actionable fix (--sandbox docker). Everything an operator needs to
    # resolve the misconfig should be in this single line.
    assert "my_scenario" in msg
    assert "'none'" in msg
    assert "'host'" in msg
    assert "--sandbox docker" in msg


def test_host_validate_runs_before_setup_in_call_order() -> None:
    # Document the contract that callers must call ``validate_profile``
    # before ``setup`` -- ``setup`` itself does not re-validate (cheap
    # behaviour, no I/O), so a caller that forgets to validate would
    # build a SandboxHandle for an unenforceable profile and then run
    # the agent with weaker isolation than the profile declared. The
    # runner orchestrator's _build_sandbox_for_scenario does call them
    # in the right order; this test pins that setup() will NOT
    # double-check, so the orchestrator's call order is the single
    # source of truth.
    provider = HostSandboxProvider()
    profile = SandboxProfile(provider="host", network_policy="none")
    ctx = SandboxContext(workspace_dir=Path("/tmp/x"), scenario_name="s")
    handle = provider.setup(profile, ctx)
    assert handle.profile is profile
    assert handle.context is ctx
