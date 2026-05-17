# (c) JFrog Ltd. (2026)

"""DockerSandboxProvider unit tests.

Pure-policy tests: no real ``docker`` invocation. The provider's ``wrap``
is the security-critical surface (it builds the ``docker run`` argv); the
tests pin the exact flags so a regression that drops ``--cap-drop=ALL``
or that forwards env values verbatim shows up as a failing assertion.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from belt.runner.sandbox.base import SandboxContext, SandboxHandle
from belt.runner.sandbox.docker import DockerSandboxError, DockerSandboxProvider
from belt.scenario import SandboxProfile


def _handle(
    profile: SandboxProfile | None = None,
    workspace: Path = Path("/tmp/work"),
    required_env: frozenset[str] = frozenset(),
) -> SandboxHandle:
    profile = profile or SandboxProfile(provider="docker", image="agent-belt-sandbox-cursor:dev")
    ctx = SandboxContext(
        workspace_dir=workspace,
        agent_required_env=required_env,
        scenario_name="s",
    )
    return SandboxHandle(profile=profile, context=ctx, state={})


def test_validate_rejects_missing_image() -> None:
    # ``image`` is required for the docker provider: no image, no
    # container, no isolation. The check runs in ``validate_profile``
    # (not ``setup``) so the error surfaces before any docker subprocess
    # spawns and works on hosts without docker installed -- the typed
    # config error must not require a working docker daemon to fire.
    from belt.runner.sandbox.base import SandboxConfigError

    provider = DockerSandboxProvider()
    profile = SandboxProfile(provider="docker", image=None)
    ctx = SandboxContext(workspace_dir=Path("/tmp/x"), scenario_name="my_scenario")
    with pytest.raises(SandboxConfigError) as exc:
        provider.validate_profile(profile, ctx)
    msg = str(exc.value)
    assert "my_scenario" in msg
    assert "image" in msg
    assert "docker" in msg


def test_validate_accepts_full_profile_without_docker_installed() -> None:
    # ``validate_profile`` is environment-independent: a host without the
    # docker binary on PATH must still be able to surface a clean config
    # error for a malformed profile (and accept a well-formed one). Only
    # ``setup`` requires the docker daemon.
    provider = DockerSandboxProvider()
    profile = SandboxProfile(provider="docker", image="img:tag", network_policy="none")
    ctx = SandboxContext(workspace_dir=Path("/tmp/x"), scenario_name="s")
    with patch("belt.runner.sandbox.docker._docker_available", return_value=False):
        provider.validate_profile(profile, ctx)


def test_setup_rejects_missing_docker_binary() -> None:
    provider = DockerSandboxProvider()
    profile = SandboxProfile(provider="docker", image="img:tag")
    ctx = SandboxContext(workspace_dir=Path("/tmp/x"), scenario_name="s")
    with patch("belt.runner.sandbox.docker._docker_available", return_value=False):
        with pytest.raises(DockerSandboxError, match="docker"):
            provider.setup(profile, ctx)


def test_wrap_applies_hardening_flags() -> None:
    provider = DockerSandboxProvider()
    cmd, cwd, env = provider.wrap(_handle(), cmd=["agent", "hi"], cwd="/tmp/work", env={"PATH": "/usr/bin"})

    # Cmd starts with `docker run` and the canonical hardening switches.
    assert cmd[:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--workdir=/work" in cmd
    # Worktree bind mount lands at the canonical /work path.
    assert "-v" in cmd
    bind_idx = cmd.index("-v")
    assert cmd[bind_idx + 1] == "/tmp/work:/work:rw"
    # Image is positioned just before the agent argv.
    image_idx = cmd.index("agent-belt-sandbox-cursor:dev")
    assert cmd[image_idx + 1 :] == ["agent", "hi"]
    # Host cwd is cleared so `docker run` runs in the invoker's cwd.
    assert cwd is None
    # Env is forwarded by reference -- docker reads the values from the
    # caller's process env, the framework never reads them itself.
    assert env == {"PATH": "/usr/bin"}


def test_wrap_forwards_env_passthrough_by_name_only() -> None:
    provider = DockerSandboxProvider()
    profile = SandboxProfile(
        provider="docker",
        image="img:tag",
        env_passthrough=["CURSOR_API_KEY", "WORKSPACE_TOKEN"],
    )
    handle = _handle(profile=profile, required_env=frozenset({"PATH", "HOME"}))
    cmd, _, _ = provider.wrap(
        handle,
        cmd=["agent"],
        cwd="/tmp/work",
        env={
            "PATH": "/usr/bin",
            "HOME": "/root",
            "CURSOR_API_KEY": "sk-leaky",
            "WORKSPACE_TOKEN": "tok",
            "BANNED_SECRET": "leak",
        },
    )

    # ``-e NAME`` (no value) reaches docker for the union of agent-required
    # and scenario-passthrough names. The framework also injects exactly
    # one ``-e HOME=<workdir>`` with a fixed value (so agents that write
    # under ``$HOME`` land inside the writable worktree mount instead of
    # the read-only rootfs); the value is framework-controlled, not
    # caller-supplied, so this is not an exfil channel.
    e_flag_pairs = [(cmd[i], cmd[i + 1]) for i, x in enumerate(cmd) if x == "-e"]
    by_name_only = sorted(v for _, v in e_flag_pairs if "=" not in v)
    by_name_with_value = sorted(v for _, v in e_flag_pairs if "=" in v)
    assert by_name_only == ["CURSOR_API_KEY", "HOME", "PATH", "WORKSPACE_TOKEN"]
    assert by_name_with_value == ["HOME=/home/agent"]
    # ``BANNED_SECRET`` was in env but not in the union -> never reaches docker.
    assert "BANNED_SECRET" not in cmd
    # Values never leak -- only names appear in the docker argv (apart
    # from the framework-fixed HOME injection above).
    assert "sk-leaky" not in cmd
    assert "leak" not in cmd


def test_wrap_emits_add_host_with_resolved_ip_for_each_allowed_host(monkeypatch) -> None:
    # ``--add-host name:0.0.0.0`` BLOCKS the host (resolves to loopback);
    # the provider must resolve each name on the host (where DNS works)
    # and pass the real IP so the in-container resolver can reach it
    # without an outbound DNS query. Pinning a literal ``0.0.0.0`` here
    # would silently BLOCK each named host (resolves to loopback) while
    # the bridge network's full outbound masks the bug for any name
    # also reachable via the default resolver.
    fake_ips = {"api.cursor.sh": "1.2.3.4", "api.openai.com": "5.6.7.8"}
    monkeypatch.setattr("belt.runner.sandbox.docker.socket.gethostbyname", lambda h: fake_ips[h])

    provider = DockerSandboxProvider()
    profile = SandboxProfile(
        provider="docker",
        image="img:tag",
        allowed_hosts=["api.cursor.sh", "api.openai.com"],
    )
    cmd, _, _ = provider.wrap(_handle(profile=profile), cmd=["agent"], cwd="/tmp/work", env={})

    add_host_indices = [i for i, x in enumerate(cmd) if x == "--add-host"]
    add_host_values = [cmd[i + 1] for i in add_host_indices]
    assert add_host_values == ["api.cursor.sh:1.2.3.4", "api.openai.com:5.6.7.8"]


def test_wrap_skips_unresolvable_allowed_host(monkeypatch, caplog) -> None:
    # An unresolvable allowed_host must not abort the run -- bridge
    # network outbound still works, the entry just isn't pre-pinned.
    # The skip is logged so operators can debug DNS issues.
    def _bad_resolve(host: str) -> str:
        raise OSError("nodename nor servname provided")

    monkeypatch.setattr("belt.runner.sandbox.docker.socket.gethostbyname", _bad_resolve)

    provider = DockerSandboxProvider()
    profile = SandboxProfile(
        provider="docker",
        image="img:tag",
        allowed_hosts=["nonexistent.invalid"],
    )
    cmd, _, _ = provider.wrap(_handle(profile=profile), cmd=["agent"], cwd="/tmp/work", env={})

    assert "--add-host" not in cmd


def test_wrap_strips_absolute_host_path_from_cmd0() -> None:
    # Regression: agent adapters resolve their CLI to an absolute host
    # path (e.g. /Users/x/.local/bin/cursor-agent). That path does not
    # exist inside the container, so docker exec failed with
    # "stat /Users/.../cursor-agent: no such file or directory".
    # The provider must strip cmd[0] to its basename and rely on the
    # container PATH (where the BYOI image installs the agent).
    provider = DockerSandboxProvider()
    cmd, _, _ = provider.wrap(
        _handle(),
        cmd=["/Users/anyone/.local/bin/cursor-agent", "-p", "hi"],
        cwd="/tmp/work",
        env={},
    )
    image_idx = cmd.index("agent-belt-sandbox-cursor:dev")
    # First argv element after the image is the in-container command.
    # Must be the bare basename, NOT the host absolute path.
    assert cmd[image_idx + 1 :] == ["cursor-agent", "-p", "hi"]


def test_wrap_leaves_relative_or_bare_cmd0_unchanged() -> None:
    # When the agent adapter passes a bare binary name (e.g. just
    # "claude"), the provider must NOT rewrite it -- there's nothing to
    # strip and the container PATH already resolves it.
    provider = DockerSandboxProvider()
    cmd, _, _ = provider.wrap(
        _handle(),
        cmd=["claude", "--print", "hi"],
        cwd="/tmp/work",
        env={},
    )
    image_idx = cmd.index("agent-belt-sandbox-cursor:dev")
    assert cmd[image_idx + 1 :] == ["claude", "--print", "hi"]


def test_wrap_mounts_extra_writable_paths() -> None:
    provider = DockerSandboxProvider()
    profile = SandboxProfile(
        provider="docker",
        image="img:tag",
        writable_paths=["/var/cache/agent"],
    )
    cmd, _, _ = provider.wrap(_handle(profile=profile), cmd=["agent"], cwd="/tmp/work", env={})

    # Worktree mount + each writable_paths entry, in that order.
    mounts = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-v"]
    assert mounts == ["/tmp/work:/work:rw", "/var/cache/agent:/var/cache/agent:rw"]


def test_wrap_image_is_immediately_followed_by_in_container_argv() -> None:
    # The relative position of the image vs the in-container command
    # is part of the docker CLI grammar: ``docker run [flags] IMAGE
    # [CMD ...]``. A regression that injects another flag between
    # ``IMAGE`` and ``CMD`` would make docker interpret CMD[0] as a
    # flag argument and fail with an opaque error. Pin the contract.
    provider = DockerSandboxProvider()
    cmd, _, _ = provider.wrap(_handle(), cmd=["agent", "--print", "hi"], cwd="/tmp/work", env={})
    image = "agent-belt-sandbox-cursor:dev"
    image_idx = cmd.index(image)
    assert cmd[image_idx + 1 :] == ["agent", "--print", "hi"]
    # Nothing after the image starts with "-" (it would be parsed as
    # a flag to the in-container process, not docker, but the
    # invariant is "everything after IMAGE is the agent's argv").
    # We check the position rather than the prefix because some agent
    # CLIs do start their argv with a flag (``cursor-agent -p ...``).


def test_wrap_home_injection_present_even_with_no_passthrough_env() -> None:
    # ``-e HOME=/work`` is unconditional: it does not depend on the
    # invoker's HOME being in the env or in passthrough. A scenario
    # with an empty env_passthrough still gets HOME pinned, so any
    # agent that touches ``$HOME`` lands in the writable mount
    # regardless of scenario configuration.
    provider = DockerSandboxProvider()
    cmd, _, _ = provider.wrap(_handle(), cmd=["agent"], cwd="/tmp/work", env={})
    assert "HOME=/home/agent" in cmd
    home_idx = cmd.index("HOME=/home/agent")
    assert cmd[home_idx - 1] == "-e", "HOME=/home/agent must be a -e value"
    # The HOME path MUST be backed by a tmpfs mount so the value is
    # actually writable (read-only rootfs blocks everything else).
    assert any(x.startswith("--tmpfs=/home/agent") for x in cmd), f"HOME path must have a tmpfs backing: {cmd}"


def test_wrap_passthrough_value_never_appears_in_argv() -> None:
    # Defence-in-depth: even if a future change accidentally switches
    # to ``-e NAME=value``, the asserted-value test below catches it.
    # We pick a sentinel that would only appear if the value (not just
    # the name) were forwarded.
    provider = DockerSandboxProvider()
    profile = SandboxProfile(
        provider="docker",
        image="img:tag",
        env_passthrough=["TOP_SECRET"],
    )
    cmd, _, _ = provider.wrap(
        _handle(profile=profile),
        cmd=["agent"],
        cwd="/tmp/work",
        env={"TOP_SECRET": "BELT-LEAK-CANARY"},
    )
    # The NAME appears (as a -e arg). The VALUE must not.
    assert "TOP_SECRET" in cmd
    assert "BELT-LEAK-CANARY" not in cmd
    assert not any("BELT-LEAK-CANARY" in token for token in cmd)


def test_wrap_required_env_dropped_when_not_in_invoker_env() -> None:
    # An agent declares it needs ``ANTHROPIC_API_KEY`` (required_env),
    # but the invoker did not export it. The provider must NOT emit
    # ``-e ANTHROPIC_API_KEY`` -- doing so would make docker error
    # with "env var not set" and mask the real "you forgot to export
    # your API key" diagnostic. The agent's own preflight is what
    # surfaces the missing-credential error to the user.
    provider = DockerSandboxProvider()
    handle = _handle(required_env=frozenset({"ANTHROPIC_API_KEY"}))
    cmd, _, _ = provider.wrap(handle, cmd=["agent"], cwd="/tmp/work", env={"PATH": "/usr/bin"})
    assert "ANTHROPIC_API_KEY" not in cmd


def test_wrap_argv_order_flags_before_image_argv_after(_handle_=_handle) -> None:
    # The full docker grammar contract: every ``--`` flag MUST appear
    # before the image; cmd[0]+ MUST appear after. A regression that
    # appended a flag at the end (e.g. trailing ``--rm``) would make
    # docker parse it as the in-container command's argv.
    provider = DockerSandboxProvider()
    cmd, _, _ = provider.wrap(_handle_(), cmd=["agent", "hi"], cwd="/tmp/work", env={})
    image = "agent-belt-sandbox-cursor:dev"
    image_idx = cmd.index(image)
    pre_image = cmd[2:image_idx]  # skip "docker", "run"
    post_image = cmd[image_idx + 1 :]
    # Every pre-image token is either a recognised flag, a -v/-e value,
    # or an --add-host value. None of them should be the agent argv.
    assert "agent" not in pre_image
    assert "hi" not in pre_image
    assert post_image == ["agent", "hi"]


def test_provider_error_messages_are_actionable() -> None:
    # Operator-facing error messages must name the offending field and
    # suggest the fix path. A bare ``ValueError`` here would force the
    # operator to dig through code to find why the run aborted. The two
    # error families split by failure source: missing-image is a profile
    # problem (``SandboxConfigError`` from ``validate_profile``), missing
    # docker daemon is an environment problem (``DockerSandboxError`` from
    # ``setup``).
    from belt.runner.sandbox.base import SandboxConfigError

    provider = DockerSandboxProvider()
    ctx = SandboxContext(workspace_dir=Path("/tmp/x"), scenario_name="my-scn")

    profile = SandboxProfile(provider="docker", image=None)
    with pytest.raises(SandboxConfigError) as exc:
        provider.validate_profile(profile, ctx)
    msg = str(exc.value)
    assert "my-scn" in msg, "scenario name must appear so operator knows which file"
    assert "image" in msg
    assert "_config.json" in msg or "sandbox.image" in msg

    profile = SandboxProfile(provider="docker", image="x")
    with patch("belt.runner.sandbox.docker._docker_available", return_value=False):
        with pytest.raises(DockerSandboxError) as exc:
            provider.setup(profile, ctx)
    assert "docker" in str(exc.value).lower()
    assert "PATH" in str(exc.value) or "install" in str(exc.value).lower()
    assert "host" in str(exc.value), "fallback path (--sandbox host) must be suggested"


# ----------------------------------------------------------------------
# Network policy: open (default) vs none (kernel-enforced no network)
# ----------------------------------------------------------------------


def test_wrap_default_network_policy_does_not_emit_network_flag() -> None:
    # ``open`` (the default) leaves Docker's default network in place.
    # The provider must NOT emit ``--network=...`` so docker uses its
    # built-in bridge network -- regression guard against a future
    # change that pins ``--network=bridge`` (would override per-host
    # docker daemon defaults like custom networks for proxies/CI).
    provider = DockerSandboxProvider()
    cmd, _, _ = provider.wrap(_handle(), cmd=["agent"], cwd="/tmp/work", env={})
    assert not any(
        t.startswith("--network=") for t in cmd
    ), f"unexpected --network flag in default-policy argv: {cmd!r}"


def test_wrap_network_policy_none_emits_network_none_flag() -> None:
    # ``none`` is the kernel-enforced zero-network mode: the container
    # gets its own network namespace with only loopback, so any
    # outbound socket call fails with EHOSTUNREACH/ENETUNREACH at the
    # kernel level. The provider must emit exactly ``--network=none``
    # in the docker argv. Pin the flag spelling -- ``--net=none``
    # also works on docker but isn't the documented contract and a
    # silent switch would make grep-for-docs harder.
    provider = DockerSandboxProvider()
    profile = SandboxProfile(provider="docker", image="img:tag", network_policy="none")
    cmd, _, _ = provider.wrap(_handle(profile=profile), cmd=["agent"], cwd="/tmp/work", env={})
    assert "--network=none" in cmd
    image_idx = cmd.index("img:tag")
    network_idx = cmd.index("--network=none")
    assert network_idx < image_idx, "--network=none must appear before the image (it is a docker run flag)"


def test_wrap_network_policy_none_drops_add_host_entries() -> None:
    # ``--add-host`` paired with ``--network=none`` is rejected by
    # docker (no interface to register the entry against). Even if
    # docker accepted it, the entries would be dead weight. The
    # provider must drop them silently when the network is closed --
    # the parser already documents that ``allowed_hosts`` is ignored
    # in this mode.
    provider = DockerSandboxProvider()
    profile = SandboxProfile(
        provider="docker",
        image="img:tag",
        network_policy="none",
        allowed_hosts=["api.openai.com"],
    )
    cmd, _, _ = provider.wrap(_handle(profile=profile), cmd=["agent"], cwd="/tmp/work", env={})
    assert "--add-host" not in cmd, f"--add-host must not appear under network_policy=none: {cmd!r}"
    assert "--network=none" in cmd


def test_wrap_network_policy_none_keeps_other_hardening_intact() -> None:
    # Belt-and-suspenders: turning off the network must NOT silently
    # turn off any other hardening. A regression that branched
    # docker_cmd construction differently for ``none`` could drop
    # ``--cap-drop=ALL`` etc.
    provider = DockerSandboxProvider()
    profile = SandboxProfile(provider="docker", image="img:tag", network_policy="none")
    cmd, _, _ = provider.wrap(_handle(profile=profile), cmd=["agent"], cwd="/tmp/work", env={})
    for required in (
        "--rm",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        "--workdir=/work",
        "--network=none",
    ):
        assert required in cmd, f"{required!r} missing under network_policy=none: {cmd!r}"
    assert any(t.startswith("--tmpfs=/home/agent") for t in cmd)
