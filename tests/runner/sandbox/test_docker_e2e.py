# (c) JFrog Ltd. (2026)

"""End-to-end Docker sandbox tests -- REAL container execution.

These tests pin every security-critical invariant by actually running
the wrapped command through ``docker run`` and asserting on observable
container behaviour, not on argv. Pure unit tests in
``test_docker_provider.py`` pin the argv shape; this file pins
"the argv we generate produces the isolation we promise".

Why both layers:

- **Argv tests catch flag drops fast.** A regression that removes
  ``--cap-drop=ALL`` shows up in ``test_wrap_applies_hardening_flags``
  with no docker dependency.
- **E2E tests catch integration drift slow.** A regression where a flag
  is present in argv but ineffective in practice (wrong ordering, wrong
  value, masked by a later flag, contradicted by image USER, etc.) only
  shows up by actually executing the container. We learned this the
  hard way: argv tests passed cleanly while real eval runs hit two
  separate "container exec failed" classes (host abs path in cmd[0],
  read-only rootfs blocking ``$HOME`` writes) that the mocked tests
  could not surface.

These tests:

- Use ``agent-belt-sandbox-generic:dev`` (the framework's reference
  image, no agent CLI inside) and execute small ``python3 -c "..."``
  payloads. We are testing the sandbox machinery, not any specific
  agent; using a real agent would couple the tests to that agent's
  install matrix and auth model, neither of which is what we are
  trying to verify here.
- ``skipif`` when docker or the reference image is unavailable so the
  default ``pytest`` run on a developer machine without docker still
  passes. CI builds the image then runs this module unconditionally.
- Each test runs one container with ``--rm`` so there is nothing to
  clean up. Tests do not share state between containers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from belt.runner.sandbox.base import SandboxContext, SandboxHandle
from belt.runner.sandbox.docker import DockerSandboxProvider
from belt.scenario import SandboxProfile

REFERENCE_IMAGE = "agent-belt-sandbox-generic:dev"


def _docker_image_present(image: str) -> bool:
    """Return True iff the docker daemon already has the image locally.

    Pure-shell check: ``docker image inspect`` returns 0 when the image
    is present, non-zero (with a "no such image" message) otherwise.
    """
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(  # noqa: S603,S607 - fixed argv
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_image_present(REFERENCE_IMAGE),
    reason=(
        f"docker or reference image '{REFERENCE_IMAGE}' not available; "
        "build with `docker build -t agent-belt-sandbox-generic:dev "
        "-f examples/sandbox-images/Dockerfile.generic .` to enable e2e tests."
    ),
)


def _handle(
    workspace: Path,
    *,
    profile: SandboxProfile | None = None,
    required_env: frozenset[str] = frozenset(),
    scenario: str = "e2e",
) -> SandboxHandle:
    profile = profile or SandboxProfile(provider="docker", image=REFERENCE_IMAGE)
    ctx = SandboxContext(
        workspace_dir=workspace,
        agent_required_env=required_env,
        scenario_name=scenario,
    )
    return SandboxHandle(profile=profile, context=ctx, state={})


def _run_in_sandbox(
    workspace: Path,
    payload: str,
    *,
    profile: SandboxProfile | None = None,
    required_env: frozenset[str] = frozenset(),
    extra_env: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    """Wrap ``python3 -c <payload>`` through the docker provider and run it.

    Returns the ``CompletedProcess`` so tests can assert on stdout,
    stderr, and return code. The function deliberately does NOT raise
    on non-zero exit -- many tests verify *failure modes* (e.g. write
    to read-only rootfs MUST fail), so the test asserts on the exit
    code itself.
    """
    provider = DockerSandboxProvider()
    handle = _handle(workspace, profile=profile, required_env=required_env)
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    cmd, cwd, env_out = provider.wrap(
        handle,
        cmd=["python3", "-c", payload],
        cwd=str(workspace),
        env=env,
    )
    return subprocess.run(  # noqa: S603 - argv is constructed by the provider under test
        cmd,
        cwd=cwd,
        env=env_out,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ----------------------------------------------------------------------
# Container actually launches and produces output
# ----------------------------------------------------------------------


def test_e2e_container_launches_and_returns_stdout(tmp_path: Path) -> None:
    result = _run_in_sandbox(tmp_path, "print('hello-from-sandbox')")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "hello-from-sandbox"


def test_e2e_python_runtime_is_present_and_recent(tmp_path: Path) -> None:
    # Pins the BYOI contract for the generic image: it MUST ship a
    # Python interpreter the runner can rely on for portable test
    # payloads. If a future image bump drops python, this test fails
    # loudly instead of every other e2e test failing for opaque reasons.
    result = _run_in_sandbox(
        tmp_path,
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
    )
    assert result.returncode == 0, result.stderr
    major, minor = (int(x) for x in result.stdout.strip().split("."))
    assert (major, minor) >= (3, 10), f"need python>=3.10, got {major}.{minor}"


def test_e2e_exit_code_propagates_through_subprocess(tmp_path: Path) -> None:
    # If a container exits non-zero, the host subprocess.run must
    # surface that exact code so the orchestrator can flag the run as
    # failed. A previous regression mistakenly translated all docker
    # errors to exit code 1, hiding the agent's real signal.
    result = _run_in_sandbox(tmp_path, "import sys; sys.exit(42)")
    assert result.returncode == 42


# ----------------------------------------------------------------------
# Filesystem isolation: read-only rootfs + only worktree writable
# ----------------------------------------------------------------------


def test_e2e_rootfs_is_readonly_writes_to_etc_fail(tmp_path: Path) -> None:
    # ``--read-only`` rootfs MUST block writes outside the bind mounts.
    # The container has /etc populated by the image but the layer is
    # read-only at runtime. A successful write here means the flag was
    # silently dropped from the argv.
    result = _run_in_sandbox(
        tmp_path,
        "open('/etc/agent-belt-leak', 'w').write('pwned')",
    )
    assert result.returncode != 0, "rootfs write must fail under --read-only"
    assert (
        "Read-only file system" in result.stderr or "PermissionError" in result.stderr or "OSError" in result.stderr
    ), f"expected read-only error, got: {result.stderr!r}"


def test_e2e_workdir_is_writable_and_persists_to_host(tmp_path: Path) -> None:
    # The worktree bind mount is the ONLY writable path by design.
    # Confirm: writes from the container land on the host worktree
    # (proves the bind mount actually points where we said it did)
    # AND the in-container write path resolves to /work (the canonical
    # workdir; tests that depend on this constant will fail loudly if
    # it drifts).
    payload = (
        "import os; " "p='/work/e2e-write.txt'; " "open(p,'w').write('host-readable'); " "print(os.path.realpath(p))"
    )
    result = _run_in_sandbox(tmp_path, payload)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/work/e2e-write.txt"
    # File must materialise on the HOST at the bind-mount source.
    landed = tmp_path / "e2e-write.txt"
    assert landed.exists(), f"file did not land on host at {landed}"
    assert landed.read_text() == "host-readable"


def test_e2e_writable_paths_extra_mount_is_writable_and_isolated(tmp_path: Path) -> None:
    # ``writable_paths`` widens the trust boundary: each entry becomes
    # a writable bind mount at the same absolute path inside the
    # container. We verify both halves: (a) the extra path IS writable
    # from the container and (b) writes show up on the host at the
    # source. Uses a dedicated tmp dir so the test does not collide
    # with the worktree mount.
    extra = tmp_path / "extra-cache"
    extra.mkdir()
    profile = SandboxProfile(
        provider="docker",
        image=REFERENCE_IMAGE,
        writable_paths=[str(extra)],
    )
    payload = f"open('{extra}/cached', 'w').write('extra-writable')"
    result = _run_in_sandbox(tmp_path, payload, profile=profile)
    assert result.returncode == 0, result.stderr
    assert (extra / "cached").read_text() == "extra-writable"


def test_e2e_writes_outside_worktree_and_extras_fail(tmp_path: Path) -> None:
    # Negative half of the writable_paths contract: a path NOT in
    # writable_paths and NOT the worktree must reject writes. /tmp
    # inside the container is on the read-only rootfs (no tmpfs is
    # mounted by the framework today), so this is the right test.
    result = _run_in_sandbox(
        tmp_path,
        "open('/tmp/should-fail', 'w').write('x')",
    )
    assert result.returncode != 0, "/tmp write must fail under --read-only"


# ----------------------------------------------------------------------
# Capability isolation: --cap-drop=ALL + no-new-privileges
# ----------------------------------------------------------------------


def test_e2e_runs_as_unprivileged_user_not_root(tmp_path: Path) -> None:
    # The reference image declares ``USER belt`` (uid 1000); the runner
    # does not currently override it (the docstring mentions
    # ``--user 1000:1000`` belt-and-suspenders, but the canonical
    # source of truth is the image USER). Either way, the container
    # MUST NOT run as root -- a regression that drops USER from the
    # image or adds ``--user 0`` to the runner would break this.
    result = _run_in_sandbox(tmp_path, "import os; print(os.getuid())")
    assert result.returncode == 0, result.stderr
    uid = int(result.stdout.strip())
    assert uid != 0, "container ran as root (uid 0); image USER setting is broken"


def test_e2e_capability_dropped_mount_blocked(tmp_path: Path) -> None:
    # ``--cap-drop=ALL`` strips CAP_SYS_ADMIN; the container should be
    # unable to mount anything. We can't actually call mount(2) from
    # python easily, but we can attempt to read /proc/self/status and
    # confirm CapEff is empty (all hex zeros) -- a tighter, more
    # portable assertion than relying on a specific syscall failure
    # path.
    payload = (
        "import re; "
        "txt=open('/proc/self/status').read(); "
        "m=re.search(r'CapEff:\\s+([0-9a-fA-F]+)', txt); "
        "print(m.group(1) if m else 'NOMATCH')"
    )
    result = _run_in_sandbox(tmp_path, payload)
    assert result.returncode == 0, result.stderr
    cap_eff = result.stdout.strip()
    # CapEff is a 16-hex-char bitmask; "all zero" means no capabilities.
    assert cap_eff != "NOMATCH"
    assert int(cap_eff, 16) == 0, f"capabilities not dropped: CapEff={cap_eff}"


# ----------------------------------------------------------------------
# Env passthrough: by-name only, no leaks of unlisted vars
# ----------------------------------------------------------------------


def test_e2e_env_passthrough_value_delivered_when_listed(tmp_path: Path) -> None:
    # Listed name + value present in invoker env -> value reaches the
    # container. Sanity check on the ``-e NAME`` (no value) docker
    # syntax that the provider emits.
    profile = SandboxProfile(
        provider="docker",
        image=REFERENCE_IMAGE,
        env_passthrough=["BELT_E2E_TEST_TOKEN"],
    )
    result = _run_in_sandbox(
        tmp_path,
        "import os; print(os.environ.get('BELT_E2E_TEST_TOKEN', 'MISSING'))",
        profile=profile,
        extra_env={"BELT_E2E_TEST_TOKEN": "value-" + uuid.uuid4().hex[:8]},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith("value-")


def test_e2e_env_not_in_passthrough_does_not_leak(tmp_path: Path) -> None:
    # Critical security invariant: a secret-shaped env var that is
    # present in the invoker's process env but NOT declared in
    # ``env_passthrough`` (and not in the agent's required_env)
    # must not appear in the container.
    secret = "secret-value-" + uuid.uuid4().hex
    profile = SandboxProfile(
        provider="docker",
        image=REFERENCE_IMAGE,
        env_passthrough=["BELT_E2E_ALLOWED"],
    )
    result = _run_in_sandbox(
        tmp_path,
        "import os; print(os.environ.get('BELT_E2E_BANNED', 'ABSENT'))",
        profile=profile,
        extra_env={
            "BELT_E2E_ALLOWED": "ok",
            "BELT_E2E_BANNED": secret,
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ABSENT", f"unlisted env var leaked into container: {result.stdout!r}"
    # Belt-and-suspenders: the secret must not appear anywhere in the
    # container's view of stderr either (some image entrypoints echo env).
    assert secret not in result.stdout
    assert secret not in result.stderr


def test_e2e_home_is_writable_tmpfs_inside_container(tmp_path: Path) -> None:
    # Invariant: agents that write to ``~/.cache``, ``~/.config``,
    # ``~/.cursor/projects`` etc. must succeed without polluting the
    # host worktree. The runner injects ``-e HOME=/home/agent`` plus
    # a per-run tmpfs at the same path so HOME writes (a) succeed on
    # the read-only rootfs, (b) stay ephemeral (vanish on container
    # exit), and (c) never appear in the worktree diff. The split
    # between HOME (tmpfs) and CWD (/work bind) is the contract;
    # collapsing them would either break read-only or pollute diffs.
    result = _run_in_sandbox(
        tmp_path,
        "import os, pathlib; "
        "h=os.environ['HOME']; "
        "p=pathlib.Path(h)/'agent-cache.txt'; "
        "p.write_text('home-write-ok'); "
        "print(h, p.read_text(), sep='|')",
    )
    assert result.returncode == 0, result.stderr
    home_value, content = result.stdout.strip().split("|")
    assert home_value == "/home/agent"
    assert content == "home-write-ok"
    # The host worktree MUST NOT see the cache file -- HOME is a
    # tmpfs, not a bind mount. This is the regression assertion that
    # would have caught the old HOME=/work polluting the diff.
    assert not (tmp_path / "agent-cache.txt").exists(), "tmpfs-backed HOME leaked file into the worktree"


# ----------------------------------------------------------------------
# Cmd[0] basename rewrite (regression: host abs paths exec-failed)
# ----------------------------------------------------------------------


def test_e2e_absolute_host_path_in_cmd_resolves_via_container_path(tmp_path: Path) -> None:
    # Regression for the bug found by the very first real eval run:
    # cursor adapter resolves cmd[0] to ``/Users/.../cursor-agent``;
    # docker tries to exec that absolute path inside the container,
    # which obviously does not exist. The provider rewrites cmd[0] to
    # its basename and relies on the container PATH. Here we simulate
    # that by passing a fake host abs path whose basename is
    # ``python3`` (which IS on PATH inside the image).
    provider = DockerSandboxProvider()
    handle = _handle(tmp_path)
    cmd, cwd, env = provider.wrap(
        handle,
        cmd=["/Users/imaginary/.local/bin/python3", "-c", "print('basename-resolved')"],
        cwd=str(tmp_path),
        env=dict(os.environ),
    )
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "basename-resolved"


# ----------------------------------------------------------------------
# Container lifecycle: --rm cleans up
# ----------------------------------------------------------------------


def test_e2e_rm_flag_removes_container_after_exit(tmp_path: Path) -> None:
    # Each ``docker run --rm`` MUST remove the container on exit.
    # Without --rm the host accumulates dead containers indefinitely
    # (one per scenario per agent per run -> very fast leak in CI).
    # We assert by counting containers for the image before/after.
    label_before = subprocess.run(  # noqa: S603,S607
        ["docker", "ps", "-a", "--filter", f"ancestor={REFERENCE_IMAGE}", "-q"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    count_before = len([line for line in label_before.stdout.splitlines() if line.strip()])

    result = _run_in_sandbox(tmp_path, "print('lifecycle-test')")
    assert result.returncode == 0, result.stderr

    label_after = subprocess.run(  # noqa: S603,S607
        ["docker", "ps", "-a", "--filter", f"ancestor={REFERENCE_IMAGE}", "-q"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    count_after = len([line for line in label_after.stdout.splitlines() if line.strip()])
    assert count_after == count_before, (
        f"container leak: {count_before} before, {count_after} after run " f"(--rm should have removed it)"
    )


def test_e2e_workdir_inside_container_is_canonical_workdir(tmp_path: Path) -> None:
    # ``--workdir=/work`` is the contract that scenarios expect: the
    # CWD the agent sees is /work, regardless of where the host
    # invoked docker from. Tests that depend on relative paths in
    # scenario inputs (e.g. ``cat scenario.json``) rely on this.
    result = _run_in_sandbox(tmp_path, "import os; print(os.getcwd())")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/work"


# ----------------------------------------------------------------------
# Stress: multiple sequential runs share no state
# ----------------------------------------------------------------------


def test_e2e_two_runs_in_separate_workspaces_are_isolated(tmp_path: Path) -> None:
    # Regression: a previous container layout shared /tmp across runs
    # via a host bind. Verify here that two consecutive runs cannot
    # see each other's worktree state -- container A writes a file in
    # workspace A, container B in workspace B does NOT see it.
    ws_a = tmp_path / "a"
    ws_b = tmp_path / "b"
    ws_a.mkdir()
    ws_b.mkdir()

    write = _run_in_sandbox(ws_a, "open('/work/secret-a', 'w').write('a-only')")
    assert write.returncode == 0, write.stderr
    assert (ws_a / "secret-a").exists()

    peek = _run_in_sandbox(
        ws_b,
        "import os; print('PRESENT' if os.path.exists('/work/secret-a') else 'ABSENT')",
    )
    assert peek.returncode == 0, peek.stderr
    assert peek.stdout.strip() == "ABSENT", "workspace B saw workspace A's file"


def test_e2e_artifacts_written_in_one_run_visible_to_next_run_in_same_workspace(
    tmp_path: Path,
) -> None:
    # Multi-turn behaviour: a single workspace persists across turn
    # boundaries (each turn is a fresh container with the same bind).
    # This is the foundation of multi-turn scoring -- if a turn writes
    # state that the next turn cannot see, scenarios that build on
    # earlier turns would score as if every turn started cold.
    write = _run_in_sandbox(tmp_path, "open('/work/multi-turn-state.json','w').write('{\"turn\": 1}')")
    assert write.returncode == 0, write.stderr

    read = _run_in_sandbox(
        tmp_path,
        "import json; print(json.load(open('/work/multi-turn-state.json'))['turn'])",
    )
    assert read.returncode == 0, read.stderr
    assert read.stdout.strip() == "1"


# ----------------------------------------------------------------------
# Network policy: kernel-enforced ``--network=none`` actually blocks outbound
# ----------------------------------------------------------------------


def test_e2e_network_policy_open_default_can_reach_loopback(tmp_path: Path) -> None:
    # Sanity baseline for the negative test below: under the default
    # ``open`` policy the container has a working network stack
    # (loopback at minimum), so opening a TCP socket succeeds. If
    # this ever fails, the negative test would be measuring the
    # wrong thing.
    payload = (
        "import socket\n"
        "s = socket.socket()\n"
        "try:\n"
        "    s.connect(('127.0.0.1', 1))\n"
        "except (ConnectionRefusedError, OSError) as e:\n"
        "    print(type(e).__name__)\n"
        "else:\n"
        "    print('CONNECTED')\n"
    )
    result = _run_in_sandbox(tmp_path, payload)
    assert result.returncode == 0, result.stderr
    # ConnectionRefusedError = port closed but stack reachable; that
    # proves loopback is up. EHOSTUNREACH/ENETUNREACH would mean the
    # stack itself is gone, which is what the ``none`` test asserts.
    assert result.stdout.strip() in (
        "ConnectionRefusedError",
        "OSError",
    ), f"loopback should reach a closed port, got {result.stdout!r}"


def test_e2e_network_policy_none_blocks_outbound_tcp_at_kernel_level(tmp_path: Path) -> None:
    # The single most important assertion of this whole feature:
    # ``--network=none`` actually prevents outbound TCP. We try to
    # connect to a public IP (1.1.1.1:443, Cloudflare's resolver,
    # always reachable from a default bridge network). Under
    # ``open`` the connect would either succeed or fail with TLS
    # noise; under ``none`` it MUST fail with a kernel-level
    # network-down error: EHOSTUNREACH, ENETUNREACH, or
    # OSError("Network is unreachable"). If a future change made
    # this connect succeed (e.g. accidentally swapping in
    # ``--network=bridge``), this test fails loudly.
    profile = SandboxProfile(
        provider="docker",
        image=REFERENCE_IMAGE,
        network_policy="none",
    )
    payload = (
        "import socket, errno\n"
        "s = socket.socket()\n"
        "s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 443))\n"
        "    print('CONNECTED')\n"
        "except OSError as e:\n"
        "    name = errno.errorcode.get(e.errno, str(e.errno))\n"
        "    print(f'BLOCKED:{name}')\n"
    )
    result = _run_in_sandbox(tmp_path, payload, profile=profile)
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    assert out.startswith("BLOCKED:"), f"outbound should be blocked, got {out!r}"
    # Kernel surfaces the block via one of these errno codes; pin
    # the set so a regression that only fails by *timeout* (which
    # could mean the packet went out but got dropped late) shows
    # up as a test failure.
    blocked_errno = out.split(":", 1)[1]
    assert blocked_errno in (
        "ENETUNREACH",
        "EHOSTUNREACH",
        "EPERM",
    ), f"expected kernel-level network-down errno, got {blocked_errno!r}"


def test_e2e_network_policy_none_blocks_dns_resolution(tmp_path: Path) -> None:
    # Defence-in-depth: if a misconfiguration left DNS reachable
    # while blocking TCP, an exfil channel via DNS-over-UDP queries
    # would still exist. Resolving any public hostname under
    # ``--network=none`` MUST fail because there is no network
    # interface to send the UDP query out of.
    profile = SandboxProfile(
        provider="docker",
        image=REFERENCE_IMAGE,
        network_policy="none",
    )
    payload = (
        "import socket\n"
        "try:\n"
        "    print(socket.gethostbyname('cloudflare.com'))\n"
        "except (OSError, socket.gaierror) as e:\n"
        "    print(f'DNS_BLOCKED:{type(e).__name__}')"
    )
    result = _run_in_sandbox(tmp_path, payload, profile=profile)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith("DNS_BLOCKED:"), f"DNS resolution should be blocked, got {result.stdout!r}"


def test_e2e_network_policy_none_loopback_still_present(tmp_path: Path) -> None:
    # ``--network=none`` does NOT remove the loopback interface
    # (Linux always provides ``lo`` for the network namespace).
    # This matters because some agents/tools bind to 127.0.0.1
    # for in-process IPC; if loopback were missing, those would
    # break in the sandboxed mode for no security gain.
    profile = SandboxProfile(
        provider="docker",
        image=REFERENCE_IMAGE,
        network_policy="none",
    )
    payload = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.bind(('127.0.0.1', 0))\n"
        "s.listen(1)\n"
        "port = s.getsockname()[1]\n"
        "c = socket.socket()\n"
        "c.connect(('127.0.0.1', port))\n"
        "print('LOOPBACK_OK')"
    )
    result = _run_in_sandbox(tmp_path, payload, profile=profile)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LOOPBACK_OK"


def test_e2e_network_policy_none_filesystem_workdir_still_writable(tmp_path: Path) -> None:
    # Belt-and-suspenders: closing the network must not
    # accidentally close the worktree. An offline scenario's
    # whole point is to write code locally.
    profile = SandboxProfile(
        provider="docker",
        image=REFERENCE_IMAGE,
        network_policy="none",
    )
    result = _run_in_sandbox(
        tmp_path,
        "open('/work/offline-edit.py','w').write('print(\"ok\")')",
        profile=profile,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "offline-edit.py").read_text() == 'print("ok")'


def test_e2e_concurrent_runs_do_not_collide_on_container_name(tmp_path: Path) -> None:
    # Docker auto-generates container names when --name is not passed
    # (which is what the provider does). Two parallel runs must not
    # conflict. We approximate parallelism by running two containers
    # back-to-back and confirming no "name in use" error -- if the
    # provider ever started passing a fixed --name, this would fail.
    ws_a = tmp_path / "para-a"
    ws_b = tmp_path / "para-b"
    ws_a.mkdir()
    ws_b.mkdir()

    a = _run_in_sandbox(ws_a, "print('A')")
    b = _run_in_sandbox(ws_b, "print('B')")
    assert a.returncode == 0 and a.stdout.strip() == "A", a.stderr
    assert b.returncode == 0 and b.stdout.strip() == "B", b.stderr
