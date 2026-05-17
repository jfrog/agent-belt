# (c) JFrog Ltd. (2026)

"""Docker sandbox provider -- container isolation via the host ``docker`` CLI.

Implementation choices and their rationale:

- **No ``docker-py`` SDK dependency.** Adding the SDK pulls a transitive tree
  that frequently lights up SCA scanners. ``subprocess.run(["docker", ...])``
  is enough for ``run`` / ``stop`` / ``rm`` and keeps belt's wheel surface
  unchanged. The cost is one ``shutil.which("docker")`` call up front and
  parsing of ``docker``'s exit codes; the benefit is zero new dependencies.
- **Filesystem + capability isolation are first-class.** The wrapper sets
  ``--cap-drop=ALL``, ``--security-opt=no-new-privileges``, ``--read-only``
  rootfs, ``--user 1000:1000``, and bind-mounts only the worktree (plus
  scenario-declared ``writable_paths``). An agent inside the container
  cannot read ``$HOME`` on the host, cannot escalate, cannot mount, cannot
  ptrace, cannot write outside the worktree.
- **Network policy has two kernel-enforced modes plus a v1 hint.**
  ``SandboxProfile.network_policy='open'`` (default) leaves Docker's
  bridge network in place so real LLM calls work end-to-end out of the
  box; ``allowed_hosts`` materialises as ``--add-host`` entries (a v1
  best-effort hint that helps DNS but does NOT block other outbound
  traffic). ``SandboxProfile.network_policy='none'`` adds
  ``--network=none`` so the container has no network namespace
  interfaces other than loopback -- any outbound syscall fails at the
  kernel level with EHOSTUNREACH/ENETUNREACH, no userspace bypass is
  possible. Use ``none`` for offline scenarios (local code edits, no
  LLM/API needed). Hostname-level allowlisting on top of ``open`` (the
  partial-allowlist case) is the planned next iteration; doing it
  properly requires a privileged sidecar or root-owned daemons that
  belt cannot ship safely. See ``SANDBOXING.md`` -> "Network policy"
  for the precise enforcement model and the future-work issue.
- **Env passthrough is by exact name only.** No wildcards, no shell
  expansion. The runner unions ``ctx.agent_required_env`` (declared by the
  agent class) with ``profile.env_passthrough`` (declared by the scenario
  author); only names in that union reach the container, and they reach it
  via ``-e NAME`` (Docker reads the value from the invoker's env, the value
  is never echoed by belt).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from typing import Any

from loguru import logger

from belt.errors import BeltError
from belt.runner.sandbox.base import BaseSandboxProvider, SandboxConfigError, SandboxContext, SandboxHandle
from belt.scenario import SandboxProfile


class DockerSandboxError(BeltError):
    """Raised when the docker provider cannot set up or wrap a scenario at runtime.

    Use for environment-level failures the framework cannot detect from the
    profile alone (docker daemon missing, container start fails, etc.).
    Profile-coherence failures (missing image, contradictory fields) raise
    :class:`SandboxConfigError` from :meth:`validate_profile` instead, so
    the misconfiguration is caught before any subprocess work begins.
    """


# Single source of truth for the container's working directory and the
# in-container path that the host worktree is bind-mounted to. Anything that
# moves these has to update SANDBOXING.md and the showcase scenario together.
_CONTAINER_WORKDIR = "/work"

# In-container HOME for the agent. Backed by a per-run tmpfs so writes to
# ``~/.cache``, ``~/.config``, ``~/.cursor/projects`` etc. (a) succeed on the
# read-only rootfs, (b) stay ephemeral (vanish at container exit, never
# reach the host), and (c) DO NOT pollute the worktree diff -- a previous
# implementation used HOME=/work, which made cursor-agent's compile cache
# show up as workspace edits in ``files_modified``. The size cap is generous
# enough for typical agent caches but bounded so a runaway write can't
# exhaust container memory.
_CONTAINER_HOME = "/home/agent"
_HOME_TMPFS_SIZE = "256m"


def _docker_available() -> bool:
    """Return True iff the host has a working ``docker`` binary on PATH."""
    return shutil.which("docker") is not None


class DockerSandboxProvider(BaseSandboxProvider):
    """Run agent subprocesses inside a Docker container.

    The image is BYOI: scenarios declare ``SandboxProfile.image``. The
    framework ships one reference Dockerfile in ``examples/sandbox-images/``
    plus a worked example for cursor; the framework owns no agent-specific
    images.
    """

    @classmethod
    def name(cls) -> str:
        return "docker"

    def validate_profile(self, profile: SandboxProfile, ctx: SandboxContext) -> None:
        # Profile-coherence check: every field needed to build the docker
        # argv must be present. Environment-level checks (docker daemon
        # reachable) live in :meth:`setup` because the runner calls
        # ``validate_profile`` for every scenario, on every host, even
        # those without docker installed -- the typed config error must
        # not require a working docker installation to surface.
        if not profile.image:
            raise SandboxConfigError(
                f"sandbox profile for scenario '{ctx.scenario_name}' uses provider='docker' "
                "but no 'image' was specified. Set sandbox.image in the scenario group's "
                "_config.json (e.g. 'agent-belt-sandbox-cursor:dev'). See SANDBOXING.md."
            )

    def setup(self, profile: SandboxProfile, ctx: SandboxContext) -> SandboxHandle:
        if not _docker_available():
            raise DockerSandboxError(
                "sandbox profile requests provider='docker' but the 'docker' binary is not "
                "on PATH. Install Docker, or re-run with --sandbox host. See SANDBOXING.md."
            )
        return SandboxHandle(profile=profile, context=ctx, state={})

    def wrap(
        self,
        handle: SandboxHandle,
        *,
        cmd: list[str],
        cwd: str | None,
        env: dict[str, str],
    ) -> tuple[list[str], str | None, dict[str, str]]:
        profile = handle.profile
        ctx = handle.context

        # Forward only env-var names in the union of (agent-declared required)
        # and (scenario-declared passthrough). The values stay in the
        # invoker's env -- ``docker`` reads them via ``-e NAME`` rather than
        # ``-e NAME=value``, so belt itself never echoes a secret.
        passthrough_names = frozenset(ctx.agent_required_env) | frozenset(profile.env_passthrough)

        docker_cmd: list[str] = [
            "docker",
            "run",
            "--rm",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            f"--workdir={_CONTAINER_WORKDIR}",
            # Ephemeral HOME on tmpfs: writable, memory-only, bounded,
            # auto-discarded on container exit. See ``_CONTAINER_HOME``
            # comment for why this is preferred over HOME=/work.
            f"--tmpfs={_CONTAINER_HOME}:rw,size={_HOME_TMPFS_SIZE},mode=1777",
        ]

        # Network policy. ``none`` is kernel-enforced (Linux network
        # namespace with no interfaces other than loopback): any
        # outbound syscall fails with EHOSTUNREACH/ENETUNREACH at the
        # kernel level, no userspace bypass is possible. Use it for
        # scenarios that operate purely on local files. ``open`` (the
        # default) keeps Docker's bridge network so real LLM-using
        # agents can reach their providers; outbound restriction at
        # the hostname level is the planned next iteration (see
        # SANDBOXING.md). When ``none`` is requested, ``allowed_hosts``
        # cannot meaningfully take effect (--add-host needs a network
        # to point at) and is silently dropped from the argv -- the
        # parser already documents this.
        if profile.network_policy == "none":
            docker_cmd += ["--network=none"]

        # Bind-mount the worktree as the container's working directory. The
        # ``:rw`` suffix is explicit so a misconfigured image with a
        # read-only root cannot accidentally mount the worktree read-only.
        docker_cmd += ["-v", f"{ctx.workspace_dir}:{_CONTAINER_WORKDIR}:rw"]

        # Extra writable host paths declared by the scenario, mapped into the
        # container at the same absolute path. Used sparingly: every entry
        # widens the trust boundary.
        for path in profile.writable_paths:
            docker_cmd += ["-v", f"{path}:{path}:rw"]

        # ``allowed_hosts`` is documented as a v1 best-effort hint. Each
        # entry is resolved on the host (where DNS is known to work) and
        # injected as a ``--add-host name:ip`` entry so the in-container
        # resolver returns the real IP without needing an outbound DNS
        # query. Unresolvable names are skipped with a warning rather
        # than raising -- v1 keeps the bridge network's full outbound
        # access, so a missing /etc/hosts entry just falls back to the
        # default resolver. v1 does NOT enforce a hard outbound
        # allowlist via iptables (see SANDBOXING.md "Network policy"),
        # which is the planned next iteration.
        # Skip --add-host entries when there is no network: docker
        # rejects ``--add-host`` combined with ``--network=none`` (no
        # interface to register the entry against), and even if it
        # didn't, the entries would be useless without an interface.
        allowed_hosts: list[str] = [] if profile.network_policy == "none" else list(profile.allowed_hosts)
        for host in allowed_hosts:
            try:
                ip = socket.gethostbyname(host)
            except OSError as exc:
                logger.warning(
                    "sandbox.docker: could not resolve allowed_host %r on the host (%s); " "skipping --add-host entry",
                    host,
                    exc,
                )
                continue
            docker_cmd += ["--add-host", f"{host}:{ip}"]

        for name in sorted(passthrough_names):
            if name in env:
                docker_cmd += ["-e", name]

        # Force HOME to the per-run tmpfs so agents that write state
        # under ``$HOME`` (cursor-agent's ``~/.cursor/projects/<cwd>``,
        # node's ``~/.npm`` cache, gh's ``~/.config/gh``, etc.) land
        # in an ephemeral, memory-backed directory rather than the
        # worktree (where they would pollute the diff) or the
        # read-only rootfs (where they would fail). ``-e HOME=`` (with
        # value) is the one exception to the by-name-only env policy:
        # the value is fixed by the framework, not sourced from the
        # invoker's env, so there is no secret-leak vector.
        docker_cmd += ["-e", f"HOME={_CONTAINER_HOME}"]

        # Image goes after all flags, before the actual command.
        docker_cmd += [profile.image]

        # Rewrite cmd[0] to its basename. Agent adapters resolve their CLI
        # to an absolute host path (e.g. /Users/.../cursor-agent), but that
        # path almost never exists inside the container -- the image
        # installs the agent at its own location (typically /usr/local/bin).
        # Stripping to the basename relies on the container's PATH, which
        # is the contract the BYOI image must satisfy. The cmd[1:] argv
        # carries the actual prompt/flags and stays verbatim.
        in_container_cmd = list(cmd)
        if in_container_cmd and os.path.isabs(in_container_cmd[0]):
            in_container_cmd[0] = os.path.basename(in_container_cmd[0])
        docker_cmd += in_container_cmd

        # The container has its own working directory; the host-side cwd from
        # the agent's Popen call is meaningless and must be cleared so docker
        # itself runs in the invoker's cwd (where the docker socket and
        # config live). The host env is passed verbatim because docker reads
        # the values it needs by name (-e NAME).
        return docker_cmd, None, env

    def teardown(self, handle: SandboxHandle) -> None:
        # ``docker run --rm`` removes the container on exit. There is no
        # per-scenario state to clean up at the framework layer; future
        # changes (custom networks, persistent volumes) attach cleanup logic
        # to ``handle.state`` and reverse it here.
        return None


def docker_version() -> str | None:
    """Return ``docker --version`` first line, or ``None`` if unavailable.

    Used by ``belt doctor`` to surface sandbox provider readiness without
    raising on hosts where docker is not installed.
    """
    if not _docker_available():
        return None
    try:
        result = subprocess.run(  # noqa: S603,S607 - fixed argv
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("docker --version failed: {}", e)
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip().split("\n", 1)[0] or None


# Used by tests to bypass real Docker invocations without touching subprocess
# at the agent layer. Public to the test suite only.
_DOCKER_AVAILABLE: Any = _docker_available


__all__ = ["DockerSandboxError", "DockerSandboxProvider", "docker_version"]
