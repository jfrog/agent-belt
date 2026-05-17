# (c) JFrog Ltd. (2026)

"""Sandbox provider abstract base class.

A provider is the policy for "how do we execute this agent subprocess with
isolation?". The framework owns one method on the call chain:

    runner -> provider.validate_profile(profile, ctx)        # raises on misconfig
    runner -> provider.setup(profile, ctx) -> handle
    runner -> SandboxedSpawner(provider, handle) -> agent._spawner
    agent  -> self._spawner.popen(cmd, cwd=..., env=..., ...) -> Popen
                  └── internally: provider.wrap(handle, cmd, cwd, env) -> (cmd', cwd', env')
    runner -> provider.teardown(handle)

The base class is intentionally minimal -- four methods, one entity (one of
the four, ``validate_profile``, has a no-op default so providers that accept
every coherent profile do not need to implement it). Plugins that want
richer policy (e.g. Firecracker microVMs, gVisor, k8s Jobs) add it in their
own subclass without changing this signature.

The ``setup`` / ``teardown`` pair name matches the convention used by
:class:`belt.agent.base.BaseAgentAdapter` and by pytest / unittest fixtures,
so readers carry one mental model across the codebase.

``validate_profile`` is the security-critical extension point: a provider
that cannot enforce a particular field on :class:`SandboxProfile` (e.g.
``HostSandboxProvider`` cannot enforce ``network_policy='none'`` because
it has no isolation layer) overrides it to raise
:class:`SandboxConfigError`. The runner calls it before ``setup`` so the
scenario aborts with an actionable message instead of silently running
without the requested isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from belt.errors import BeltError
from belt.scenario import SandboxProfile


class SandboxConfigError(BeltError):
    """Raised when a sandbox profile asks for isolation the chosen provider cannot enforce.

    The runner catches this at the per-scenario boundary and surfaces a
    single actionable line: which scenario, which provider, which field,
    and what to change. Use it instead of a soft warning whenever the
    alternative is to silently run with weaker isolation than the
    profile declared -- a misleading "looks sandboxed" UX is exactly
    the failure mode this typed error exists to prevent.
    """


@dataclass
class SandboxContext:
    """Runtime info the provider needs to bind a sandbox to a scenario.

    Provided by the runner at ``setup()`` time; never reaches the agent.
    Carries the per-scenario worktree path (which becomes the only writable
    bind mount in container-style providers) plus the agent's declared
    required env-var names (so the provider can union them with the
    scenario-level ``SandboxProfile.env_passthrough`` allow-list).
    """

    workspace_dir: Path
    agent_required_env: frozenset[str] = field(default_factory=frozenset)
    scenario_name: str = ""


@dataclass
class SandboxHandle:
    """Opaque per-scenario state returned by :meth:`BaseSandboxProvider.setup`.

    The runner passes the same handle into each ``wrap()`` and the final
    ``teardown()``. The framework treats it as a black box; provider
    subclasses store whatever state they need (container ID, network name,
    bind-mount list, etc.).
    """

    profile: SandboxProfile
    context: SandboxContext
    state: dict[str, Any] = field(default_factory=dict)


class BaseSandboxProvider(ABC):
    """Policy interface for executing agent subprocesses inside a sandbox.

    Subclasses are loaded via the ``belt.sandbox_providers`` entry-point
    group (see ``runner/sandbox/registry.py``). Two ship with the framework:
    :class:`HostSandboxProvider` (no-op pass-through) and
    :class:`DockerSandboxProvider` (container isolation via the host
    ``docker`` CLI).
    """

    @classmethod
    def name(cls) -> str:
        """Short identifier used in ``--sandbox NAME`` / ``BELT_SANDBOX_PROVIDER``."""
        return cls.__name__.lower().replace("sandboxprovider", "")

    def validate_profile(self, profile: SandboxProfile, ctx: SandboxContext) -> None:
        """Raise :class:`SandboxConfigError` if this provider cannot enforce ``profile``.

        Default: accept everything. Providers that have an isolation layer
        capable of honouring every documented field on :class:`SandboxProfile`
        keep the default; providers that cannot enforce a specific field
        override and raise an actionable error naming the scenario, the
        field, and the fix.

        Called once per scenario, before :meth:`setup`. Must be cheap
        (no I/O, no subprocess) and side-effect-free: the runner calls
        it inside the per-scenario try/except, so an exception aborts
        only the offending scenario and leaves the rest of the run
        intact.
        """
        return None

    @abstractmethod
    def setup(self, profile: SandboxProfile, ctx: SandboxContext) -> SandboxHandle:
        """Set up per-scenario sandbox state.

        Called once per scenario, before any agent subprocess spawns. The
        provider may build a container, allocate a network, set up bind
        mounts -- whatever it needs to be ready for ``wrap()`` calls. Must
        return a :class:`SandboxHandle` carrying the profile, context, and
        any provider-private state.

        Pairs with :meth:`teardown`; the runner always calls ``teardown``
        from a ``finally`` block so a partial setup still gets cleaned up.

        Raises a typed framework error (subclass of
        :class:`belt.errors.BeltError`) on misconfiguration so the runner
        can surface a single actionable line.
        """

    @abstractmethod
    def wrap(
        self,
        handle: SandboxHandle,
        *,
        cmd: list[str],
        cwd: str | None,
        env: dict[str, str],
    ) -> tuple[list[str], str | None, dict[str, str]]:
        """Rewrite ``(cmd, cwd, env)`` so :func:`subprocess.Popen` lands inside the sandbox.

        Pure-policy: no side effects. Called once per agent subprocess spawn
        (i.e. once per :meth:`BaseAgentAdapter.execute` for streaming agents,
        more for agents that fan out). The provider returns the rewritten
        triple; the spawner forwards it to ``Popen`` verbatim.

        The local provider returns the inputs unchanged. The docker provider
        prepends ``docker run --rm --cap-drop=ALL --read-only -v <ws>:/work
        -w /work -e KEY1 -e KEY2 <image>`` and clears the host-side ``cwd``
        (the container has its own working directory). ``env`` is filtered
        down to the union of ``ctx.agent_required_env`` and
        ``profile.env_passthrough`` so secrets the scenario did not opt
        into never reach the container.
        """

    @abstractmethod
    def teardown(self, handle: SandboxHandle) -> None:
        """Release per-scenario sandbox resources.

        Idempotent: safe to call twice (the runner does call it from a
        ``finally`` block). Errors are logged and swallowed so a teardown
        bug never masks a real scenario failure.
        """


__all__ = [
    "BaseSandboxProvider",
    "SandboxConfigError",
    "SandboxContext",
    "SandboxHandle",
]
