# (c) JFrog Ltd. (2026)

"""Host sandbox provider -- no-op pass-through.

The default. Behaviour-identical to running an agent without any sandbox
machinery: ``setup`` returns an empty handle, ``wrap`` returns its inputs
unchanged, ``teardown`` does nothing. Useful as a sentinel so the rest of
the runner can treat "no sandbox" and "sandbox X" through one code path.

Named ``host`` (not ``local``, ``none``, or ``off``) because the name
should describe what the user gets: the agent runs on the host, with the
host's user, the host's filesystem, the host's network, and the host's
secrets. Anyone reading ``--sandbox host`` should immediately recognise
that no isolation is in effect.
"""

from __future__ import annotations

from belt.runner.sandbox.base import BaseSandboxProvider, SandboxConfigError, SandboxContext, SandboxHandle
from belt.scenario import SandboxProfile


class HostSandboxProvider(BaseSandboxProvider):
    """No-op provider: agent runs on the host with the invoking user's privileges.

    Documented honestly: this provides zero kernel-enforced isolation. It
    exists so ``--sandbox host`` is a real choice the user makes (rather
    than a hidden default that would mask the threat model). For real
    isolation use ``--sandbox docker`` and see ``SANDBOXING.md``.
    """

    @classmethod
    def name(cls) -> str:
        return "host"

    def validate_profile(self, profile: SandboxProfile, ctx: SandboxContext) -> None:
        # The host provider has no isolation layer: it spawns the agent as
        # the invoking user, on the host kernel, with the host network. Any
        # ``SandboxProfile`` field whose semantics require a sandbox to
        # enforce is rejected here rather than silently ignored, so a
        # scenario authored as ``provider=host, network_policy=none`` cannot
        # silently downgrade to "running with full network" the way it
        # would with a soft warning. Other isolation-bearing fields
        # (``writable_paths``, ``env_passthrough``, ``allowed_hosts``)
        # default to "no extra isolation" already; only ``network_policy``
        # has a non-default value that demands kernel enforcement, so it
        # is the only field this method needs to gate today. New fields
        # added to ``SandboxProfile`` that require enforcement must be
        # added to the rejection list below.
        if profile.network_policy != "open":
            raise SandboxConfigError(
                f"sandbox profile for scenario '{ctx.scenario_name}' declares "
                f"network_policy={profile.network_policy!r} but the chosen provider is "
                "'host', which has no isolation layer and cannot enforce any network "
                "policy. Either re-run with --sandbox docker (which honours "
                "network_policy via --network=none, kernel-enforced), or remove "
                "network_policy from the scenario group's _config.json."
            )

    def setup(self, profile: SandboxProfile, ctx: SandboxContext) -> SandboxHandle:
        return SandboxHandle(profile=profile, context=ctx, state={})

    def wrap(
        self,
        handle: SandboxHandle,
        *,
        cmd: list[str],
        cwd: str | None,
        env: dict[str, str],
    ) -> tuple[list[str], str | None, dict[str, str]]:
        return cmd, cwd, env

    def teardown(self, handle: SandboxHandle) -> None:  # pragma: no cover - trivial
        return None


__all__ = ["HostSandboxProvider"]
