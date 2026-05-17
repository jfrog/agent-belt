# (c) JFrog Ltd. (2026)

"""Subprocess spawning interface and reference implementations.

Three implementations:

- :class:`SubprocessRunner` -- abstract; one method, ``popen``.
- :class:`LocalSpawner` -- forwards directly to :func:`subprocess.Popen`.
  Default for every agent until the runner overrides it. Behaviour-identical
  to a direct ``Popen`` call so agents under the local spawner are
  indistinguishable from their pre-sandbox behaviour.
- :class:`SandboxedSpawner` -- delegates to a :class:`BaseSandboxProvider`
  via its already-prepared :class:`SandboxHandle`. The provider's
  ``wrap()`` rewrites the agent's argv (and may rewrite cwd / env) so the
  resulting :class:`subprocess.Popen` call lands inside an OS-level sandbox.

The seam is intentionally narrow: agents pass exactly the same kwargs they
would pass to ``Popen``. Anything richer than that (TTYs, pipes, custom
streams) belongs in a future revision -- keep the surface tiny so the seam
stays compatible across plugin spawners.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from typing import IO, TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from belt.runner.sandbox.base import BaseSandboxProvider, SandboxHandle


class SubprocessRunner(ABC):
    """Abstract spawner injected onto each agent by the runner.

    Implementations forward to :func:`subprocess.Popen`, optionally rewriting
    the command (e.g. prepending ``docker run ...``) before spawn. Agents
    never branch on which spawner they have; the runner is the only caller
    that decides.
    """

    @abstractmethod
    def popen(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        stdout: int | IO[Any] | None = None,
        stderr: int | IO[Any] | None = None,
        text: bool = False,
        start_new_session: bool = False,
        **kwargs: Any,
    ) -> "subprocess.Popen[Any]":
        """Spawn ``cmd`` and return a :class:`subprocess.Popen`.

        Mirrors :func:`subprocess.Popen` for the kwargs agents actually use.
        Anything passed via ``**kwargs`` is forwarded verbatim to ``Popen`` by
        the local spawner; sandbox spawners may filter or rewrite it.
        """


class LocalSpawner(SubprocessRunner):
    """Pass-through spawner -- behaviour-identical to a direct ``Popen`` call."""

    def popen(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        stdout: int | IO[Any] | None = None,
        stderr: int | IO[Any] | None = None,
        text: bool = False,
        start_new_session: bool = False,
        **kwargs: Any,
    ) -> "subprocess.Popen[Any]":
        return subprocess.Popen(  # noqa: S603 - cmd is the agent's own argv
            cmd,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdout=stdout,
            stderr=stderr,
            text=text,
            start_new_session=start_new_session,
            **kwargs,
        )


class SandboxedSpawner(SubprocessRunner):
    """Spawner that wraps each ``cmd`` via a prepared sandbox handle.

    The provider's ``wrap()`` returns the rewritten argv (e.g. with a
    ``docker run --rm --cap-drop=ALL ...`` prefix). The original ``cwd`` is
    handled by the provider (a Docker provider sets ``-w /work`` after
    bind-mounting the host worktree); ``env`` is filtered down to the
    provider's allow-list before being passed in as ``-e KEY`` flags. Agents
    are unaware of any of this -- they call ``self._spawner.popen(cmd, ...)``
    with exactly the kwargs they would pass to a local spawn.
    """

    def __init__(self, provider: "BaseSandboxProvider", handle: "SandboxHandle") -> None:
        self._provider = provider
        self._handle = handle

    def popen(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        stdout: int | IO[Any] | None = None,
        stderr: int | IO[Any] | None = None,
        text: bool = False,
        start_new_session: bool = False,
        **kwargs: Any,
    ) -> "subprocess.Popen[Any]":
        wrapped_cmd, wrapped_cwd, wrapped_env = self._provider.wrap(
            self._handle,
            cmd=cmd,
            cwd=cwd,
            env=dict(env) if env is not None else {},
        )
        return subprocess.Popen(  # noqa: S603 - cmd is rewritten by trusted provider
            wrapped_cmd,
            cwd=wrapped_cwd,
            env=wrapped_env,
            stdout=stdout,
            stderr=stderr,
            text=text,
            start_new_session=start_new_session,
            **kwargs,
        )


__all__ = ["LocalSpawner", "SandboxedSpawner", "SubprocessRunner"]
