# (c) JFrog Ltd. (2026)

"""Subprocess-spawning abstraction injected by the runner into agents.

Agents call ``self._spawner.popen(...)`` instead of ``subprocess.Popen(...)``
directly. The default implementation (:class:`LocalSpawner`) is a transparent
pass-through, so agents that never touch ``_spawner`` keep working unchanged.
The runner replaces it with a :class:`SandboxedSpawner` when the user opts
into ``--sandbox docker`` (or any third-party provider), and the agent never
notices.

This is the cross-tier seam between agents (CLI invocation + output parsing)
and the runner (process isolation policy). Agents stay sandbox-unaware;
runner owns sandboxing entirely.
"""

from belt.runner.process.spawner import LocalSpawner, SandboxedSpawner, SubprocessRunner

__all__ = ["LocalSpawner", "SandboxedSpawner", "SubprocessRunner"]
