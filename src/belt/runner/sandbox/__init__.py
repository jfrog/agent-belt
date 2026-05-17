# (c) JFrog Ltd. (2026)

"""Sandbox provider extension point.

Providers implement OS-level isolation for agent subprocesses. The framework
ships two: ``host`` (no-op, today's behaviour) and ``docker`` (container
isolation via the ``docker`` CLI). Third-party providers register through
the ``belt.sandbox_providers`` entry-point group; see ``SANDBOXING.md`` and
``PLUGGABILITY.md`` for the contract.
"""

from belt.runner.sandbox.base import BaseSandboxProvider, SandboxHandle
from belt.runner.sandbox.docker import DockerSandboxProvider
from belt.runner.sandbox.host import HostSandboxProvider
from belt.runner.sandbox.registry import available_sandbox_providers, get_sandbox_provider

__all__ = [
    "BaseSandboxProvider",
    "DockerSandboxProvider",
    "HostSandboxProvider",
    "SandboxHandle",
    "available_sandbox_providers",
    "get_sandbox_provider",
]
