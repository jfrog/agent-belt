# (c) JFrog Ltd. (2026)

"""Sandbox provider registry tests.

The registry is the single resolver consulted by ``commands/run.py``,
``commands/doctor.py``, and the orchestrator. A regression here cascades
everywhere a provider is selected by name.
"""

from __future__ import annotations

import pytest

from belt.runner.sandbox.docker import DockerSandboxProvider
from belt.runner.sandbox.host import HostSandboxProvider
from belt.runner.sandbox.registry import available_sandbox_providers, get_sandbox_provider


def test_builtins_available_under_canonical_names() -> None:
    names = available_sandbox_providers()
    assert "host" in names
    assert "docker" in names


def test_get_sandbox_provider_resolves_builtins() -> None:
    assert get_sandbox_provider("host") is HostSandboxProvider
    assert get_sandbox_provider("docker") is DockerSandboxProvider


def test_get_sandbox_provider_rejects_unknown_with_available_list() -> None:
    with pytest.raises(KeyError) as exc:
        get_sandbox_provider("nonexistent-provider-xyz")
    msg = str(exc.value)
    # The error names every available provider so the user knows what to
    # type instead of guessing.
    assert "nonexistent-provider-xyz" in msg
    assert "host" in msg
    assert "docker" in msg
