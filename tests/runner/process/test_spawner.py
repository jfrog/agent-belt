# (c) JFrog Ltd. (2026)

"""SubprocessRunner tests.

LocalSpawner is the default spawner injected onto every agent at import
time, so any behavioural drift propagates to every agent. SandboxedSpawner
is the seam through which provider-policy reaches subprocess.Popen.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from belt.runner.process.spawner import LocalSpawner, SandboxedSpawner
from belt.runner.sandbox.base import SandboxContext
from belt.runner.sandbox.host import HostSandboxProvider
from belt.scenario import SandboxProfile


def test_local_spawner_calls_subprocess_popen_directly() -> None:
    # Use a real subprocess against the host python to prove end-to-end
    # plumbing works without mocking. ``-c "..."`` keeps it portable.
    spawner = LocalSpawner()
    proc = spawner.popen([sys.executable, "-c", "print('hi')"], stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(timeout=10)
    assert out.strip() == "hi"


def test_local_spawner_forwards_env_dict_unchanged() -> None:
    spawner = LocalSpawner()
    with patch("belt.runner.process.spawner.subprocess.Popen") as mock_popen:
        spawner.popen(["echo"], cwd="/tmp", env={"PATH": "/x"}, stdout=subprocess.PIPE)
    mock_popen.assert_called_once()
    kwargs = mock_popen.call_args.kwargs
    assert kwargs["cwd"] == "/tmp"
    assert kwargs["env"] == {"PATH": "/x"}
    # ``text``, ``stdout``, ``start_new_session`` are forwarded with their
    # documented defaults so an agent that omits them keeps current
    # behaviour. Defaults: text=False, start_new_session=False.
    assert kwargs["text"] is False
    assert kwargs["start_new_session"] is False


def test_sandboxed_spawner_invokes_provider_wrap_then_popen() -> None:
    provider = MagicMock()
    provider.wrap.return_value = (["wrapped", "argv"], None, {"X": "y"})
    handle = MagicMock()
    spawner = SandboxedSpawner(provider, handle)
    with patch("belt.runner.process.spawner.subprocess.Popen") as mock_popen:
        spawner.popen(["agent"], cwd="/tmp/work", env={"PATH": "/x"})
    provider.wrap.assert_called_once_with(handle, cmd=["agent"], cwd="/tmp/work", env={"PATH": "/x"})
    mock_popen.assert_called_once()
    args = mock_popen.call_args.args
    kwargs = mock_popen.call_args.kwargs
    # Wrapped triple flows verbatim into Popen.
    assert args[0] == ["wrapped", "argv"]
    assert kwargs["cwd"] is None
    assert kwargs["env"] == {"X": "y"}


def test_sandboxed_spawner_with_host_provider_is_behaviour_identical() -> None:
    # Belt-and-braces: a SandboxedSpawner wrapping HostSandboxProvider
    # must behave the same as a LocalSpawner. Without this guarantee the
    # orchestrator's branch-free injection would silently change behaviour
    # for ``provider: host`` scenarios.
    provider = HostSandboxProvider()
    handle = provider.setup(SandboxProfile(), SandboxContext(workspace_dir=Path("/tmp/work")))
    spawner = SandboxedSpawner(provider, handle)
    proc = spawner.popen([sys.executable, "-c", "print('hi')"], stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(timeout=10)
    assert out.strip() == "hi"
