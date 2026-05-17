# (c) JFrog Ltd. (2026)

"""Wrapped ``git`` subprocess helpers.

The single way belt shells out to ``git``. Concentrating these
calls means a uniform timeout policy, a single ``shutil.which("git")``
result feeding every caller, and one place to add the inevitable
"record exactly which git command we ran" instrumentation later.

Public API (deliberately small):

- :func:`git_available` - resolved path of the ``git`` binary or
  ``None``.
- :func:`run_git` - low-level. Returns a
  :class:`subprocess.CompletedProcess` (so the caller can branch on
  ``returncode`` / ``stderr``) or ``None`` if ``git`` is missing or
  the call timed out. Never raises.
- :func:`git_text` - convenience over :func:`run_git`. Returns stripped
  stdout on a clean exit, ``None`` on any failure (missing binary,
  timeout, non-zero exit, empty output). Used by best-effort metadata
  collectors (HEAD SHA, ``status --porcelain``) where the surrounding
  card field just degrades to ``null`` if anything goes wrong.

Non-goals: higher-level operations (worktree create/remove, init +
seed commit, diff capture). Those compose :func:`run_git` and live in
the workspace manager.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger

# Default per-call timeout for "metadata" git operations (rev-parse,
# status --porcelain). Worktree operations and init flows run longer
# and should pass an explicit ``timeout=`` to :func:`run_git`.
_DEFAULT_GIT_TIMEOUT = 5


def git_available() -> Optional[str]:
    """Return the resolved path of the ``git`` binary, or ``None``.

    Thin wrapper over :func:`shutil.which` kept here so callers can
    grep for a single canonical name.
    """
    return shutil.which("git")


def run_git(
    *args: str,
    cwd: Path | str | None = None,
    timeout: float = _DEFAULT_GIT_TIMEOUT,
    check: bool = False,
    extra_env: dict[str, str] | None = None,
) -> Optional[subprocess.CompletedProcess[str]]:
    """Run ``git <args>`` and return the completed process, or ``None``.

    Returns ``None`` if:

    - ``git`` is not on ``PATH`` (caller should fall back gracefully -
      the surrounding feature is best-effort);
    - the call exceeded ``timeout`` (caller treats as missing data).

    On non-zero exit codes the function returns the
    :class:`subprocess.CompletedProcess` so the caller can inspect
    ``stderr``; pass ``check=True`` to raise
    :class:`subprocess.CalledProcessError` instead (matching
    :func:`subprocess.run`).

    Always passes ``capture_output=True`` and ``text=True`` -
    everywhere we shell out to git we want stdout / stderr as strings.
    """
    git = git_available()
    if not git:
        return None
    env = None
    if extra_env is not None:
        import os

        env = {**os.environ, **extra_env}
    try:
        return subprocess.run(
            [git, *args],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.debug("git {} in {} timed out after {}s", " ".join(args), cwd, timeout)
        return None


def git_text(
    *args: str,
    cwd: Path | str | None = None,
    timeout: float = _DEFAULT_GIT_TIMEOUT,
) -> Optional[str]:
    """Convenience: return stripped stdout of ``git <args>``, or ``None``.

    ``None`` covers every failure mode (binary missing, timeout,
    non-zero exit, empty output) so callers do
    ``value = git_text(...)`` and treat ``None`` as "no data".
    Used by the benchmark-card collector and the per-fixture
    provenance capture - both of which are best-effort.
    """
    proc = run_git(*args, cwd=cwd, timeout=timeout)
    if proc is None or proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


__all__ = ["git_available", "git_text", "run_git"]
