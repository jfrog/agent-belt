# (c) JFrog Ltd. (2026)

"""Unit tests for :mod:`belt._git`.

Pinning behaviour: every git syscall in the project (worktree manager,
benchmark-card collector, per-fixture provenance, run-time diff capture)
goes through :func:`run_git` / :func:`git_text`. A drift back to inline
``subprocess.run(["git", ...])`` would lose the unified timeout policy
and the "best-effort returns ``None``" contract these tests assert.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from belt._git import git_available, git_text, run_git


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with one commit so the helpers have something to read."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--allow-empty", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


class TestGitAvailable:
    def test_returns_path_when_git_installed(self):
        # Pre-condition: every dev/CI environment has git on PATH. If
        # this assertion ever fails we have a far bigger problem than a
        # unit test - the project doesn't ship a fallback.
        path = git_available()
        assert path is not None
        assert "git" in path

    def test_returns_none_when_path_lacks_git(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PATH", "")
        assert git_available() is None


class TestRunGit:
    def test_returns_completed_process_on_success(self, repo: Path):
        result = run_git("rev-parse", "HEAD", cwd=repo)
        assert result is not None
        assert result.returncode == 0
        assert len(result.stdout.strip()) == 40  # SHA-1

    def test_returns_completed_process_on_nonzero_exit(self, repo: Path):
        # Caller can inspect stderr; non-zero is *not* the same as None.
        result = run_git("no-such-subcommand", cwd=repo)
        assert result is not None
        assert result.returncode != 0

    def test_returns_none_when_git_not_on_path(self, repo: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PATH", "")
        assert run_git("rev-parse", "HEAD", cwd=repo) is None

    def test_returns_none_on_timeout(self, repo: Path, monkeypatch: pytest.MonkeyPatch):
        # We don't want a flaky real-process timeout - patch
        # :func:`subprocess.run` to raise :class:`TimeoutExpired` directly.
        from belt import _git as _git_mod

        def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise subprocess.TimeoutExpired(cmd=args[0] if args else [], timeout=kwargs.get("timeout", 0))

        monkeypatch.setattr(_git_mod.subprocess, "run", _raise)
        assert run_git("rev-parse", "HEAD", cwd=repo, timeout=5) is None

    def test_check_true_raises_called_process_error(self, repo: Path):
        with pytest.raises(subprocess.CalledProcessError):
            run_git("no-such-subcommand", cwd=repo, check=True)

    def test_extra_env_overrides_inherited_value(self, repo: Path):
        # ``GIT_AUTHOR_NAME`` is honoured by ``commit``; verify the
        # extra_env path makes it through.
        result = run_git(
            "-c",
            "user.email=test@test",
            "commit",
            "--allow-empty",
            "-m",
            "x",
            cwd=repo,
            extra_env={
                "GIT_AUTHOR_NAME": "TestAuthor",
                "GIT_AUTHOR_EMAIL": "a@a",
                "GIT_COMMITTER_NAME": "T",
                "GIT_COMMITTER_EMAIL": "t@t",
            },
        )
        assert result is not None and result.returncode == 0
        log = run_git("log", "-1", "--format=%an", cwd=repo)
        assert log is not None and log.stdout.strip() == "TestAuthor"


class TestGitText:
    def test_returns_stripped_stdout_on_success(self, repo: Path):
        sha = git_text("rev-parse", "HEAD", cwd=repo)
        assert sha is not None
        assert len(sha) == 40
        assert "\n" not in sha

    def test_returns_none_on_empty_stdout(self, repo: Path):
        # ``git status --porcelain`` on a clean repo emits nothing;
        # collapse to None matches the "no data" contract used by the
        # benchmark-card collector.
        assert git_text("status", "--porcelain", cwd=repo) is None

    def test_returns_none_on_nonzero_exit(self, repo: Path):
        assert git_text("no-such-subcommand", cwd=repo) is None

    def test_returns_none_when_git_not_on_path(self, repo: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PATH", "")
        assert git_text("rev-parse", "HEAD", cwd=repo) is None
