# (c) JFrog Ltd. (2026)

"""Tests for WorkspaceManager - git worktree-based scenario isolation."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from belt.runner.workspace import WorkspaceManager


def _init_git_repo(path: Path, files: dict[str, str] | None = None) -> None:
    """Initialize a git repo with an initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, capture_output=True, check=True)
    for name, content in (files or {"README.md": "# Test"}).items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(content)
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


class TestWorkspaceManagerInit:
    def test_auto_inits_non_git_directory(self, tmp_path: Path):
        non_git = tmp_path / "not-a-repo"
        non_git.mkdir()
        (non_git / "hello.txt").write_text("content")
        mgr = WorkspaceManager(non_git)
        assert (non_git / ".git").exists()
        assert mgr._auto_initialized is True
        mgr.cleanup_all()

    def test_accepts_valid_git_repo(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        mgr = WorkspaceManager(repo)
        assert mgr._base == repo.resolve()
        mgr.cleanup_all()


class TestAcquireRelease:
    def test_acquire_creates_worktree(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"src/main.py": "print('hello')", "README.md": "# Test"})
        mgr = WorkspaceManager(repo)

        ws = mgr.acquire("test_scenario")
        try:
            assert ws.is_dir()
            assert (ws / "src" / "main.py").read_text() == "print('hello')"
            assert (ws / "README.md").read_text() == "# Test"
        finally:
            mgr.cleanup_all()

    def test_release_removes_worktree(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        mgr = WorkspaceManager(repo)

        ws = mgr.acquire("scenario_a")
        assert ws.is_dir()
        mgr.release(ws)
        # Worktree directory should be removed
        assert not ws.exists()
        mgr.cleanup_all()

    def test_release_captures_diff(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"code.py": "def old(): pass"})
        mgr = WorkspaceManager(repo)

        ws = mgr.acquire("diff_test")
        (ws / "code.py").write_text("def new(): pass")

        git_diff, files_modified = mgr.release(ws)
        assert git_diff is not None
        assert "new" in git_diff
        assert "code.py" in files_modified
        mgr.cleanup_all()

    def test_release_captures_new_files(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        mgr = WorkspaceManager(repo)

        ws = mgr.acquire("new_file_test")
        (ws / "new_file.txt").write_text("hello")

        git_diff, files_modified = mgr.release(ws)
        assert "new_file.txt" in files_modified
        mgr.cleanup_all()

    def test_no_diff_when_unchanged(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        mgr = WorkspaceManager(repo)

        ws = mgr.acquire("clean_scenario")
        git_diff, files_modified = mgr.release(ws)
        assert git_diff is None
        assert files_modified == []
        mgr.cleanup_all()

    def test_acquire_with_ref(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"v1.txt": "version 1"})

        # Create a second commit
        (repo / "v2.txt").write_text("version 2")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "v2"], cwd=repo, capture_output=True, check=True)

        # Get the first commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD~1"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        first_commit = result.stdout.strip()

        mgr = WorkspaceManager(repo, ref=first_commit)
        ws = mgr.acquire("old_ref")
        try:
            assert (ws / "v1.txt").exists()
            assert not (ws / "v2.txt").exists()
        finally:
            mgr.cleanup_all()


class TestCapturedDiff:
    def test_capture_diff_without_release(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"file.py": "original"})
        mgr = WorkspaceManager(repo)

        ws = mgr.acquire("capture_test")
        (ws / "file.py").write_text("modified")

        git_diff, files_modified = mgr.capture_diff(ws)
        assert git_diff is not None
        assert "file.py" in files_modified
        # Worktree should still exist
        assert ws.is_dir()
        mgr.cleanup_all()

    def test_capture_staged_changes(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"staged.py": "old"})
        mgr = WorkspaceManager(repo)

        ws = mgr.acquire("staged_test")
        (ws / "staged.py").write_text("new content")
        subprocess.run(["git", "add", "staged.py"], cwd=ws, capture_output=True, check=True)

        git_diff, files_modified = mgr.capture_diff(ws)
        assert git_diff is not None
        assert "new content" in git_diff
        assert "staged.py" in files_modified
        mgr.cleanup_all()


class TestParallelWorkers:
    def test_parallel_worktrees_are_isolated(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"shared.txt": "original"})
        mgr = WorkspaceManager(repo)

        ws1 = mgr.acquire("worker_1")
        ws2 = mgr.acquire("worker_2")

        try:
            # Different paths
            assert ws1 != ws2

            # Modify independently
            (ws1 / "shared.txt").write_text("worker 1 edit")
            (ws2 / "shared.txt").write_text("worker 2 edit")

            assert (ws1 / "shared.txt").read_text() == "worker 1 edit"
            assert (ws2 / "shared.txt").read_text() == "worker 2 edit"

            # Original repo unchanged
            assert (repo / "shared.txt").read_text() == "original"
        finally:
            mgr.cleanup_all()

    def test_concurrent_acquire_release(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"data.txt": "base"})
        mgr = WorkspaceManager(repo)

        results: list[tuple[str | None, list[str]]] = []

        def _worker(idx: int) -> tuple[str | None, list[str]]:
            ws = mgr.acquire(f"parallel_{idx}")
            (ws / "data.txt").write_text(f"worker {idx}")
            return mgr.release(ws)

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(_worker, i) for i in range(4)]
                results = [f.result() for f in futures]

            for diff, files in results:
                assert diff is not None
                assert "data.txt" in files
        finally:
            mgr.cleanup_all()


class TestCleanup:
    def test_cleanup_all_removes_everything(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        mgr = WorkspaceManager(repo)

        ws1 = mgr.acquire("cleanup_1")
        ws2 = mgr.acquire("cleanup_2")
        assert ws1.is_dir()
        assert ws2.is_dir()

        mgr.cleanup_all()
        assert not ws1.exists()
        assert not ws2.exists()
        assert mgr._worktrees == []

    def test_cleanup_idempotent(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        mgr = WorkspaceManager(repo)

        mgr.acquire("idem_test")
        mgr.cleanup_all()
        mgr.cleanup_all()  # should not raise

    def test_cleanup_handles_already_removed(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        mgr = WorkspaceManager(repo)

        ws = mgr.acquire("manual_rm")
        import shutil

        shutil.rmtree(ws)

        mgr.cleanup_all()  # should not raise


class TestAutoInitDriftGate:
    """Refuse stale auto-initialised fixtures.

    The gate is narrowly scoped to repos ``belt`` itself initialised
    (identified by ``.git/belt-init-marker``). User-managed repos and
    ``fixture_repo`` clones must be unaffected so authors keep full
    ownership of their working tree.
    """

    def test_auto_init_writes_marker_sidecar(self, tmp_path: Path):
        non_git = tmp_path / "fixture"
        non_git.mkdir()
        (non_git / "hello.txt").write_text("v1")
        mgr = WorkspaceManager(non_git)
        try:
            marker = non_git / ".git" / "belt-init-marker"
            assert marker.exists(), "marker must be dropped so future runs can detect belt-owned repos"
            # Sanity: marker lives inside ``.git/`` so it never leaks into
            # worktree checkouts.
            ws = mgr.acquire("scn")
            try:
                assert not (ws / "belt-init-marker").exists()
            finally:
                mgr.release(ws)
        finally:
            mgr.cleanup_all()

    def test_drift_in_belt_owned_repo_is_refused(self, tmp_path: Path):
        from belt.runner.workspace import WorkspaceError

        non_git = tmp_path / "fixture"
        non_git.mkdir()
        (non_git / "hello.txt").write_text("v1")
        # First run: belt initialises and drops the marker.
        mgr1 = WorkspaceManager(non_git)
        mgr1.cleanup_all()

        # Author edits a fixture file outside git (the bug scenario).
        (non_git / "hello.txt").write_text("v2 - drifted")

        # Second run: must refuse with remediation hint rather than
        # silently running scenarios against the stale snapshot.
        with pytest.raises(WorkspaceError) as exc:
            WorkspaceManager(non_git)
        msg = str(exc.value)
        assert "uncommitted changes" in msg
        assert str(non_git) in msg
        assert "git add -A" in msg, "must point the author at the fix"

    def test_clean_belt_owned_repo_passes(self, tmp_path: Path):
        non_git = tmp_path / "fixture"
        non_git.mkdir()
        (non_git / "hello.txt").write_text("v1")
        WorkspaceManager(non_git).cleanup_all()
        # No drift on the second run -> gate is silent.
        mgr2 = WorkspaceManager(non_git)
        mgr2.cleanup_all()

    def test_user_managed_repo_is_not_gated(self, tmp_path: Path):
        """Pre-existing git repos (no marker) bypass the gate entirely.

        The author owns these and may legitimately have a dirty tree
        while iterating; belt must not lecture them.
        """
        repo = tmp_path / "user-repo"
        _init_git_repo(repo, {"code.py": "v1"})
        # User edits without committing - this is THEIR repo.
        (repo / "code.py").write_text("v2 in progress")
        # No exception: the marker is absent so the gate is inert.
        mgr = WorkspaceManager(repo)
        mgr.cleanup_all()

    def test_marker_lives_inside_git_dir(self, tmp_path: Path):
        """Marker must be inside ``.git/`` so ``git`` ignores it.

        Otherwise it would pollute every diff and trip the very drift
        gate we just installed.
        """
        non_git = tmp_path / "fixture"
        non_git.mkdir()
        (non_git / "hello.txt").write_text("v1")
        mgr = WorkspaceManager(non_git)
        try:
            assert (non_git / ".git" / "belt-init-marker").is_file()
            # A clean ``git status`` proves the marker isn't seen.
            r = subprocess.run(
                ["git", "status", "--porcelain"], cwd=non_git, capture_output=True, text=True, check=True
            )
            assert r.stdout.strip() == ""
        finally:
            mgr.cleanup_all()
