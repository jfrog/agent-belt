# (c) JFrog Ltd. (2026)

"""Tests for ``GroupConfig.fixture_repo`` and ``GroupConfig.resources``.

These exercise the workspace pieces that absorb skill-eval-style
workflows: clone a foreign repo as the workspace base, then install
versioned resource payloads into each per-scenario worktree before the
agent runs.
"""

from __future__ import annotations

import hashlib
import tarfile
import zipfile
from pathlib import Path

import pytest

from belt._git import git_available, run_git
from belt.runner.workspace import (
    WorkspaceError,
    WorkspaceManager,
    clone_fixture_repo,
    install_resources,
    resolve_local_resource_source,
)
from belt.scenario import GroupConfig, Resource


def _init_git_repo(path: Path, file_name: str = "README.md", content: str = "hello") -> None:
    path.mkdir(parents=True, exist_ok=True)
    run_git("init", "-q", cwd=path, timeout=30, check=True)
    (path / file_name).write_text(content)
    run_git("add", "-A", cwd=path, timeout=30, check=True)
    run_git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=t@e",
        "commit",
        "-q",
        "-m",
        "initial",
        cwd=path,
        timeout=30,
        check=True,
    )


pytestmark = pytest.mark.skipif(not git_available(), reason="git not on PATH")


# clone_fixture_repo


def test_clone_fixture_repo_clones_local_repo(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dest = tmp_path / "cache" / "out"
    _init_git_repo(src)
    clone_fixture_repo(str(src), "HEAD", dest)
    assert (dest / "README.md").read_text() == "hello"


def test_clone_fixture_repo_rejects_unsafe_ref(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dest = tmp_path / "out"
    _init_git_repo(src)
    with pytest.raises(WorkspaceError):
        clone_fixture_repo(str(src), "../../etc/passwd", dest)


def test_clone_fixture_repo_rejects_existing_destination(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dest = tmp_path / "out"
    _init_git_repo(src)
    dest.mkdir()
    with pytest.raises(WorkspaceError):
        clone_fixture_repo(str(src), "HEAD", dest)


def test_clone_fixture_repo_checks_out_named_ref(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dest = tmp_path / "out"
    _init_git_repo(src, "first.txt", "v1")
    run_git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=t@e",
        "checkout",
        "-q",
        "-b",
        "feature",
        cwd=src,
        timeout=10,
        check=True,
    )
    (src / "second.txt").write_text("v2")
    run_git("add", "-A", cwd=src, timeout=10, check=True)
    run_git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=t@e",
        "commit",
        "-q",
        "-m",
        "v2",
        cwd=src,
        timeout=10,
        check=True,
    )
    clone_fixture_repo(str(src), "feature", dest)
    assert (dest / "second.txt").read_text() == "v2"


# install_resources


def test_install_resources_copies_local_files(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    src = tmp_path / "skill.md"
    src.write_text("skill body")
    locks = install_resources(
        workspace,
        [Resource(kind="file", source=str(src), dest=".skills/skill.md", version="0.1.0")],
    )
    assert (workspace / ".skills" / "skill.md").read_text() == "skill body"
    assert locks[0]["version"] == "0.1.0"
    assert locks[0]["source_sha256"] == hashlib.sha256(b"skill body").hexdigest()
    assert locks[0]["name"] == "skill.md"


def test_install_resources_copies_directories(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    src_dir = tmp_path / "skill"
    (src_dir / "inner").mkdir(parents=True)
    (src_dir / "SKILL.md").write_text("hi")
    (src_dir / "inner" / "ref.md").write_text("ref")
    install_resources(
        workspace,
        [Resource(kind="file", source=str(src_dir), dest=".skills/skill")],
    )
    assert (workspace / ".skills" / "skill" / "SKILL.md").read_text() == "hi"
    assert (workspace / ".skills" / "skill" / "inner" / "ref.md").read_text() == "ref"


def test_install_resources_preserves_symlinks_in_directory(tmp_path: Path) -> None:
    """A symlink inside a ``kind: file`` directory source must NOT be
    dereferenced. ``shutil.copytree`` defaults to following links, which
    would let a hostile (or accidentally symlinked) source tree
    exfiltrate the link target's contents into the workspace.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("S3CRET")

    src_dir = tmp_path / "skill"
    src_dir.mkdir()
    (src_dir / "SKILL.md").write_text("body")
    (src_dir / "leak").symlink_to(secret)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    install_resources(
        workspace,
        [Resource(kind="file", source=str(src_dir), dest=".skills/skill")],
    )

    leak = workspace / ".skills" / "skill" / "leak"
    assert leak.is_symlink(), "symlink must be preserved, not followed"
    # The link target's contents must not be present in the workspace as
    # a regular file alongside the symlink.
    assert not (workspace / ".skills" / "skill" / "secret.txt").exists()


def test_install_resources_preserves_symlinked_file_source(tmp_path: Path) -> None:
    """A ``kind: file`` source that *is* a symlink must land in the
    workspace as a symlink, not as a copy of the link target.
    ``shutil.copy2`` defaults to ``follow_symlinks=True``.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("S3CRET")
    link = tmp_path / "link.txt"
    link.symlink_to(secret)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    install_resources(
        workspace,
        [Resource(kind="file", source=str(link), dest=".skills/link.txt")],
    )

    dest = workspace / ".skills" / "link.txt"
    assert dest.is_symlink(), "file-source symlink must be preserved, not dereferenced"


def test_install_resources_extracts_zip(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("SKILL.md", "zip body")
        zf.writestr("nested/file.txt", "nested")
    install_resources(workspace, [Resource(kind="archive", source=str(archive), dest=".skills/bundle")])
    assert (workspace / ".skills" / "bundle" / "SKILL.md").read_text() == "zip body"
    assert (workspace / ".skills" / "bundle" / "nested" / "file.txt").read_text() == "nested"


def test_install_resources_extracts_tar_gz(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "f.txt").write_text("tarred")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src_dir / "f.txt", arcname="f.txt")
    install_resources(workspace, [Resource(kind="archive", source=str(archive), dest="bundle")])
    assert (workspace / "bundle" / "f.txt").read_text() == "tarred"


def test_install_resources_rejects_path_traversal_dest(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    src = tmp_path / "f.txt"
    src.write_text("x")
    with pytest.raises(WorkspaceError):
        install_resources(workspace, [Resource(kind="file", source=str(src), dest="../escape.txt")])


def test_install_resources_rejects_absolute_dest(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    src = tmp_path / "f.txt"
    src.write_text("x")
    with pytest.raises(WorkspaceError):
        install_resources(workspace, [Resource(kind="file", source=str(src), dest="/etc/passwd")])


def test_install_resources_rejects_unknown_kind(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    src = tmp_path / "f.txt"
    src.write_text("x")
    with pytest.raises(WorkspaceError):
        install_resources(workspace, [Resource(kind="symlink", source=str(src), dest="f.txt")])


# ──────────────────────────────────────────────────────────────────────────
# resolve_local_resource_source + install_resources(config_dir=)
#
# Bare local ``resources[].source`` paths anchor against the scenario's
# ``_config.json`` directory, not process CWD. Without this, the bundled
# showcase only worked when the user happened to ``cd`` into the scenario
# directory. Pin the behaviour from multiple angles so a future "cleanup"
# cannot quietly re-break it.
# ──────────────────────────────────────────────────────────────────────────


def test_resolve_local_resource_source_anchors_relative_path_to_config_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / "scenarios" / "g"
    config_dir.mkdir(parents=True)
    target = tmp_path / "shared" / "skill.md"
    target.parent.mkdir()
    target.write_text("hi")
    resolved = resolve_local_resource_source("../../shared/skill.md", config_dir)
    assert Path(resolved) == target.resolve()


def test_resolve_local_resource_source_passes_through_url_schemes(tmp_path: Path) -> None:
    for url in (
        "https://example.com/skill.zip",
        "http://example.com/skill.zip",
        "file:///abs/path/skill.zip",
    ):
        assert resolve_local_resource_source(url, tmp_path) == url


def test_resolve_local_resource_source_passes_through_ssh_shortcut(tmp_path: Path) -> None:
    assert resolve_local_resource_source("git@example.com:user/repo.git", tmp_path) == "git@example.com:user/repo.git"


def test_resolve_local_resource_source_keeps_absolute_paths(tmp_path: Path) -> None:
    abs_src = tmp_path / "abs.txt"
    abs_src.write_text("x")
    resolved = resolve_local_resource_source(str(abs_src), tmp_path / "elsewhere")
    assert Path(resolved) == abs_src


def test_install_resources_resolves_relative_source_against_config_dir(tmp_path: Path) -> None:
    """Repro of the bundled-showcase failure on `main`.

    Before the fix, ``Path("../../shared/skill.md")`` was resolved against
    process CWD inside ``_fetch_source``. Running from the repo root
    therefore failed even when the file existed at a stable scenario-
    relative location -- exactly what the showcase scenario writes.
    """
    config_dir = tmp_path / "scenarios" / "g"
    config_dir.mkdir(parents=True)
    target = tmp_path / "shared" / "skill.md"
    target.parent.mkdir()
    target.write_text("skill body")

    workspace = tmp_path / "ws"
    workspace.mkdir()

    locks = install_resources(
        workspace,
        [
            Resource(
                kind="file",
                source="../../shared/skill.md",
                dest=".skills/skill.md",
                version="0.1.0",
            )
        ],
        config_dir=config_dir,
    )
    assert (workspace / ".skills" / "skill.md").read_text() == "skill body"
    assert locks[0]["source"] == "../../shared/skill.md"  # lock records the *original* string
    assert locks[0]["source_sha256"] == hashlib.sha256(b"skill body").hexdigest()


def test_install_resources_keeps_absolute_source_when_config_dir_passed(tmp_path: Path) -> None:
    config_dir = tmp_path / "scenarios" / "g"
    config_dir.mkdir(parents=True)
    abs_src = tmp_path / "absolute.txt"
    abs_src.write_text("body")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    install_resources(
        workspace,
        [Resource(kind="file", source=str(abs_src), dest="abs.txt")],
        config_dir=config_dir,
    )
    assert (workspace / "abs.txt").read_text() == "body"


def test_install_resources_rejects_zip_with_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "evil")
    with pytest.raises(WorkspaceError):
        install_resources(workspace, [Resource(kind="archive", source=str(archive), dest="bundle")])


def test_install_resources_rejects_unsupported_scheme(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(WorkspaceError):
        install_resources(workspace, [Resource(kind="file", source="ftp://example.com/x", dest="x.txt")])


# End-to-end: WorkspaceManager + clone + install


def test_worktree_acquire_against_cloned_fixture_with_resources(tmp_path: Path) -> None:
    src = tmp_path / "src"
    cache = tmp_path / "cache" / "demo"
    _init_git_repo(src)
    clone_fixture_repo(str(src), "HEAD", cache)

    skill = tmp_path / "skill.md"
    skill.write_text("install me")

    mgr = WorkspaceManager(cache, ref="HEAD")
    try:
        worktree = mgr.acquire("scenario-1")
        try:
            install_resources(
                worktree,
                [Resource(kind="file", source=str(skill), dest=".skills/skill.md", version="0.1.0")],
            )
            assert (worktree / "README.md").read_text() == "hello"
            assert (worktree / ".skills" / "skill.md").read_text() == "install me"
        finally:
            mgr.release(worktree)
    finally:
        mgr.cleanup_all()


def test_group_config_accepts_both_fixture_repo_and_working_dir() -> None:
    """The orchestrator phase enforces mutual exclusion. Field-level the
    schema allows both for forward compatibility (e.g. validators that
    want to log a warning instead of a hard failure)."""
    gc = GroupConfig(agent="cursor", fixture_repo="https://example/x.git", working_dir="../ws")
    assert gc.fixture_repo and gc.working_dir


# End-to-end via the setup_groups orchestrator hook


def test_setup_groups_clones_fixture_repo_into_run_dir(tmp_path: Path) -> None:
    """``_prepare_group_fixtures`` populates ``ctx.group_fixtures`` for groups with ``fixture_repo``."""
    from argparse import Namespace

    from belt.progress import RunnerProgress
    from belt.runner.context import MatchedGroup, RunContext
    from belt.runner.phases.setup_groups import _prepare_group_fixtures

    src = tmp_path / "src"
    _init_git_repo(src)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    group_dir = tmp_path / "groups" / "demo"
    group_dir.mkdir(parents=True)
    gc = GroupConfig(agent="cursor", fixture_repo=str(src), fixture_ref="HEAD")

    ctx = RunContext(
        args=Namespace(),
        scenarios_root=tmp_path,
        matched_groups=[MatchedGroup(group_dir=group_dir, config=gc, scenarios=[], name="demo")],
        agent_args={},
        outcomes_root=tmp_path,
        run_dir=run_dir,
        workspace=tmp_path,
        progress=RunnerProgress(plain=True),
    )

    _prepare_group_fixtures(ctx)

    assert "demo" in ctx.group_fixtures
    assert ctx.group_fixtures["demo"].is_dir()
    assert (ctx.group_fixtures["demo"] / "README.md").read_text() == "hello"


def _captured_source_via_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture_repo: str,
) -> str:
    """Run ``_prepare_group_fixtures`` with a stub ``clone_fixture_repo`` and
    return the ``source`` value the runner would have passed to git.

    Stubbing isolates the resolution logic from real git execution: we
    only care which string reaches ``clone_fixture_repo``. The real-clone
    path stays covered by ``test_setup_groups_clones_fixture_repo_into_run_dir``.
    """
    from argparse import Namespace

    from belt.progress import RunnerProgress
    from belt.runner.context import MatchedGroup, RunContext
    from belt.runner.phases import setup_groups as sg

    captured: dict[str, str] = {}

    def _stub(source: str, ref: str, dest: Path) -> None:
        captured["source"] = source
        dest.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(sg, "clone_fixture_repo", _stub)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    group_dir = tmp_path / "groups" / "demo"
    group_dir.mkdir(parents=True)
    gc = GroupConfig(agent="cursor", fixture_repo=fixture_repo, fixture_ref="HEAD")
    ctx = RunContext(
        args=Namespace(),
        scenarios_root=tmp_path,
        matched_groups=[MatchedGroup(group_dir=group_dir, config=gc, scenarios=[], name="demo")],
        agent_args={},
        outcomes_root=tmp_path,
        run_dir=run_dir,
        workspace=tmp_path,
        progress=RunnerProgress(plain=True),
    )
    sg._prepare_group_fixtures(ctx)
    assert "demo" not in ctx.failed_groups, f"unexpected setup failure for {fixture_repo!r}"
    return captured["source"]


def test_fixture_repo_relative_path_resolves_against_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare ``../foo`` is resolved against the process CWD before reaching git.

    Locks in the fix for the bug: ``git clone`` runs with ``cwd`` pointing at
    the per-run cache, so a bare relative path used to silently fail.
    """
    sibling = tmp_path / "fixture"
    sibling.mkdir()
    cwd = tmp_path / "scenarios"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    captured = _captured_source_via_prepare(monkeypatch, tmp_path, "../fixture")
    assert captured == str(sibling.resolve())


def test_fixture_repo_dot_relative_path_resolves_against_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``./fixture`` resolves to ``<cwd>/fixture`` (absolute, normalised)."""
    target = tmp_path / "fixture"
    target.mkdir()
    monkeypatch.chdir(tmp_path)
    captured = _captured_source_via_prepare(monkeypatch, tmp_path, "./fixture")
    assert captured == str(target.resolve())


def test_fixture_repo_absolute_path_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An absolute local path is preserved (resolve() is a no-op for symlink-free abs paths)."""
    target = tmp_path / "fixture"
    target.mkdir()
    captured = _captured_source_via_prepare(monkeypatch, tmp_path, str(target))
    assert captured == str(target.resolve())


def test_fixture_repo_https_url_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``https://`` URLs reach git byte-for-byte - no path resolution attempted."""
    url = "https://github.com/jfrog/agent-belt.git"
    captured = _captured_source_via_prepare(monkeypatch, tmp_path, url)
    assert captured == url


def test_fixture_repo_ssh_shortcut_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCP-style ``git@host:path`` shortcuts pass through (urlparse reports no scheme)."""
    shortcut = "git@github.com:jfrog/belt.git"
    captured = _captured_source_via_prepare(monkeypatch, tmp_path, shortcut)
    assert captured == shortcut


def test_fixture_repo_file_url_passes_through_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``file:///abs/path`` URLs reach git verbatim.

    RFC 8089 reserves ``file://`` for absolute host paths; the user wrote a
    URL on purpose, so we do not second-guess them by resolving its inside
    against CWD. Locks the pass-through behaviour against future "fixes".
    """
    url = "file:///opt/fixtures/repo"
    captured = _captured_source_via_prepare(monkeypatch, tmp_path, url)
    assert captured == url


def test_fixture_repo_relative_path_with_dotdot_traverses_intentionally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``..`` segments in a bare local ``fixture_repo`` traverse by design.

    Same threat model as ``working_dir`` (which also uses ``Path(...).resolve()``):
    the scenario author is trusted, clone access is bounded by what the
    running user can read, and the schema documents the field as a "git URL
    or local path". A future "security hardening" that blocks ``..`` would
    reintroduce the very bug the resolver fixes - this test pins the
    intent so that decision stays explicit.
    """
    deep = tmp_path / "a" / "b" / "scenarios"
    deep.mkdir(parents=True)
    target = tmp_path / "outside"
    target.mkdir()
    monkeypatch.chdir(deep)
    captured = _captured_source_via_prepare(monkeypatch, tmp_path, "../../../outside")
    assert captured == str(target.resolve())


def test_setup_groups_rejects_fixture_repo_with_working_dir(tmp_path: Path) -> None:
    from argparse import Namespace

    from belt.progress import RunnerProgress
    from belt.runner.context import MatchedGroup, RunContext
    from belt.runner.phases.setup_groups import _prepare_group_fixtures

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    group_dir = tmp_path / "groups" / "demo"
    group_dir.mkdir(parents=True)
    gc = GroupConfig(agent="cursor", fixture_repo="https://example/x.git", working_dir="../ws")
    ctx = RunContext(
        args=Namespace(),
        scenarios_root=tmp_path,
        matched_groups=[MatchedGroup(group_dir=group_dir, config=gc, scenarios=[], name="demo")],
        agent_args={},
        outcomes_root=tmp_path,
        run_dir=run_dir,
        workspace=tmp_path,
        progress=RunnerProgress(plain=True),
    )
    _prepare_group_fixtures(ctx)
    assert "demo" in ctx.failed_groups
