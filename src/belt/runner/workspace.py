# (c) JFrog Ltd. (2026)

"""Workspace isolation via git worktrees.

Provides per-scenario isolated working directories for code-editing scenarios.
Each scenario gets its own git worktree checked out at a configurable ref,
ensuring edits from one scenario never leak into another - even under
parallel execution with --workers N.

Usage by the orchestrator:
    mgr = WorkspaceManager(base_repo, ref="HEAD")
    ws = mgr.acquire("group__scenario_name")
    # ... run scenario with cwd=ws ...
    diff, files = mgr.release(ws)
    mgr.cleanup_all()  # safety net in finally/atexit
"""

from __future__ import annotations

import atexit
import hashlib
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from loguru import logger

from belt._git import git_available, run_git
from belt.scenario import Resource

_init_locks: dict[Path, threading.Lock] = {}
_init_locks_guard = threading.Lock()

# Sidecar file dropped inside ``.git/`` for repos that ``belt`` itself
# auto-initialised. Used by ``_check_auto_init_drift`` to scope the drift
# gate to ``belt``-owned fixtures and leave user-managed repos /
# ``fixture_repo`` clones untouched. Lives inside ``.git/`` so it never
# leaks into worktree checkouts.
_AUTO_INIT_MARKER = "belt-init-marker"

# `git worktree add` writes to `.git/worktrees/<name>/{HEAD, commondir, gitdir}`
# without locking. Two concurrent invocations against the same base repo can
# race on metadata creation, leaving a half-written `commondir` that the next
# read sees as `fatal: failed to read .git/worktrees/<name>/commondir: Success`.
# Serialise the subprocess call per base-repo path. Keyed by repo (not global)
# so unrelated WorkspaceManagers do not block each other.
_worktree_add_locks: dict[Path, threading.Lock] = {}
_worktree_add_locks_guard = threading.Lock()

_SAFE_REF_RE = re.compile(r"^[a-zA-Z0-9_./@\-]{1,200}$")
_DOTDOT_RE = re.compile(r"(^|/)\.\.(/|$)")
_LABEL_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]+")
# Matches the SCP-style ``user@host:path`` shortcut that ``git clone``
# treats as remote SSH. ``urlparse`` reports an empty scheme for these,
# so we need an explicit detector to keep them out of CWD-resolution.
_SSH_SHORTCUT_RE = re.compile(r"^[\w.-]+@[\w.-]+:")
_FETCH_TIMEOUT_SECONDS = 60
# 256 MiB cap on remote-fetched resource payloads. A hostile mirror that
# never closes the connection would otherwise fill the runner's disk.
_FETCH_MAX_BYTES = 256 * 1024 * 1024


class WorkspaceError(Exception):
    """Raised when workspace operations fail."""


def clone_fixture_repo(source: str, ref: str, dest: Path) -> None:
    """Clone ``source`` (URL or local path) into ``dest`` and check out ``ref``.

    Designed to be called once per group from :func:`setup_groups` so the
    cost is amortised across all scenarios in the group. ``dest`` must not
    already exist; the caller owns cache management. Refs are validated
    through the same allow-list as :class:`WorkspaceManager` so an
    attacker-controlled scenario file cannot smuggle ``;`` or ``--`` into
    ``git checkout``.
    """
    if not git_available():
        raise WorkspaceError("git not found on PATH")
    if not _SAFE_REF_RE.match(ref) or _DOTDOT_RE.search(ref):
        raise WorkspaceError(f"Invalid fixture_ref: {ref!r}. Use a commit SHA, tag, or branch name.")
    if dest.exists():
        raise WorkspaceError(f"fixture clone destination already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning fixture {} -> {}", source, dest)
    result = run_git("clone", "--quiet", source, str(dest), cwd=dest.parent, timeout=300)
    if result is None:
        raise WorkspaceError(f"git clone of {source} timed out")
    if result.returncode != 0:
        raise WorkspaceError(f"git clone of {source} failed: {result.stderr.strip()}")
    if ref != "HEAD":
        result = run_git("checkout", "--quiet", ref, cwd=dest, timeout=60)
        if result is None or result.returncode != 0:
            detail = "timeout" if result is None else result.stderr.strip()
            raise WorkspaceError(f"git checkout {ref} in cloned fixture failed: {detail}")


def _safe_dest_in(workspace: Path, dest: str) -> Path:
    """Resolve ``dest`` inside ``workspace`` and reject path-traversal escapes."""
    if not dest:
        raise WorkspaceError("Resource dest must be a non-empty path")
    if Path(dest).is_absolute():
        raise WorkspaceError(f"Resource dest must be relative: {dest!r}")
    if _DOTDOT_RE.search(dest):
        raise WorkspaceError(f"Resource dest contains parent-traversal segment: {dest!r}")
    full = (workspace / dest).resolve()
    try:
        full.relative_to(workspace.resolve())
    except ValueError as e:
        raise WorkspaceError(f"Resource dest escapes worktree: {dest!r}") from e
    return full


def _sha256_of_path(path: Path) -> str:
    """Return SHA-256 of a file or, for directories, a sorted manifest of file digests."""
    hasher = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(64 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    if path.is_dir():
        for entry in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = entry.relative_to(path).as_posix().encode("utf-8")
            hasher.update(rel + b"\x00")
            with entry.open("rb") as fh:
                for chunk in iter(lambda: fh.read(64 * 1024), b""):
                    hasher.update(chunk)
        return hasher.hexdigest()
    return ""


def resolve_local_fixture_repo(value: str) -> str:
    """Resolve a ``fixture_repo`` value to what ``git clone`` should receive.

    Bare local paths (``./fixture``, ``../foo``, ``my-repo``) are resolved
    against the current process working directory because
    :func:`clone_fixture_repo` runs ``git clone`` with ``cwd`` set to the
    per-run cache directory, not the user's CWD. Without this step a
    relative ``fixture_repo`` that works from the shell silently fails.

    Anything with an explicit URL scheme - ``https://``, ``file://``,
    ``git://``, ``ssh://``, etc. - passes through byte-for-byte. Users
    writing a URL are signalling URL semantics and we should not
    second-guess them; in particular, ``file://`` is RFC 8089-only for
    absolute paths, so resolving it against CWD would invent behaviour
    git itself does not assume. SCP-style ``user@host:path`` shortcuts
    also pass through (urlparse does not recognise their scheme).

    Threat model note: ``..`` segments traverse intentionally (same model
    as ``working_dir``, which also calls ``Path(...).resolve()``). The
    schema documents ``fixture_repo`` as a "git URL or local path"
    supplied by the scenario author, who is trusted; clone access is
    bounded by what the running user can read. Don't add a path-traversal
    block here without revisiting that model.

    Symlink note: ``Path.resolve()`` follows symlinks, matching ``cd``
    semantics. A bare local path that traverses through a symlink will
    land on the symlink's target before reaching ``git clone``. Run
    scenarios from sources you trust: a hostile scenario JSON could
    plant a symlink in your CWD to redirect the runner at any directory
    the running user can read, exposing its contents to the agent under
    test. Defense-in-depth (e.g. resolved-path sanity check) is a
    separate hardening track; the resolver itself does not impose it
    so legitimate symlinked-fixture layouts keep working.
    """
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme:
        return value
    if _SSH_SHORTCUT_RE.match(value):
        return value
    return str(Path(value).resolve())


def resolve_local_resource_source(value: str, config_dir: Path) -> str:
    """Resolve a ``resources[].source`` value to what ``_fetch_source`` should receive.

    Sister of :func:`resolve_local_fixture_repo`. Bare local paths
    (``./skill.zip``, ``../shared/file.txt``, ``data/blob``) are resolved
    against ``config_dir`` -- the directory containing the scenario
    ``_config.json`` -- because resources are scenario assets authored
    alongside the config. Resolving against process CWD instead (the
    original behaviour) made the bundled showcase only succeed when the
    user happened to ``cd`` into the scenario directory, contradicting the
    example command in the scenario's own description.

    Anything with an explicit URL scheme -- ``https://``, ``http://``,
    ``file://`` -- passes through byte-for-byte. ``file://`` is RFC
    8089-only for absolute paths so anchoring it would invent behaviour;
    HTTP(S) is downloaded by :func:`_fetch_source`. SCP-style
    ``user@host:path`` shortcuts also pass through (``urlparse`` reports
    an empty scheme for them, and they are not valid local paths).

    Threat model note: ``..`` segments traverse intentionally, matching
    the established :func:`resolve_local_fixture_repo` model. The schema
    documents ``source`` as scenario-author-supplied; read access is
    bounded by what the running user can read. See ``docs/glossary/
    SCENARIOS.md`` "Trust boundary for ``resources``" for the
    full discussion.
    """
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme:
        return value
    if _SSH_SHORTCUT_RE.match(value):
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((config_dir / p).resolve())


def _fetch_source(source: str, *, config_dir: Path | None = None) -> tuple[Path, bool, str]:
    """Resolve ``source`` to a local path. Returns ``(path, is_temp, sha256)``.

    Local paths and ``file://`` URLs are returned in place. ``http(s)://``
    URLs are downloaded to a temp file with a strict size cap. Other
    schemes are rejected because ``urllib`` would otherwise happily handle
    ``ftp:`` and ``data:`` against expectations.

    ``config_dir`` is the scenario's ``_config.json`` directory; bare
    local paths anchor against it via :func:`resolve_local_resource_source`.
    Pass ``None`` only from contexts that have no scenario anchor (e.g.
    direct unit tests against the helper); callers in the runner always
    pass it.
    """
    parsed = urllib.parse.urlparse(source)
    scheme = parsed.scheme.lower()
    if scheme in ("", "file"):
        if scheme == "":
            resolved = resolve_local_resource_source(source, config_dir) if config_dir is not None else source
            path = Path(resolved)
        else:
            path = Path(parsed.path)
        if not path.exists():
            raise WorkspaceError(f"Resource source not found: {source}")
        return path, False, _sha256_of_path(path)
    if scheme not in ("http", "https"):
        raise WorkspaceError(f"Unsupported resource source scheme {scheme!r}: {source}")
    tmp = Path(tempfile.mkstemp(prefix="belt-resource-")[1])
    hasher = hashlib.sha256()
    try:
        # urlopen with a timeout is safe; we cap downloaded bytes to avoid
        # an unbounded write in the worst case (slow infinite stream).
        with urllib.request.urlopen(  # nosec B310 - scheme allow-listed above (http/https only)
            source, timeout=_FETCH_TIMEOUT_SECONDS
        ) as resp:  # noqa: S310 -- scheme allow-listed above
            total = 0
            with tmp.open("wb") as fh:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _FETCH_MAX_BYTES:
                        raise WorkspaceError(f"Resource source exceeded {_FETCH_MAX_BYTES} bytes: {source}")
                    hasher.update(chunk)
                    fh.write(chunk)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return tmp, True, hasher.hexdigest()


def _extract_archive(source: Path, dest_dir: Path) -> None:
    """Extract ``source`` into ``dest_dir`` rejecting path-traversal entries.

    Tar / zip "evil" entries that point outside the destination (``..``
    segments or absolute paths) are refused so an attacker-published
    archive cannot drop files into the belt install or the user's
    home dir.
    """
    name = source.name.lower()
    dest_resolved = dest_dir.resolve()
    if name.endswith(".zip"):
        with zipfile.ZipFile(source) as zf:
            for info in zf.infolist():
                target = (dest_dir / info.filename).resolve()
                try:
                    target.relative_to(dest_resolved)
                except ValueError as e:
                    raise WorkspaceError(f"Refusing zip entry that escapes dest: {info.filename!r}") from e
            zf.extractall(dest_dir)  # nosec B202 - entries pre-validated against dest_resolved above  # noqa: S202
        return
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        if name.endswith(".tar.bz2"):
            mode = "r:bz2"
        elif name.endswith((".tar.gz", ".tgz")):
            mode = "r:gz"
        else:
            mode = "r:"
        with tarfile.open(source, mode) as tf:
            for member in tf.getmembers():
                target = (dest_dir / member.name).resolve()
                try:
                    target.relative_to(dest_resolved)
                except ValueError as e:
                    raise WorkspaceError(f"Refusing tar entry that escapes dest: {member.name!r}") from e
            try:
                # nosec B202 - members pre-validated above; filter="data" is the
                # Python 3.12+ safe extractor that additionally rejects symlinks /
                # device files / abs paths even if validation drifted.
                tf.extractall(dest_dir, filter="data")  # type: ignore[call-arg]  # nosec B202
            except TypeError:  # pragma: no cover -- Python < 3.12 has no filter kwarg
                tf.extractall(dest_dir)  # nosec B202 - entries pre-validated above  # noqa: S202
        return
    raise WorkspaceError(f"Unsupported archive format for resource: {source.name}")


def install_resources(
    workspace: Path,
    resources: list[Resource],
    *,
    config_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Install ``resources`` into ``workspace`` and return per-resource lock entries.

    Lock entries record ``name``, ``kind``, ``source``, ``dest``, optional
    ``version``, and SHA-256 of the source bytes so a reviewer can pin
    "result X corresponds to skill v0.91 sha256:abcd...". Unknown ``kind``
    values raise :class:`WorkspaceError` so a typo does not silently
    install nothing.

    ``config_dir`` anchors bare local ``resources[].source`` paths -- see
    :func:`resolve_local_resource_source`. Runner callers pass the
    scenario's ``_config.json`` directory; ``None`` is only for direct
    unit tests where the source is already absolute or a URL.
    """
    locks: list[dict[str, Any]] = []
    for resource in resources:
        full_dest = _safe_dest_in(workspace, resource.dest)
        full_dest.parent.mkdir(parents=True, exist_ok=True)
        local_source, is_temp, sha256 = _fetch_source(resource.source, config_dir=config_dir)
        try:
            if resource.kind == "file":
                # Preserve symlinks rather than following them. ``shutil``
                # defaults dereference symlinks in both ``copytree`` and
                # ``copy2``, which would let a hostile (or accidentally
                # symlinked) source tree exfiltrate the link target's
                # contents into the agent's workspace. Run scenarios from
                # sources you trust; the harness defends against the
                # accidental case by preserving links rather than
                # rewriting them.
                if local_source.is_dir():
                    if full_dest.exists():
                        shutil.rmtree(full_dest)
                    shutil.copytree(local_source, full_dest, symlinks=True)
                else:
                    shutil.copy2(local_source, full_dest, follow_symlinks=False)
            elif resource.kind == "archive":
                full_dest.mkdir(parents=True, exist_ok=True)
                _extract_archive(local_source, full_dest)
            else:
                raise WorkspaceError(f"Unknown resource kind {resource.kind!r} for {resource.source}")
        finally:
            if is_temp:
                local_source.unlink(missing_ok=True)
        locks.append(
            {
                "name": resource.name or Path(resource.dest).name or resource.kind,
                "kind": resource.kind,
                "source": resource.source,
                "dest": resource.dest,
                "version": resource.version,
                "source_sha256": sha256,
            }
        )
    return locks


class WorkspaceManager:
    """Manages per-scenario isolated workspaces using git worktrees."""

    def __init__(self, base_repo: Path, ref: str = "HEAD"):
        self._base = base_repo.resolve()
        self._ref = ref
        self._worktrees: list[Path] = []
        self._wt_lock = threading.Lock()
        self._tmp_root: Path | None = None
        self._auto_initialized = False
        # Path of any auto-initialised .git directory we own and may remove on
        # cleanup. Only set when we ran ``git init`` ourselves.
        self._auto_initialized_git_dir: Path | None = None

        if not _SAFE_REF_RE.match(ref) or _DOTDOT_RE.search(ref):
            raise WorkspaceError(f"Invalid workspace_ref: {ref!r}. " "Use a commit SHA, tag, or branch name.")

        if not git_available():
            raise WorkspaceError("git not found on PATH")

        if not (self._base / ".git").exists() and not self._base.joinpath(".git").is_file():
            self._auto_init_repo()
        else:
            # Repo already exists. Re-check whether ``belt`` itself
            # initialised it on a prior run (marker present) and refuse
            # to operate against a stale fixture snapshot. User-managed
            # repos (no marker) are unaffected.
            self._check_auto_init_drift()

        atexit.register(self.cleanup_all)

    def _auto_init_repo(self) -> None:
        """Auto-initialize a non-git directory as a git repo with an initial commit.

        Required for worktree-based isolation when the fixture directory
        is shipped as plain files (e.g. inside the belt repo tree).
        Thread-safe: parallel scenarios sharing the same base_repo wait
        on a per-path lock so only one init runs.
        """
        with _init_locks_guard:
            if self._base not in _init_locks:
                _init_locks[self._base] = threading.Lock()
            lock = _init_locks[self._base]

        with lock:
            if (self._base / ".git").exists():
                # A concurrent caller (or the constructor's existing-repo
                # branch) already ran the drift check; nothing to do.
                return
            # Refuse to auto-init a directory we don't own. A hostile shared
            # ``working_dir`` could otherwise trick us into running
            # ``git init`` in a place we shouldn't write;
            # ``--allow-external-working-dir`` opts back in for users who
            # have already vetted the path.
            try:
                base_stat = self._base.stat()
                import os as _os

                if hasattr(_os, "geteuid") and base_stat.st_uid != _os.geteuid():
                    raise WorkspaceError(
                        f"Refusing to auto-init git repo at {self._base}: directory is owned "
                        f"by a different user (uid={base_stat.st_uid}). Pre-init the repo or "
                        f"choose a different working_dir."
                    )
            except FileNotFoundError as e:
                raise WorkspaceError(f"working_dir does not exist: {self._base}") from e
            logger.info("Auto-initializing git repo in {}", self._base)
            try:
                run_git("init", cwd=self._base, timeout=30, check=True)
                run_git("add", "-A", cwd=self._base, timeout=30, check=True)
                run_git(
                    "-c",
                    "user.name=belt",
                    "-c",
                    "user.email=belt@local",
                    "commit",
                    "-m",
                    "initial fixture state",
                    cwd=self._base,
                    timeout=30,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise WorkspaceError(f"Failed to auto-init git repo in {self._base}: {e.stderr.strip()}")
            # Drop an ownership marker INSIDE ``.git/`` so it never leaks
            # into worktrees (git checkout doesn't materialise contents of
            # the ``.git`` dir). The marker lets future ``belt`` runs detect
            # that *we* initialised this repo and gate against fixture
            # drift; foreign repos without the marker are left alone.
            try:
                (self._base / ".git" / _AUTO_INIT_MARKER).write_text("1\n")
            except OSError as e:
                raise WorkspaceError(f"Failed to write auto-init marker in {self._base / '.git'}: {e}") from e

        self._auto_initialized = True
        self._auto_initialized_git_dir = self._base / ".git"

    def _check_auto_init_drift(self) -> None:
        """Refuse spinning off worktrees against a stale auto-initialised fixture.

        Triggered when ``.git`` already exists. The guard is scoped to repos
        ``belt`` itself initialised (identified by the
        ``.git/belt-init-marker`` sidecar): for user-managed repos and
        ``fixture_repo`` clones the marker is absent and we return without
        inspection. For ``belt``-owned repos, ``git status --porcelain``
        decides: if the working tree drifted since the initial fixture
        commit, raise with a remediation hint rather than silently running
        scenarios against the stale snapshot.
        """
        marker = self._base / ".git" / _AUTO_INIT_MARKER
        if not marker.exists():
            return
        try:
            result = run_git("status", "--porcelain", cwd=self._base, timeout=10, check=True)
        except subprocess.CalledProcessError:
            # Tolerate transient git failure: the surrounding worktree spin
            # also calls git and will surface a clearer error on real
            # breakage. Better to proceed than to wedge belt on a flaky
            # ``git status``.
            return
        if result is None:
            return
        if result.stdout.strip():
            raise WorkspaceError(
                f"Fixture {self._base} has uncommitted changes since the last "
                f"`belt` run that auto-initialised it. Refusing to spin off worktrees "
                f"against the stale snapshot. To refresh:\n"
                f"  cd {self._base} && git add -A && git commit -m 'refresh fixture'"
            )

    @staticmethod
    def _sanitize_label(label: str) -> str:
        """Reduce ``label`` to filesystem-safe characters.

        Path separators, parent-traversal markers, control characters, and
        whitespace are all rewritten to ``_``. The result is also length-capped
        so we don't try to mkdir a path longer than typical filesystem limits.
        Falls back to ``scenario`` if the input collapses to an empty string.
        """
        cleaned = _LABEL_UNSAFE_RE.sub("_", label).strip("._")
        cleaned = cleaned.replace("..", "_")
        if not cleaned:
            cleaned = "scenario"
        return cleaned[:200]

    def _ensure_tmp_root(self) -> Path:
        if self._tmp_root is None:
            self._tmp_root = Path(tempfile.mkdtemp(prefix="belt-ws-"))
            logger.debug("Workspace tmp root: {}", self._tmp_root)
        return self._tmp_root

    def acquire(self, label: str) -> Path:
        """Create an isolated worktree for a scenario.

        Returns the worktree path, ready for use as a subprocess cwd. The
        ``label`` is sanitised down to a conservative ASCII alphabet - any
        path-traversal sequence (``..``), embedded separator, or control
        character is rewritten before being used as a directory name. This is
        a defence-in-depth backstop on top of :class:`belt.scenario.Scenario`
        already validating its own ``name``.
        """
        safe_label = self._sanitize_label(label)
        tmp_root = self._ensure_tmp_root()
        worktree_path = tmp_root / safe_label
        # Confirm we landed under tmp_root and not somewhere unexpected.
        try:
            worktree_path.resolve().relative_to(tmp_root.resolve())
        except ValueError as e:
            raise WorkspaceError(f"Refusing to acquire worktree outside tmp root: {label!r}") from e

        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)

        with _worktree_add_locks_guard:
            wt_lock = _worktree_add_locks.setdefault(self._base, threading.Lock())

        with wt_lock:
            result = run_git(
                "worktree",
                "add",
                "--detach",
                str(worktree_path),
                self._ref,
                cwd=self._base,
                timeout=60,
            )
        if result is None:
            raise WorkspaceError("git worktree add timed out after 60s")
        if result.returncode != 0:
            raise WorkspaceError(f"git worktree add failed (rc={result.returncode}): {result.stderr.strip()}")

        with self._wt_lock:
            self._worktrees.append(worktree_path)
        logger.debug("Acquired worktree: {} → {}", label, worktree_path)
        return worktree_path

    def capture_diff(self, worktree: Path) -> tuple[str | None, list[str]]:
        """Capture git diff and modified file list from a worktree.

        Stages all changes (including new untracked files) before diffing,
        so the diff includes both modifications and new file additions.
        This is safe because the worktree is ephemeral.

        Returns (git_diff_text, files_modified_list). Either may be empty
        if no changes were made.
        """
        git_diff: str | None = None
        files_modified: list[str] = []

        # Stage everything so new files appear in the diff
        run_git("add", "-A", cwd=worktree, timeout=30)

        diff_result = run_git("diff", "--cached", "--no-color", cwd=worktree, timeout=30)
        if diff_result is None:
            logger.warning("git diff timed out in {}", worktree)
        elif diff_result.returncode == 0 and diff_result.stdout.strip():
            git_diff = diff_result.stdout

        names_result = run_git("diff", "--cached", "--name-only", cwd=worktree, timeout=30)
        if names_result is None:
            logger.warning("git diff --name-only timed out in {}", worktree)
        elif names_result.returncode == 0 and names_result.stdout.strip():
            files_modified = sorted(names_result.stdout.strip().splitlines())

        return git_diff, files_modified

    def release(self, worktree: Path) -> tuple[str | None, list[str]]:
        """Capture diff, then remove the worktree.

        Returns (git_diff_text, files_modified_list).
        """
        git_diff, files_modified = self.capture_diff(worktree)

        result = run_git(
            "worktree",
            "remove",
            "--force",
            str(worktree),
            cwd=self._base,
            timeout=30,
        )
        if result is None or result.returncode != 0:
            detail = "timeout" if result is None else result.stderr.strip()
            logger.warning("git worktree remove failed for {}: {}", worktree, detail)
            shutil.rmtree(worktree, ignore_errors=True)

        with self._wt_lock:
            if worktree in self._worktrees:
                self._worktrees.remove(worktree)

        return git_diff, files_modified

    def cleanup_all(self) -> None:
        """Remove all worktrees created by this manager (crash safety net)."""
        for wt in list(self._worktrees):
            result = run_git(
                "worktree",
                "remove",
                "--force",
                str(wt),
                cwd=self._base,
                timeout=10,
            )
            if result is None or result.returncode != 0:
                shutil.rmtree(wt, ignore_errors=True)
        self._worktrees.clear()

        if self._tmp_root and self._tmp_root.exists():
            shutil.rmtree(self._tmp_root, ignore_errors=True)
            self._tmp_root = None

        run_git("worktree", "prune", cwd=self._base, timeout=10)

        # Auto-initialized .git dirs are left in place - .gitignore covers them.
        # Removing mid-run would break other parallel scenarios sharing the same base.
