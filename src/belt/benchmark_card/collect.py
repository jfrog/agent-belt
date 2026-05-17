# (c) JFrog Ltd. (2026)

"""Static-provenance collectors written into ``run_meta.json``.

These run before any scenario executes; their outputs are purely a
function of the host environment, the belt install, and the user's
invocation. The aggregate phase consumes them via
:func:`belt.benchmark_card.build_card`.
"""

from __future__ import annotations

import hashlib
import platform
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from loguru import logger

from belt._git import git_text
from belt._redact import safe_environ
from belt.constants import RUNTIME_INFO_FILE, SCORE_FILE

from .entities import (
    AgentIdentity,
    AgentProvenance,
    BeltProvenance,
    CliIdentity,
    HostProvenance,
    Invocation,
    JudgeProvenance,
    ScenarioFile,
)
from .io import read_json


def collect_belt_provenance(repo_root: Path | None = None) -> BeltProvenance:
    """Identify the running belt install.

    Resolution order:

    1. ``importlib.metadata.version("agent-belt")`` for the package
       version.
    2. If ``repo_root`` is supplied and contains a ``.git`` directory,
       also capture the HEAD SHA and dirty-state via ``git rev-parse`` /
       ``git status --porcelain``. This is the editable-install case;
       a wheel install has no ``.git`` and degrades to
       ``install_kind="wheel"``.
    """
    try:
        version = importlib_metadata.version("agent-belt")
    except importlib_metadata.PackageNotFoundError:
        version = "0.0.0+unknown"

    git_sha: str | None = None
    git_dirty: bool | None = None
    install_kind = "wheel"
    if repo_root is not None and (repo_root / ".git").exists():
        install_kind = "editable"
        git_sha = git_text("rev-parse", "HEAD", cwd=repo_root)
        status = git_text("status", "--porcelain", cwd=repo_root)
        if status is not None:
            git_dirty = bool(status)
    elif repo_root is not None:
        # Path was supplied but is not a git checkout - treat as
        # unknown rather than wheel, so a misconfiguration shows up
        # plainly.
        install_kind = "unknown"

    return BeltProvenance(
        version=version,
        install_kind=install_kind,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )


_DEPS_OF_INTEREST: tuple[str, ...] = (
    "agent-belt",
    "pydantic",
    "loguru",
    "rich",
    "filelock",
    "httpx",
    "backoff",
    "python-dotenv",
    "pyyaml",
)


def collect_host_provenance() -> HostProvenance:
    """Capture OS + Python runtime + versions of declared direct dependencies.

    Package selection mirrors ``[project] dependencies`` in
    ``pyproject.toml`` so the card reflects the actual install, not an
    opinion about which packages matter. Missing packages are silently
    skipped (a stripped-down install will simply have a smaller dict).
    """
    pkg_versions: dict[str, str] = {}
    for name in _DEPS_OF_INTEREST:
        try:
            pkg_versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue

    return HostProvenance(
        os=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        package_versions=pkg_versions,
    )


def collect_invocation(argv: list[str], parsed_args: dict[str, Any], cwd: str) -> Invocation:
    """Snapshot the user's invocation for the card.

    Both ``argv`` and ``parsed_args`` are passed in already-redacted by
    the caller (the run-phase entry point owns the policy of which
    argparse fields to expose). ``env`` is always sourced via
    :func:`belt._redact.safe_environ` so secret-name leaks are
    impossible regardless of caller.
    """
    return Invocation(
        argv=list(argv),
        parsed_args=dict(parsed_args),
        cwd=cwd,
        env=safe_environ(),
    )


def hash_scenario_files(
    scenarios_root: Path,
    scenario_paths: list[Path],
) -> list[ScenarioFile]:
    """SHA-256 every scenario JSON that the run is about to execute.

    Hashes are computed on the raw bytes of the file as it lives on
    disk - no JSON re-serialisation, so byte-identical files always
    agree. Paths that fail to read (deleted between resolution and
    hashing, permission error) are silently dropped from the manifest
    rather than aborting the run; the card reflects what was actually
    hashable.
    """
    out: list[ScenarioFile] = []
    root = scenarios_root.resolve()
    for path in scenario_paths:
        try:
            data = path.read_bytes()
        except OSError as e:
            logger.debug("scenario file unreadable, skipping: {} ({})", path, e)
            continue
        try:
            relpath = str(path.resolve().relative_to(root))
        except ValueError:
            relpath = str(path)
        out.append(ScenarioFile(relpath=relpath, sha256=hashlib.sha256(data).hexdigest()))
    out.sort(key=lambda s: s.relpath)
    return out


def collect_runtime_info_sidecars(run_dir: Path) -> list[AgentProvenance]:
    """Aggregate per-scenario ``_runtime_info.json`` sidecars into per-group records.

    Sidecars are written by the orchestrator once per scenario (after
    ``agent.setup()``) in the canonical two-level shape
    (``agent.{name,adapter_class,args,auth_signals}`` /
    ``cli.{binary_path,version}``). Per-group deduplication keeps the
    card readable even on cross-agent runs that visit each group's
    agent many times.

    A multi-agent run produces one ``AgentProvenance`` per group;
    dedup is keyed on ``group`` (the canonical
    ``MatchedGroup.name``), so distinct groups - even ones running the
    same adapter class - each surface in the card.
    """
    by_group: dict[str, AgentProvenance] = {}
    for sidecar in sorted(run_dir.rglob(RUNTIME_INFO_FILE)):
        data = read_json(sidecar)
        if not data:
            continue
        group = data.get("group") or sidecar.parent.parent.name
        if group in by_group:
            # First sighting wins. All instances within the same group
            # share the same agent class; later scenarios cannot
            # disagree without a bug in setup_groups.
            continue
        agent_block = data.get("agent") or {}
        cli_block = data.get("cli") or {}
        try:
            by_group[group] = AgentProvenance(
                group=group,
                agent=AgentIdentity(
                    name=agent_block.get("name", "unknown"),
                    adapter_class=agent_block.get("adapter_class", "unknown"),
                    args=dict(agent_block.get("args") or {}),
                    auth_signals=list(agent_block.get("auth_signals") or []),
                ),
                cli=CliIdentity(
                    binary_path=cli_block.get("binary_path"),
                    version=cli_block.get("version"),
                ),
            )
        except Exception as e:
            logger.debug("Skipping malformed runtime_info sidecar {}: {}", sidecar, e)
    return sorted(by_group.values(), key=lambda a: a.group)


def collect_judges(run_dir: Path) -> list[JudgeProvenance]:
    """Discover unique LLM judge backends used across all scored scenarios.

    Read from ``score.json`` files produced by the scorer. Backends are
    keyed by ``(provider, model, base_url)`` so a run that mixes a
    primary judge with an OpenAI-compatible secondary surfaces both.
    ``base_url`` is recorded as the scorer left it; when it's a real
    URL the scorer is expected to have already redacted it via
    :mod:`belt._redact`.
    """
    seen: dict[tuple[str, str, str], JudgeProvenance] = {}
    for score_path in run_dir.rglob(SCORE_FILE):
        data = read_json(score_path)
        if not data:
            continue
        llm = (data.get("scores") or {}).get("llm") or {}
        usage = llm.get("usage") or {}
        backends = usage.get("backends") if isinstance(usage, dict) else None
        if not isinstance(backends, list):
            continue
        for b in backends:
            if not isinstance(b, dict):
                continue
            key = (
                str(b.get("provider", "")),
                str(b.get("model", "")),
                str(b.get("base_url", "")),
            )
            if key in seen:
                continue
            seen[key] = JudgeProvenance(
                provider=key[0] or "unknown",
                model=key[1] or "unknown",
                base_url=key[2] or None,
                dimensions=sorted(d for d in (llm.get("dimensions") or []) if isinstance(d, str)),
            )
    return list(seen.values())
