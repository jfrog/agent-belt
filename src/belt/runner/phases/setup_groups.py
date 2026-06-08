# (c) JFrog Ltd. (2026)

"""Phase 2 - set up agent groups in parallel; clean up orphan resources.

``setup_groups`` runs ``agent.check_available`` + ``health_check`` +
``setup_group`` in a thread pool, populating ``ctx.group_states`` and
registering the run with the manifest.

``cleanup_orphans`` walks the manifest for previously-crashed runs and
invokes the relevant agent's ``teardown_group`` to release any leaked
remote state before the new run starts.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger

from belt import envvars
from belt._git import git_available, run_git
from belt._io import write_json
from belt.agent.registry import get_agent_class
from belt.constants import MANIFEST_FILE
from belt.manifest import Manifest
from belt.runner.context import RunContext, create_agent
from belt.runner.workspace import WorkspaceError, clone_fixture_repo, resolve_local_fixture_repo
from belt.scenario import GroupConfig

_RUN_FIXTURES_FILE = "run_fixtures.json"
_SETUP_ERRORS_FILE = "setup_errors.json"


def _write_setup_errors_sidecar(ctx: RunContext) -> None:
    """Persist per-group setup-failure records for the aggregator to consume.

    Written to ``<run_dir>/setup_errors.json`` as a JSON array of
    ``{"group": name, "scenarios": [scenario_name, ...], "error":
    message}`` so ``belt aggregate`` can fold it into
    ``AggregatedResults.setup_errors`` without having to re-run setup
    or reach back into RunContext. Absent when no group failed setup.

    The matched-group list is the source of truth for which scenarios
    were *meant* to run; a failed group still has its scenarios listed
    here even though no per-scenario score artifact exists on disk.
    Order matches the order ``setup_errors`` was populated in, which is
    the order groups were rejected in.
    """
    if not ctx.failed_groups:
        return
    scenarios_by_group: dict[str, list[str]] = {}
    for mg in ctx.matched_groups:
        scenarios_by_group.setdefault(mg.name, []).extend(s.name for s in mg.scenarios)
    records = []
    for group_name in ctx.setup_errors:
        records.append(
            {
                "group": group_name,
                "scenarios": scenarios_by_group.get(group_name, []),
                "error": ctx.setup_errors[group_name],
            }
        )
    # Any group rejected without a recorded cause (defensive fallback;
    # should not happen with the in-tree rejection sites covered) still
    # surfaces under a placeholder so downstream consumers see every
    # failed group, not just the well-formed ones.
    for group_name in ctx.failed_groups - set(ctx.setup_errors):
        records.append(
            {
                "group": group_name,
                "scenarios": scenarios_by_group.get(group_name, []),
                "error": "group setup failed",
            }
        )
    write_json(ctx.run_dir / _SETUP_ERRORS_FILE, records)


def _external_working_dir_message(working_dir: Path, scenarios_root: Path) -> str:
    """Render the user-facing error for a ``working_dir`` outside the scenarios root.

    Extracted so the wording is testable in isolation and stays in sync
    with the ``--allow-external-working-dir`` opt-in copy on the CLI.
    Both escape hatches and the security intent are named explicitly so
    a first-time reader knows which knob to turn next.
    """
    return (
        f"working_dir '{working_dir}' resolves outside the scenarios root "
        f"'{scenarios_root}'. This is a security guardrail against "
        f"auto-initialising a git repo in unrelated directories. To "
        f"proceed, either:\n"
        f"       - opt in: re-run with --allow-external-working-dir "
        f"(or set {envvars.ALLOW_EXTERNAL_WORKING_DIR}=1), or\n"
        f"       - scope away: re-run with --scenarios pointing at a "
        f"different group."
    )


def _check_external_working_dir_gate(ctx: RunContext) -> None:
    """Reject groups whose ``working_dir`` escapes the scenarios root.

    ``working_dir`` lives on ``GroupConfig`` (``_config.json``) and applies
    to every scenario in the group by definition - its validity is a group
    property, not a per-scenario one. Validating here means N scenarios
    in the same misconfigured group surface one banner line instead of N
    identical raises inside ``run_scenarios._run_one``.

    Default DENY: groups are marked failed and ``run_scenarios`` then writes
    ``ScenarioResult(error="group setup failed")`` for each member without
    creating ``turn_0_cli.txt`` sentinels. Opting in via
    ``--allow-external-working-dir`` (or ``BELT_ALLOW_EXTERNAL_WORKING_DIR=1``)
    silences this gate but logs a single ``WARNING`` so the relaxation is
    visible in ``eval.log``.

    Skips groups that use ``fixture_repo`` (the clone lives under
    ``<run_dir>/_fixtures``, which is always inside the run root) and
    groups whose ``workspace_isolation`` is ``"none"`` (no auto-init
    happens; ``--allow-inplace`` is the relevant opt-in there).
    """
    allowed = bool(getattr(ctx.args, "allow_external_working_dir", False)) or envvars.is_truthy(
        envvars.ALLOW_EXTERNAL_WORKING_DIR
    )
    scenarios_root = ctx.scenarios_root.resolve()
    for mg in ctx.matched_groups:
        gc = mg.config
        if not gc.working_dir:
            continue
        if gc.fixture_repo:
            continue
        if gc.workspace_isolation != "git-worktree":
            continue
        resolved = (mg.group_dir / gc.working_dir).resolve()
        if resolved.is_relative_to(scenarios_root):
            continue
        if allowed:
            logger.warning(
                "Group {}: working_dir {} is outside the scenarios root {} "
                "(allowed via --allow-external-working-dir)",
                mg.name,
                resolved,
                scenarios_root,
            )
            continue
        cause = _external_working_dir_message(resolved, scenarios_root)
        ctx.failed_groups.add(mg.name)
        ctx.setup_errors[mg.name] = cause
        ctx.progress.console.print(f"\n  [red]\u2717[/red] {mg.name}: {cause}")


def _check_inplace_gate(ctx: RunContext) -> None:
    """Reject groups with ``workspace_isolation: "none"`` unless opted in.

    ``workspace_isolation: "none"`` disables per-scenario worktrees, so any
    agent edits land in the harness CWD instead of an isolated copy. The
    schema (``GroupConfig.workspace_isolation``) already restricts the field
    to ``"git-worktree"`` or ``"none"`` so typos cannot silently disable
    isolation; this gate adds the runtime opt-in for the one valid value
    that *intentionally* skips it.

    Default DENY: groups are marked failed and a message tells the user to
    re-run with ``--allow-inplace`` (or set ``BELT_ALLOW_INPLACE=1``).
    Mirrors the existing ``--allow-external-working-dir`` guardrail in
    ``run_scenarios.py`` and the ``fixture_repo + working_dir`` mutex
    enforced below.
    """
    allowed = bool(getattr(ctx.args, "allow_inplace", False)) or envvars.is_truthy(envvars.ALLOW_INPLACE)
    if allowed:
        # When the operator opts in, isolation is off for every group with
        # ``workspace_isolation: "none"``. Print a yellow banner per group
        # so the disabled-isolation status is visible in stdout AND in
        # ``eval.log`` - readers of the artifacts can grep for it later.
        for mg in ctx.matched_groups:
            if mg.config.workspace_isolation != "none":
                continue
            ctx.progress.console.print(
                f"\n  [yellow]\u26a0[/yellow] {mg.name}: workspace_isolation: 'none' "
                f"is active (--allow-inplace). The agent operates in the harness "
                f"working directory; edits land outside any per-scenario worktree."
            )
        return
    for mg in ctx.matched_groups:
        if mg.config.workspace_isolation != "none":
            continue
        cause = (
            "workspace_isolation: 'none' disables per-scenario worktrees; "
            "the agent operates without isolation. Re-run with --allow-inplace "
            f"(or set {envvars.ALLOW_INPLACE}=1) to opt in."
        )
        ctx.failed_groups.add(mg.name)
        ctx.setup_errors[mg.name] = cause
        ctx.progress.console.print(f"\n  [red]\u2717[/red] {mg.name}: {cause}")


def _check_verify_gate(ctx: RunContext) -> None:
    """Reject groups whose scenarios declare ``verify`` unless opted in and isolated.

    A ``verify`` block (``Turn.verify`` or ``Scenario.verify``) runs an
    author-supplied command in the worktree, so it is default-deny - exactly
    like ``--allow-inplace`` and ``--allow-external-working-dir``. Two
    conditions are enforced here, before any agent runs:

    1. **Opt-in.** ``--allow-verify-exec`` (or ``BELT_ALLOW_VERIFY_EXEC=1``)
       must be set; otherwise the group is refused with a remediation line.
    2. **Isolated worktree.** ``verify`` only runs inside a per-scenario
       worktree (``working_dir`` or ``fixture_repo`` with
       ``workspace_isolation: git-worktree``); without one the command would
       execute in the operator CWD, so the group is refused.

    See ``docs/glossary/SECURITY-MODEL.md`` for the threat model.
    """
    allowed = bool(getattr(ctx.args, "allow_verify_exec", False)) or envvars.is_truthy(envvars.ALLOW_VERIFY_EXEC)
    for mg in ctx.matched_groups:
        if mg.name in ctx.failed_groups:
            continue
        has_verify = any(scn.verify is not None or any(t.verify is not None for t in scn.turns) for scn in mg.scenarios)
        if not has_verify:
            continue
        if not allowed:
            cause = (
                "a scenario declares `verify` (deterministic command execution). "
                f"Re-run with --allow-verify-exec (or set {envvars.ALLOW_VERIFY_EXEC}=1) to opt in."
            )
            ctx.failed_groups.add(mg.name)
            ctx.setup_errors[mg.name] = cause
            ctx.progress.console.print(f"\n  [red]\u2717[/red] {mg.name}: {cause}")
            continue
        gc = mg.config
        has_worktree = bool(gc.working_dir or gc.fixture_repo) and gc.workspace_isolation == "git-worktree"
        if not has_worktree:
            cause = (
                "`verify` requires an isolated worktree. Set working_dir (or fixture_repo) "
                "with workspace_isolation: git-worktree so the command runs in a per-scenario "
                "copy, not the operator's working directory."
            )
            ctx.failed_groups.add(mg.name)
            ctx.setup_errors[mg.name] = cause
            ctx.progress.console.print(f"\n  [red]\u2717[/red] {mg.name}: {cause}")


def _prepare_group_fixtures(ctx: RunContext) -> None:
    """Clone every group's ``fixture_repo`` into a per-run cache directory.

    Runs once before group setup so a network failure short-circuits the
    whole run rather than every scenario re-discovering it. The cache lives
    under ``<run_dir>/_fixtures/<group-name>`` so concurrent runs never
    collide on the same path. Groups without ``fixture_repo`` are skipped.

    A group that sets both ``fixture_repo`` and ``working_dir`` is rejected
    here (mutual exclusion enforced at the orchestrator boundary, not at
    the schema layer, so plugins remain free to add forward-compatible
    fields).
    """
    cache_root = ctx.run_dir / "_fixtures"
    for mg in ctx.matched_groups:
        gc = mg.config
        if not gc.fixture_repo:
            continue
        if gc.working_dir:
            cause = "``fixture_repo`` and ``working_dir`` are mutually exclusive."
            ctx.failed_groups.add(mg.name)
            ctx.setup_errors[mg.name] = cause
            ctx.progress.console.print(f"\n  [red]\u2717[/red] {mg.name}: {cause}")
            continue
        dest = cache_root / mg.group_dir.name
        # ``git clone`` runs with cwd set to the per-run cache directory, so a
        # relative ``fixture_repo`` like ``../foo`` would resolve against that
        # cache (and fail) instead of the user's CWD. Bare local paths are
        # pre-resolved here; URL forms and SSH shortcuts pass through.
        fixture_source = resolve_local_fixture_repo(gc.fixture_repo)
        try:
            clone_fixture_repo(fixture_source, gc.fixture_ref, dest)
        except WorkspaceError as e:
            # Same rationale as the generic setup-failure branch below:
            # the human-facing banner is the canonical surface; the
            # forensic copy lives in ``eval.log`` at ``DEBUG``.
            logger.debug("Fixture clone failed for {}: {}", mg.name, e)
            ctx.failed_groups.add(mg.name)
            ctx.setup_errors[mg.name] = str(e)
            ctx.progress.console.print(f"\n  [red]\u2717[/red] {mg.name}: {e}")
            continue
        ctx.group_fixtures[mg.name] = dest


def _capture_fixture_provenance_for_group(group_dir: Path, gc: GroupConfig) -> dict[str, Any]:
    """Snapshot per-group fixture identity at the moment groups finish setting up.

    For ``workspace_isolation == "git-worktree"`` groups we resolve the
    ``working_dir`` to its current HEAD SHA and dirty-file count. For
    plain-directory fixtures the SHA fields stay ``None`` and ``tracked``
    is ``False`` - the card still records the path so a reviewer can see
    "this run hit working_dir=X, untracked".

    Errors do not raise: a missing ``git`` binary, a non-repo directory,
    or a slow ``git status`` falls back to ``tracked: False``. The card is
    a best-effort artifact, never a precondition for a successful run.
    """
    # ``working_dir`` in ``_config.json`` is resolved relative to the group
    # directory by the runtime (``run_scenarios.py``), so the recorded
    # provenance must use the same anchor - ``Path(...).resolve()`` alone
    # would resolve a relative path against the process CWD and produce a
    # nonexistent path in the manifest.
    resolved_working_dir = (group_dir / gc.working_dir).resolve() if gc.working_dir else None
    info: dict[str, Any] = {
        "group": group_dir.name,
        "working_dir": str(resolved_working_dir) if resolved_working_dir else None,
        "tracked": False,
        "git_sha": None,
        "git_ref": gc.workspace_ref if gc.working_dir else None,
        "auto_initialized": False,
        "dirty_files": 0,
    }
    if not gc.working_dir or gc.workspace_isolation != "git-worktree":
        return info
    if not git_available():
        return info
    assert resolved_working_dir is not None  # guarded by ``gc.working_dir`` check above
    repo = resolved_working_dir
    if not (repo / ".git").exists():
        return info
    info["tracked"] = True
    sha = run_git("rev-parse", "HEAD", cwd=repo, timeout=5)
    if sha is not None and sha.returncode == 0:
        info["git_sha"] = sha.stdout.strip() or None
    elif sha is not None:
        logger.debug("git rev-parse failed in {}: {}", repo, sha.stderr.strip())
    status = run_git("status", "--porcelain", cwd=repo, timeout=5)
    if status is not None and status.returncode == 0:
        info["dirty_files"] = sum(1 for line in status.stdout.splitlines() if line.strip())
    elif status is not None:
        logger.debug("git status failed in {}: {}", repo, status.stderr.strip())
    return info


def _write_fixture_provenance(ctx: RunContext) -> None:
    """Persist per-group fixture provenance for the benchmark card.

    Written as ``<run_dir>/run_fixtures.json``. The card builder reads
    this once and folds it into ``BenchmarkCard.fixtures``. We capture
    here, after groups successfully set up, because at this point the
    worktree references are stable and the user has not yet had a chance
    to switch branches mid-run.
    """
    out: list[dict[str, Any]] = []
    for mg in ctx.matched_groups:
        if mg.name in ctx.failed_groups:
            continue
        try:
            out.append(_capture_fixture_provenance_for_group(mg.group_dir, mg.config))
        except Exception as e:
            logger.debug("fixture provenance capture failed for {}: {}", mg.name, e)
    if not out:
        return
    write_json(ctx.run_dir / _RUN_FIXTURES_FILE, out)


def cleanup_orphans(ctx: RunContext) -> None:
    """Clean up orphan resources from previous crashed runs."""
    manifest = Manifest(ctx.outcomes_root / MANIFEST_FILE)
    ctx._manifest = manifest  # type: ignore[attr-defined]

    def _cleanup_orphan(entry: dict) -> None:
        agent_name = entry.get("agent") or entry.get("agent")
        if not agent_name:
            logger.warning("Orphan entry missing agent name, skipping: {}", entry.get("group", "?"))
            return
        shared_state = entry.get("shared_state", {})
        try:
            agent_cls = get_agent_class(agent_name)
            cleanup = create_agent(agent_cls, ctx.agent_args)
            cleanup.teardown_group(shared_state)
        except Exception as e:
            logger.warning("Orphan cleanup failed for {}: {}", entry.get("group", "?"), e)

    orphans_deleted = manifest.cleanup_orphans(_cleanup_orphan)
    if orphans_deleted:
        ctx.progress.console.print(
            f"\n[yellow]Cleaned up {orphans_deleted} orphan resource(s) from previous run[/yellow]"
        )


def setup_groups(ctx: RunContext) -> int | None:
    """Set up agent groups in parallel. Returns exit code on fatal failure, None on success."""
    _check_external_working_dir_gate(ctx)
    _check_inplace_gate(ctx)
    _check_verify_gate(ctx)
    _prepare_group_fixtures(ctx)
    total_groups = len(ctx.matched_groups)
    ctx.progress.console.print(f"\nSetting up [bold]{total_groups}[/bold] group(s)...", end=" ")

    def _setup_one(name: str, group_dir: Path, gc: GroupConfig) -> tuple[str, Any]:
        agent_cls = get_agent_class(gc.agent)
        agent_cls.check_available()
        agent = create_agent(agent_cls, ctx.agent_args)
        agent.health_check()
        return name, agent.setup_group(gc, group_dir)

    with ThreadPoolExecutor(max_workers=total_groups) as executor:
        futures = {}
        for mg in ctx.matched_groups:
            # Skip groups already rejected by an earlier gate
            # (``_check_external_working_dir_gate``,
            # ``_check_inplace_gate``, ``_prepare_group_fixtures``). They
            # would otherwise pay the cost of ``check_available`` +
            # ``health_check`` + ``setup_group`` for a group whose
            # scenarios will all be reported as ``group setup failed``.
            if mg.name in ctx.failed_groups:
                continue
            futures[executor.submit(_setup_one, mg.name, mg.group_dir, mg.config)] = mg.name

        for future in as_completed(futures):
            name = futures[future]
            try:
                _, shared_state = future.result()
                ctx.group_states[name] = shared_state
                logger.info("Group {} setup: {}", name, shared_state)
            except Exception as e:
                # The same message is surfaced via ``console.print`` below as
                # the user-facing banner. Recording it as ``debug`` keeps the
                # forensic copy in ``eval.log`` (the file handler runs at
                # DEBUG) without double-printing on the terminal once the
                # ``-v`` opt-in lifts the level to INFO+ - the banner is the
                # canonical surface.
                logger.debug("Failed to set up group {}: {}", name, e)
                ctx.progress.console.print(f"\n  [red]\u2717[/red] {name}: {e}")
                ctx.failed_groups.add(name)
                ctx.setup_errors[name] = str(e)
                if ctx.args.strict:
                    ctx.progress.console.print("[red bold]--strict: aborting on first group failure[/red bold]")
                    return 1

    created = len(ctx.group_states)
    if ctx.failed_groups:
        ctx.progress.console.print(f"[yellow]{created} ready, {len(ctx.failed_groups)} failed[/yellow]")
    else:
        ctx.progress.console.print(f"[green]{created} ready[/green]")

    display_summaries: dict[str, str] = {}
    manifest_entries: dict[str, dict] = {}
    for name, shared_state in ctx.group_states.items():
        gc = ctx.group_config_by_name[name]
        agent_cls = get_agent_class(gc.agent)
        agent = create_agent(agent_cls, ctx.agent_args)
        summary = agent.group_setup_summary(shared_state)
        if summary:
            display_summaries[name] = summary
        manifest_entries[name] = {"agent": gc.agent, "shared_state": shared_state or {}}

    if display_summaries or ctx.failed_groups:
        ctx.progress.group_setup_table(display_summaries, ctx.failed_groups)

    # ``setup_errors.json`` is persisted before the early-return so a
    # run where *every* group failed setup still leaves an auditable
    # record on disk for ``belt view`` / ``belt aggregate`` to consume.
    ctx.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_setup_errors_sidecar(ctx)

    if not ctx.group_states:
        ctx.progress.console.print("[red bold]All group setups failed - aborting.[/red bold]")
        return 1

    ctx._manifest.register_run(os.getpid(), manifest_entries, str(ctx.run_dir))  # type: ignore[attr-defined]

    _write_fixture_provenance(ctx)

    # Groups that failed setup never produce a scenario turn, so their
    # progress bar would only ever show "0/N never advanced" and the
    # summary headline (e.g. "✅ 12/12") would mis-imply completeness.
    # Excluding them here keeps the live view honest; the per-group ✗
    # banner above and ``setup_errors`` in ``results.json`` are the
    # canonical surfaces for the skipped work.
    groups_for_display = [
        (mg.group_dir, mg.config, mg.scenarios) for mg in ctx.matched_groups if mg.name not in ctx.failed_groups
    ]
    workers = min(ctx.args.workers, ctx.total_scenarios)
    ctx.progress.start(groups_for_display, ctx.scenarios_root, workers, scenario_delay=ctx.args.scenario_delay)
    return None
