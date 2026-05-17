# (c) JFrog Ltd. (2026)

"""``belt gc`` - prune old run directories under ``outcomes/``.

Without retention controls, long-lived CI or developer machines accumulate
run directories indefinitely, and a single adversarial agent emitting
gigabytes of output can fill the disk shared with other jobs. This module
implements policy-based pruning that is safe to run unattended.

Pruning policy
~~~~~~~~~~~~~~
1. List immediate child directories of ``OUTCOMES_ROOT`` (each is one run).
2. Skip runs registered as live in ``.manifest.json`` - never delete an
   in-flight run, even if it predates ``--keep-last``.
3. Sort the remaining runs by mtime (newest first).
4. With ``--keep-last N``: retain the N most recent, delete the rest.
5. With ``--older-than DAYS``: also delete anything older than that wall-clock
   threshold, applied AFTER the keep-last filter.
6. ``--dry-run`` reports decisions without touching the filesystem.

Both flags compose: ``--keep-last 50 --older-than 30`` means "keep at most 50,
and within those still drop anything older than 30 days".
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from loguru import logger

from belt._ui import eprint
from belt.constants import MANIFEST_FILE, OUTCOMES_ROOT
from belt.manifest import Manifest


def _live_run_dirs(outcomes_root: Path) -> set[Path]:
    """Return the set of run dirs the manifest currently considers in-flight."""
    manifest_path = outcomes_root / MANIFEST_FILE
    if not manifest_path.exists():
        return set()
    try:
        m = Manifest(manifest_path)
    except Exception as e:
        logger.warning("manifest at {} unreadable: {}", manifest_path, e)
        return set()
    live: set[Path] = set()
    for entry in m.runs:
        run_dir = entry.get("run_dir")
        pid = entry.get("pid")
        if not run_dir or not isinstance(pid, int):
            continue
        if Manifest._is_pid_alive(pid, entry.get("create_time")):
            try:
                live.add(Path(run_dir).resolve())
            except OSError:
                continue
    return live


def _candidate_run_dirs(outcomes_root: Path) -> list[Path]:
    """List immediate child dirs of ``outcomes_root`` (skipping dotfiles)."""
    if not outcomes_root.exists():
        return []
    out: list[Path] = []
    for child in outcomes_root.iterdir():
        if child.name.startswith("."):
            continue
        if not child.is_dir():
            continue
        out.append(child)
    return out


def plan_deletions(
    outcomes_root: Path,
    *,
    keep_last: int | None,
    older_than_days: float | None,
    now: float | None = None,
) -> tuple[list[Path], list[Path]]:
    """Compute (to_delete, to_keep) for the configured policy.

    Live runs from the manifest are always kept. ``keep_last`` and
    ``older_than_days`` compose: a run is deleted if it is past the keep-last
    cutoff *or* older than the wall-clock threshold.
    """
    live = _live_run_dirs(outcomes_root)
    candidates = _candidate_run_dirs(outcomes_root)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    to_delete: list[Path] = []
    to_keep: list[Path] = []
    threshold = None
    if older_than_days is not None:
        threshold = (now if now is not None else time.time()) - older_than_days * 86400

    for idx, run in enumerate(candidates):
        try:
            resolved = run.resolve()
        except OSError:
            resolved = run
        if resolved in live:
            to_keep.append(run)
            continue
        past_keep_last = keep_last is not None and idx >= keep_last
        too_old = threshold is not None and run.stat().st_mtime < threshold
        if past_keep_last or too_old:
            to_delete.append(run)
        else:
            to_keep.append(run)
    return to_delete, to_keep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="belt gc",
        description="Prune old run directories under outcomes/.",
    )
    # Flags alphabetised by long flag name; enforced by
    # ``tests/test_cli_order.py``.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report decisions without deleting",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=50,
        help="Retain the N most recent runs (default: %(default)s; 0 disables)",
    )
    parser.add_argument(
        "--older-than",
        type=float,
        default=None,
        metavar="DAYS",
        help="Also delete runs older than this many days",
    )
    parser.add_argument(
        "--outcomes-dir",
        default=str(OUTCOMES_ROOT),
        help="Outcomes directory to clean (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    outcomes_root = Path(args.outcomes_dir).resolve()
    if not outcomes_root.exists():
        eprint(f"  outcomes dir not found: {outcomes_root}")
        return 1

    keep_last = args.keep_last if args.keep_last and args.keep_last > 0 else None
    to_delete, to_keep = plan_deletions(outcomes_root, keep_last=keep_last, older_than_days=args.older_than)

    eprint(f"  outcomes: {outcomes_root}")
    eprint(f"  total runs: {len(to_keep) + len(to_delete)}")
    eprint(f"  keeping:    {len(to_keep)}")
    eprint(f"  deleting:   {len(to_delete)}{' (dry-run)' if args.dry_run else ''}")

    failures = 0
    for run in to_delete:
        if args.dry_run:
            eprint(f"    would delete: {run.name}")
            continue
        try:
            shutil.rmtree(run)
            eprint(f"    deleted: {run.name}")
        except OSError as e:
            failures += 1
            eprint(f"    failed: {run.name} ({e})", file=sys.stderr)
    return 0 if failures == 0 else 1
