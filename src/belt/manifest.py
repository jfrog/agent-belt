# (c) JFrog Ltd. (2026)

"""Run manifest - tracks active runs and their resources.

Supports concurrent eval runs by storing per-run entries keyed by PID.
On startup, orphaned resources (from crashed runs whose PID is dead)
are cleaned up automatically. Live runs are left untouched.

All mutations are serialized via a file lock (``filelock``) adjacent to the
manifest file. This makes concurrent ``belt eval`` processes sharing the
same ``OUTCOMES_ROOT`` directory safe from manifest corruption.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock
from loguru import logger

from belt.constants import MANIFEST_FILE, OUTCOMES_ROOT


class Manifest:
    def __init__(self, path: Path | None = None):
        self._path = path or (OUTCOMES_ROOT / MANIFEST_FILE)
        self._lock = FileLock(str(self._path) + ".lock", timeout=30)
        self._data: dict = self._read()

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        return self._validate(data)

    @staticmethod
    def _validate(data: object) -> dict:
        """Reject manifests that don't match the expected shape.

        ``cleanup_orphans`` invokes a delete callback against attacker-influenced
        manifest entries. A corrupt or hostile manifest must not be able to
        feed surprising types into the callback; any deviation collapses the
        manifest to empty so cleanup is a no-op.
        """
        if not isinstance(data, dict):
            return {}
        runs = data.get("runs")
        if runs is None:
            return data
        if not isinstance(runs, list):
            data.pop("runs", None)
            return data
        clean_runs: list[dict] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            pid = run.get("pid")
            if pid is not None and not isinstance(pid, int):
                continue
            groups = run.get("groups", [])
            if not isinstance(groups, list):
                continue
            clean_groups = [g for g in groups if isinstance(g, dict)]
            run_dir = run.get("run_dir")
            if run_dir is not None and not isinstance(run_dir, str):
                continue
            create_time = run.get("create_time")
            if create_time is not None and not isinstance(create_time, (int, float)):
                create_time = None
            clean_runs.append(
                {
                    "pid": pid,
                    "create_time": create_time,
                    "run_dir": run_dir,
                    "groups": clean_groups,
                }
            )
        data["runs"] = clean_runs
        return data

    def _refresh(self) -> None:
        """Re-read on-disk state before mutating to avoid stale overwrites."""
        self._data = self._read()

    def _write(self) -> None:
        # Owner-only directory + manifest file. The manifest stores PIDs and
        # provider-specific shared-state for cleanup; treating it as sensitive
        # blocks tampering on shared hosts.
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._path.write_text(json.dumps(self._data, indent=2) + "\n")
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Acquire the file lock, refresh from disk, yield, then release."""
        with self._lock:
            self._refresh()
            yield

    @property
    def latest_run(self) -> str | None:
        return self._data.get("latest_run")

    @property
    def runs(self) -> list[dict]:
        return self._data.get("runs", [])

    def register_run(
        self,
        pid: int,
        groups: dict[str, dict],
        run_dir: str,
    ) -> None:
        """Record a new run with its PID and per-group resource info.

        ``groups`` maps group name → ``{"agent": str, "shared_state": dict}``.
        Each entry is flattened into the manifest's ``groups`` list with a ``group`` key.
        Also records the process ``create_time`` (epoch seconds) so PID-reuse
        false positives can be ruled out in :meth:`_is_pid_alive` later.
        """
        create_time = self._pid_create_time(pid)
        with self._locked():
            self._data["latest_run"] = run_dir
            runs = self._data.setdefault("runs", [])
            entries = []
            for group, info in groups.items():
                entry = {"group": group}
                entry.update(info)
                entries.append(entry)
            runs.append(
                {
                    "pid": pid,
                    "create_time": create_time,
                    "run_dir": run_dir,
                    "groups": entries,
                }
            )
            self._write()

    def unregister_run(self, pid: int) -> None:
        """Remove this run's entry after successful cleanup."""
        with self._locked():
            runs = self._data.get("runs", [])
            self._data["runs"] = [r for r in runs if r.get("pid") != pid]
            if not self._data["runs"]:
                del self._data["runs"]
            self._write()

    @staticmethod
    def _pid_create_time(pid: int) -> float | None:
        """Return process create_time (epoch seconds) if psutil is available.

        psutil is an optional dependency; when missing we degrade to the
        original ``os.kill(pid, 0)`` semantics. With psutil we record the
        creation timestamp at registration and compare it on liveness checks
        to defeat PID-reuse races (Linux ``pid_max=4194304``, Darwin ~99,999;
        on Darwin in particular reuse happens within minutes for short-lived
        runs).
        """
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            return float(psutil.Process(pid).create_time())
        except Exception:
            return None

    @classmethod
    def _is_pid_alive(cls, pid: int, expected_create_time: float | None = None) -> bool:
        """Check whether ``pid`` is still alive, defeating PID reuse if possible.

        With ``expected_create_time`` and psutil available, we compare the
        live process creation time to the manifest record (1-second tolerance
        for clock granularity); a mismatch means the original process exited
        and the kernel reissued the PID - treat as dead and cleanup orphans.
        """
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        if expected_create_time is None:
            return True
        live_create = cls._pid_create_time(pid)
        if live_create is None:
            return True
        return abs(live_create - expected_create_time) < 1.0

    def cleanup_orphans(self, delete_fn: Callable[[dict], None]) -> int:
        """Delete resources from runs whose PID is no longer alive.

        ``delete_fn`` receives the full entry dict (group, agent, shared_state).
        """
        with self._locked():
            runs = self._data.get("runs", [])
            if not runs:
                return 0

            live_runs: list[dict] = []
            deleted = 0

            for run in runs:
                pid = run.get("pid")
                expected_create = run.get("create_time")
                if pid is not None and self._is_pid_alive(pid, expected_create):
                    live_runs.append(run)
                    logger.info("Skipping live run (pid {}): {}", pid, run.get("run_dir", "?"))
                    continue

                for entry in run.get("groups", []):
                    group = entry.get("group", "?")
                    try:
                        delete_fn(entry)
                        deleted += 1
                        logger.info("Deleted orphan resource {}: {}", group, entry)
                    except Exception as e:
                        logger.warning("Failed to delete orphan {}: {}", group, e)

            self._data["runs"] = live_runs
            if not self._data["runs"]:
                self._data.pop("runs", None)
            self._write()
            return deleted
