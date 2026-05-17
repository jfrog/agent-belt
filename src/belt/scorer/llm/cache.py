# (c) JFrog Ltd. (2026)

"""Content-addressable response cache for LLM judge calls.

Caches based on SHA256 of (model, temperature, seed, system_message, dynamic_message,
schema). Changing any input invalidates the cache entry.

Cache files are stored as JSON in a `.score_cache/` directory within the outcomes root.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any

from loguru import logger

from belt import envvars

# Disk budget for the LLM-judge response cache. Without a cap, a long-running
# eval campaign can fill the disk shared with other CI jobs. Override with
# ``envvars.CACHE_MAX_BYTES``; set to 0 to disable eviction.
_DEFAULT_CACHE_MAX_BYTES = 500 * 1024 * 1024  # 500 MiB


def _cache_max_bytes() -> int:
    raw = os.environ.get(envvars.CACHE_MAX_BYTES)
    if raw is None:
        return _DEFAULT_CACHE_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer {}={!r}", envvars.CACHE_MAX_BYTES, raw)
        return _DEFAULT_CACHE_MAX_BYTES
    return max(value, 0)


class ScoreCache:
    """File-based content-addressable cache for LLM judge responses."""

    def __init__(self, cache_dir: Path, *, max_bytes: int | None = None) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._hits = 0
        self._misses = 0
        # Serialise concurrent writes to the same key from multiple worker threads.
        # The OS-level rename via os.replace() makes the visible cache entry
        # atomic per-key; this lock prevents two workers in the same process
        # racing on the same tmp filename.
        self._write_lock = threading.Lock()
        self._max_bytes = _cache_max_bytes() if max_bytes is None else max(max_bytes, 0)

    @staticmethod
    def make_key(
        model: str,
        temperature: float,
        seed: int,
        system_message: str,
        dynamic_message: str,
        schema: dict,
    ) -> str:
        """Compute SHA256 cache key from all scoring inputs."""
        payload = json.dumps(
            {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "system_message": system_message,
                "dynamic_message": dynamic_message,
                "schema": schema,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve cached response, or None on miss."""
        path = self._dir / f"{key}.json"
        if not path.exists():
            self._misses += 1
            return None
        try:
            data = json.loads(path.read_text())
            self._hits += 1
            return data
        except Exception as e:
            logger.warning("Corrupt cache entry {}: {}", key[:12], e)
            path.unlink(missing_ok=True)
            self._misses += 1
            return None

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Store a response in the cache atomically.

        Writes to a per-call temp file in the same directory, then renames it
        into place via ``os.replace`` so concurrent readers never observe a
        partial JSON file. Cleans up the temp file on any failure.
        """
        path = self._dir / f"{key}.json"
        tmp_path = self._dir / f".{key}.{secrets.token_hex(4)}.tmp"
        with self._write_lock:
            try:
                payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
                tmp_path.write_text(payload)
                os.replace(tmp_path, path)
            except Exception as e:
                logger.warning("Failed to write cache entry {}: {}", key[:12], e)
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return
            self._evict_if_over_budget()

    def _evict_if_over_budget(self) -> None:
        """Drop oldest entries (by mtime) until total size fits ``max_bytes``.

        Best-effort: a missing or unstat-able entry is skipped. Called under
        ``_write_lock`` so concurrent ``put`` calls cannot race the eviction.
        Bounds the score cache so an adversarial scenario set cannot fill the
        disk via uncached judge calls.
        """
        if self._max_bytes <= 0:
            return
        try:
            entries: list[tuple[float, int, Path]] = []
            total = 0
            for child in self._dir.iterdir():
                if not child.is_file() or not child.name.endswith(".json"):
                    continue
                try:
                    st = child.stat()
                except OSError:
                    continue
                entries.append((st.st_mtime, st.st_size, child))
                total += st.st_size
            if total <= self._max_bytes:
                return
            entries.sort(key=lambda e: e[0])  # oldest first
            for _mtime, size, victim in entries:
                if total <= self._max_bytes:
                    break
                try:
                    victim.unlink()
                    total -= size
                    logger.debug("Cache eviction: dropped {} ({} bytes)", victim.name, size)
                except OSError as e:
                    logger.debug("Cache eviction skipped {}: {}", victim.name, e)
        except OSError as e:
            logger.warning("Cache eviction scan failed: {}", e)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
