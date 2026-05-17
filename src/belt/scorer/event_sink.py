# (c) JFrog Ltd. (2026)

"""NDJSON event sink + judge → progress fan-out.

Named ``event_sink`` (not ``events``) to avoid shadowing
``scorer/llm/events.py`` which defines the ``ScoreEvent`` dataclass that
this module consumes.  Three responsibilities:

- ``NdjsonWriter`` - append-only, lock-guarded writer for ``score_stream.ndjson``.
- ``build_event_sink`` - fan-out callback (NDJSON file + LiveProgress panel).
- ``wire_event_callbacks`` - attach the callback to direct + consensus judges
  (judges inside a ``ConsensusScorer`` are rewrapped so their events carry
  the originating judge name).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

from belt.progress import ScorerProgress
from belt.scorer import BaseScorer, ConsensusScorer, LLMScorer
from belt.scorer.llm.events import ScoreEvent, format_score_event


class NdjsonWriter:
    """Append-only NDJSON file writer for score events.

    Thread-safe: ``write()`` and ``close()`` are guarded by an internal lock so
    multiple worker threads writing verdict events do not interleave at the
    buffered-text layer (CPython's GIL does not atomicise ``TextIOWrapper``
    writes across the buffer/flush boundary).
    """

    def __init__(self, fh) -> None:
        self._fh = fh
        self._lock = threading.Lock()

    @classmethod
    def from_path(cls, path: Path) -> "NdjsonWriter":
        """Open ``path`` for writing and wrap it in a lock-guarded writer.

        Mirrors ``runner.orchestrator._BoundedStreamWriter.from_path`` -
        opening here keeps the bare ``open()`` out of the call sites and
        means callers don't need to suppress lint warnings about unmanaged
        file handles (ownership transfers into the writer's ``close()``).
        """
        return cls(open(path, "w"))  # noqa: SIM115 - handle owned by NdjsonWriter

    def write(self, event: ScoreEvent) -> None:
        with self._lock:
            try:
                self._fh.write(json.dumps(event.to_dict()) + "\n")
                self._fh.flush()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass


def build_event_sink(
    progress: ScorerProgress | None,
    ndjson_writer: NdjsonWriter | None,
) -> Callable[[ScoreEvent], None]:
    """Create a callback that fans out events to progress display and/or NDJSON file."""

    def _sink(event: ScoreEvent) -> None:
        if ndjson_writer is not None:
            ndjson_writer.write(event)
        if progress is not None:
            formatted = format_score_event(event)
            progress.add_event(event.scenario, formatted)

    return _sink


def wire_event_callbacks(scorers: list[BaseScorer], callback: Callable[[ScoreEvent], None] | None) -> None:
    """Attach the event callback to all LLM scorers (direct or inside consensus)."""
    if callback is None:
        return
    llm_count = sum(1 for s in scorers if isinstance(s, LLMScorer))
    for s in scorers:
        if isinstance(s, ConsensusScorer):
            s.set_on_event(callback)
        elif isinstance(s, LLMScorer):
            if llm_count > 1:
                name = s.judge_name

                def _wrap(cb: Callable[[ScoreEvent], None], jn: str) -> Callable[[ScoreEvent], None]:
                    def _inner(event: ScoreEvent) -> None:
                        event.judge = jn
                        cb(event)

                    return _inner

                s.on_event = _wrap(callback, name)
            else:
                s.on_event = callback
