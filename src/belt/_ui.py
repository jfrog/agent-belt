# (c) JFrog Ltd. (2026)

"""Stderr-bound ``print`` helper for UI output.

Stdout is reserved for *data* output: pipeable results, a future
``--output {json,markdown,csv}`` flag, the rare command (``belt compare``)
that emits markdown by design. Everything else - status messages,
progress markers, error banners, "no scenarios found" notices,
``Loaded N scenarios``, the post-run pointer - is UI, and UI belongs on
stderr.

This matches the convention shipped by ``gh``, ``kubectl``, ``helm``,
``docker``, ``inspect``, ``promptfoo``, ``claude``, and ``cursor``: a
CLI is two things at once - an interactive tool for a human, and a
node in a Unix pipe. The only way both work is to keep stdout reserved
for what the next program in the pipeline might want. Putting the UI on
stderr is harmless even when stdout isn't piped, because terminals show
both streams.

Why a helper instead of redirecting ``sys.stdout`` once at CLI entry:
redirection is invisible at the call site, breaks the few legitimate
stdout writes (``belt compare`` markdown), and confuses anyone debugging
stream routing later. ``eprint`` is grep-able and intent-revealing.
"""

from __future__ import annotations

import sys
from typing import Any


def eprint(*args: Any, **kwargs: Any) -> None:
    """``print(...)`` bound to ``sys.stderr``.

    Callers can still pass ``file=`` to override (used by tests that
    capture into a buffer); the only behavioural change is the default
    destination.
    """
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def pluralize(n: int, singular: str, plural: str | None = None) -> str:
    """Render ``"N noun"`` with the correct singular/plural inflection.

    Avoids the ``"1 agent(s) ready"`` parenthetical that reads as careless.
    For irregular plurals pass ``plural`` explicitly (``pluralize(2, "child",
    "children")``); otherwise an ``"s"`` is appended.
    """
    if n == 1:
        return f"{n} {singular}"
    return f"{n} {plural if plural is not None else singular + 's'}"
