# (c) JFrog Ltd. (2026)

"""Pins the watch ``system/init`` rendering shape (#385 / 3.4).

The previous rendering surfaced ``🔌 Codex 5.3 · 12 tools`` which read
as "the icon means *Codex 5.3 the model* is the connection target" -
ambiguous in a row that already shows the agent name. The fix prefixes
the model with an explicit ``model:`` label so the meaning is verbatim.
"""

from __future__ import annotations

import json

from belt.commands.watch import StreamParser


def _parse(payload: dict):
    return StreamParser().parse_line(json.dumps(payload))


def test_system_init_uses_model_label():
    event = _parse(
        {
            "type": "system",
            "subtype": "init",
            "model": "Codex 5.3",
            "tools": ["a", "b", "c"],
        }
    )
    assert event is not None
    assert "model: Codex 5.3" in event.summary
    assert "3 tools" in event.summary


def test_system_init_without_tools():
    event = _parse({"type": "system", "subtype": "init", "model": "gpt-5.4-mini", "tools": []})
    assert event is not None
    assert event.summary == "model: gpt-5.4-mini"


def test_system_init_without_model():
    event = _parse({"type": "system", "subtype": "init", "tools": ["x"]})
    assert event is not None
    assert event.summary == "1 tools"


def test_system_init_empty_returns_none():
    event = _parse({"type": "system", "subtype": "init"})
    assert event is None
