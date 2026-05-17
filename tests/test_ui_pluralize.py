# (c) JFrog Ltd. (2026)

"""Tests for ``belt._ui.pluralize`` and its consumers (#385 / 2.4)."""

from __future__ import annotations

from belt._ui import pluralize


def test_pluralize_singular():
    assert pluralize(1, "agent") == "1 agent"


def test_pluralize_zero_is_plural():
    assert pluralize(0, "agent") == "0 agents"


def test_pluralize_plural():
    assert pluralize(2, "agent") == "2 agents"


def test_pluralize_irregular():
    assert pluralize(1, "child", "children") == "1 child"
    assert pluralize(3, "child", "children") == "3 children"


def test_pluralize_large_number():
    assert pluralize(100, "scenario") == "100 scenarios"


def test_pluralize_negative_treated_as_plural():
    """Defensive: negative counts (shouldn't happen) at least don't render ``-1 agent``."""
    assert pluralize(-1, "agent") == "-1 agents"
