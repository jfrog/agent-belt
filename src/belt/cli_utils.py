# (c) JFrog Ltd. (2026)

"""Shared CLI utilities for belt sub-commands."""

from __future__ import annotations

import argparse


def parse_kv_args(raw: list[str] | None) -> dict[str, str]:
    """Parse repeatable KEY=VALUE arguments into a dict."""
    if not raw:
        return {}
    result: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid KEY=VALUE format: '{item}'. Expected format: key=value")
        key, _, value = item.partition("=")
        result[key.strip()] = value.strip()
    return result
