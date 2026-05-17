# (c) JFrog Ltd. (2026)

"""Tests for shared CLI utilities."""

from __future__ import annotations

import argparse

import pytest

from belt.cli_utils import parse_kv_args


class TestParseKvArgs:
    def test_none_returns_empty(self):
        assert parse_kv_args(None) == {}

    def test_empty_list_returns_empty(self):
        assert parse_kv_args([]) == {}

    def test_single_pair(self):
        assert parse_kv_args(["key=value"]) == {"key": "value"}

    def test_multiple_pairs(self):
        result = parse_kv_args(["model=gpt-4.1", "temperature=0.0", "seed=42"])
        assert result == {"model": "gpt-4.1", "temperature": "0.0", "seed": "42"}

    def test_value_with_equals_sign(self):
        result = parse_kv_args(["url=http://host:8080/path?a=1"])
        assert result == {"url": "http://host:8080/path?a=1"}

    def test_whitespace_stripped(self):
        result = parse_kv_args(["  key  =  value  "])
        assert result == {"key": "value"}

    def test_empty_value(self):
        result = parse_kv_args(["key="])
        assert result == {"key": ""}

    def test_no_equals_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid KEY=VALUE"):
            parse_kv_args(["no_equals_here"])

    def test_duplicate_key_last_wins(self):
        result = parse_kv_args(["key=first", "key=second"])
        assert result == {"key": "second"}
