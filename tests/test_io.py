# (c) JFrog Ltd. (2026)

"""Unit tests for :mod:`belt._io`.

The card collector, the run-meta enricher, the per-fixture provenance
writer, and the runtime-info sidecar emitter all rely on the
"best-effort, never raise" contract these tests pin down. A regression
that lets a missing input bubble as :class:`FileNotFoundError` would
abort the surrounding eval; a regression that drops the trailing
newline would re-introduce noisy diffs in committed fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

from belt._io import read_json, write_json


class TestReadJson:
    def test_returns_parsed_dict(self, tmp_path: Path):
        p = tmp_path / "x.json"
        p.write_text('{"a": 1, "b": [2, 3]}')
        assert read_json(p) == {"a": 1, "b": [2, 3]}

    def test_returns_none_on_missing_file(self, tmp_path: Path):
        assert read_json(tmp_path / "nope.json") is None

    def test_returns_none_on_malformed_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("not json{")
        assert read_json(p) is None

    def test_returns_none_on_directory_path(self, tmp_path: Path):
        # Best-effort: a path that exists but is a directory must not raise.
        assert read_json(tmp_path) is None

    def test_accepts_string_path(self, tmp_path: Path):
        p = tmp_path / "x.json"
        p.write_text('{"k": "v"}')
        assert read_json(str(p)) == {"k": "v"}


class TestWriteJson:
    def test_writes_pretty_json_with_trailing_newline(self, tmp_path: Path):
        p = tmp_path / "out.json"
        assert write_json(p, {"b": 1, "a": 2}) is True
        text = p.read_text()
        # ``sort_keys=True`` is the default - ``a`` precedes ``b``.
        assert text == '{\n  "a": 2,\n  "b": 1\n}\n'

    def test_sort_keys_false_preserves_insertion_order(self, tmp_path: Path):
        p = tmp_path / "out.json"
        assert write_json(p, {"b": 1, "a": 2}, sort_keys=False) is True
        # Insertion order is preserved.
        assert p.read_text().startswith('{\n  "b": 1,')

    def test_indent_override(self, tmp_path: Path):
        p = tmp_path / "out.json"
        assert write_json(p, {"a": 1}, indent=4) is True
        assert "    " in p.read_text()

    def test_returns_false_on_unwritable_path(self, tmp_path: Path):
        # Path that points into a non-existent directory is unwritable;
        # the helper logs a debug message and returns False rather than
        # raising :class:`FileNotFoundError`.
        p = tmp_path / "nope" / "out.json"
        assert write_json(p, {"a": 1}) is False

    def test_round_trip(self, tmp_path: Path):
        p = tmp_path / "rt.json"
        data = {"nested": {"list": [1, 2, 3], "bool": True, "none": None}}
        assert write_json(p, data) is True
        assert read_json(p) == data

    def test_accepts_string_path(self, tmp_path: Path):
        p = tmp_path / "x.json"
        assert write_json(str(p), {"k": "v"}) is True
        assert json.loads(p.read_text()) == {"k": "v"}
