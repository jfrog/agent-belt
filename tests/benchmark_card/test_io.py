# (c) JFrog Ltd. (2026)

"""Disk write-out for ``benchmark_card.io``.

Exercises ``write_card`` end-to-end: both the JSON manifest and the
Markdown rendering land under the right names, and the JSON is
re-loadable.
"""

from __future__ import annotations

import json
from pathlib import Path

from belt.benchmark_card import write_card
from belt.constants import BENCHMARK_CARD_JSON_FILE, BENCHMARK_CARD_MD_FILE

from .conftest import minimal_card


class TestWriteCard:
    def test_writes_both_artifacts(self, tmp_path: Path) -> None:
        card = minimal_card()
        json_path, md_path = write_card(card, tmp_path)
        assert json_path == tmp_path / BENCHMARK_CARD_JSON_FILE
        assert md_path == tmp_path / BENCHMARK_CARD_MD_FILE
        loaded = json.loads(json_path.read_text())
        assert loaded["run_id"] == card.run_id
        md = md_path.read_text()
        assert "Benchmark Card" in md
        assert "Run identity" in md
