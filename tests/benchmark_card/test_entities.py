# (c) JFrog Ltd. (2026)

"""Pydantic schema round-trips for ``benchmark_card.entities``.

The card crosses a process boundary (aggregator writes JSON; CI bots
and external tooling read it back), so the round-trip and
default-population guarantees are part of the public contract. Other
sibling suites cover behaviour; this one pins the data contract.
"""

from __future__ import annotations

from belt.benchmark_card import BenchmarkCard
from belt.constants import SCHEMA_VERSION

from .conftest import minimal_card


class TestSchemaRoundTrip:
    def test_minimal_card_round_trips(self) -> None:
        card = minimal_card()
        encoded = card.model_dump_json()
        rebuilt = BenchmarkCard.model_validate_json(encoded)
        assert rebuilt.run_id == card.run_id
        assert rebuilt.schema_version == SCHEMA_VERSION
        assert rebuilt.host.machine == "x86_64"

    def test_unknown_optional_fields_default(self) -> None:
        card = BenchmarkCard.model_validate_json(minimal_card().model_dump_json())
        # cost_timing/summary default to empty dataclasses; never None.
        assert card.cost_timing.total_cost_usd is None
        assert card.summary.total == 0
        assert card.summary.pass_rate == 0.0
