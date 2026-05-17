# (c) JFrog Ltd. (2026)

"""Base scorer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from belt.entities import TurnOutput
from belt.scenario import Scenario
from belt.scorer.entities import ScorerResult


class BaseScorer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used as key in ScenarioScore (e.g. 'rules', 'llm')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this scorer can run (credentials, dependencies, etc.)."""

    @abstractmethod
    def score(
        self,
        scenario: Scenario,
        turn_outputs: list[TurnOutput],
    ) -> ScorerResult | None:
        """Score a scenario given its normalized turn outputs."""
