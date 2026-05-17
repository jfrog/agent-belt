# (c) JFrog Ltd. (2026)

"""Scoring strategies - per-agent definition of LLM judge dimensions and context.

Each agent provides a ScoringStrategy that tells the LLM scorer:
- Which dimensions to evaluate (and their rubric descriptions)
- What agent-specific context to include in the system message
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from belt.errors import ConfigError

# Verdict-scale literal expressed once so authoring sites and validation
# both use the same source of truth.
DimensionKind = Literal["binary", "ternary"]


@dataclass
class DimensionDef:
    """Definition of a single LLM judge scoring dimension.

    A dimension declares its **verdict scale** (``kind``) independently
    of whether the judge may also return ``inconclusive``
    (``allow_inconclusive``). The two axes are orthogonal so any of the
    four combinations is expressible:

    +-----------+---------------------+----------------------------------+
    | kind      | allow_inconclusive  | Verdicts the judge may return    |
    +===========+=====================+==================================+
    | ternary   | False (default)     | low / medium / high              |
    +-----------+---------------------+----------------------------------+
    | ternary   | True                | low / medium / high / inconclusive|
    +-----------+---------------------+----------------------------------+
    | binary    | False               | pass / fail                      |
    +-----------+---------------------+----------------------------------+
    | binary    | True                | pass / fail / inconclusive       |
    +-----------+---------------------+----------------------------------+

    ``inconclusive`` always counts as a failure in the headline pass-
    rate (so the judge has no incentive to hedge), but the aggregator
    reports it separately so reviewers can distinguish "agent did it
    wrong" from "evidence missing".

    Rubric fields (``high``/``medium``/``low``, ``pass_``/``fail``)
    have safe defaults so a dimension declared with only ``name`` and
    ``description`` still validates and renders. The ``pass_`` field
    accepts the JSON key ``pass`` via :func:`from_config_dict`; ``pass``
    is a Python keyword and cannot be used as a dataclass attribute.
    """

    name: str
    description: str = ""
    kind: DimensionKind = "ternary"
    allow_inconclusive: bool = False
    high: str = "fully meets expectations"
    medium: str = "partially meets expectations"
    low: str = "does not meet expectations"
    pass_: str = "meets expectations"
    fail: str = "does not meet expectations"
    evidence_hints: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("binary", "ternary"):
            raise ConfigError(f"DimensionDef '{self.name}': kind must be 'binary' or 'ternary', got {self.kind!r}")

    @classmethod
    def from_config_dict(cls, raw: dict) -> "DimensionDef":
        """Build a ``DimensionDef`` from a config dict accepting JSON aliases.

        Translates the JSON key ``pass`` to the dataclass attribute
        ``pass_`` (Python keyword constraint) and rejects unknown keys
        loudly so authors find typos at parse time, not at judge time.
        """
        translated = dict(raw)
        if "pass" in translated:
            translated["pass_"] = translated.pop("pass")
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(translated) - allowed)
        if unknown:
            raise ConfigError(
                f"DimensionDef '{translated.get('name', '?')}': unknown field(s) {unknown!r}. "
                f"Valid fields: {sorted(allowed)!r}"
            )
        return cls(**translated)

    @property
    def allowed_score_values(self) -> list[str]:
        """Verdict tokens this dimension may emit, in display order."""
        if self.kind == "binary":
            base = ["pass", "fail"]
        else:
            base = ["low", "medium", "high"]
        if self.allow_inconclusive:
            base.append("inconclusive")
        return base


@dataclass
class ScoringStrategy:
    """Base scoring strategy - defines dimensions and context for the LLM judge."""

    dimensions: list[DimensionDef] = field(default_factory=list)
    agent_context: str = ""

    @property
    def dimension_names(self) -> list[str]:
        return [d.name for d in self.dimensions]

    def build_dimensions_prompt(self) -> str:
        """Render dimension rubrics as markdown for the LLM system message.

        The list number for ``inconclusive`` is computed from the scale
        length (``binary`` → 3, ``ternary`` → 4) so the rubric stays a
        cleanly enumerated list. Hardcoding the trailing number is a
        rendering bug: in the ternary case it produces two items numbered
        ``3.``, which smaller judges parse as a malformed rubric.
        """
        parts: list[str] = []
        for d in self.dimensions:
            section = f"## {d.name.replace('_', ' ').title()}\n\n{d.description}\n\n"
            if d.kind == "binary":
                section += f"1. pass - {d.pass_}\n2. fail - {d.fail}\n"
                inconclusive_num = 3
            else:
                section += f"1. high - {d.high}\n2. medium - {d.medium}\n3. low - {d.low}\n"
                inconclusive_num = 4
            if d.allow_inconclusive:
                section += (
                    f"{inconclusive_num}. inconclusive - the available evidence does not let you decide. "
                    "Only pick this when the transcript or workspace does not show "
                    "behaviour relevant to this dimension; quote the gap in `reasoning`. "
                    "Inconclusive counts as a failure in the headline pass-rate.\n"
                )
            if d.evidence_hints:
                section += f"\nLook for: {d.evidence_hints}\n"
            parts.append(section)
        return "\n".join(parts)

    def build_schema(self) -> dict:
        """Generate a strict JSON schema for the LLM judge response.

        Each dimension carries its own per-dimension verdict enum so
        the judge cannot return a verdict that does not belong to that
        dimension's scale (for example, returning ``"medium"`` for a
        binary dimension). Schema is OpenAI ``strict: true`` compliant
        - every property is required and ``additionalProperties`` is
        ``false`` at every level.
        """
        properties: dict = {}
        required: list[str] = []
        defs: dict = {}

        for d in self.dimensions:
            verdict_def_name = f"{_sanitise(d.name)}_Verdict"
            defs[verdict_def_name] = {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "1-3 sentences with specific evidence from output.",
                    },
                    "score": {
                        "type": "string",
                        "enum": d.allowed_score_values,
                    },
                },
                "required": ["reasoning", "score"],
                "additionalProperties": False,
            }
            properties[d.name] = {"$ref": f"#/$defs/{verdict_def_name}"}
            required.append(d.name)

        properties["overall_pass"] = {
            "type": "boolean",
            "description": (
                "true if and only if no dimension scored low (ternary), " "fail (binary), or inconclusive."
            ),
        }
        required.append("overall_pass")

        return {
            "type": "object",
            "$defs": defs,
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


def _sanitise(name: str) -> str:
    """Make a dimension name safe to embed in a ``$defs`` key.

    JSON Schema ``$defs`` keys can be arbitrary strings, but some
    backends and tooling reject characters outside ``[A-Za-z0-9_-]``.
    Replace anything else with ``_`` so a malformed dimension name in
    a config file still produces a valid schema rather than a 400 from
    the provider.
    """
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


# ── Built-in strategies ──

GENERIC_DIMENSIONS = [
    DimensionDef(
        name="execution",
        description="Did the agent run without errors?",
        high="clean execution, no errors, all turns completed",
        medium="minor warnings but functionally correct",
        low="crashes, errors, incomplete turns",
        evidence_hints="error messages, tracebacks, incomplete output.",
    ),
    DimensionDef(
        name="trajectory",
        description="Did the agent use the correct tools and approach?",
        high="correct tools for every task, efficient approach",
        medium="right tools but suboptimal approach",
        low="wrong tools, missing actions, loops",
        evidence_hints="tool names, action sequence, unnecessary repetitions.",
    ),
    DimensionDef(
        name="response_quality",
        description=(
            "Was the final response accurate and useful? "
            "When workspace files are available, verify the agent's claims against actual file contents."
        ),
        high="accurate, coherent, well-formatted; file changes match claims",
        medium="functional but generic or partially wrong; minor discrepancies with workspace",
        low="hallucinated data, contradicts tool results or workspace files, incoherent",
        evidence_hints="response content vs tool results, workspace file contents vs claimed changes.",
    ),
    DimensionDef(
        name="efficiency",
        description="Did the agent minimize unnecessary work?",
        high="minimal turns, no redundant actions",
        medium="some redundancy or missed optimization",
        low="excessive turns, repeated actions, unnecessary clarification",
        evidence_hints="number of turns, duplicate operations.",
    ),
]

GENERIC_AGENT_CONTEXT = """You are evaluating a CLI-based AI agent.

For each scenario you receive:
1. Scenario definition - the user's messages and test metadata
2. CLI output - the raw output from the agent execution
3. Thread state - agent internal state or git diff (if available)
4. Workspace files - actual file contents captured after the agent ran (ground truth)

CRITICAL: When workspace files are present, they are the authoritative source of truth.
If the agent's CLI output claims it made changes but the workspace files show otherwise,
the workspace files are correct. Score based on what actually happened, not what the
agent said happened.
"""


def default_scoring_strategy() -> ScoringStrategy:
    return ScoringStrategy(dimensions=GENERIC_DIMENSIONS, agent_context=GENERIC_AGENT_CONTEXT)
