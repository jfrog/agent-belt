# (c) JFrog Ltd. (2026)

"""Shared fixtures for the ``tests/benchmark_card/`` mirror suite.

The card builder, renderer, and writer all need a populated
:class:`BenchmarkCard` to exercise. Centralising the minimal fixture
here keeps the per-file suites focused on the behaviour they cover and
ensures a future schema field is added in one place rather than
copy-pasted across siblings.
"""

from __future__ import annotations

from belt.benchmark_card import BeltProvenance, BenchmarkCard, HostProvenance, Invocation, ScenarioSelection


def minimal_card() -> BenchmarkCard:
    """Smallest schema-valid card the public Pydantic surface accepts.

    Sibling suites import and tweak this baseline rather than
    constructing a full card per test - the only goal is "satisfy the
    required fields"; everything else falls through to the schema's
    own defaults so this stays in lock-step with future required-field
    changes.
    """
    return BenchmarkCard(
        run_id="20260101-000000-deadbeef",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:01:00Z",
        belt=BeltProvenance(version="9.9.9", install_kind="wheel"),
        host=HostProvenance(
            os="Linux 6.0",
            machine="x86_64",
            python_version="3.12.0",
            python_implementation="CPython",
            package_versions={"agent-belt": "9.9.9"},
        ),
        invocation=Invocation(
            argv=["belt", "eval", "/tmp/scn"],
            parsed_args={"modes": "rules"},
            cwd="/tmp",
        ),
        scenarios=ScenarioSelection(scenarios_root="/tmp/scn"),
    )
