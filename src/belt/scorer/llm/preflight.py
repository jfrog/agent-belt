# (c) JFrog Ltd. (2026)

"""Parallel preflight for LLM judges.

Called by :func:`belt.scorer.pipeline.validate_scorers` before
``belt eval`` spawns its agent phase. For each configured
:class:`belt.scorer.llm.scorer.LLMScorer` (including the individual
judges inside a :class:`belt.scorer.llm.consensus.ConsensusScorer`),
issue the backend's cheapest model-callability probe in parallel and
aggregate every 4xx into a single composite error.

Why parallel: a multi-judge consensus run preflights N judges; doing
them serially would multiply the round-trip cost by N for no benefit.
Why a composite error: when two judges are misconfigured (typo +
missing key), reporting both at once means the user fixes their
config in one round-trip rather than discovering each failure one at
a time.

The 5xx-vs-4xx policy lives in :func:`_do_get_preflight` (see
:mod:`belt.scorer.llm.backend`): preflight returns silently on 5xx /
timeout / rate-limit (transient) and raises :class:`ScorerError` on
401/403/404 (config bug). This module is responsible only for fan-out
and aggregation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from loguru import logger

from belt.errors import ScorerError
from belt.scorer.base import BaseScorer
from belt.scorer.llm.consensus import ConsensusScorer
from belt.scorer.llm.scorer import LLMScorer

# Cap concurrency so a 10-judge consensus does not open 10 simultaneous
# sockets to the same provider host and trip its connection limits;
# providers also rate-limit per-IP, so a small pool keeps preflight
# friendly. The cap is intentionally lower than the typical
# ``--workers`` default because preflight runs once per ``belt eval``,
# not per scenario - throughput matters less than playing nicely with
# the provider.
_MAX_PREFLIGHT_WORKERS = 5


def collect_llm_scorers(scorers: Iterable[BaseScorer]) -> list[LLMScorer]:
    """Walk the scorer list and return every :class:`LLMScorer` that needs probing.

    Flattens :class:`ConsensusScorer` into its constituent judges so
    each provider+model pair is preflighted exactly once. The order is
    stable (consensus judges in declaration order) so the aggregated
    error message is deterministic across runs.
    """
    out: list[LLMScorer] = []
    seen: set[int] = set()
    for scorer in scorers:
        if isinstance(scorer, ConsensusScorer):
            for judge in scorer.judges:
                if id(judge) not in seen:
                    seen.add(id(judge))
                    out.append(judge)
        elif isinstance(scorer, LLMScorer):
            if id(scorer) not in seen:
                seen.add(id(scorer))
                out.append(scorer)
    return out


def _preflight_one(scorer: LLMScorer) -> tuple[str, BaseException | None]:
    """Run a single judge's preflight; return (label, error_or_None).

    Catches the backend's :class:`ScorerError` (config bug) so we can
    aggregate multiple failures into one composite error - re-raising
    eagerly would lose the second judge's diagnosis. Any non-belt
    exception (programmer error, bug in a third-party backend) is
    intentionally not caught and propagates.
    """
    backend = scorer.backend
    label = f"{scorer.judge_name} ({backend.provider_name()} / {scorer.config.model})"
    try:
        backend.preflight_model(scorer.config)
    except ScorerError as e:
        return label, e
    return label, None


def preflight_judges(scorers: Iterable[BaseScorer]) -> None:
    """Run every judge's ``preflight_model`` in parallel; raise on any 4xx.

    Iterates ``scorers``, fans out one probe per unique
    :class:`LLMScorer`, and collects results. If **any** probe raised
    :class:`ScorerError`, raise a single composite :class:`ScorerError`
    with every failure listed - so a multi-judge run reports all
    config bugs at once rather than one-per-rerun.

    No-op when there are no LLM scorers (``--modes rules``) - this
    function is safe to call unconditionally from preflight.
    """
    judges = collect_llm_scorers(scorers)
    if not judges:
        return

    failures: list[tuple[str, ScorerError]] = []
    max_workers = min(_MAX_PREFLIGHT_WORKERS, len(judges))
    logger.debug("Running judge preflight for {} judge(s) with {} worker(s)", len(judges), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_preflight_one, j): j for j in judges}
        for fut in as_completed(futures):
            label, err = fut.result()
            if err is not None:
                assert isinstance(err, ScorerError)
                failures.append((label, err))

    if not failures:
        return

    # Stable ordering for the composite message: by judge name. Without
    # this the message reshuffles on every run, which makes diffing CI
    # logs across reruns harder than it needs to be.
    failures.sort(key=lambda kv: kv[0])
    if len(failures) == 1:
        # Single failure: re-raise the original error verbatim. No
        # value in wrapping it because there's nothing to aggregate.
        raise failures[0][1]
    lines = [
        f"Judge preflight failed for {len(failures)} judge(s) before agent phase:",
        "",
    ]
    for label, err in failures:
        lines.append(f"  ── {label} ──")
        for body_line in str(err).splitlines():
            lines.append(f"  {body_line}")
        lines.append("")
    raise ScorerError("\n".join(lines).rstrip())
