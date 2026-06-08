# (c) JFrog Ltd. (2026)

"""Verify checks - assert on a deterministic command's captured result.

The runner executes a ``VerifySpec`` command in the worktree and records a
:class:`belt.entities.VerifyResult` (per-turn on ``TurnOutput.verify_result``,
per-scenario on the final turn's ``TurnOutput.scenario_verify_result``). This
module turns that result into ``CheckEntry`` rows under the ``verify``
dimension - so ``--threshold rules/verify:0`` gates on it like any other
dimension. A missing result (command did not run, e.g. the agent errored on
that turn) is reported as a tri-state skip (``passed=None``), never a false
failure.
"""

from __future__ import annotations

from typing import Optional

from belt._sanitize import sanitize
from belt.entities import VerifyResult
from belt.scenario import VerifySpec
from belt.scorer.payloads import CheckEntry

# Length of the stdout excerpt surfaced in a failure ``details`` so a user can
# see WHY the command failed (e.g. the pytest error) without opening
# ``turn_N_output.json``. Kept short so it fits a one-line table cell.
_DETAILS_STDOUT_TAIL_CHARS = 240


def _stdout_tail(stdout: str) -> str:
    """Return a single-line, sanitized tail of ``stdout`` for a failure message.

    The runner already ANSI-strips captured stdout; this additionally strips
    residual control bytes and collapses whitespace via
    :func:`belt._sanitize.sanitize` so the excerpt is safe to drop into a
    one-line ``details`` cell (the per-sink exporters still escape ``details``
    on top of this). Returns ``""`` when there is nothing useful to show.
    """
    cleaned = " ".join(sanitize(stdout or "").split())
    if not cleaned:
        return ""
    if len(cleaned) > _DETAILS_STDOUT_TAIL_CHARS:
        cleaned = "..." + cleaned[-_DETAILS_STDOUT_TAIL_CHARS:]
    return cleaned


def has_verify(spec: Optional[VerifySpec]) -> bool:
    """True when a ``verify`` block is declared for this turn / scenario."""
    return spec is not None


def check_verify(
    spec: VerifySpec,
    result: Optional[VerifyResult],
    *,
    turn_idx: Optional[int],
) -> list[CheckEntry]:
    """Assert a captured verify ``result`` against its ``spec``.

    ``turn_idx`` is the turn index for a per-turn verify, or ``None`` for a
    per-scenario (end-of-conversation) verify - a turn-less ``CheckEntry``
    that the renderers already handle. A ``None`` ``result`` means the command
    never ran; emit a single skipped check.
    """
    # Short command label so each check self-describes WHAT ran (parallels
    # ``[state] file_exists(path)``); the dimension tag is already ``verify``,
    # so the check text never repeats the word.
    cmd_label = " ".join(spec.cmd)
    if len(cmd_label) > 60:
        cmd_label = cmd_label[:57] + "..."

    if result is None:
        return [
            CheckEntry(
                dimension="verify",
                check=f"exit_code=={spec.exit_code} ({cmd_label})",
                passed=None,
                details="skipped (command did not run)",
                turn_idx=turn_idx,
            )
        ]

    results: list[CheckEntry] = []
    exit_ok = result.exit_code == spec.exit_code
    if exit_ok:
        exit_details = ""
    else:
        # Surface a short, sanitized tail of the command's output so a failed
        # run shows *why* (e.g. the pytest error) right in the report.
        exit_details = f"got exit_code {result.exit_code}"
        tail = _stdout_tail(result.stdout)
        if tail:
            exit_details += f"; stdout: {tail}"
    results.append(
        CheckEntry(
            dimension="verify",
            check=f"exit_code=={spec.exit_code} ({cmd_label})",
            passed=exit_ok,
            details=exit_details,
            turn_idx=turn_idx,
        )
    )
    for substring in spec.output_contains:
        found = substring in (result.stdout or "")
        results.append(
            CheckEntry(
                dimension="verify",
                check=f"stdout contains {substring!r}",
                passed=found,
                details="" if found else "substring not found in stdout",
                turn_idx=turn_idx,
            )
        )
    return results
