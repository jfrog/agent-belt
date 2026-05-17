# (c) JFrog Ltd. (2026)

"""Documentation parity for the agent error-type taxonomy.

`ERROR_TYPES` in :mod:`belt.entities` is the single source of
truth for the cross-phase contract carried in
`TurnOutput.error_type`. Documentation under
`docs/glossary/OUTCOMES.md` must list every token, and must not
list any token that is not in the constant - otherwise consumers
write code against documented tokens that the framework never emits
(or rely on emitted tokens that the docs never advertise).

Mirrors the cheap-and-fast pattern used by other doc-parity tests
(providers, agents): runs in CI on every PR.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from belt.entities import ERROR_TYPES

DOC_PATH = Path(__file__).parent.parent / "docs" / "glossary" / "OUTCOMES.md"


def _docfile_text() -> str:
    assert DOC_PATH.is_file(), f"missing doc file: {DOC_PATH}"
    return DOC_PATH.read_text()


class TestErrorTypeDocParity:
    @pytest.mark.parametrize("token", sorted(ERROR_TYPES))
    def test_token_documented(self, token: str) -> None:
        """Every constant in ``ERROR_TYPES`` appears verbatim in the docs.

        We check for the bare token (``authentication_failed``) rather
        than its constant name (``AUTHENTICATION_FAILED``) because the
        token is what surfaces in JSON artifacts and CLI output.
        """
        text = _docfile_text()
        assert f"`{token}`" in text, (
            f"Token {token!r} (from belt.entities.ERROR_TYPES) is not "
            f"documented in {DOC_PATH.name}. Add it to the error-type table "
            f"under '## Agent runtime errors (`error_type`)'."
        )

    def test_no_undocumented_tokens_advertised(self) -> None:
        """No token in the doc is missing from ``ERROR_TYPES``.

        Catches the inverse direction: a doc table row claims a
        token the framework never emits (typo, removed feature, future
        plan that wasn't implemented). The check is strict for the
        documented set under the error-type heading; tokens that
        appear elsewhere in prose (e.g. mentioned in passing) are
        ignored to keep the parity check focused on the contract
        table.
        """
        import re

        text = _docfile_text()
        # Locate the section: from "## Agent runtime errors" through
        # the next ``## `` heading (exclusive). The error-type
        # contract table is fully contained inside this section.
        section_match = re.search(
            r"##\s+(?:\d+\.\s+)?Agent runtime errors.*?(?=\n## )",
            text,
            re.DOTALL,
        )
        assert section_match, (
            f"Could not locate 'Agent runtime errors' section in {DOC_PATH.name}. "
            f"Restore the heading - the doc-parity test depends on it."
        )
        section = section_match.group(0)

        # Pull tokens from inline-code spans inside the section
        # (e.g. ``authentication_failed``). The intent is to match
        # token-shaped strings only, so we restrict to lowercase +
        # underscore.
        documented = set(re.findall(r"`([a-z][a-z_]*)`", section))
        # Strip non-token strings that legitimately appear in prose
        # within the section (field names, type names, JSON keys).
        non_tokens = {
            "agent_errors",
            "by_error_type",
            "scenarios_with_errors",
            "scenarios_total",
            "vacuous_passes",
            "remediation",
            "per_scenario",
            "scenario",
            "passed",
            "vacuous_pass",
            "error_types",
            "first_reply_text",
            "error_type",
            "has_error",
            "results",
            "json",
            "null",
            "false",
            "true",
            "normalize_error_type",
            "fetch_results",
            "belt",
            # ``task_quality`` sub-block field names + bucket labels.
            "task_quality",
            "attempted",
            "env_failed",
            "env_failed_agent",
            "env_failed_judge",
            "completed",
            "task_failed",
            "pct",
            "environmental",
            "task",
            # ``judge_errors`` sibling block field names.
            "judge_errors",
            # Vendor-specific error codes quoted in prose alongside the
            # canonical ``model_unavailable`` token to show users what
            # raw provider output maps to. Not error_type tokens.
            "model_not_found",
        }
        candidate_tokens = documented - non_tokens
        unknown = candidate_tokens - ERROR_TYPES
        assert not unknown, (
            f"{DOC_PATH.name} advertises tokens not in ERROR_TYPES: "
            f"{sorted(unknown)}. Either add them to "
            f"belt.entities.ERROR_TYPES or remove them from the "
            f"doc table. Tokens that legitimately appear in prose "
            f"(field names, type names) should be added to the "
            f"non_tokens set in this test."
        )
