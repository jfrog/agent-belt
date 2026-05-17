# (c) JFrog Ltd. (2026)

"""Tests for file-diff scoring rules (files_modified_*, git_diff_contains)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from belt.entities import Scenario, Turn, TurnExpectation, TurnOutput
from belt.scorer.rules import RuleBasedScorer


def _make_output(
    git_diff: str | None = None,
    files_modified: list[str] | None = None,
) -> TurnOutput:
    return TurnOutput(
        raw_cli="some output",
        has_reply=True,
        reply_text="done",
        git_diff=git_diff,
        files_modified=files_modified or [],
    )


def _make_scenario(expect: TurnExpectation) -> Scenario:
    return Scenario(
        name="diff_test",
        description="Test file diff scoring",
        turns=[Turn(message="edit the code", expect=expect)],
    )


def _score(expect: TurnExpectation, output: TurnOutput) -> dict:
    scorer = RuleBasedScorer()
    scenario = _make_scenario(expect)
    result = scorer.score(scenario, [output])
    assert result is not None
    checks = result.data.checks
    return {c.check: c.model_dump(mode="json") for c in checks}


class TestFilesModifiedAny:
    def test_passes_when_any_match(self):
        output = _make_output(files_modified=["src/main.py", "tests/test_main.py"])
        expect = TurnExpectation(files_modified_any=["src/main.py", "src/other.py"])
        checks = _score(expect, output)
        assert checks["files_modified_any(src/main.py,src/other.py)"]["passed"] is True

    def test_fails_when_none_match(self):
        output = _make_output(files_modified=["unrelated.txt"])
        expect = TurnExpectation(files_modified_any=["src/main.py"])
        checks = _score(expect, output)
        assert checks["files_modified_any(src/main.py)"]["passed"] is False

    def test_fails_when_no_files_modified(self):
        output = _make_output(files_modified=[])
        expect = TurnExpectation(files_modified_any=["anything.py"])
        checks = _score(expect, output)
        assert checks["files_modified_any(anything.py)"]["passed"] is False

    def test_skipped_when_not_set(self):
        output = _make_output(files_modified=["file.py"])
        expect = TurnExpectation()
        checks = _score(expect, output)
        assert not any("files_modified" in k for k in checks)


class TestFilesModifiedExact:
    def test_passes_on_exact_match(self):
        output = _make_output(files_modified=["a.py", "b.py"])
        expect = TurnExpectation(files_modified_exact=["a.py", "b.py"])
        checks = _score(expect, output)
        assert checks["files_modified_exact"]["passed"] is True

    def test_fails_on_missing_file(self):
        output = _make_output(files_modified=["a.py"])
        expect = TurnExpectation(files_modified_exact=["a.py", "b.py"])
        checks = _score(expect, output)
        assert checks["files_modified_exact"]["passed"] is False
        assert "missing" in checks["files_modified_exact"]["details"]

    def test_fails_on_extra_file(self):
        output = _make_output(files_modified=["a.py", "b.py", "c.py"])
        expect = TurnExpectation(files_modified_exact=["a.py", "b.py"])
        checks = _score(expect, output)
        assert checks["files_modified_exact"]["passed"] is False
        assert "extra" in checks["files_modified_exact"]["details"]


class TestFilesNotModified:
    def test_passes_when_file_not_touched(self):
        output = _make_output(files_modified=["safe.py"])
        expect = TurnExpectation(files_not_modified=["config.json"])
        checks = _score(expect, output)
        assert checks["file_not_modified(config.json)"]["passed"] is True

    def test_fails_when_file_was_modified(self):
        output = _make_output(files_modified=["config.json", "other.py"])
        expect = TurnExpectation(files_not_modified=["config.json"])
        checks = _score(expect, output)
        assert checks["file_not_modified(config.json)"]["passed"] is False


class TestGitDiffContains:
    def test_passes_when_substring_found(self):
        output = _make_output(git_diff="--- a/file.py\n+++ b/file.py\n+def new_function():")
        expect = TurnExpectation(git_diff_contains=["new_function"])
        checks = _score(expect, output)
        assert checks["git_diff_contains(new_function)"]["passed"] is True

    def test_fails_when_substring_not_found(self):
        output = _make_output(git_diff="--- a/file.py\n+++ b/file.py\n+def other():")
        expect = TurnExpectation(git_diff_contains=["new_function"])
        checks = _score(expect, output)
        assert checks["git_diff_contains(new_function)"]["passed"] is False

    def test_fails_when_no_diff(self):
        output = _make_output(git_diff=None)
        expect = TurnExpectation(git_diff_contains=["anything"])
        checks = _score(expect, output)
        assert checks["git_diff_contains(anything)"]["passed"] is False
        assert "not found in diff" in checks["git_diff_contains(anything)"]["details"]

    def test_multiple_substrings(self):
        diff = "+import pytest\n+def test_add():\n+    assert add(1, 2) == 3"
        output = _make_output(git_diff=diff)
        expect = TurnExpectation(git_diff_contains=["import pytest", "test_add"])
        checks = _score(expect, output)
        assert checks["git_diff_contains(import pytest)"]["passed"] is True
        assert checks["git_diff_contains(test_add)"]["passed"] is True


class TestDirectoryPathRejection:
    """Directory-shaped paths would silently pass against the flat
    modified-files list. Reject them at load time with a clear hint."""

    @pytest.mark.parametrize(
        "field",
        ["files_modified_any", "files_modified_exact", "files_not_modified"],
    )
    def test_rejects_trailing_slash(self, field):
        with pytest.raises(ValidationError) as exc_info:
            TurnExpectation(**{field: ["src/billing/"]})
        msg = str(exc_info.value)
        assert "directory-shaped paths are not supported" in msg
        assert "src/billing/" in msg

    def test_rejects_trailing_backslash(self):
        with pytest.raises(ValidationError):
            TurnExpectation(files_not_modified=["src\\billing\\"])

    def test_accepts_specific_file_paths(self):
        TurnExpectation(
            files_modified_any=["src/billing/handler.py"],
            files_modified_exact=["a.py", "b.py"],
            files_not_modified=["pyproject.toml", "go.mod"],
        )

    def test_rejects_mixed_valid_and_directory(self):
        with pytest.raises(ValidationError) as exc_info:
            TurnExpectation(files_not_modified=["pyproject.toml", "src/"])
        assert "src/" in str(exc_info.value)
