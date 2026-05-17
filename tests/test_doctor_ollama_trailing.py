# (c) JFrog Ltd. (2026)

"""Pin the Ollama line's ``+N more`` trailing suffix (#385 / 3.2).

When Ollama reports >5 pulled models, doctor used to truncate to the
first 5 with no hint that more existed - so the user couldn't tell the
difference between "I have exactly 5 models" and "I have 200 models, 5
shown". The ``+N more`` suffix closes that ambiguity.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from belt.commands.doctor import _check_ollama


def _mock_response(models_count: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"models": [{"name": f"m{i}"} for i in range(models_count)]}
    return resp


def test_ollama_lists_models_under_threshold():
    """Up to 5 models: no trailing suffix."""
    with patch("httpx.get", return_value=_mock_response(3)):
        result = _check_ollama()
    assert result.ok
    assert "3 models" in result.detail
    assert "more" not in result.detail


def test_ollama_singular_when_one_model():
    """Pluralization: ``1 model`` not ``1 model(s)``."""
    with patch("httpx.get", return_value=_mock_response(1)):
        result = _check_ollama()
    assert result.ok
    assert "1 model:" in result.detail
    assert "models" not in result.detail.split(":")[0]


def test_ollama_trailing_more_when_truncated():
    """>5 models: ``+N more`` so the user knows the list is partial."""
    with patch("httpx.get", return_value=_mock_response(7)):
        result = _check_ollama()
    assert result.ok
    assert "7 models:" in result.detail
    assert "+2 more" in result.detail


def test_ollama_no_trailing_when_exactly_five():
    """Exactly 5 means everything is shown - no suffix."""
    with patch("httpx.get", return_value=_mock_response(5)):
        result = _check_ollama()
    assert result.ok
    assert "5 models:" in result.detail
    assert "more" not in result.detail
