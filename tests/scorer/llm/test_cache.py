# (c) JFrog Ltd. (2026)

"""Unit tests for scorer.cache - content-addressable response cache."""

from __future__ import annotations

from belt.scorer.llm.cache import ScoreCache


class TestScoreCache:
    def test_make_key_deterministic(self):
        k1 = ScoreCache.make_key("gpt-4.1", 0.0, 42, "sys", "dyn", {"type": "object"})
        k2 = ScoreCache.make_key("gpt-4.1", 0.0, 42, "sys", "dyn", {"type": "object"})
        assert k1 == k2
        assert len(k1) == 64  # SHA256 hex

    def test_different_inputs_produce_different_keys(self):
        k1 = ScoreCache.make_key("gpt-4.1", 0.0, 42, "sys", "dyn", {"type": "object"})
        k2 = ScoreCache.make_key("gpt-4.1", 0.5, 42, "sys", "dyn", {"type": "object"})
        assert k1 != k2

    def test_model_change_invalidates(self):
        k1 = ScoreCache.make_key("gpt-4.1", 0.0, 42, "sys", "dyn", {})
        k2 = ScoreCache.make_key("claude-sonnet-4-5", 0.0, 42, "sys", "dyn", {})
        assert k1 != k2

    def test_miss_returns_none(self, tmp_path):
        cache = ScoreCache(tmp_path / "cache")
        assert cache.get("nonexistent") is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_put_then_get(self, tmp_path):
        cache = ScoreCache(tmp_path / "cache")
        key = ScoreCache.make_key("m", 0.0, 1, "s", "d", {})
        data = {"verdict": {"overall_pass": True}, "usage": {"prompt_tokens": 100}}

        cache.put(key, data)
        result = cache.get(key)

        assert result is not None
        assert result["verdict"]["overall_pass"] is True
        assert result["usage"]["prompt_tokens"] == 100
        assert cache.hits == 1
        assert cache.misses == 0

    def test_hit_rate(self, tmp_path):
        cache = ScoreCache(tmp_path / "cache")
        key = ScoreCache.make_key("m", 0.0, 1, "s", "d", {})
        cache.put(key, {"verdict": {}})

        cache.get(key)  # hit
        cache.get("miss1")  # miss
        cache.get("miss2")  # miss

        assert cache.hits == 1
        assert cache.misses == 2
        assert abs(cache.hit_rate - 1 / 3) < 0.01

    def test_hit_rate_zero_total(self, tmp_path):
        cache = ScoreCache(tmp_path / "cache")
        assert cache.hit_rate == 0.0

    def test_corrupt_entry_removed(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        key = "abc123"
        (cache_dir / f"{key}.json").write_text("not valid json{{{")

        cache = ScoreCache(cache_dir)
        result = cache.get(key)

        assert result is None
        assert cache.misses == 1
        assert not (cache_dir / f"{key}.json").exists()

    def test_creates_directory(self, tmp_path):
        cache_dir = tmp_path / "nested" / "cache" / "dir"
        assert not cache_dir.exists()
        ScoreCache(cache_dir)
        assert cache_dir.exists()
