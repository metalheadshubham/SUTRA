"""Tests for agent_handoff.cache — TTL, persistence, key generation."""

import json
import time
import tempfile
import os

import pytest
from agent_handoff.cache import DemonstrationCache


class TestDemonstrationCache:
    def test_set_and_get(self):
        cache = DemonstrationCache()
        cache.set("key1", ["demo1", "demo2"])
        result = cache.get("key1")
        assert result == ["demo1", "demo2"]

    def test_get_missing_key(self):
        cache = DemonstrationCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = DemonstrationCache(default_ttl=1)
        cache.set("key1", ["demo1"])
        assert cache.get("key1") == ["demo1"]

        # Simulate expiry
        cache._store["key1"]["expires_at"] = time.time() - 1
        assert cache.get("key1") is None

    def test_no_ttl(self):
        cache = DemonstrationCache(default_ttl=0)
        cache.set("key1", ["demo1"])
        assert cache.get("key1") == ["demo1"]

    def test_custom_ttl_per_entry(self):
        cache = DemonstrationCache(default_ttl=3600)
        cache.set("key1", ["demo1"], ttl=0)  # No expiry
        assert cache._store["key1"]["expires_at"] is None

    def test_make_key_deterministic(self):
        cache = DemonstrationCache()
        k1 = cache.make_key("hello world")
        k2 = cache.make_key("hello world")
        assert k1 == k2

    def test_make_key_different_queries(self):
        cache = DemonstrationCache()
        k1 = cache.make_key("query one")
        k2 = cache.make_key("query two")
        assert k1 != k2

    def test_make_key_with_profile(self):
        cache = DemonstrationCache()
        k1 = cache.make_key("query", profile=None)
        k2 = cache.make_key("query", profile="expert")
        assert k1 != k2

    def test_contains(self):
        cache = DemonstrationCache()
        cache.set("key1", ["demo1"])
        assert "key1" in cache
        assert "key2" not in cache

    def test_len(self):
        cache = DemonstrationCache()
        assert len(cache) == 0
        cache.set("a", ["x"])
        cache.set("b", ["y"])
        assert len(cache) == 2

    def test_clear(self):
        cache = DemonstrationCache()
        cache.set("a", ["x"])
        cache.clear()
        assert len(cache) == 0

    def test_save_and_load(self):
        cache = DemonstrationCache()
        cache.set("key1", ["demo1", "demo2"])
        cache.set("key2", ["demo3"])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            cache.save(path)
            assert os.path.exists(path)

            # Load into a new cache
            new_cache = DemonstrationCache()
            new_cache.load(path)
            assert new_cache.get("key1") == ["demo1", "demo2"]
            assert new_cache.get("key2") == ["demo3"]
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        cache = DemonstrationCache()
        cache.load("/tmp/nonexistent_cache_file_12345.json")
        assert len(cache) == 0

    def test_repr(self):
        cache = DemonstrationCache(default_ttl=7200)
        r = repr(cache)
        assert "DemonstrationCache" in r
        assert "7200" in r
