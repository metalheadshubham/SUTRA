"""Tests for agent_handoff.utils — hashing, truncation, formatting."""

import pytest
from agent_handoff.utils import hash_query, truncate_text, format_demonstrations


class TestHashQuery:
    def test_deterministic(self):
        h1 = hash_query("hello")
        h2 = hash_query("hello")
        assert h1 == h2

    def test_different_inputs(self):
        h1 = hash_query("hello")
        h2 = hash_query("world")
        assert h1 != h2

    def test_returns_hex_string(self):
        h = hash_query("test")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest length

    def test_with_profile(self):
        h1 = hash_query("query", profile=None)
        h2 = hash_query("query", profile="expert")
        assert h1 != h2

    def test_empty_string(self):
        h = hash_query("")
        assert isinstance(h, str)
        assert len(h) == 64


class TestTruncateText:
    def test_short_text_unchanged(self):
        text = "Short text"
        assert truncate_text(text, max_chars=100) == text

    def test_exact_length(self):
        text = "a" * 100
        assert truncate_text(text, max_chars=100) == text

    def test_truncation(self):
        text = "word " * 1000  # 5000 chars
        result = truncate_text(text, max_chars=100)
        assert len(result) <= 101  # +1 for ellipsis char
        assert result.endswith("…")

    def test_breaks_at_word_boundary(self):
        text = "hello world this is a long sentence that should be truncated"
        result = truncate_text(text, max_chars=30)
        assert result.endswith("…")
        # Should not end mid-word
        assert not result[-2].isalpha() or result[-1] == "…"

    def test_default_max_chars(self):
        short = "short"
        assert truncate_text(short) == short


class TestFormatDemonstrations:
    def test_basic(self):
        result = format_demonstrations(["Alpha", "Beta"])
        assert "Example 1:" in result
        assert "Alpha" in result
        assert "Example 2:" in result
        assert "Beta" in result

    def test_empty_list(self):
        result = format_demonstrations([])
        assert "no demonstrations" in result.lower()

    def test_single_demo(self):
        result = format_demonstrations(["Only one"])
        assert "Example 1:" in result
        assert "Only one" in result
