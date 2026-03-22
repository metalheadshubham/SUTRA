"""Tests for agent_handoff.parser — XML parsing and regex fallback."""

import pytest
from agent_handoff.parser import parse_answer, parse_demonstrations, strip_tags


# ── parse_answer ─────────────────────────────────────────────────────────────

class TestParseAnswer:
    def test_clean_xml(self):
        text = "<answer>Hello world</answer>"
        assert parse_answer(text) == "Hello world"

    def test_multiline(self):
        text = "<answer>\nLine 1\nLine 2\n</answer>"
        result = parse_answer(text)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_with_surrounding_text(self):
        text = "Some preamble\n<answer>The real answer</answer>\nSome postamble"
        assert parse_answer(text) == "The real answer"

    def test_no_answer_tag(self):
        text = "Just a plain response with no tags"
        result = parse_answer(text)
        assert result == "Just a plain response with no tags"

    def test_case_insensitive(self):
        text = "<Answer>Mixed case</Answer>"
        # Regex fallback handles case insensitivity
        result = parse_answer(text)
        assert "Mixed case" in result

    def test_malformed_xml_falls_back_to_regex(self):
        text = "<answer>Good answer</answer><unclosed>"
        result = parse_answer(text)
        assert "Good answer" in result

    def test_empty_answer_tag(self):
        text = "<answer></answer>"
        result = parse_answer(text)
        assert result == ""


# ── parse_demonstrations ─────────────────────────────────────────────────────

class TestParseDemonstrations:
    def test_clean_xml(self):
        text = """
        <demonstrations>
            <demonstration>Example 1</demonstration>
            <demonstration>Example 2</demonstration>
            <demonstration>Example 3</demonstration>
        </demonstrations>
        """
        demos = parse_demonstrations(text)
        assert len(demos) == 3
        assert demos[0] == "Example 1"
        assert demos[1] == "Example 2"
        assert demos[2] == "Example 3"

    def test_single_demonstration(self):
        text = "<demonstrations><demonstration>Only one</demonstration></demonstrations>"
        demos = parse_demonstrations(text)
        assert len(demos) == 1
        assert demos[0] == "Only one"

    def test_no_demonstrations(self):
        text = "No demonstrations here at all"
        demos = parse_demonstrations(text)
        assert demos == []

    def test_demonstrations_without_inner_tags(self):
        text = "<demonstrations>\nLine A\nLine B\nLine C\n</demonstrations>"
        demos = parse_demonstrations(text)
        assert len(demos) == 3

    def test_loose_demonstration_tags(self):
        text = """
        Some output here
        <demonstration>Loose example 1</demonstration>
        More text
        <demonstration>Loose example 2</demonstration>
        """
        demos = parse_demonstrations(text)
        assert len(demos) == 2

    def test_with_full_model_output(self):
        text = """
        <answer>Here is my detailed answer about the topic.</answer>

        <demonstrations>
            <demonstration>
                If asked about X, respond with Y format.
            </demonstration>
            <demonstration>
                When explaining Z, use analogies.
            </demonstration>
            <demonstration>
                For code questions, include type hints.
            </demonstration>
        </demonstrations>
        """
        demos = parse_demonstrations(text)
        assert len(demos) == 3
        assert "type hints" in demos[2]

    def test_malformed_xml_falls_back(self):
        text = "<demonstrations><demonstration>Works</demonstration><broken"
        demos = parse_demonstrations(text)
        # Should at least try regex and find the one valid demo
        assert len(demos) >= 1


# ── strip_tags ───────────────────────────────────────────────────────────────

class TestStripTags:
    def test_basic(self):
        assert strip_tags("<b>bold</b>") == "bold"

    def test_nested(self):
        assert strip_tags("<a><b>text</b></a>") == "text"

    def test_no_tags(self):
        assert strip_tags("plain text") == "plain text"

    def test_self_closing(self):
        assert strip_tags("before<br/>after") == "beforeafter"

    def test_complex_model_output(self):
        text = "<answer>Hello</answer> <demonstrations><demonstration>Ex</demonstration></demonstrations>"
        result = strip_tags(text)
        assert "<" not in result
        assert ">" not in result
