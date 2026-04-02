"""Tests for agent_handoff.protocol — dataclass creation and serialization."""

import json

import pytest
from agent_handoff.protocol import HandoffPacket, HandoffResult


class TestHandoffPacket:
    def test_creation(self):
        packet = HandoffPacket(query="test query", demonstrations=["d1", "d2"])
        assert packet.query == "test query"
        assert len(packet.demonstrations) == 2
        assert "created_at" in packet.metadata

    def test_defaults(self):
        packet = HandoffPacket(query="q")
        assert packet.demonstrations == []
        assert isinstance(packet.metadata, dict)

    def test_to_dict(self):
        packet = HandoffPacket(query="q", demonstrations=["d1"])
        d = packet.to_dict()
        assert d["query"] == "q"
        assert d["demonstrations"] == ["d1"]

    def test_to_json(self):
        packet = HandoffPacket(query="q", demonstrations=["d1", "d2"])
        j = packet.to_json()
        parsed = json.loads(j)
        assert parsed["query"] == "q"
        assert len(parsed["demonstrations"]) == 2

    def test_from_dict(self):
        data = {"query": "q", "demonstrations": ["a", "b"], "metadata": {"k": "v"}}
        packet = HandoffPacket.from_dict(data)
        assert packet.query == "q"
        assert packet.demonstrations == ["a", "b"]
        assert packet.metadata["k"] == "v"

    def test_from_json_roundtrip(self):
        original = HandoffPacket(
            query="test", demonstrations=["x", "y"], metadata={"source": "test"}
        )
        j = original.to_json()
        restored = HandoffPacket.from_json(j)
        assert restored.query == original.query
        assert restored.demonstrations == original.demonstrations
        assert restored.metadata["source"] == "test"

    def test_custom_metadata(self):
        packet = HandoffPacket(
            query="q",
            demonstrations=[],
            metadata={"model": "llama3.2:3b", "score": 0.95},
        )
        assert packet.metadata["model"] == "llama3.2:3b"
        assert packet.metadata["score"] == 0.95
        # created_at should still be added
        assert "created_at" in packet.metadata


class TestHandoffResult:
    def test_creation(self):
        result = HandoffResult(query="q", answer="a", model_a="m1", model_b="m2")
        assert result.query == "q"
        assert result.answer == "a"
        assert result.cache_hit is False
        assert result.latency_ms == 0.0

    def test_to_dict(self):
        result = HandoffResult(
            query="q", answer="a", demonstrations=["d1"], latency_ms=123.4
        )
        d = result.to_dict()
        assert d["latency_ms"] == 123.4
        assert d["demonstrations"] == ["d1"]

    def test_to_json(self):
        result = HandoffResult(query="q", answer="a")
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["query"] == "q"
