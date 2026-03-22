"""
protocol.py — Data classes for structured handoff messages.

Defines the core data structures used to pass demonstrations
and metadata between models in a handoff chain.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json


@dataclass
class HandoffPacket:
    """A packet of demonstrations extracted from one model's output,
    ready to be forwarded to the next model in the chain.

    Attributes:
        query: The original user query.
        demonstrations: List of demonstration strings extracted from the model output.
        metadata: Arbitrary metadata (model names, timestamps, scores, etc.).
    """

    query: str
    demonstrations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the packet to a plain dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the packet to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffPacket":
        """Reconstruct a HandoffPacket from a dictionary."""
        return cls(
            query=data["query"],
            demonstrations=data.get("demonstrations", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, raw: str) -> "HandoffPacket":
        """Reconstruct a HandoffPacket from a JSON string."""
        return cls.from_dict(json.loads(raw))


@dataclass
class HandoffResult:
    """The final result of a complete handoff pipeline run.

    Attributes:
        query: The original user query.
        answer: The final answer produced by the last model in the chain.
        demonstrations: The demonstration set that was used.
        model_a: Name/tag of the first model (demonstration generator).
        model_b: Name/tag of the second model (final responder).
        cache_hit: Whether the demonstrations were served from cache.
        latency_ms: Total wall-clock time of the handoff in milliseconds.
        token_counts: Optional dict with prompt/completion token counts per model.
    """

    query: str
    answer: str
    demonstrations: List[str] = field(default_factory=list)
    model_a: str = ""
    model_b: str = ""
    cache_hit: bool = False
    latency_ms: float = 0.0
    token_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class CouncilResult:
    """The result of a complete council pipeline run (4 stages).

    Attributes:
        query: The original user query.
        answer_a: Stage 1a output — first candidate solution (temp=0.3).
        answer_b: Stage 1b output — second candidate solution (temp=0.8).
        critique: Stage 2 output — comparative review of both solutions.
        synthesis: Stage 3 output — final best implementation.
        model_small: Name/tag of the small model (stages 1a, 1b, 2).
        model_large: Name/tag of the large model (stage 3).
        latency_ms: Dict mapping stage name to wall-clock ms, plus ``"total"``.
        token_counts: Dict mapping stage name to completion token count.
    """

    query: str = ""
    answer_a: str = ""
    answer_b: str = ""
    critique: str = ""
    synthesis: str = ""
    model_small: str = ""
    model_large: str = ""
    latency_ms: Dict[str, float] = field(default_factory=dict)
    token_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
