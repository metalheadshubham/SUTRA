"""
S.U.T.R.A — Structured Universal Transfer via Retrieval Adaptation
==================================================================

A lightweight library for orchestrating structured demonstration handoffs
and council pipelines between small language models via Ollama.

Quick start::

    from agent_handoff import AgentHandoff

    handoff = AgentHandoff(model_a="llama3.2:3b", model_b="mistral:7b")
    result = handoff.run("Write a safe Rust function to read a file")
    print(result)

Council mode::

    from agent_handoff import CouncilHandoff

    council = CouncilHandoff(model_small="qwen2.5-coder:3b", model_large="qwen2.5-coder:7b")
    result = council.run("Implement binary search")
    print(result)
"""

from agent_handoff.protocol import HandoffPacket, HandoffResult, CouncilResult
from agent_handoff.cache import DemonstrationCache
from agent_handoff.handoff import AgentHandoff, CouncilHandoff
from agent_handoff.agent import AgentLoop, AgentResult

__all__ = [
    "AgentHandoff",
    "CouncilHandoff",
    "AgentLoop",
    "AgentResult",
    "DemonstrationCache",
    "HandoffPacket",
    "HandoffResult",
    "CouncilResult",
]

__version__ = "0.2.0"
