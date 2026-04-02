"""
handoff.py — Main AgentHandoff orchestrator.

Coordinates the two-model demonstration-passing pipeline:
  Query  →  Model A (generates answer + demonstrations)
         →  Parser (extracts demonstrations)
         →  Cache  (store / retrieve)
         →  Model B (generates final answer using demonstrations)
"""

import time
import logging
from typing import Dict, List, Optional

import ollama

from agent_handoff.protocol import HandoffPacket, HandoffResult, CouncilResult
from agent_handoff.parser import parse_answer, parse_demonstrations
from agent_handoff.cache import DemonstrationCache
from agent_handoff.templates import format_prompt_a, format_prompt_b
from agent_handoff.utils import hash_query, truncate_text

logger = logging.getLogger(__name__)


class AgentHandoff:
    """Orchestrate a structured demonstration handoff between two models.

    Args:
        model_a: Ollama model tag for the demonstration generator
                 (e.g. ``"llama3.2:3b"``).
        model_b: Ollama model tag for the final responder
                 (e.g. ``"mistral:7b"``).
        cache: Optional :class:`DemonstrationCache` instance.
               A default cache is created if ``None``.
        templates: Optional dict with keys ``"prompt_a"`` and/or ``"prompt_b"``
                   to override the default prompt templates.
        ollama_host: Optional Ollama server URL (defaults to ``http://localhost:11434``).

    Example::

        handoff = AgentHandoff(model_a="llama3.2:3b", model_b="mistral:7b")
        result = handoff.run("Write a safe Rust function to read a file")
        print(result)
    """

    def __init__(
        self,
        model_a: str,
        model_b: str,
        cache: Optional[DemonstrationCache] = None,
        templates: Optional[Dict[str, str]] = None,
        ollama_host: Optional[str] = None,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.cache = cache or DemonstrationCache()
        self._template_a = (templates or {}).get("prompt_a")
        self._template_b = (templates or {}).get("prompt_b")

        # Configure Ollama client
        if ollama_host:
            self._client = ollama.Client(host=ollama_host)
        else:
            self._client = ollama.Client()

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, query: str, use_cache: bool = True) -> str:
        """Execute the full handoff pipeline and return the final answer string.

        This is the high-level convenience method.  For more control, use
        :meth:`generate_demonstrations` and :meth:`generate_final` directly.
        """
        result = self.run_detailed(query, use_cache=use_cache)
        return result.answer

    def run_detailed(self, query: str, use_cache: bool = True) -> HandoffResult:
        """Execute the full handoff pipeline and return a :class:`HandoffResult`
        with timing, token counts, and cache-hit information.
        """
        t0 = time.perf_counter()
        cache_hit = False
        token_counts: Dict[str, int] = {}

        # ── Step 1: Obtain demonstrations ────────────────────────────────
        cache_key = self.cache.make_key(query)
        demonstrations: Optional[List[str]] = None

        if use_cache:
            demonstrations = self.cache.get(cache_key)
            if demonstrations:
                cache_hit = True
                logger.info("Cache HIT for query (key=%s…)", cache_key[:12])

        if not demonstrations:
            logger.info("Cache MISS — generating demonstrations via %s", self.model_a)
            packet, a_tokens = self._call_model_a(query)
            demonstrations = packet.demonstrations
            token_counts.update(a_tokens)

            if use_cache and demonstrations:
                self.cache.set(cache_key, demonstrations)

        # ── Step 2: Generate final answer ────────────────────────────────
        answer, b_tokens = self._call_model_b(query, demonstrations)
        token_counts.update(b_tokens)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return HandoffResult(
            query=query,
            answer=answer,
            demonstrations=demonstrations,
            model_a=self.model_a,
            model_b=self.model_b,
            cache_hit=cache_hit,
            latency_ms=round(elapsed_ms, 2),
            token_counts=token_counts,
        )

    def generate_demonstrations(self, query: str) -> HandoffPacket:
        """Run only Model A and return a :class:`HandoffPacket` with demonstrations.

        Useful when chaining more than two models or when you want to
        inspect / modify demonstrations before passing them forward.
        """
        packet, _ = self._call_model_a(query)
        return packet

    def generate_final(self, query: str, demonstrations: List[str]) -> str:
        """Run only Model B, given pre-existing demonstrations.

        Useful for the second half of a manual chain.
        """
        answer, _ = self._call_model_b(query, demonstrations)
        return answer

    # ── Internal helpers ─────────────────────────────────────────────────

    def _call_model_a(self, query: str) -> tuple[HandoffPacket, Dict[str, int]]:
        """Prompt Model A and parse its output into a HandoffPacket."""
        prompt = format_prompt_a(query, template=self._template_a)
        logger.debug("Model A prompt (%d chars): %s…", len(prompt), prompt[:120])

        try:
            response = self._client.generate(model=self.model_a, prompt=prompt, format="json")
            raw_output: str = response.get("response", "")
        except Exception as exc:
            logger.error("Model A call failed: %s", exc)
            raise RuntimeError(f"Failed to call model_a ({self.model_a}): {exc}") from exc

        # Parse
        demonstrations = parse_demonstrations(raw_output)
        if not demonstrations:
            logger.warning(
                "No demonstrations parsed from Model A output; "
                "falling back to full output as single demonstration."
            )
            demonstrations = [truncate_text(raw_output, max_chars=2000)]

        packet = HandoffPacket(
            query=query,
            demonstrations=demonstrations,
            metadata={
                "source_model": self.model_a,
            },
        )

        tokens = {
            "model_a_prompt_tokens": response.get("prompt_eval_count", 0),
            "model_a_completion_tokens": response.get("eval_count", 0),
        }

        return packet, tokens

    def _call_model_b(
        self, query: str, demonstrations: List[str]
    ) -> tuple[str, Dict[str, int]]:
        """Prompt Model B with demonstrations and return the final answer."""
        prompt = format_prompt_b(query, demonstrations, template=self._template_b)
        logger.debug("Model B prompt (%d chars): %s…", len(prompt), prompt[:120])

        try:
            response = self._client.generate(model=self.model_b, prompt=prompt)
            raw_output: str = response.get("response", "")
        except Exception as exc:
            logger.error("Model B call failed: %s", exc)
            raise RuntimeError(f"Failed to call model_b ({self.model_b}): {exc}") from exc

        # Model B's output is the final answer — strip any leftover tags
        answer = parse_answer(raw_output)

        tokens = {
            "model_b_prompt_tokens": response.get("prompt_eval_count", 0),
            "model_b_completion_tokens": response.get("eval_count", 0),
        }

        return answer, tokens

    def __repr__(self) -> str:
        return (
            f"AgentHandoff(model_a={self.model_a!r}, model_b={self.model_b!r}, "
            f"cache={self.cache!r})"
        )


# ── Council pipeline (v0.2) ─────────────────────────────────────────────────

class CouncilHandoff:
    """Orchestrate a 4-stage council pipeline between two models.

    The pipeline generates diversity via two answers at different temperatures,
    critiques both, then synthesizes the best solution with a larger model.
    Every stage passes ``keep_alive=0`` to force Ollama to evict the model
    from RAM before the next stage loads.

    Stages::

        1a  small model  temp=0.3  →  Answer A    →  unload
        1b  small model  temp=0.8  →  Answer B    →  unload
        2   small model  temp=0.3  →  Critique    →  unload
        3   large model  temp=0.2  →  Synthesis   →  unload

    Args:
        model_small: Ollama model tag for stages 1a, 1b, and 2.
        model_large: Ollama model tag for stage 3 (synthesis).
        cache: Optional :class:`DemonstrationCache` instance.
        ollama_host: Optional Ollama server URL.
    """

    def __init__(
        self,
        model_small: str,
        model_large: str,
        cache: Optional[DemonstrationCache] = None,
        ollama_host: Optional[str] = None,
    ) -> None:
        self.model_small = model_small
        self.model_large = model_large
        self.cache = cache or DemonstrationCache()

        if ollama_host:
            self._client = ollama.Client(host=ollama_host)
        else:
            self._client = ollama.Client()

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, query: str) -> str:
        """Execute the full council pipeline and return the synthesis string."""
        result = self.run_detailed(query)
        return result.synthesis

    def run_detailed(self, query: str) -> CouncilResult:
        """Execute all 4 stages and return a :class:`CouncilResult`."""
        from agent_handoff.templates import (
            ANSWER_PROMPT, CRITIQUE_PROMPT, SYNTHESIS_PROMPT,
        )

        total_t0 = time.perf_counter()
        latency: Dict[str, float] = {}
        tokens: Dict[str, int] = {}

        # ── Stage 1a: Answer A (small, temp=0.3) ────────────────────────
        t0 = time.perf_counter()
        answer_a = self._generate(
            model=self.model_small,
            prompt=ANSWER_PROMPT.format(query=query),
            temperature=0.3,
        )
        latency["stage_1a"] = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("Stage 1a complete: %d chars in %.0fms", len(answer_a), latency["stage_1a"])

        # ── Stage 1b: Answer B (small, temp=0.8) ────────────────────────
        t0 = time.perf_counter()
        answer_b = self._generate(
            model=self.model_small,
            prompt=ANSWER_PROMPT.format(query=query),
            temperature=0.8,
        )
        latency["stage_1b"] = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("Stage 1b complete: %d chars in %.0fms", len(answer_b), latency["stage_1b"])

        # ── Stage 2: Critique (small, temp=0.3) ─────────────────────────
        t0 = time.perf_counter()
        critique = self._generate(
            model=self.model_small,
            prompt=CRITIQUE_PROMPT.format(
                query=query, solution_a=answer_a, solution_b=answer_b,
            ),
            temperature=0.3,
        )
        latency["stage_2"] = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("Stage 2 complete: %d chars in %.0fms", len(critique), latency["stage_2"])

        # ── Stage 3: Synthesis (large, temp=0.2) ─────────────────────────
        t0 = time.perf_counter()
        synthesis = self._generate(
            model=self.model_large,
            prompt=SYNTHESIS_PROMPT.format(
                query=query, solution_a=answer_a, solution_b=answer_b,
                critique=critique,
            ),
            temperature=0.2,
        )
        latency["stage_3"] = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("Stage 3 complete: %d chars in %.0fms", len(synthesis), latency["stage_3"])

        total_ms = round((time.perf_counter() - total_t0) * 1000, 2)
        latency["total"] = total_ms

        return CouncilResult(
            query=query,
            answer_a=answer_a,
            answer_b=answer_b,
            critique=critique,
            synthesis=synthesis,
            model_small=self.model_small,
            model_large=self.model_large,
            latency_ms=latency,
            token_counts=tokens,
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _generate(self, model: str, prompt: str, temperature: float) -> str:
        """Call Ollama with ``keep_alive=0`` to force RAM eviction after use."""
        try:
            response = self._client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature},
                keep_alive=0,
            )
            return response.get("response", "")
        except Exception as exc:
            logger.error("Council stage failed (%s): %s", model, exc)
            raise RuntimeError(f"Council stage failed ({model}): {exc}") from exc

    def _cache_key(self, query: str) -> str:
        """Generate a cache key scoped to the model pair."""
        return hash_query(query, profile=f"{self.model_small}:{self.model_large}")

    def __repr__(self) -> str:
        return (
            f"CouncilHandoff(model_small={self.model_small!r}, "
            f"model_large={self.model_large!r})"
        )
