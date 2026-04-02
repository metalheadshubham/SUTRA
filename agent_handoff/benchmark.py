"""
benchmark.py — Compare handoff method vs. raw context passing.

Measures latency, token usage, and optionally quality for a set of
test queries.  Run with::

    python -m agent_handoff.benchmark --model-a llama3.2:3b --model-b mistral:7b

Requires a running Ollama instance with the specified models pulled.
"""

import argparse
import logging
import time
from typing import Dict, List, Any

import ollama

from agent_handoff.handoff import AgentHandoff
from agent_handoff.cache import DemonstrationCache

logger = logging.getLogger(__name__)

# ── Test queries ─────────────────────────────────────────────────────────────

CODING_QUERIES: List[str] = [
    "Write a Python function that merges two sorted lists into one sorted list without using built-in sort.",
    "Implement a thread-safe singleton pattern in Java with lazy initialization.",
    "Write a Rust function that reads a file and returns its contents as a String, handling all errors properly.",
    "Create a SQL query that finds the second highest salary from an employees table.",
    "Write a recursive TypeScript function to flatten an arbitrarily nested array.",
]

EXPLANATION_QUERIES: List[str] = [
    "Explain the difference between concurrency and parallelism with real-world analogies.",
    "What is the CAP theorem and how does it apply to distributed databases?",
    "Describe how garbage collection works in Go compared to Java.",
    "Explain the concept of monads in functional programming in simple terms.",
    "What are the trade-offs between microservices and monolithic architectures?",
]


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_raw(client: ollama.Client, model: str, query: str) -> Dict[str, Any]:
    """Baseline: send query directly to Model B with no demonstrations."""
    t0 = time.perf_counter()
    try:
        response = client.generate(model=model, prompt=query)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "answer": response.get("response", ""),
            "latency_ms": round(elapsed, 2),
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
        }
    except Exception as exc:
        logger.error("Raw call failed: %s", exc)
        return {"answer": "", "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0}


def run_handoff(handoff: AgentHandoff, query: str) -> Dict[str, Any]:
    """Handoff method: Model A → demonstrations → Model B."""
    result = handoff.run_detailed(query, use_cache=False)
    return {
        "answer": result.answer,
        "latency_ms": result.latency_ms,
        "prompt_tokens": sum(
            v for k, v in result.token_counts.items() if "prompt" in k
        ),
        "completion_tokens": sum(
            v for k, v in result.token_counts.items() if "completion" in k
        ),
        "demonstrations": len(result.demonstrations),
    }


def benchmark(
    model_a: str,
    model_b: str,
    queries: List[str] | None = None,
    host: str | None = None,
) -> List[Dict[str, Any]]:
    """Run the full benchmark and return a list of result rows.

    Each row contains: query (truncated), method, latency_ms,
    prompt_tokens, completion_tokens.
    """
    if queries is None:
        queries = CODING_QUERIES + EXPLANATION_QUERIES

    client = ollama.Client(host=host) if host else ollama.Client()
    cache = DemonstrationCache(default_ttl=0)  # No TTL for benchmark
    handoff = AgentHandoff(
        model_a=model_a,
        model_b=model_b,
        cache=cache,
        ollama_host=host,
    )

    rows: List[Dict[str, Any]] = []

    for i, query in enumerate(queries, 1):
        short_q = query[:60] + ("…" if len(query) > 60 else "")
        print(f"\n[{i}/{len(queries)}] {short_q}")

        # --- Raw baseline ---
        print("  ▸ Raw …", end=" ", flush=True)
        raw = run_raw(client, model_b, query)
        print(f"{raw['latency_ms']:.0f} ms")
        rows.append({
            "query": short_q,
            "method": "raw",
            "latency_ms": raw["latency_ms"],
            "prompt_tokens": raw["prompt_tokens"],
            "completion_tokens": raw["completion_tokens"],
        })

        # --- Handoff ---
        print("  ▸ Handoff …", end=" ", flush=True)
        ho = run_handoff(handoff, query)
        print(f"{ho['latency_ms']:.0f} ms  ({ho.get('demonstrations', 0)} demos)")
        rows.append({
            "query": short_q,
            "method": "handoff",
            "latency_ms": ho["latency_ms"],
            "prompt_tokens": ho["prompt_tokens"],
            "completion_tokens": ho["completion_tokens"],
        })

    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    """Pretty-print benchmark results as a table."""
    header = f"{'Query':<62} {'Method':<9} {'Latency(ms)':>12} {'Prompt Tok':>11} {'Compl Tok':>10}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for row in rows:
        print(
            f"{row['query']:<62} "
            f"{row['method']:<9} "
            f"{row['latency_ms']:>12.1f} "
            f"{row['prompt_tokens']:>11} "
            f"{row['completion_tokens']:>10}"
        )
    print(sep)

    # Averages
    for method in ("raw", "handoff"):
        subset = [r for r in rows if r["method"] == method]
        if not subset:
            continue
        avg_lat = sum(r["latency_ms"] for r in subset) / len(subset)
        avg_pt = sum(r["prompt_tokens"] for r in subset) / len(subset)
        avg_ct = sum(r["completion_tokens"] for r in subset) / len(subset)
        print(
            f"{'AVG ' + method.upper():<62} "
            f"{'':9} "
            f"{avg_lat:>12.1f} "
            f"{avg_pt:>11.0f} "
            f"{avg_ct:>10.0f}"
        )
    print(sep)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S.U.T.R.A Benchmark — handoff vs. raw context passing"
    )
    parser.add_argument("--model-a", default="llama3.2:3b", help="Demonstration model")
    parser.add_argument("--model-b", default="mistral:7b", help="Final-answer model")
    parser.add_argument("--host", default=None, help="Ollama server URL")
    parser.add_argument(
        "--coding-only", action="store_true", help="Run only coding queries"
    )
    parser.add_argument(
        "--explain-only", action="store_true", help="Run only explanation queries"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    queries = None
    if args.coding_only:
        queries = CODING_QUERIES
    elif args.explain_only:
        queries = EXPLANATION_QUERIES

    rows = benchmark(
        model_a=args.model_a,
        model_b=args.model_b,
        queries=queries,
        host=args.host,
    )
    print_table(rows)


if __name__ == "__main__":
    main()
