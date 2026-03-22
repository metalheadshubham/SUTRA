# S.U.T.R.A

### Structured Universal Transfer via Retrieval Adaptation

> A lightweight Python library for orchestrating structured demonstration handoffs between small language models — enabling chains of 2B–7B models to collectively approach SOTA-level performance.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AgentHandoff.run()                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐ │
│   │  Query    │────▶│ Model A  │────▶│  Parser  │────▶│  Cache   │ │
│   │          │     │ (3B)     │     │          │     │          │  │
│   └──────────┘     └──────────┘     └──────────┘     └────┬─────┘ │
│                                                            │       │
│                     ┌──────────┐     ┌──────────┐          │       │
│                     │  Output  │◀────│ Model B  │◀─────────┘       │
│                     │          │     │ (7B)     │  demonstrations  │
│                     └──────────┘     └──────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Model A** generates an answer *and* 3 behavioral demonstrations.  
The **parser** extracts those demonstrations into a `HandoffPacket`.  
**Model B** receives the demonstrations as few-shot examples alongside the original query — producing a higher-quality final answer.

---

## Before / After

### ❌ Raw Context Passing
```python
import ollama

# Just forward the raw query — Model B has no style guidance
response = ollama.generate(model="mistral:7b", prompt="Write a safe Rust function to read a file")
print(response["response"])
```

### ✅ Structured Handoff (S.U.T.R.A)
```python
from agent_handoff import AgentHandoff

handoff = AgentHandoff(model_a="llama3.2:3b", model_b="mistral:7b")
result = handoff.run("Write a safe Rust function to read a file")
print(result)  # Model B's answer, guided by Model A's demonstrations
```

Model B receives curated demonstrations that prime it for the right style, format, and reasoning — without any extra training.

---

## Installation

```bash
# Clone and install
git clone <your-repo-url>
cd S.U.T.R.A
pip install -r requirements.txt

# Ensure Ollama is running with your models
ollama pull llama3.2:3b
ollama pull mistral:7b
```

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally.

---

## Quick Start

```python
from agent_handoff import AgentHandoff, DemonstrationCache

# Basic usage
handoff = AgentHandoff(model_a="llama3.2:3b", model_b="mistral:7b")
answer = handoff.run("Explain the CAP theorem with real-world analogies")
print(answer)

# Detailed result with metrics
result = handoff.run_detailed("Write a Python merge sort")
print(f"Answer: {result.answer[:200]}…")
print(f"Latency: {result.latency_ms:.0f} ms")
print(f"Cache hit: {result.cache_hit}")
print(f"Demonstrations used: {len(result.demonstrations)}")
```

### Manual Chaining (Advanced)

```python
# Step 1: Generate demonstrations only
packet = handoff.generate_demonstrations("Implement a binary search tree in Python")
print(f"Got {len(packet.demonstrations)} demonstrations")

# Step 2: Use demonstrations with any model
final = handoff.generate_final("Implement a binary search tree in Python", packet.demonstrations)
print(final)
```

---

## Caching

Demonstrations are cached by default (SHA-256 hash of the query, 1-hour TTL).

```python
from agent_handoff import AgentHandoff, DemonstrationCache

# Custom cache with 24h TTL
cache = DemonstrationCache(default_ttl=86400)
handoff = AgentHandoff(model_a="llama3.2:3b", model_b="mistral:7b", cache=cache)

# First call generates & caches demonstrations
handoff.run("Explain monads")

# Second call is faster — demonstrations served from cache
handoff.run("Explain monads")

# Persist cache to disk
cache.save("demo_cache.json")

# Load cache in a new session
new_cache = DemonstrationCache()
new_cache.load("demo_cache.json")
```

---

## Benchmark

Compare handoff vs. raw context passing across 10 test queries:

```bash
# Default models
python -m agent_handoff.benchmark

# Custom models
python -m agent_handoff.benchmark --model-a qwen2.5:3b --model-b deepseek-coder:6.7b

# Coding queries only
python -m agent_handoff.benchmark --coding-only

# Custom Ollama host
python -m agent_handoff.benchmark --host http://192.168.1.100:11434
```

Output:

```
Query                                                          Method    Latency(ms)  Prompt Tok  Compl Tok
──────────────────────────────────────────────────────────────────────────────────────────────────────────
Write a Python function that merges two sorted lists…          raw          1234.5          45        312
Write a Python function that merges two sorted lists…          handoff      3456.7         182        298
...
AVG RAW                                                                     1456.3          52        287
AVG HANDOFF                                                                 3891.2         195        301
```

---

## Custom Templates

Override the default prompts to tailor the handoff for your domain:

```python
handoff = AgentHandoff(
    model_a="llama3.2:3b",
    model_b="mistral:7b",
    templates={
        "prompt_a": "You are a security expert. Answer in <answer> tags, "
                    "then give 3 vulnerability examples in <demonstration> tags "
                    "inside <demonstrations>.\nQuery: {query}",
        "prompt_b": "Security examples:\n{demonstrations}\n\n"
                    "Apply the same security mindset to answer:\n{query}",
    },
)
```

---

## Project Structure

```
agent_handoff/
├── __init__.py       # Package exports
├── __main__.py       # python -m agent_handoff.benchmark
├── protocol.py       # HandoffPacket, HandoffResult dataclasses
├── parser.py         # XML + regex extraction of answer/demonstrations
├── cache.py          # In-memory cache with TTL + file persistence
├── templates.py      # Prompt templates for Model A and Model B
├── handoff.py        # AgentHandoff orchestrator
├── benchmark.py      # Latency/token comparison script
└── utils.py          # hash_query, truncate_text helpers
tests/
├── test_parser.py    # Parser unit tests
├── test_cache.py     # Cache unit tests
├── test_protocol.py  # Protocol unit tests
└── test_utils.py     # Utils unit tests
```

---

## Core Principles

| Principle | Detail |
|---|---|
| **Zero-training** | Pure inference — no fine-tuning, no RLHF |
| **Low-resource** | Designed for 8 GB RAM, CPU-only with 2B–7B models |
| **Extensible** | Chain 3+ models by composing `generate_demonstrations` / `generate_final` |
| **Production-ready** | Type hints, docstrings, logging, error handling throughout |

---

## Future Plans

- **Multi-hop chaining** — Chain N models where each refines the previous demonstrations
- **Quality scoring** — Automated evaluation of handoff vs. raw outputs using a judge model
- **Async pipeline** — Parallel demonstration generation across multiple queries
- **Demonstration selection** — Intelligent filtering of the most relevant demonstrations
- **Council mode** — Multiple models vote on the best demonstrations before forwarding

---

## License

MIT
