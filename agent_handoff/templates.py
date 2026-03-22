"""
templates.py — Prompt templates for the handoff and council pipelines.

Legacy (v0.1):
  PROMPT_A instructs the demonstration-generating model to output JSON.
  PROMPT_B instructs the final-answer model using those demonstrations.

Council (v0.2):
  ANSWER_PROMPT   — Stage 1a/1b: step-by-step solution generation.
  CRITIQUE_PROMPT — Stage 2: comparative review of two candidate solutions.
  SYNTHESIS_PROMPT — Stage 3: synthesize the single best implementation.
"""

PROMPT_A: str = """\
You are a helpful coding assistant. Answer the query, then provide exactly \
3 short demonstrations (code snippets or explanations) that show the same \
pattern applied to similar problems.

Respond with a JSON object using this exact schema — no other text:

{{
  "answer": "<your answer here>",
  "demonstrations": [
    "<demonstration 1>",
    "<demonstration 2>",
    "<demonstration 3>"
  ]
}}

Query: {query}
"""

PROMPT_B: str = """\
Here are examples of the desired output style and reasoning:

{demonstrations}

Now answer the following query using the same style, depth, and format \
demonstrated above.

Query: {query}
"""


def format_prompt_a(query: str, template: str | None = None) -> str:
    """Build the Model A prompt from *query* using *template* (or the default)."""
    tpl = template or PROMPT_A
    return tpl.format(query=query)


def format_prompt_b(
    query: str,
    demonstrations: list[str],
    template: str | None = None,
) -> str:
    """Build the Model B prompt, injecting formatted demonstrations."""
    tpl = template or PROMPT_B
    demo_text = "\n\n".join(
        f"Example {i}:\n{demo}" for i, demo in enumerate(demonstrations, 1)
    )
    return tpl.format(query=query, demonstrations=demo_text)


# ── Council pipeline prompts (v0.2) ─────────────────────────────────────────

ANSWER_PROMPT: str = """\
You are an expert programmer. Solve the following problem step by step, \
then output the complete solution.

Think through the problem carefully:
1. Understand the requirements and constraints
2. Consider edge cases
3. Write clean, correct code

Problem:
{query}
"""

CRITIQUE_PROMPT: str = """\
You are a rigorous code reviewer. Two candidate solutions were written for \
the same problem. Review BOTH solutions carefully.

For each solution, analyze:
1. **Correctness** — Does it produce the right output for all valid inputs?
2. **Logic errors** — Any off-by-one, wrong operator, missing return, infinite loop?
3. **Edge cases missed** — Empty input, single element, negative numbers, large input, None/null?

Be specific. Quote the exact lines that are wrong and explain why.

### Problem
{query}

### Solution A
{solution_a}

### Solution B
{solution_b}
"""

SYNTHESIS_PROMPT: str = """\
You are an expert programmer. You have two candidate solutions and a \
detailed critique of both. Your job is to write the single best \
implementation that fixes ALL issues identified in the critique.

### Problem
{query}

### Solution A
{solution_a}

### Solution B
{solution_b}

### Critique
{critique}

Write the final, corrected implementation. Output ONLY the function — \
no explanation, no markdown fences, no extra text. Just the code.
"""