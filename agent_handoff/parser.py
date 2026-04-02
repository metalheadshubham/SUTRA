"""
parser.py — Extract answer and demonstrations from model output.

Primary path: parse JSON produced by Ollama's ``format="json"`` mode.
Fallback path: regex extraction for backward compatibility with XML output.
Designed to be resilient to the messy, inconsistent output of small LLMs.
"""

import json
import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Regex patterns (fallback for legacy XML output) ──────────────────────────

_ANSWER_RE = re.compile(
    r"<answer>(.*?)</\s*answer>", re.DOTALL | re.IGNORECASE
)
_DEMONSTRATIONS_BLOCK_RE = re.compile(
    r"<demonstrations>(.*?)</\s*demonstrations>", re.DOTALL | re.IGNORECASE
)
_SINGLE_DEMO_RE = re.compile(
    r"<demonstration>(.*?)</\s*demonstration>", re.DOTALL | re.IGNORECASE
)
_ANY_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


# ── JSON helpers ─────────────────────────────────────────────────────────────

def _escape_newlines_in_strings(s: str) -> str:
    """Escape raw newlines inside JSON string values.

    Ollama's format='json' guarantees structural validity but not properly
    escaped string contents — code blocks with literal newlines break
    json.loads. This walks the raw text and escapes newlines that appear
    inside string literals.
    """
    result = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            result.append(ch)
            escape = False
        elif ch == '\\':
            result.append(ch)
            escape = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif ch == '\n' and in_string:
            result.append('\\n')
        else:
            result.append(ch)
    return ''.join(result)


def _try_json_parse(text: str) -> Optional[dict]:
    """Try to parse *text* as JSON.

    Handles:
    - Markdown code fences (```json ... ```)
    - Raw newlines inside JSON string values (common with code-generating models)
    """
    cleaned = text.strip()

    # Strip optional markdown code fence
    if cleaned.startswith("```"):
        idx = cleaned.find("\n")
        cleaned = cleaned[idx + 1:] if idx != -1 else ""
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip().removesuffix("```").rstrip()

    # Fix unescaped newlines inside JSON string values
    cleaned = _escape_newlines_in_strings(cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parse failed: {e}")
    return None


# ── Public API ───────────────────────────────────────────────────────────────

def parse_answer(text: str) -> str:
    """Extract the answer from model output.

    Tries JSON first (``{"answer": "..."}``), then falls back to XML regex.
    Returns the full text (with tags stripped) if nothing is found.
    """
    # JSON path (primary)
    data = _try_json_parse(text)
    if data is not None:
        answer = data.get("answer")
        if answer and isinstance(answer, str):
            return answer.strip()

    # XML regex fallback
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()

    logger.warning("No answer found via JSON or XML; returning full text stripped of tags.")
    return strip_tags(text).strip()


def parse_demonstrations(text: str) -> List[str]:
    """Extract demonstrations from model output.

    Tries JSON first (``{"demonstrations": ["...", ...]}``), then falls back
    to the old XML regex approach. Returns an empty list if nothing is found.
    """
    # JSON path (primary)
    data = _try_json_parse(text)
    if data is not None:
        demos = data.get("demonstrations")
        if isinstance(demos, list):
            cleaned = [
                d.strip() for d in demos
                if isinstance(d, str) and d.strip()
            ]
            if cleaned:
                return cleaned

    # XML regex fallback
    block_match = _DEMONSTRATIONS_BLOCK_RE.search(text)
    if block_match:
        block = block_match.group(1)
        demos = [m.group(1).strip() for m in _SINGLE_DEMO_RE.finditer(block)]
        if demos:
            return demos
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if lines:
            logger.warning(
                "Found <demonstrations> block but no <demonstration> tags; "
                "splitting on newlines."
            )
            return lines

    # Last resort — loose <demonstration> tags anywhere
    demos = [m.group(1).strip() for m in _SINGLE_DEMO_RE.finditer(text)]
    if demos:
        logger.warning(
            "No <demonstrations> wrapper found; extracted loose <demonstration> tags."
        )
        return demos

    logger.warning("No demonstrations found in model output.")
    return []


def strip_tags(text: str) -> str:
    """Remove all XML/HTML-style tags from *text*."""
    return _ANY_TAG_RE.sub("", text)