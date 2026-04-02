"""
utils.py — Lightweight helper functions.

Hashing, text truncation, and demonstration formatting utilities
shared across the library.
"""

import hashlib
from typing import List


def hash_query(query: str, profile: str | None = None) -> str:
    """Return a deterministic SHA-256 hex digest for *query*.

    If *profile* is provided it is appended to the hash input so that
    the same query with different profiles produces different keys.
    """
    seed = query if profile is None else f"{query}||{profile}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncate *text* to at most *max_chars* characters.

    Tries to break at a whitespace boundary to avoid splitting words.
    Appends ``…`` when truncation occurs.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars]
    # Try to break at last space
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated.rstrip() + "…"


def format_demonstrations(demonstrations: List[str]) -> str:
    """Format a list of demonstration strings into numbered examples."""
    if not demonstrations:
        return "(no demonstrations available)"
    return "\n\n".join(
        f"Example {i}:\n{demo}" for i, demo in enumerate(demonstrations, 1)
    )
