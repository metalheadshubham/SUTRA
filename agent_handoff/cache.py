"""
cache.py — Demonstration cache with TTL and optional file persistence.

Stores demonstration sets keyed by a SHA-256 hash of the query (or any
user-supplied key).  Supports time-based expiry and JSON serialization
for persistence across runs.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from agent_handoff.utils import hash_query

logger = logging.getLogger(__name__)


class DemonstrationCache:
    """In-memory cache for demonstration sets with optional TTL and file persistence.

    Each entry stores ``(demonstrations, expiry_timestamp)``.

    Args:
        default_ttl: Default time-to-live in seconds for cached entries.
                     Set to ``0`` or ``None`` to disable expiry.
    """

    def __init__(self, default_ttl: int = 3600) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl

    # ── Core operations ──────────────────────────────────────────────────

    def get(self, key: str) -> Optional[List[str]]:
        """Retrieve cached demonstrations by *key*.

        Returns ``None`` if the key is missing or the entry has expired.
        """
        entry = self._store.get(key)
        if entry is None:
            return None

        # Check TTL
        expires_at = entry.get("expires_at")
        if expires_at is not None and time.time() > expires_at:
            logger.debug("Cache entry expired for key=%s", key[:16])
            del self._store[key]
            return None

        return entry["demonstrations"]

    def set(
        self,
        key: str,
        demos: List[str],
        ttl: Optional[int] = None,
    ) -> None:
        """Store demonstrations under *key*.

        Args:
            key: Cache key (typically from :func:`hash_query`).
            demos: List of demonstration strings.
            ttl: Time-to-live in seconds.  Falls back to ``default_ttl``.
                 Pass ``0`` to store without expiry.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = (time.time() + effective_ttl) if effective_ttl else None
        self._store[key] = {
            "demonstrations": demos,
            "expires_at": expires_at,
            "cached_at": time.time(),
        }
        logger.debug("Cached %d demos for key=%s (ttl=%s)", len(demos), key[:16], effective_ttl)

    def make_key(self, query: str, profile: Optional[str] = None) -> str:
        """Generate a deterministic cache key from a query (and optional profile)."""
        seed = query if profile is None else f"{query}||{profile}"
        return hash_query(seed)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Number of entries currently in the cache (including expired)."""
        return len(self._store)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the cache to a JSON file at *path*."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self._store, fh, indent=2)
        logger.info("Cache saved to %s (%d entries)", path, len(self._store))

    def load(self, path: str) -> None:
        """Load cache entries from a JSON file at *path*.

        Existing in-memory entries are **merged** (file entries win on conflict).
        """
        filepath = Path(path)
        if not filepath.exists():
            logger.warning("Cache file not found: %s", path)
            return
        with open(filepath, "r", encoding="utf-8") as fh:
            data: Dict[str, Dict[str, Any]] = json.load(fh)
        self._store.update(data)
        logger.info("Cache loaded from %s (%d entries)", path, len(data))

    # ── Dunder helpers ───────────────────────────────────────────────────

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"DemonstrationCache(entries={self.size}, default_ttl={self.default_ttl})"
