"""Lightweight in-memory cache utilities."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple


class MemoryCache:
    """A minimal in-memory TTL cache for reusing intermediate results.

    The cache is process-local and is intended for lightweight helper use, not
    durable persistence.

    Attributes:
        _store: Mapping of cache keys to ``(value, expires_at)`` pairs.
    """

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._store: Dict[str, Tuple[Any, Optional[float]]] = {}

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value under a key.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Optional time-to-live in seconds. ``None`` means no expiry.
        """
        expires_at = time.time() + ttl if ttl else None
        self._store[key] = (value, expires_at)

    def get(self, key: str) -> Optional[Any]:
        """Return a cached value if present and not expired.

        Args:
            key: Cache key.

        Returns:
            Cached value, or ``None`` when the key is missing or expired.
        """
        value = self._store.get(key)
        if not value:
            return None
        payload, expires_at = value
        if expires_at and expires_at < time.time():
            del self._store[key]
            return None
        return payload

    def clear(self) -> None:
        """Remove all cached values."""
        self._store.clear()
