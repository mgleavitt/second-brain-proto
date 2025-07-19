"""Simple in-memory cache implementation for query results.

This module provides a SimpleCache class that stores query results in memory
with configurable TTL (time-to-live) and maximum size limits.
"""

import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class SimpleCache:
    """Simple in-memory cache for query results."""

    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_hours = ttl_hours

    def _get_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _is_expired(self, timestamp: str) -> bool:
        """Check if a cache entry is expired."""
        try:
            entry_time = datetime.fromisoformat(timestamp)
            return datetime.now() - entry_time > timedelta(hours=self.ttl_hours)
        except (ValueError, TypeError):
            return True

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get a cached result for a query."""
        key = self._get_key(query)

        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check if expired
        if self._is_expired(entry['timestamp']):
            del self.cache[key]
            return None

        return entry['data']

    def set(self, query: str, data: Dict[str, Any]) -> None:
        """Cache a query result."""
        key = self._get_key(query)

        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'query': query
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()

    def remove_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry['timestamp'])
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self.remove_expired()  # Clean up expired entries

        return {
            'total_entries': len(self.cache),
            'max_size': self.max_size,
            'ttl_hours': self.ttl_hours
        }
