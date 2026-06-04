"""In-memory LRU cache for proof-snippet verification.

Provides :class:`ProofCache` (the key-value store) and two wrapper clients
(:class:`CachedClient`, :class:`AsyncCachedClient`) that implement
:class:`ProofAssistantClient` / :class:`AsyncProofAssistantClient` by
intercepting :meth:`check` and serving previously-seen snippets from the
cache instead of making a network call.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional

from loguru import logger

from .base import AsyncProofAssistantClient, ProofAssistantClient

# ---------------------------------------------------------------------------
# Cache entry & store
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """Result stored for a single proof snippet."""

    success: bool
    response: dict[str, Any]
    error: Optional[str] = None


def _snippet_key(snippet: str) -> str:
    """Deterministic hash for a proof snippet."""
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()


class ProofCache:
    """Fixed-size LRU store keyed by ``sha256(proof_snippet)``.

    Thread-safety is **not** guaranteed — callers serialise through the
    same event loop or process.
    """

    def __init__(self, maxsize: int = 4096) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    # -- public API --------------------------------------------------------

    def get(self, snippet: str) -> Optional[CacheEntry]:
        key = _snippet_key(snippet)
        entry = self._data.get(key)
        if entry is not None:
            self._hits += 1
            self._data.move_to_end(key)  # LRU refresh
        else:
            self._misses += 1
        return entry

    def put(self, snippet: str, entry: CacheEntry) -> None:
        key = _snippet_key(snippet)
        self._data[key] = entry
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)  # evict LRU

    def has(self, snippet: str) -> bool:
        return _snippet_key(snippet) in self._data

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": self.size}

    def clear(self) -> None:
        self._data.clear()
        self._hits = 0
        self._misses = 0


# ---------------------------------------------------------------------------
# Wrapper clients
# ---------------------------------------------------------------------------


def _build_synthetic_response(
    entries: List[CacheEntry],
) -> SimpleNamespace:
    """Reconstruct a ``response`` object from cached entries.

    Downstream code accesses ``response.results[i].response`` and
    ``response.results[i].error`` — this satisfies both.
    """
    results = [
        SimpleNamespace(response=e.response, error=e.error) for e in entries
    ]
    return SimpleNamespace(results=results)


class CachedClient(ProofAssistantClient):
    """Synchronous wrapper that caches :meth:`check` results.

    Usage:
        real = LeanClientAdapter(...)
        cached = CachedClient(real, cache=ProofCache())
        cached.check(snips=["...", "..."])   # first call → real.check()
        cached.check(snips=["...", "..."])   # second call → cache hit
    """

    def __init__(
        self,
        inner: ProofAssistantClient,
        cache: Optional[ProofCache] = None,
    ) -> None:
        self._inner = inner
        self._cache = cache or ProofCache()

    # -- ProofAssistantClient ----------------------------------------------

    def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> Any:
        if not snips:
            return _build_synthetic_response([])

        # Partition into cached / uncached, preserving original order.
        cached_entry_by_idx: dict[int, CacheEntry] = {}
        uncached_idx_to_snip: dict[int, str] = {}

        for i, snip in enumerate(snips):
            entry = self._cache.get(snip)
            if entry is not None:
                cached_entry_by_idx[i] = entry
            else:
                uncached_idx_to_snip[i] = snip

        # All cached — skip network entirely.
        if not uncached_idx_to_snip:
            logger.debug(f"Cache FULL hit for {len(snips)} snippet(s).")
            return _build_synthetic_response(
                [cached_entry_by_idx[i] for i in range(len(snips))]
            )

        # Send only uncached to the inner client.
        uncached_list = [
            uncached_idx_to_snip[idx] for idx in sorted(uncached_idx_to_snip)
        ]
        inner_resp = self._inner.check(
            snips=uncached_list,
            timeout=timeout,
            show_progress=show_progress,
            batch_size=batch_size,
            max_workers=max_workers,
        )

        # Cache results from the inner response.
        if inner_resp and hasattr(inner_resp, "results"):
            for pos_in_resp, orig_idx in enumerate(
                sorted(uncached_idx_to_snip)
            ):
                raw = inner_resp.results[pos_in_resp]
                snip = uncached_idx_to_snip[orig_idx]
                resp_dict = raw.response if hasattr(raw, "response") else {}
                err = getattr(raw, "error", None)
                if isinstance(resp_dict, dict):
                    success = self._inner.is_success_response(resp_dict)
                else:
                    success = False
                entry = CacheEntry(
                    success=success, response=resp_dict, error=err
                )
                self._cache.put(snip, entry)
                cached_entry_by_idx[orig_idx] = entry

        # Build final response in original order.
        merged = [cached_entry_by_idx[i] for i in range(len(snips))]
        return _build_synthetic_response(merged)

    def is_success_response(self, response: Any) -> bool:
        return self._inner.is_success_response(response)

    def close(self) -> None:
        self._inner.close()

    @property
    def cache(self) -> ProofCache:
        return self._cache


class AsyncCachedClient(AsyncProofAssistantClient):
    """Asynchronous wrapper that caches :meth:`check` results."""

    def __init__(
        self,
        inner: AsyncProofAssistantClient,
        cache: Optional[ProofCache] = None,
    ) -> None:
        self._inner = inner
        self._cache = cache or ProofCache()

    # -- AsyncProofAssistantClient -----------------------------------------

    async def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> Any:
        if not snips:
            return _build_synthetic_response([])

        cached_entry_by_idx: dict[int, CacheEntry] = {}
        uncached_idx_to_snip: dict[int, str] = {}

        for i, snip in enumerate(snips):
            entry = self._cache.get(snip)
            if entry is not None:
                cached_entry_by_idx[i] = entry
            else:
                uncached_idx_to_snip[i] = snip

        if not uncached_idx_to_snip:
            logger.debug(f"Async cache FULL hit for {len(snips)} snippet(s).")
            return _build_synthetic_response(
                [cached_entry_by_idx[i] for i in range(len(snips))]
            )

        uncached_list = [
            uncached_idx_to_snip[idx] for idx in sorted(uncached_idx_to_snip)
        ]
        inner_resp = await self._inner.check(
            snips=uncached_list,
            timeout=timeout,
            show_progress=show_progress,
            batch_size=batch_size,
            max_workers=max_workers,
        )

        if inner_resp and hasattr(inner_resp, "results"):
            for pos_in_resp, orig_idx in enumerate(
                sorted(uncached_idx_to_snip)
            ):
                raw = inner_resp.results[pos_in_resp]
                snip = uncached_idx_to_snip[orig_idx]
                resp_dict = raw.response if hasattr(raw, "response") else {}
                err = getattr(raw, "error", None)
                if isinstance(resp_dict, dict):
                    success = self._inner.is_success_response(resp_dict)
                else:
                    success = False
                entry = CacheEntry(
                    success=success, response=resp_dict, error=err
                )
                self._cache.put(snip, entry)
                cached_entry_by_idx[orig_idx] = entry

        merged = [cached_entry_by_idx[i] for i in range(len(snips))]
        return _build_synthetic_response(merged)

    def is_success_response(self, response: Any) -> bool:
        return self._inner.is_success_response(response)

    async def close(self) -> None:
        await self._inner.close()

    @property
    def cache(self) -> ProofCache:
        return self._cache


__all__ = [
    "CacheEntry",
    "CachedClient",
    "AsyncCachedClient",
    "ProofCache",
]
