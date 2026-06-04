"""Tests for the proof-snippet verification cache.

Covers:
- :class:`ProofCache` basic operations (get / put / has / stats / LRU)
- :class:`CachedClient` deduplication (same snips → one inner call)
- Synthetic response reconstruction (cached results look like real ones)
- ``enable_cache=False`` — the factory returns a raw (uncached) client
- Async variant via :class:`AsyncCachedClient`
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

from treethink import (
    ClientArgs,
    FormalLanguage,
    LeanClientAdapter,
    ProofCache,
    create_client,
)
from treethink.clients.base import ProofAssistantClient
from treethink.clients.cache import (
    AsyncCachedClient,
    CachedClient,
    CacheEntry,
    _build_synthetic_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CountingClient(ProofAssistantClient):
    """Mock client that counts how many times ``check()`` is called."""

    def __init__(self, *, succeed: bool = True):
        self.call_count = 0
        self._succeed = succeed

    def check(
        self,
        *,
        snips: List[str],
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        results = [
            SimpleNamespace(
                response={
                    "proof_finished": self._succeed,
                    "messages": [],
                },
                error=None,
            )
            for _ in snips
        ]
        return SimpleNamespace(results=results)

    def is_success_response(self, response: dict) -> bool:
        return bool(response.get("proof_finished"))


# ---------------------------------------------------------------------------
# ProofCache
# ---------------------------------------------------------------------------


class TestProofCache:
    def test_basic_put_get_has(self):
        c = ProofCache(maxsize=10)
        snippet = "theorem foo : True := by trivial"

        assert not c.has(snippet)
        assert c.get(snippet) is None
        assert c.stats["misses"] == 1

        e = CacheEntry(success=True, response={"proof_finished": True})
        c.put(snippet, e)

        assert c.has(snippet)
        got = c.get(snippet)
        assert got is not None
        assert got.success is True
        assert c.stats["hits"] == 1

    def test_lru_eviction(self):
        c = ProofCache(maxsize=3)
        for i in range(5):
            c.put(f"snippet_{i}", CacheEntry(success=True, response={}))
        # Oldest two entries should be evicted.
        assert not c.has("snippet_0")
        assert not c.has("snippet_1")
        assert c.has("snippet_4")  # newest

    def test_clear(self):
        c = ProofCache(maxsize=10)
        c.put("a", CacheEntry(success=True, response={}))
        c.put("b", CacheEntry(success=False, response={}))
        assert c.size == 2
        c.clear()
        assert c.size == 0
        assert c.stats["hits"] == 0
        assert c.stats["misses"] == 0


# ---------------------------------------------------------------------------
# CachedClient (sync)
# ---------------------------------------------------------------------------


class TestCachedClient:
    def test_deduplicates_identical_batches(self):
        """Same snips twice → only one inner call."""
        inner = CountingClient()
        cache = ProofCache()
        wrapped = CachedClient(inner, cache=cache)

        snips = ["proof A", "proof B"]
        first = wrapped.check(snips=snips)
        assert inner.call_count == 1

        second = wrapped.check(snips=snips)
        assert inner.call_count == 1, "Should NOT call inner again"

        # Both responses should indicate success.
        for resp in (first, second):
            assert hasattr(resp, "results")
            assert len(resp.results) == 2
            assert all(r.response["proof_finished"] for r in resp.results)

    def test_partial_hit(self):
        """Only uncached snips are sent to inner."""
        inner = CountingClient()
        cache = ProofCache()
        wrapped = CachedClient(inner, cache=cache)

        # First call: both go to inner.
        wrapped.check(snips=["s1"])
        assert inner.call_count == 1

        # Second call: s1 cached, s2 new.
        wrapped.check(snips=["s1", "s2"])
        assert inner.call_count == 2, "Should only send uncached 's2'"

    def test_all_cached_skip_network(self):
        """When all cached, inner.check() is never called."""
        inner = CountingClient()
        cache = ProofCache()
        wrapped = CachedClient(inner, cache=cache)

        # Prime cache manually.
        cache.put(
            "prime",
            CacheEntry(success=True, response={"proof_finished": True}),
        )

        wrapped.check(snips=["prime"])
        assert inner.call_count == 0

    def test_empty_snips(self):
        """Empty snip list → synthetic empty response, no inner call."""
        inner = CountingClient()
        wrapped = CachedClient(inner)
        resp = wrapped.check(snips=[])
        assert inner.call_count == 0
        assert hasattr(resp, "results")
        assert len(resp.results) == 0

    def test_is_success_response_delegates(self):
        """is_success_response() is passed through to inner."""
        inner = CountingClient(succeed=True)
        wrapped = CachedClient(inner)
        assert wrapped.is_success_response({"proof_finished": True}) is True

    def test_cache_stats_track_hits_misses(self):
        inner = CountingClient()
        cache = ProofCache()
        wrapped = CachedClient(inner, cache=cache)

        wrapped.check(snips=["new"])
        assert cache.stats["misses"] == 1

        wrapped.check(snips=["new"])
        assert cache.stats["hits"] == 1
        assert cache.stats["size"] == 1


# ---------------------------------------------------------------------------
# Factory integration — enable_cache toggle
# ---------------------------------------------------------------------------


class TestFactoryCacheIntegration:
    def test_enable_cache_true_returns_cached_client(self):
        client = create_client(
            FormalLanguage.LEAN4,
            ClientArgs(
                lean_server_url="http://localhost:8000", enable_cache=True
            ),
        )
        assert isinstance(client, CachedClient), (
            f"Expected CachedClient, got {type(client)}"
        )

    def test_enable_cache_false_returns_raw_client(self):
        client = create_client(
            FormalLanguage.LEAN4,
            ClientArgs(
                lean_server_url="http://localhost:8000", enable_cache=False
            ),
        )
        assert isinstance(client, LeanClientAdapter), (
            f"Expected LeanClientAdapter (no cache), got {type(client)}"
        )


# ---------------------------------------------------------------------------
# AsyncCachedClient
# ---------------------------------------------------------------------------


class AsyncCountingClient:
    """Mock async client — satisfies :class:`AsyncProofAssistantClient` interface."""

    def __init__(self, *, succeed: bool = True):
        self.call_count = 0
        self._succeed = succeed

    async def check(
        self,
        *,
        snips: List[str],
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        results = [
            SimpleNamespace(
                response={
                    "proof_finished": self._succeed,
                    "messages": [],
                },
                error=None,
            )
            for _ in snips
        ]
        return SimpleNamespace(results=results)

    def is_success_response(self, response: dict) -> bool:
        return bool(response.get("proof_finished"))


class TestAsyncCachedClient:
    @pytest.mark.anyio
    async def test_deduplicates_identical_batches(self):
        inner = AsyncCountingClient()
        cache = ProofCache()
        wrapped = AsyncCachedClient(inner, cache=cache)

        snips = ["proof A", "proof B"]
        first = await wrapped.check(snips=snips)
        assert inner.call_count == 1

        second = await wrapped.check(snips=snips)
        assert inner.call_count == 1

        for resp in (first, second):
            assert hasattr(resp, "results")
            assert len(resp.results) == 2

    @pytest.mark.anyio
    async def test_partial_hit(self):
        inner = AsyncCountingClient()
        cache = ProofCache()
        wrapped = AsyncCachedClient(inner, cache=cache)

        await wrapped.check(snips=["s1"])
        assert inner.call_count == 1

        await wrapped.check(snips=["s1", "s2"])
        assert inner.call_count == 2

    @pytest.mark.anyio
    async def test_empty_snips(self):
        inner = AsyncCountingClient()
        wrapped = AsyncCachedClient(inner)
        resp = await wrapped.check(snips=[])
        assert inner.call_count == 0
        assert len(resp.results) == 0


# ---------------------------------------------------------------------------
# Synthetic response reconstruction helpers
# ---------------------------------------------------------------------------


class TestBuildSyntheticResponse:
    def test_matches_real_response_shape(self):
        """Synthetic response has the same shape as a real one."""
        from treethink.clients.cache import CacheEntry

        entries = [
            CacheEntry(
                success=True,
                response={"proof_finished": True},
            ),
            CacheEntry(
                success=False,
                response={"proof_finished": False, "error": "boom"},
                error="boom",
            ),
        ]
        resp = _build_synthetic_response(entries)
        assert resp.results[0].response["proof_finished"] is True
        assert resp.results[0].error is None
        assert resp.results[1].response["proof_finished"] is False
        assert resp.results[1].error == "boom"
