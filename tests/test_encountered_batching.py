"""Tests for encountered-termination batching.

Verifies:
- ``batch_size=1`` → immediate callback (backward compatible)
- ``batch_size > 1`` → closure that accumulates and flushes at threshold
- ``flush_encountered_batch()`` handles remaining pending after search
- Pending list is empty after flush
- Disabled config returns ``None`` callback
- Config propagation from ``TreeThinkArgs`` → ``ReplRuntime``
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Any, List

import pytest

from treethink import (
    ClientArgs,
    TerminationOnEncounterConfig,
    TreeThinkArgs,
)
from treethink.clients.base import ProofAssistantClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMethod:
    """Minimal stand-in for a search method."""

    class RootNode:
        pass

    root_node = RootNode()

    @staticmethod
    def traverse_to_root(node, include_root: bool = True) -> str:
        return f"proof {node}"

    @staticmethod
    def parse_proof(proof: str) -> str:
        return proof


class CountingClient(ProofAssistantClient):
    """Mock client that records all submitted batches."""

    def __init__(self, *, succeed: bool = True):
        self.calls: List[List[str]] = []
        self._succeed = succeed

    def check(
        self,
        *,
        snips: List[str],
        **kwargs: Any,
    ) -> Any:
        self.calls.append(snips)
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
# Config propagation
# ---------------------------------------------------------------------------


class TestConfigPropagation:
    def test_default_batch_size_is_one(self):
        """Default config keeps batch_size=1 (backward compat)."""
        cfg = TerminationOnEncounterConfig()
        assert cfg.batch_size == 1

    def test_custom_batch_size_propagates_to_runtime(self):
        args = TreeThinkArgs(
            termination_str="```",
            termination_on_encounter=TerminationOnEncounterConfig(
                enabled=True,
                batch_size=5,
            ),
        )
        rt = args.build_repl_runtime()
        assert rt.encountered_config.batch_size == 5
        assert rt._encountered_batch_size == 5

    def test_disabled_returns_none_callback(self):
        args = TreeThinkArgs(
            termination_str="```",
            termination_on_encounter=TerminationOnEncounterConfig(
                enabled=False,
            ),
        )
        rt = args.build_repl_runtime()
        cb = rt.build_termination_callback(FakeMethod())
        assert cb is None

    def test_no_termination_str_disables_repl(self):
        """Without a termination_str, repl_enabled=False disables config."""
        args = TreeThinkArgs(
            termination_on_encounter=TerminationOnEncounterConfig(
                enabled=True,
            ),
        )
        rt = args.build_repl_runtime()
        # repl_enabled is False because termination_str is None
        assert not rt.encountered_config.enabled


# ---------------------------------------------------------------------------
# Callback shape
# ---------------------------------------------------------------------------


class TestCallbackShape:
    def test_batch_size_one_returns_partial(self):
        """batch_size=1 → partial function (immediate path)."""
        args = TreeThinkArgs(
            termination_str="```",
            client_args=ClientArgs(lean_server_url="http://localhost:8000"),
            termination_on_encounter=TerminationOnEncounterConfig(
                enabled=True,
                batch_size=1,
            ),
        )
        rt = args.build_repl_runtime()
        cb = rt.build_termination_callback(FakeMethod())
        assert isinstance(cb, partial), f"Expected partial, got {type(cb)}"

    def test_batch_size_greater_one_returns_closure(self):
        """batch_size > 1 → custom callable (batching path)."""
        args = TreeThinkArgs(
            termination_str="```",
            client_args=ClientArgs(lean_server_url="http://localhost:8000"),
            termination_on_encounter=TerminationOnEncounterConfig(
                enabled=True,
                batch_size=3,
            ),
        )
        rt = args.build_repl_runtime()
        cb = rt.build_termination_callback(FakeMethod())
        assert callable(cb)
        assert not isinstance(cb, partial), "Expected closure, got partial"


# ---------------------------------------------------------------------------
# Batching behaviour
# ---------------------------------------------------------------------------


class TestBatchingBehaviour:
    @pytest.fixture
    def runtime(self):
        """ReplRuntime with batch_size=2 and a mock client injected."""
        args = TreeThinkArgs(
            termination_str="```",
            termination_on_encounter=TerminationOnEncounterConfig(
                enabled=True,
                batch_size=2,
            ),
        )
        rt = args.build_repl_runtime()
        rt.client = CountingClient(succeed=True)
        return rt

    def test_accumulates_then_flushes(self, runtime):
        method = FakeMethod()
        cb = runtime.build_termination_callback(method)

        # First node → accumulate (1 < 2)
        r1 = cb("a")
        assert r1 is None
        assert len(runtime._pending_encountered) == 1
        assert len(runtime.client.calls) == 0

        # Second node → flush (2 >= 2)
        r2 = cb("b")
        assert r2 is not None, "Should return proof path on flush"
        assert len(runtime._pending_encountered) == 0
        assert len(runtime.client.calls) == 1
        assert runtime.client.calls[0] == ["proof a", "proof b"]

    def test_flush_encountered_batch_after_search(self, runtime):
        """flush_encountered_batch() handles remaining pending."""
        method = FakeMethod()
        cb = runtime.build_termination_callback(method)

        cb("x")  # accumulate
        assert len(runtime._pending_encountered) == 1

        result = runtime.flush_encountered_batch(method)
        assert result is not None, "Should flush remaining"
        assert len(runtime._pending_encountered) == 0
        assert len(runtime.client.calls) == 1
        assert runtime.client.calls[0] == ["proof x"]

    def test_flush_with_nothing_pending(self, runtime):
        result = runtime.flush_encountered_batch(FakeMethod())
        assert result is None
        assert len(runtime.client.calls) == 0

    def test_multiple_batches(self, runtime):
        """With batch_size=2, 5 nodes → 3 flushes (2+2+1)."""
        method = FakeMethod()
        cb = runtime.build_termination_callback(method)

        for node in ["a", "b"]:
            cb(node)
        assert len(runtime.client.calls) == 1  # first batch

        for node in ["c", "d"]:
            cb(node)
        assert len(runtime.client.calls) == 2  # second batch

        cb("e")
        assert len(runtime.client.calls) == 2  # not flushed yet

        runtime.flush_encountered_batch(method)
        assert len(runtime.client.calls) == 3  # third flush (remaining)

    def test_failed_proof_penalises_node(self, runtime):
        """When the proof fails, the node gets win_value = -inf."""
        # Replace with a client that fails all proofs
        from types import SimpleNamespace

        from treethink.clients.base import ProofAssistantClient

        class FailingClient(ProofAssistantClient):
            def check(self, *, snips, **kw):
                results = [
                    SimpleNamespace(
                        response={
                            "proof_finished": False,
                            "error": "wrong",
                        },
                        error="wrong",
                    )
                    for _ in snips
                ]
                return SimpleNamespace(results=results)

            def is_success_response(self, resp):
                return False

        runtime.client = FailingClient()
        method = FakeMethod()
        cb = runtime.build_termination_callback(method)

        class FakeNode:
            win_value = 0.0

        node_a = FakeNode()
        node_b = FakeNode()

        cb(node_a)
        cb(node_b)  # flush — both fail

        assert node_a.win_value == float("-inf")
        assert node_b.win_value == float("-inf")


# ---------------------------------------------------------------------------
# Cache interaction
# ---------------------------------------------------------------------------


class TestCacheInteraction:
    def test_cached_client_wraps_sync_client(self):
        """ReplRuntime._sync_client returns a client wrapped in CachedClient
        when enable_cache=True. The batching flush automatically benefits."""
        from treethink.clients.cache import CachedClient

        args = TreeThinkArgs(
            termination_str="```",
            client_args=ClientArgs(
                lean_server_url="http://localhost:8000",
                enable_cache=True,
            ),
            termination_on_encounter=TerminationOnEncounterConfig(
                enabled=True,
                batch_size=2,
            ),
        )
        rt = args.build_repl_runtime()
        client = rt._sync_client()
        assert isinstance(client, CachedClient), (
            f"Expected CachedClient, got {type(client)}"
        )
