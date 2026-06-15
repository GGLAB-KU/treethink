"""Lightweight REPL runtime that wires termination checks to the correct
language-specific client and exposes a narrow API consumed by `TreeThink`.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Optional

from loguru import logger

from .client_factory import create_async_client, create_client
from .clients.base import AsyncProofAssistantClient, ProofAssistantClient
from .clients.cache import ProofCache
from .termination import check_terminated_paths, check_termination_encountered
from .utils.args import (
    ClientArgs,
    TerminationOnEncounterConfig,
    TerminationOnPathsConfig,
    TreeThinkArgs,
)
from .utils.enums import FormalLanguage


@dataclass
class ReplRuntime:
    """Coordinates proof-assistant clients and termination callbacks.

    Owns sync and async REPL clients and exposes callbacks consumed by
    ``TreeThink.generate()`` / ``async_generate()``.  Manages a shared
    :class:`ProofCache` when caching is enabled.
    """

    language: FormalLanguage
    client_args: ClientArgs
    encountered_config: TerminationOnEncounterConfig
    paths_config: TerminationOnPathsConfig

    client: Optional[ProofAssistantClient] = None
    async_client: Optional[AsyncProofAssistantClient] = None

    # Internal: shared LRU cache (created when enable_cache=True)
    _cache: Optional[ProofCache] = None

    # Encountered-termination batching
    _encountered_batch_size: int = 1
    _pending_encountered: List[tuple] = field(default_factory=list)
    # each entry: (method, node, proof_path, parsed_snippet)

    # -- factory -----------------------------------------------------------

    @classmethod
    def from_treethink_args(
        cls,
        args: TreeThinkArgs,
        termination_str: Optional[str] = None,
    ) -> "ReplRuntime":
        repl_enabled = bool(termination_str)
        encountered_enabled = (
            args.termination_on_encounter.enabled and repl_enabled
        )
        paths_enabled = args.termination_on_paths.enabled and repl_enabled

        # Create shared cache when caching is enabled.
        cache: Optional[ProofCache] = None
        if args.client_args.enable_cache and repl_enabled:
            cache = ProofCache(maxsize=args.client_args.cache_maxsize)

        return cls(
            language=args.language,
            client_args=args.client_args,
            encountered_config=TerminationOnEncounterConfig(
                enabled=encountered_enabled,
                batch_size=args.termination_on_encounter.batch_size,
            ),
            paths_config=TerminationOnPathsConfig(
                enabled=paths_enabled,
                max_repl=args.termination_on_paths.max_repl,
            ),
            _cache=cache,
            _encountered_batch_size=args.termination_on_encounter.batch_size,
        )

    # -- properties --------------------------------------------------------

    @property
    def needs_repl(self) -> bool:
        return self.encountered_config.enabled or self.paths_config.enabled

    # -- client helpers ----------------------------------------------------

    def _sync_client(self) -> ProofAssistantClient:
        if self.client is None:
            self.client = create_client(
                self.language,
                self.client_args,
                cache=self._cache,
            )
        return self.client

    def _async_client(self) -> AsyncProofAssistantClient:
        if self.async_client is None:
            self.async_client = create_async_client(
                self.language,
                self.client_args,
                cache=self._cache,
            )
        return self.async_client

    # -- encountered-termination batching ---------------------------------

    def _flush_pending_encountered(self, method) -> Optional[str]:
        """Flush accumulated encountered-termination proofs to the REPL.

        Returns the *proof path* (str) of the first verified proof, or
        ``None`` if none verified.  Failed nodes are penalised with
        ``win_value = float("-inf")`` just as the immediate-send path does.
        """
        if not self._pending_encountered:
            return None

        batch = self._pending_encountered
        self._pending_encountered = []

        nodes: List[Any] = [entry[1] for entry in batch]
        proof_paths: List[str] = [entry[2] for entry in batch]
        snips: List[str] = [entry[3] for entry in batch]

        logger.debug(
            f"Flushing encountered-termination batch ({len(snips)} snippet(s))."
        )

        client = self._sync_client()
        try:
            resp = client.check(
                snips=snips,
                timeout=self.client_args.timeout,
                show_progress=False,
                batch_size=self.client_args.batch_size,
                max_workers=self.client_args.num_proc,
            )
        except Exception as e:
            logger.error(f"REPL batch (encountered) failed: {e}")
            return None

        if not resp or not hasattr(resp, "results"):
            logger.warning("REPL batch (encountered) returned no results.")
            return None

        for idx, result in enumerate(resp.results):
            if client.is_success_response(result.response):
                logger.success(
                    f"REPL batch (encountered) found a solution at index {idx}!"
                )
                return proof_paths[idx]
            # Penalise failed node (same as immediate-send path).
            if idx < len(nodes):
                nodes[idx].win_value = float("-inf")

        logger.debug("No valid solution in encountered-termination batch.")
        return None

    async def _async_flush_pending_encountered(self, method) -> Optional[str]:
        """Async variant of :meth:`_flush_pending_encountered`."""
        if not self._pending_encountered:
            return None

        batch = self._pending_encountered
        self._pending_encountered = []

        nodes: List[Any] = [entry[1] for entry in batch]
        proof_paths: List[str] = [entry[2] for entry in batch]
        snips: List[str] = [entry[3] for entry in batch]

        logger.debug(
            "Async flushing encountered-termination batch "
            f"({len(snips)} snippet(s))."
        )

        client = self._async_client()
        try:
            resp = await client.check(
                snips=snips,
                timeout=self.client_args.timeout,
                show_progress=False,
                batch_size=self.client_args.batch_size,
                max_workers=self.client_args.num_proc,
            )
        except Exception as e:
            logger.error(f"Async REPL batch (encountered) failed: {e}")
            return None

        if not resp or not hasattr(resp, "results"):
            logger.warning(
                "Async REPL batch (encountered) returned no results."
            )
            return None

        for idx, result in enumerate(resp.results):
            if client.is_success_response(result.response):
                logger.success(
                    "Async REPL batch (encountered) found a solution "
                    f"at index {idx}!"
                )
                return proof_paths[idx]
            if idx < len(nodes):
                nodes[idx].win_value = float("-inf")

        logger.debug(
            "No valid solution in async encountered-termination batch."
        )
        return None

    def flush_encountered_batch(self, method) -> Optional[str]:
        """Flush any remaining pending encountered-termination proofs.

        Call this **after** the search loop completes so that proofs
        collected but not yet sent to the REPL are still verified.

        Returns the verified proof path or ``None``.
        """
        from .utils.enums import BestAnswerReason

        result = self._flush_pending_encountered(method)
        if result is not None:
            method._best_answer = result
            method.best_answer_reason = BestAnswerReason.CHECKED_AND_TRUE
        return result

    async def async_flush_encountered_batch(self, method) -> Optional[str]:
        """Async variant of :meth:`flush_encountered_batch`."""
        from .utils.enums import BestAnswerReason

        result = await self._async_flush_pending_encountered(method)
        if result is not None:
            method._best_answer = result
            method.best_answer_reason = BestAnswerReason.CHECKED_AND_TRUE
        return result

    # -- termination callbacks ---------------------------------------------

    def build_termination_callback(self, method, async_mode: bool = False):
        """Return a callable suitable for ``termination_encountered_fn``.

        When ``_encountered_batch_size > 1``, the callable accumulates
        proof snippets and only sends them to the REPL once the batch
        threshold is reached.  Otherwise it sends immediately (legacy
        behaviour).
        """
        if not self.encountered_config.enabled:
            return None

        is_batching = self._encountered_batch_size > 1

        # ----------  Sync path  ----------
        if not async_mode:
            if not is_batching:
                # Legacy path — send on every call.
                return partial(
                    check_termination_encountered,
                    method=method,
                    client=self._sync_client(),
                    timeout=self.client_args.timeout,
                    num_proc=self.client_args.num_proc,
                    batch_size=self.client_args.batch_size,
                )

            # Batching path — accumulate then flush.
            def _batch_callback(node):
                proof_path = method.traverse_to_root(node, include_root=True)
                parsed = method.parse_proof(proof_path)
                self._pending_encountered.append(
                    (method, node, proof_path, parsed)
                )
                if (
                    len(self._pending_encountered)
                    >= self._encountered_batch_size
                ):
                    return self._flush_pending_encountered(method)
                return None

            return _batch_callback

        # ----------  Async path  ----------
        if not is_batching:
            # Legacy path — send immediately.
            async def _immediate_async_callback(node):
                client = self._async_client()
                proof_path = method.traverse_to_root(node, include_root=True)
                parsed = method.parse_proof(proof_path)
                try:
                    resp = await client.check(
                        snips=[parsed],
                        timeout=self.client_args.timeout,
                        show_progress=False,
                        batch_size=self.client_args.batch_size,
                        max_workers=self.client_args.num_proc,
                    )
                except Exception as e:
                    logger.error(f"Async REPL (encountered) failed: {e}")
                    return None
                if (
                    resp
                    and hasattr(resp, "results")
                    and resp.results[0].response
                    and client.is_success_response(resp.results[0].response)
                ):
                    logger.info("Async REPL found a solution (encountered)!")
                    return proof_path
                return None

            return _immediate_async_callback

        # Batching async path.
        async def _async_batch_callback(node):
            proof_path = method.traverse_to_root(node, include_root=True)
            parsed = method.parse_proof(proof_path)
            self._pending_encountered.append((method, node, proof_path, parsed))
            if len(self._pending_encountered) >= self._encountered_batch_size:
                return await self._async_flush_pending_encountered(method)
            return None

        return _async_batch_callback

    def check_terminated_paths(self, method):
        """Batch-verify terminated leaves (sync)."""
        if not self.paths_config.enabled:
            return None
        return check_terminated_paths(
            method,
            client=self._sync_client(),
            timeout=self.client_args.timeout,
            num_proc=self.client_args.num_proc,
            batch_size=self.client_args.batch_size,
            max_repl=self.paths_config.max_repl,
        )

    async def async_check_terminated_paths(self, method):
        """Batch-verify terminated leaves (async)."""
        if not self.paths_config.enabled:
            return None

        client = self._async_client()
        node = method.root_node

        leaves = method.find_leaves(node)
        terminated_leaves = [
            leaf for leaf in leaves if leaf.is_termination_node
        ]

        if not terminated_leaves:
            logger.debug("No terminated leaves found (async).")
            return None

        if len(terminated_leaves) >= self.paths_config.max_repl:
            terminated_leaves = sorted(
                terminated_leaves, key=lambda x: x.win_value, reverse=True
            )[: self.paths_config.max_repl]

        proof_paths = [
            method.traverse_to_root(leaf) for leaf in terminated_leaves
        ]
        snips = [method.parse_proof(proof) for proof in proof_paths]

        try:
            resp = await client.check(
                snips=snips,
                timeout=self.client_args.timeout,
                show_progress=False,
                batch_size=self.client_args.batch_size,
                max_workers=self.client_args.num_proc,
            )
        except Exception as e:
            logger.error(f"Async REPL batch check failed: {e}")
            return None

        if not resp or not hasattr(resp, "results"):
            logger.warning("Async REPL returned no results.")
            return None

        for idx, result in enumerate(resp.results):
            if client.is_success_response(result.response):
                logger.success(f"Async REPL found a solution at index {idx}!")
                return proof_paths[idx]

        return None


__all__ = ["ReplRuntime"]
