"""Lightweight REPL runtime that wires termination checks to the correct
language-specific client and exposes a narrow API consumed by `TreeThink`.
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional

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
    """Thin coordinator that owns proof-assistant clients and exposes
    termination-callback helpers consumed by `TreeThink`.

    If :attr:`client_args.enable_cache` is ``True``, a single shared
    :class:`ProofCache` instance is created and used by both the sync
    and async clients, so that a proof verified via ``check_termination_encountered``
    (sync) is also cached for ``async_check_terminated_paths``.
    """

    language: FormalLanguage
    client_args: ClientArgs
    encountered_config: TerminationOnEncounterConfig
    paths_config: TerminationOnPathsConfig

    client: Optional[ProofAssistantClient] = None
    async_client: Optional[AsyncProofAssistantClient] = None

    # Internal: shared LRU cache (created when enable_cache=True)
    _cache: Optional[ProofCache] = None

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
            ),
            paths_config=TerminationOnPathsConfig(
                enabled=paths_enabled,
                max_repl=args.termination_on_paths.max_repl,
            ),
            _cache=cache,
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

    # -- termination callbacks ---------------------------------------------

    def build_termination_callback(self, method, async_mode: bool = False):
        """Return a callable suitable for ``termination_encountered_fn``."""
        if not self.encountered_config.enabled:
            return None

        if async_mode:
            # Build a coroutine that uses the async client inline.
            async def _async_callback(node):
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

            return _async_callback

        # Sync
        return partial(
            check_termination_encountered,
            client=self._sync_client(),
            timeout=self.client_args.timeout,
            num_proc=self.client_args.num_proc,
            batch_size=self.client_args.batch_size,
        )

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
