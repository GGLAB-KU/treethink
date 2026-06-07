"""Lean 4 client adapters that wrap the external ``kimina_client`` package.

These adapters implement :class:`ProofAssistantClient` /
:class:`AsyncProofAssistantClient` so the rest of TreeThink can treat
Lean identically to any other formal language.
"""

from typing import Any, List

from ..base import AsyncProofAssistantClient, ProofAssistantClient
from .proof_utils import has_error_response


class LeanClientAdapter(ProofAssistantClient):
    """Synchronous adapter around ``KiminaClient``."""

    def __init__(self, lean_server_url: str = "http://localhost:8000") -> None:
        from kimina_client import KiminaClient

        self._client = KiminaClient(lean_server_url)

    # -- ProofAssistantClient interface -----------------------------------

    def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> Any:
        return self._client.check(
            snips=snips,
            timeout=timeout,
            show_progress=show_progress,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    def is_success_response(self, response: dict) -> bool:
        return not has_error_response(response, accept_sorry=False)

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()


class AsyncLeanClientAdapter(AsyncProofAssistantClient):
    """Asynchronous adapter around ``AsyncKiminaClient``."""

    def __init__(self, lean_server_url: str = "http://localhost:8000") -> None:
        from kimina_client import AsyncKiminaClient

        self._client = AsyncKiminaClient(api_url=lean_server_url)

    # -- AsyncProofAssistantClient interface -------------------------------

    async def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> Any:
        return await self._client.check(
            snips=snips,
            timeout=timeout,
            show_progress=show_progress,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    def is_success_response(self, response: dict) -> bool:
        return not has_error_response(response, accept_sorry=False)

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await self._client.close()


__all__ = ["LeanClientAdapter", "AsyncLeanClientAdapter"]
