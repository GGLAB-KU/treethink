"""Lean 4 client adapters that wrap the external ``kimina_client`` package.

These adapters implement :class:`ProofAssistantClient` /
:class:`AsyncProofAssistantClient` so the rest of TreeThink can treat
Lean identically to any other formal language.
"""

from typing import Any, List
from loguru import logger
from ..base import (
    AsyncProofAssistantClient,
    ProofAssistantClient,
    ProofStateInfo,
)
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
        if response is None:
            return False
        return not has_error_response(response, accept_sorry=False)

    def extract_proof_state(
        self,
        proof_string: str,
        response: Any = None,
    ) -> ProofStateInfo:
        """Extract tactic/goal information from a Lean infotree response.

        Uses the Kimina server's ``infotree`` (embedded in the *response*
        dict) together with :func:`extract_data` and
        :func:`split_proof_header` to determine the last applied tactic,
        its open goals, and its solved goals.
        """
        if response is None or not isinstance(response, dict):
            logger.warning("Failed to extract proof state: no response given.")
            return ProofStateInfo()
        infotree = response.get("infotree")
        if not infotree:
            error_msg = response.get("error")
            return ProofStateInfo(error_message=error_msg)

        from .client.infotree import extract_data
        from .proof_utils import split_proof_header

        try:
            _header, body = split_proof_header(proof_string)
            intervals = extract_data(infotree, body)
            if not intervals:
                return ProofStateInfo()
            last = intervals[-1]
            return ProofStateInfo(
                applied_tactic=last.get("tactic"),
                open_goals=last.get("goalsAfter"),
                closed_goals=last.get("goalsBefore"),
            )
        except Exception as exc:
            logger.warning(
                f"Failed to extract proof state from infotree: {exc}"
            )
            return ProofStateInfo(error_message=str(exc))

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
