"""Language-agnostic proof-assistant client interfaces.

All REPL / termination clients must implement these ABCs so that the
termination layer can verify proof snippets without knowing which
formal language (Lean 4, Rocq, Isabelle, …) is in use.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class ProofAssistantClient(ABC):
    """Synchronous interface for a proof-assistant REPL client."""

    @abstractmethod
    def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> Any:
        """Batch-verify one or more proof snippets.

        Returns an object with a ``results`` attribute whose items expose
        the per-snippet verification outcome (language-specific format).
        """
        ...

    @abstractmethod
    def is_success_response(self, response: Any) -> bool:
        """Return ``True`` when *response* indicates a fully-verified proof."""
        ...

    def close(self) -> None:
        """Release any held resources (optional hook)."""


class AsyncProofAssistantClient(ABC):
    """Asynchronous interface for a proof-assistant REPL client."""

    @abstractmethod
    async def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> Any:
        """Async batch-verify one or more proof snippets.

        Returns an object with a ``results`` attribute.
        """
        ...

    @abstractmethod
    def is_success_response(self, response: Any) -> bool:
        """Return ``True`` when *response* indicates a fully-verified proof."""
        ...

    async def close(self) -> None:
        """Release any held resources (optional hook)."""


__all__ = ["ProofAssistantClient", "AsyncProofAssistantClient"]
