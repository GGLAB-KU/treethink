"""Language-agnostic proof-assistant client interfaces.

All REPL / termination clients must implement these ABCs so that the
termination layer can verify proof snippets without knowing which
formal language (Lean 4, Rocq, Isabelle, …) is in use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class SnippetResult:
    """One snippet's verification outcome.

    ``response`` holds the language-specific payload interpreted by the
    owning client's :meth:`ProofAssistantClient.is_success_response`.
    """

    response: Any


@dataclass
class CheckResponse:
    """Return type of :meth:`ProofAssistantClient.check`.

    A language-agnostic container: ``results[i]`` is the
    :class:`SnippetResult` for the ``i``-th input snippet (order preserved).
    """

    results: List[SnippetResult] = field(default_factory=list)


@dataclass
class ProofStateInfo:
    """Per-snippet proof-state information extracted by the client.

    Fields are ``None`` when the information is not available (e.g. the
    client or language backend does not support fine-grained extraction).

    Attributes:
        applied_tactic: The tactic applied at the current proof step.
        open_goals: Goals remaining after applying the tactic.
        closed_goals: Goals that were solved by the tactic.
        error_message: Any error message from the proof assistant.
    """

    applied_tactic: str | None = None
    open_goals: str | None = None
    closed_goals: str | None = None
    error_message: str | None = None


class ProofAssistantClient(ABC):
    """Synchronous interface for a proof-assistant REPL client.

    All language-specific clients (Lean 4, Rocq, Isabelle) implement this
    ABC so the termination layer and evaluators can verify proof snippets
    without knowing which formal language is in use.

    To add a new language backend, implement this interface and register
    in ``client_factory.py``.
    """

    @abstractmethod
    def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> "CheckResponse":
        """Batch-verify one or more proof snippets.

        Returns a :class:`CheckResponse` whose ``results[i].response`` holds
        the per-snippet verification outcome (language-specific payload).
        """
        ...

    @abstractmethod
    def is_success_response(self, response: Any) -> bool:
        """Return ``True`` when *response* indicates a fully-verified proof."""
        ...

    def extract_proof_state(
        self,
        proof_string: str,
        response: Any = None,
    ) -> ProofStateInfo:
        """Extract tactic/goal information from *proof_string*.

        The optional *response* is the language-specific payload from
        :meth:`check` — if not provided the client may re-verify the
        snippet to obtain the information.

        The default implementation returns an empty :class:`ProofStateInfo`
        (all fields ``None``).  Subclasses that support fine-grained
        proof-state extraction should override this method.
        """
        return ProofStateInfo()

    def close(self) -> None:
        """Release any held resources (optional hook)."""


class AsyncProofAssistantClient(ABC):
    """Asynchronous interface for a proof-assistant REPL client.

    Like :class:`ProofAssistantClient` but with ``async check()`` for
    non-blocking proof verification.
    """

    @abstractmethod
    async def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> "CheckResponse":
        """Async batch-verify one or more proof snippets.

        Returns a :class:`CheckResponse` (see :meth:`ProofAssistantClient.check`).
        """
        ...

    @abstractmethod
    def is_success_response(self, response: Any) -> bool:
        """Return ``True`` when *response* indicates a fully-verified proof."""
        ...

    def extract_proof_state(
        self,
        proof_string: str,
        response: Any = None,
    ) -> ProofStateInfo:
        """Extract tactic/goal information from *proof_string*.

        The optional *response* is the language-specific payload from
        :meth:`check` — if not provided the client may re-verify the
        snippet to obtain the information.

        The default implementation returns an empty :class:`ProofStateInfo`
        (all fields ``None``).  Subclasses that support fine-grained
        proof-state extraction should override this method.
        """
        return ProofStateInfo()

    async def close(self) -> None:
        """Release any held resources (optional hook)."""


__all__ = [
    "AsyncProofAssistantClient",
    "CheckResponse",
    "ProofAssistantClient",
    "ProofStateInfo",
    "SnippetResult",
]
