"""Isabelle proof assistant client (stub — not yet implemented)."""

from typing import Any, List

from ..base import ProofAssistantClient


class IsabelleClient(ProofAssistantClient):
    """Placeholder Isabelle client.

    Once an Isabelle REPL server is available, implement the ``check``
    method and ``is_success_response`` logic here.
    """

    def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> Any:
        raise NotImplementedError("Isabelle client is not yet implemented.")

    def is_success_response(self, response: Any) -> bool:
        raise NotImplementedError("Isabelle client is not yet implemented.")


__all__ = ["IsabelleClient"]
