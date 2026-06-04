"""Thin factory that creates a proof-assistant client from language + args."""

from .clients.base import AsyncProofAssistantClient, ProofAssistantClient
from .clients.coq.rocq import RocqBatchClient
from .clients.isabelle.client import IsabelleClient
from .clients.lean.adapter import AsyncLeanClientAdapter, LeanClientAdapter
from .utils.args import ClientArgs
from .utils.enums import FormalLanguage


def create_client(
    language: FormalLanguage,
    client_args: ClientArgs,
) -> ProofAssistantClient:
    """Build a synchronous proof-assistant client for *language*."""
    match language:
        case FormalLanguage.LEAN4:
            return LeanClientAdapter(
                lean_server_url=client_args.lean_server_url
                or "http://localhost:8000",
            )
        case FormalLanguage.RCOQ:
            return RocqBatchClient(
                host=client_args.host or "127.0.0.1",
                port=client_args.port or 5000,
                workspace_dir=client_args.workspace_dir or ".",
                theorem_name=client_args.theorem_name or "__eval",
                statement=client_args.statement or "True",
                prelude=client_args.prelude,
            )
        case FormalLanguage.ISABELLE:
            return IsabelleClient()
        case _:
            raise ValueError(f"Unsupported formal language: {language}")


def create_async_client(
    language: FormalLanguage,
    client_args: ClientArgs,
) -> AsyncProofAssistantClient:
    """Build an asynchronous proof-assistant client for *language*."""
    match language:
        case FormalLanguage.LEAN4:
            return AsyncLeanClientAdapter(
                lean_server_url=client_args.lean_server_url
                or "http://localhost:8000",
            )
        case FormalLanguage.RCOQ:
            raise NotImplementedError("Rocq async client is not yet supported.")
        case FormalLanguage.ISABELLE:
            raise NotImplementedError(
                "Isabelle async client is not yet supported."
            )
        case _:
            raise ValueError(f"Unsupported formal language: {language}")


__all__ = ["create_client", "create_async_client"]
