"""Thin factory that creates a proof-assistant client from language + args."""

from typing import Optional

from .clients.base import AsyncProofAssistantClient, ProofAssistantClient
from .clients.cache import (
    AsyncCachedClient,
    CachedClient,
    ProofCache,
)
from .utils.args import ClientArgs
from .utils.enums import FormalLanguage


def _create_raw_client(
    language: FormalLanguage,
    client_args: ClientArgs,
) -> ProofAssistantClient:
    """Build a synchronous client **without** cache wrapping."""
    match language:
        case FormalLanguage.LEAN4:
            from .clients.lean.adapter import LeanClientAdapter

            return LeanClientAdapter(
                lean_server_url=client_args.lean_server_url
                or "http://localhost:8000",
            )
        case FormalLanguage.ROCQ:
            from .clients.coq.rocq import RocqBatchClient

            return RocqBatchClient(
                host=client_args.host or "127.0.0.1",
                port=client_args.port or 5000,
                workspace_dir=client_args.workspace_dir or ".",
                theorem_name=client_args.theorem_name or "__eval",
                statement=client_args.statement or "True",
                prelude=client_args.prelude,
            )
        case FormalLanguage.ISABELLE:
            from .clients.isabelle.client import IsabelleClient

            return IsabelleClient(
                session=client_args.isabelle_session or "HOL",
                imports=client_args.isabelle_imports or "Main",
                server_log=client_args.isabelle_server_log,
            )
        case _:
            raise ValueError(f"Unsupported formal language: {language}")


def _create_raw_async_client(
    language: FormalLanguage,
    client_args: ClientArgs,
) -> AsyncProofAssistantClient:
    """Build an asynchronous client **without** cache wrapping."""
    match language:
        case FormalLanguage.LEAN4:
            from .clients.lean.adapter import AsyncLeanClientAdapter

            return AsyncLeanClientAdapter(
                lean_server_url=client_args.lean_server_url
                or "http://localhost:8000",
            )
        case FormalLanguage.ROCQ:
            raise NotImplementedError("Rocq async client is not yet supported.")
        case FormalLanguage.ISABELLE:
            raise NotImplementedError(
                "Isabelle async client is not yet supported."
            )
        case _:
            raise ValueError(f"Unsupported formal language: {language}")


# -- Public API (with optional cache wrapping) ---------------------------


def create_client(
    language: FormalLanguage,
    client_args: ClientArgs,
    cache: Optional[ProofCache] = None,
) -> ProofAssistantClient:
    """Build a synchronous proof-assistant client for *language*.

    Selects the appropriate client based on the ``FormalLanguage`` enum
    and wraps it with :class:`CachedClient` if caching is enabled.
    """
    inner = _create_raw_client(language, client_args)
    cache_to_use = (
        cache
        if cache is not None
        else (
            ProofCache(maxsize=client_args.cache_maxsize)
            if client_args.enable_cache
            else None
        )
    )
    if cache_to_use is not None:
        return CachedClient(inner, cache=cache_to_use)
    return inner


def create_async_client(
    language: FormalLanguage,
    client_args: ClientArgs,
    cache: Optional[ProofCache] = None,
) -> AsyncProofAssistantClient:
    """Build an async proof-assistant client for *language*.

    Selects the appropriate async client based on the ``FormalLanguage``
    enum and wraps it with :class:`AsyncCachedClient` if caching is enabled.
    Raises ``NotImplementedError`` for languages without async support
    (Rocq, Isabelle).
    """
    inner = _create_raw_async_client(language, client_args)
    cache_to_use = (
        cache
        if cache is not None
        else (
            ProofCache(maxsize=client_args.cache_maxsize)
            if client_args.enable_cache
            else None
        )
    )
    if cache_to_use is not None:
        return AsyncCachedClient(inner, cache=cache_to_use)
    return inner


__all__ = ["create_client", "create_async_client"]
