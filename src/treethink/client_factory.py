"""Thin factory that creates a proof-assistant client from language + args."""

from typing import Optional

from .clients.base import AsyncProofAssistantClient, ProofAssistantClient
from .clients.cache import (
    AsyncCachedClient,
    CachedClient,
    ProofCache,
)
from .clients.coq.rocq import RocqBatchClient
from .clients.isabelle.client import IsabelleClient
from .clients.lean.adapter import AsyncLeanClientAdapter, LeanClientAdapter
from .utils.args import ClientArgs
from .utils.enums import FormalLanguage


def _create_raw_client(
    language: FormalLanguage,
    client_args: ClientArgs,
) -> ProofAssistantClient:
    """Build a synchronous client **without** cache wrapping."""
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


def _create_raw_async_client(
    language: FormalLanguage,
    client_args: ClientArgs,
) -> AsyncProofAssistantClient:
    """Build an asynchronous client **without** cache wrapping."""
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


# -- Public API (with optional cache wrapping) ---------------------------


def create_client(
    language: FormalLanguage,
    client_args: ClientArgs,
    cache: Optional[ProofCache] = None,
) -> ProofAssistantClient:
    """Build a synchronous proof-assistant client for *language*.

    If *cache* is provided (or ``client_args.enable_cache`` is ``True``),
    the raw client is wrapped with :class:`CachedClient`.
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
    """Build an asynchronous proof-assistant client for *language*.

    If *cache* is provided (or ``client_args.enable_cache`` is ``True``),
    the raw client is wrapped with :class:`AsyncCachedClient`.
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
