from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping

from kimina_client import AsyncKiminaClient, KiminaClient

from .clients.coq.rocq import RocqBatchClient
from .termination import (
    async_repl_encountered_termination,
    async_repl_terminated_paths,
    repl_encountered_termination,
    repl_terminated_paths,
)

SYNC_REPL_FUNCTIONS = {
    "repl_encountered_termination": repl_encountered_termination,
    "repl_terminated_paths": repl_terminated_paths,
}

ASYNC_REPL_FUNCTIONS = {
    "async_repl_encountered_termination": async_repl_encountered_termination,
    "async_repl_terminated_paths": async_repl_terminated_paths,
}


class ReplBackendBase(ABC):
    name: str

    @abstractmethod
    def resolve_sync_function(self, function_name: str) -> Callable:
        raise NotImplementedError

    @abstractmethod
    def resolve_async_function(self, function_name: str) -> Callable:
        raise NotImplementedError

    @abstractmethod
    def create_sync_client(
        self, repl_args: Any, backend_args: Mapping[str, Any]
    ):
        raise NotImplementedError

    @abstractmethod
    def create_async_client(
        self, repl_args: Any, backend_args: Mapping[str, Any]
    ):
        raise NotImplementedError

    def build_call_kwargs(
        self,
        *,
        mode: str,
        method: Any,
        repl_args: Any,
        strategy_args: Any,
        backend_args: Mapping[str, Any],
        client: Any,
    ) -> Dict[str, Any]:
        kwargs = {
            "method": method,
            "client": client,
            "timeout": repl_args.get("timeout"),
            "num_proc": repl_args.get("num_proc"),
            "batch_size": repl_args.get("batch_size"),
        }
        if mode == "terminated_paths":
            kwargs["max_repl"] = strategy_args.max_repl
        return kwargs


class KiminaReplBackend(ReplBackendBase):
    name = "kimina"

    def resolve_sync_function(self, function_name: str) -> Callable:
        try:
            return SYNC_REPL_FUNCTIONS[function_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Kimina sync REPL function '{function_name}'."
            ) from exc

    def resolve_async_function(self, function_name: str) -> Callable:
        try:
            return ASYNC_REPL_FUNCTIONS[function_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Kimina async REPL function '{function_name}'."
            ) from exc

    def create_sync_client(
        self, repl_args: Any, backend_args: Mapping[str, Any]
    ):
        client_cls = backend_args.get("sync_client_cls", KiminaClient)
        return client_cls(repl_args.lean_server_url)

    def create_async_client(
        self, repl_args: Any, backend_args: Mapping[str, Any]
    ):
        client_cls = backend_args.get("async_client_cls", AsyncKiminaClient)
        return client_cls(repl_args.lean_server_url)


class RocqReplBackend(ReplBackendBase):
    name = "rocq"

    def resolve_sync_function(self, function_name: str) -> Callable:
        try:
            return SYNC_REPL_FUNCTIONS[function_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Rocq sync REPL function '{function_name}'."
            ) from exc

    def resolve_async_function(self, function_name: str) -> Callable:
        try:
            return ASYNC_REPL_FUNCTIONS[function_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Rocq async REPL function '{function_name}'."
            ) from exc

    def create_sync_client(
        self, repl_args: Any, backend_args: Mapping[str, Any]
    ):
        client_cls = backend_args.get("sync_client_cls", RocqBatchClient)
        client_kwargs = dict(backend_args.get("client_kwargs", {}))
        client_kwargs.setdefault(
            "workspace_dir", repl_args.get("workspace_dir")
        )
        client_kwargs.setdefault(
            "theorem_name", backend_args.get("theorem_name", "__eval")
        )
        client_kwargs.setdefault(
            "statement", backend_args.get("statement", "True")
        )
        client_kwargs.setdefault("prelude", backend_args.get("prelude"))
        return client_cls(
            repl_args.get("host"), repl_args.get("port"), **client_kwargs
        )

    def create_async_client(
        self, repl_args: Any, backend_args: Mapping[str, Any]
    ):
        raise NotImplementedError(
            "Rocq REPL backend currently supports sync verification only."
        )


REPL_BACKENDS: Dict[str, ReplBackendBase] = {
    KiminaReplBackend.name: KiminaReplBackend(),
    RocqReplBackend.name: RocqReplBackend(),
}


def get_repl_backend(name: str) -> ReplBackendBase:
    try:
        return REPL_BACKENDS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown REPL backend '{name}'. Available: {list(REPL_BACKENDS)}"
        ) from exc
