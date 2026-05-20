from dataclasses import dataclass
from functools import partial
from typing import Optional

from loguru import logger

from .repl_backends import ReplBackendBase, get_repl_backend
from .utils.args import LeanREPLArgs, ReplStrategyArgs


@dataclass
class ReplHookRuntime:
    mode: str
    config: ReplStrategyArgs
    backend: Optional[ReplBackendBase] = None
    client: Optional[object] = None
    async_client: Optional[object] = None

    def __post_init__(self):
        if self.backend is None:
            self.backend = get_repl_backend(self.config.backend_name)

    @property
    def enabled(self) -> bool:
        return bool(self.config and self.config.enabled)

    def _function_name(self, async_mode: bool) -> str:
        return (
            self.config.async_fn_name
            if async_mode
            else self.config.sync_fn_name
        )

    def _resolve_function(self, async_mode: bool):
        function_name = self._function_name(async_mode)
        if async_mode:
            return self.backend.resolve_async_function(function_name)
        return self.backend.resolve_sync_function(function_name)

    def _ensure_client(self, async_mode: bool):
        repl_args = self.config.repl_args
        if repl_args is None:
            raise ValueError(
                f"REPL mode '{self.mode}' is enabled but has no repl_args configured."
            )

        if async_mode:
            if self.async_client is None:
                self.async_client = self.backend.create_async_client(
                    repl_args,
                    self.config.backend_args,
                )
            return self.async_client

        if self.client is None:
            self.client = self.backend.create_sync_client(
                repl_args,
                self.config.backend_args,
            )
        return self.client

    def _build_kwargs(self, method, async_mode: bool) -> dict:
        repl_args = self.config.repl_args
        return self.backend.build_call_kwargs(
            mode=self.mode,
            method=method,
            repl_args=repl_args,
            strategy_args=self.config,
            backend_args=self.config.backend_args,
            client=self._ensure_client(async_mode),
        )

    def build_partial(self, method, async_mode: bool = False):
        if not self.enabled:
            return None

        return partial(
            self._resolve_function(async_mode),
            **self._build_kwargs(method, async_mode),
        )

    def run(self, method, async_mode: bool = False):
        partial_fn = self.build_partial(method, async_mode=async_mode)
        if partial_fn is None:
            return None
        return partial_fn()


@dataclass
class ReplRuntime:
    encountered_termination: ReplHookRuntime
    terminated_paths: ReplHookRuntime

    @classmethod
    def from_treethink_args(
        cls, treethink_args, termination_str: Optional[str] = None
    ):
        shared_repl_args = treethink_args.repl_args
        repl_enabled = bool(termination_str)

        encountered_config = cls._normalize_config(
            config=treethink_args.repl_encountered_termination_args,
            legacy_enabled=treethink_args.repl_encountered_termination
            and repl_enabled,
            fallback_repl_args=shared_repl_args,
            fallback_max_repl=treethink_args.max_repl,
            default_sync_fn_name="repl_encountered_termination",
            default_async_fn_name="async_repl_encountered_termination",
            repl_enabled=repl_enabled,
        )
        terminated_paths_config = cls._normalize_config(
            config=treethink_args.repl_terminated_paths_args,
            legacy_enabled=treethink_args.repl_terminated_paths
            and repl_enabled,
            fallback_repl_args=shared_repl_args,
            fallback_max_repl=treethink_args.max_repl,
            default_sync_fn_name="repl_terminated_paths",
            default_async_fn_name="async_repl_terminated_paths",
            repl_enabled=repl_enabled,
        )

        logger.debug(
            "Built REPL runtime: encountered={}, terminated_paths={}",
            encountered_config,
            terminated_paths_config,
        )

        return cls(
            encountered_termination=ReplHookRuntime(
                mode="encountered",
                config=encountered_config,
            ),
            terminated_paths=ReplHookRuntime(
                mode="terminated_paths",
                config=terminated_paths_config,
            ),
        )

    @staticmethod
    def _normalize_config(
        config: Optional[ReplStrategyArgs],
        legacy_enabled: bool,
        fallback_repl_args: Optional[LeanREPLArgs],
        fallback_max_repl: int,
        default_sync_fn_name: str,
        default_async_fn_name: str,
        repl_enabled: bool,
    ) -> ReplStrategyArgs:
        if config is None:
            config = ReplStrategyArgs(
                enabled=legacy_enabled and repl_enabled,
                backend_name="kimina",
                backend_args={},
                repl_args=fallback_repl_args,
                max_repl=fallback_max_repl,
                sync_fn_name=default_sync_fn_name,
                async_fn_name=default_async_fn_name,
            )
        else:
            config = ReplStrategyArgs(
                enabled=(config.enabled or legacy_enabled) and repl_enabled,
                backend_name=config.backend_name or "kimina",
                backend_args=config.backend_args or {},
                repl_args=config.repl_args or fallback_repl_args,
                max_repl=config.max_repl
                if config.max_repl is not None
                else fallback_max_repl,
                sync_fn_name=config.sync_fn_name or default_sync_fn_name,
                async_fn_name=config.async_fn_name or default_async_fn_name,
            )

        if config.enabled and config.repl_args is None:
            raise ValueError(
                f"REPL strategy '{default_sync_fn_name}' is enabled but no repl_args were provided."
            )

        return config

    @property
    def needs_repl(self) -> bool:
        return (
            self.encountered_termination.enabled
            or self.terminated_paths.enabled
        )

    def build_termination_callback(self, method, async_mode: bool = False):
        return self.encountered_termination.build_partial(
            method,
            async_mode=async_mode,
        )

    def check_terminated_paths(self, method):
        return self.terminated_paths.run(method, async_mode=False)

    async def async_check_terminated_paths(self, method):
        partial_fn = self.terminated_paths.build_partial(
            method, async_mode=True
        )
        if partial_fn is None:
            return None
        return await partial_fn()


__all__ = ["ReplRuntime", "ReplHookRuntime"]
