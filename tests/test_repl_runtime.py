import unittest
from unittest.mock import patch

from treethink import LeanREPLArgs, ReplStrategyArgs, TreeThinkArgs
from treethink import repl_runtime as repl_runtime_module
from treethink.repl_backends import REPL_BACKENDS, ReplBackendBase


def dummy_sync_function(*, method, client, timeout, num_proc, batch_size):
    return "dummy-sync"


async def dummy_async_function(
    *, method, client, timeout, num_proc, batch_size
):
    return "dummy-async"


class DummyBackend(ReplBackendBase):
    name = "dummy"

    def resolve_sync_function(self, function_name: str):
        return dummy_sync_function

    def resolve_async_function(self, function_name: str):
        return dummy_async_function

    def create_sync_client(self, repl_args, backend_args):
        return {"kind": "sync-client", "url": repl_args.lean_server_url}

    def create_async_client(self, repl_args, backend_args):
        return {"kind": "async-client", "url": repl_args.lean_server_url}


class TestReplRuntime(unittest.TestCase):
    def test_runtime_stays_disabled_without_termination_marker(self):
        args = TreeThinkArgs(
            repl_args=LeanREPLArgs(lean_server_url="http://localhost:8000"),
            repl_encountered_termination=True,
        )

        runtime = args.build_repl_runtime()

        self.assertFalse(runtime.encountered_termination.enabled)
        self.assertFalse(runtime.terminated_paths.enabled)

    def test_build_runtime_from_legacy_flags(self):
        args = TreeThinkArgs(
            termination_str="```",
            repl_args=LeanREPLArgs(lean_server_url="http://localhost:8000"),
            repl_terminated_paths=False,
            repl_encountered_termination=True,
        )

        runtime = args.build_repl_runtime()

        self.assertTrue(runtime.encountered_termination.enabled)
        self.assertFalse(runtime.terminated_paths.enabled)

    def test_runtime_uses_separate_strategy_configs(self):
        args = TreeThinkArgs(
            termination_str="```",
            repl_args=None,
            repl_terminated_paths=False,
            repl_encountered_termination=False,
            repl_encountered_termination_args=ReplStrategyArgs(
                enabled=True,
                repl_args=LeanREPLArgs(lean_server_url="http://encountered"),
                sync_fn_name="repl_encountered_termination",
                async_fn_name="async_repl_encountered_termination",
            ),
            repl_terminated_paths_args=ReplStrategyArgs(
                enabled=True,
                repl_args=LeanREPLArgs(lean_server_url="http://paths"),
                max_repl=3,
                sync_fn_name="repl_terminated_paths",
                async_fn_name="async_repl_terminated_paths",
            ),
        )

        runtime = args.build_repl_runtime()

        self.assertTrue(runtime.encountered_termination.enabled)
        self.assertTrue(runtime.terminated_paths.enabled)
        self.assertEqual(
            runtime.encountered_termination.config.repl_args.lean_server_url,
            "http://encountered",
        )
        self.assertEqual(
            runtime.terminated_paths.config.repl_args.lean_server_url,
            "http://paths",
        )

    def test_runtime_builds_hooks_from_registry(self):
        args = TreeThinkArgs(
            termination_str="```",
            repl_args=LeanREPLArgs(lean_server_url="http://localhost:8000"),
            repl_terminated_paths=False,
            repl_encountered_termination_args=ReplStrategyArgs(
                enabled=True,
                repl_args=LeanREPLArgs(lean_server_url="http://localhost:8001"),
                sync_fn_name="repl_encountered_termination",
                async_fn_name="async_repl_encountered_termination",
            ),
        )

        runtime = args.build_repl_runtime()

        with patch.object(repl_runtime_module, "KiminaClient") as mock_client:
            callback = runtime.build_termination_callback(method=object())

        self.assertEqual(callback.func.__name__, "repl_encountered_termination")
        mock_client.assert_called_once_with("http://localhost:8001")

    def test_runtime_uses_registered_backend(self):
        args = TreeThinkArgs(
            termination_str="```",
            repl_args=LeanREPLArgs(lean_server_url="http://dummy"),
            repl_encountered_termination_args=ReplStrategyArgs(
                enabled=True,
                backend_name="dummy",
                repl_args=LeanREPLArgs(lean_server_url="http://dummy"),
                sync_fn_name="anything",
                async_fn_name="anything_async",
            ),
        )

        with patch.dict(REPL_BACKENDS, {"dummy": DummyBackend()}):
            runtime = args.build_repl_runtime()
            callback = runtime.build_termination_callback(method=object())

        self.assertEqual(callback.func, dummy_sync_function)
        self.assertEqual(callback.keywords["client"]["kind"], "sync-client")


if __name__ == "__main__":
    unittest.main()
