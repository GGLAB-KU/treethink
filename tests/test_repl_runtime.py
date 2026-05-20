import unittest
from unittest.mock import patch

from treethink import LeanREPLArgs, ReplStrategyArgs, TreeThinkArgs
from treethink import repl_runtime as repl_runtime_module


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


if __name__ == "__main__":
    unittest.main()
