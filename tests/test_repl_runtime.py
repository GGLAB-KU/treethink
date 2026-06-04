"""Tests for the refactored ReplRuntime (language-agnostic client creation)."""

import unittest

from treethink import (
    ClientArgs,
    TerminationOnEncounterConfig,
    TerminationOnPathsConfig,
    TreeThinkArgs,
)
from treethink.utils.enums import FormalLanguage


class TestReplRuntime(unittest.TestCase):
    def test_runtime_stays_disabled_without_termination_marker(self):
        args = TreeThinkArgs(
            client_args=ClientArgs(lean_server_url="http://localhost:8000"),
            termination_on_encounter=TerminationOnEncounterConfig(enabled=True),
        )

        runtime = args.build_repl_runtime()

        self.assertFalse(runtime.encountered_config.enabled)
        self.assertFalse(runtime.paths_config.enabled)

    def test_runtime_enabled_with_marker(self):
        args = TreeThinkArgs(
            termination_str="```",
            client_args=ClientArgs(lean_server_url="http://localhost:8000"),
        )

        runtime = args.build_repl_runtime()

        self.assertTrue(runtime.encountered_config.enabled)
        self.assertTrue(runtime.paths_config.enabled)

    def test_runtime_uses_separate_configs(self):
        args = TreeThinkArgs(
            termination_str="```",
            client_args=ClientArgs(lean_server_url="http://encountered"),
            termination_on_encounter=TerminationOnEncounterConfig(enabled=True),
            termination_on_paths=TerminationOnPathsConfig(
                enabled=True, max_repl=3
            ),
        )

        runtime = args.build_repl_runtime()

        self.assertTrue(runtime.encountered_config.enabled)
        self.assertTrue(runtime.paths_config.enabled)
        self.assertEqual(runtime.paths_config.max_repl, 3)
        self.assertEqual(
            runtime.client_args.lean_server_url, "http://encountered"
        )

    def test_rocq_language_client_created(self):
        args = TreeThinkArgs(
            termination_str="```",
            language=FormalLanguage.RCOQ,
            client_args=ClientArgs(
                host="127.0.0.1", port=5000, workspace_dir="/tmp/test"
            ),
        )

        runtime = args.build_repl_runtime()

        self.assertEqual(runtime.language, FormalLanguage.RCOQ)
        self.assertTrue(runtime.needs_repl)

    def test_needs_repl_false_when_all_disabled(self):
        args = TreeThinkArgs(
            termination_str="```",
            termination_on_encounter=TerminationOnEncounterConfig(
                enabled=False
            ),
            termination_on_paths=TerminationOnPathsConfig(enabled=False),
        )

        runtime = args.build_repl_runtime()
        self.assertFalse(runtime.needs_repl)


if __name__ == "__main__":
    unittest.main()
