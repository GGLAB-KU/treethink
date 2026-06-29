"""Unit tests for ``IsabelleClient.extract_proof_state`` parsing.

These do NOT need a running Isabelle server: ``extract_proof_state`` is
called with an explicit ``check()``-style response dict, so the client is
constructed via ``__new__`` to skip the server-spawning ``__init__``.
"""

import unittest

from treethink.clients.base import ProofStateInfo
from treethink.clients.isabelle.client import IsabelleClient


def _client() -> IsabelleClient:
    # Bypass __init__ (which starts an Isabelle server); the methods under
    # test only touch the passed response + proof string.
    return IsabelleClient.__new__(IsabelleClient)


class TestExtractProofState(unittest.TestCase):
    def test_failure_with_goal_block(self):
        proof = 'lemma demo: "rev (rev xs) = xs"\n  apply (rule refl)'
        response = {
            "ok": False,
            "errors": [
                "Failed to apply proof method\\<^here>:\n"
                "goal (1 subgoal):\n 1. rev (rev xs) = xs"
            ],
        }
        info = _client().extract_proof_state(proof, response=response)

        self.assertIsInstance(info, ProofStateInfo)
        self.assertEqual(info.applied_tactic, "apply (rule refl)")
        self.assertEqual(info.error_message, "Failed to apply proof method")
        self.assertIn("goal (1 subgoal)", info.open_goals)
        self.assertIn("rev (rev xs) = xs", info.open_goals)
        self.assertNotIn("\\<^here>", info.open_goals)
        self.assertIsNone(info.closed_goals)

    def test_success_no_errors(self):
        proof = 'lemma demo: "(1::nat) + 1 = 2"\n  by simp'
        info = _client().extract_proof_state(
            proof, response={"ok": True, "errors": []}
        )
        self.assertEqual(info.applied_tactic, "by simp")
        self.assertEqual(info.open_goals, "")
        self.assertIsNone(info.error_message)

    def test_error_without_goal_block(self):
        proof = 'lemma demo: "True"\n  apply simp'
        response = {
            "ok": False,
            "errors": [
                'Bad context for command "end"\\<^here> -- using reset state'
            ],
        }
        info = _client().extract_proof_state(proof, response=response)
        self.assertIsNone(info.open_goals)
        self.assertIn("Bad context", info.error_message)
        self.assertNotIn("\\<^here>", info.error_message)

    def test_applied_tactic_is_last_nonempty_line(self):
        proof = 'lemma x: "P"\n  apply foo\n  apply bar\n\n'
        info = _client().extract_proof_state(
            proof, response={"ok": True, "errors": []}
        )
        self.assertEqual(info.applied_tactic, "apply bar")

    def test_multiple_errors_first_goal_used(self):
        response = {
            "ok": False,
            "errors": [
                "Failed to apply proof method\\<^here>:\n"
                "goal (1 subgoal):\n 1. A",
                "Failed to finish proof\\<^here>:\ngoal (1 subgoal):\n 1. A",
            ],
        }
        info = _client().extract_proof_state(
            'lemma x: "A"\n  apply blast', response=response
        )
        self.assertIn("1. A", info.open_goals)
        self.assertIn("Failed to apply proof method", info.error_message)


if __name__ == "__main__":
    unittest.main()
