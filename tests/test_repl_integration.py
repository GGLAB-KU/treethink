import os
import tempfile
import unittest

import pytest
from loguru import logger

from tests.common import (
    FirstPosOthersNegEvaluator,
    PreferTerminationChildPolicy,
    SetStrPolicy,
)
from treethink import (  # noqa
    ClientArgs,
    TerminationOnPathsConfig,
    TreeThink,
    TreeThinkArgs,
)
from treethink.graph import (  # noqa
    extract_solution_from_graphviz,
    save_tree_to_txt,
)
from treethink.methods import BFTS, AlphaZeroMCTS, Node  # noqa
from treethink.utils.enums import FinalDecisionMode, FormalLanguage

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LEAN_TESTS", "0") == "0",
    reason="RUN_LEAN_TESTS not set. Lean tests require a running kimina-lean-server.",
)


class TestREPLIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".dot"
        )
        self.temp_file_path = self.temp_file.name
        self._global_child_counter = 0

        self.treethink_args = TreeThinkArgs(
            method_name="BFTS",
            max_children=5,
            expansion_count=10,
            timeout=60,
            graph_path=None,
            termination_str="```\n",
            language=FormalLanguage.LEAN4,
            store_method_class=False,
            store_graph_stats=True,
            remove_duplicate_children=True,
            client_args=ClientArgs(),
            beam_width=2,
            exploration_weight=1.414213,
            final_decision_mode=FinalDecisionMode.CLEAR_FRONTIER,
        )

    def test_repl_terminated_paths(self):
        """Test of REPL checking finished proof trajectories."""

        # Testing repl_terminated_paths
        self.treethink_args.termination_on_paths.enabled = True
        self.treethink_args.graph_path = (
            "tests/outputs/repl_terminated_paths.txt"
        )

        # Sample proof from DeepSeekProverV2 on minif2f
        proof_begin = "Complete the following lean code:\n```\nimport Mathlib\nimport Aesop\n\n\nopen BigOperators\nopen Real\nopen Nat\nopen Topology\ntheorem mathd_algebra_478\n  (b h v : \u211d)\n  (h\u2080 : 0 < b \u2227 0 < h \u2227 0 < v)\n  (h\u2081 : v = 1 / 3 * (b * h))\n  (h\u2082 : b = 30)\n  (h\u2083 : h = 13 / 2) :\n  v = 65 := by\n"
        proof_cont = "  rw [h\u2081]\n  norm_num [h\u2082, h\u2083]\n  <;> ring\n  <;> norm_num\n  <;> linarith\n```\n"

        method = AlphaZeroMCTS(
            root_node=Node(
                "root", termination_str=self.treethink_args.termination_str
            ),
            policy=SetStrPolicy(text=proof_cont, num_child=5),
            evaluator=FirstPosOthersNegEvaluator(),
        )
        treethink = TreeThink(method=method, treethink_args=self.treethink_args)

        output = treethink.generate(proof_begin)
        logger.debug(f"treethink output: {output}")
        self.assertTrue(output.checked_and_true)

    def test_repl_encountered_termination(self):
        """Test of REPL checking encountered termination trajectories."""
        self.treethink_args.termination_on_encounter.enabled = True
        self.treethink_args.graph_path = (
            "tests/outputs/repl_encountered_termination.txt"
        )

        # Sample proof from DeepSeekProverV2 on minif2f
        proof_begin = "Complete the following lean code:\n```\nimport Mathlib\nimport Aesop\n\n\nopen BigOperators\nopen Real\nopen Nat\nopen Topology\ntheorem mathd_algebra_478\n  (b h v : \u211d)\n  (h\u2080 : 0 < b \u2227 0 < h \u2227 0 < v)\n  (h\u2081 : v = 1 / 3 * (b * h))\n  (h\u2082 : b = 30)\n  (h\u2083 : h = 13 / 2) :\n  v = 65 := by\n"
        proof_cont = "  rw [h\u2081]\n  norm_num [h\u2082, h\u2083]\n  <;> ring\n  <;> norm_num\n  <;> linarith\n```\n"  # noqa: F841

        method = BFTS(
            root_node=Node("root"),
            policy=PreferTerminationChildPolicy(),
            evaluator=FirstPosOthersNegEvaluator(),
        )
        treethink = TreeThink(method=method, treethink_args=self.treethink_args)
        output = treethink.generate(proof_begin)
        logger.debug(f"treethink output: {output}")
        self.assertFalse(output.checked_and_true)
