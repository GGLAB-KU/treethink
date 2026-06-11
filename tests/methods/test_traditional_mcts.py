"""Tests for TraditionalMCTS with rollout and REPL evaluation."""

import sys
import tempfile
import unittest

from loguru import logger

from tests.common import SimpleChildPolicy
from treethink.graph import save_tree_to_txt
from treethink.methods import Node, TraditionalMCTS
from treethink.utils.enums import FinalDecisionMode


def _int_evaluator(nodes, method):
    """Simple evaluator returning sequential integer scores."""
    if isinstance(nodes, Node):
        return [0]
    return [i + 1 for i in range(len(nodes))]


class TestTraditionalMCTS(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".dot"
        )
        self.temp_file_path = self.temp_file.name

    def test_basic_simulate_fallback(self):
        """When no rollout support, falls back to direct child evaluation."""
        root = Node("root")
        mcts = TraditionalMCTS(
            root_node=None,
            policy=SimpleChildPolicy(),
            evaluator=_int_evaluator,
            exploration_weight=0.0,
        )
        mcts.set_root_node(root)
        mcts.simulate(expansion_count=3)
        self.assertIsNotNone(mcts.best_answer)
        self.assertGreater(len(root.children), 0)

    def test_children_receive_win_values(self):
        """Children should receive win_values after expansion + evaluation."""
        root = Node("root")
        mcts = TraditionalMCTS(
            root_node=None,
            policy=SimpleChildPolicy(),
            evaluator=_int_evaluator,
            exploration_weight=0.0,
        )
        mcts.set_root_node(root)
        mcts.simulate(expansion_count=1)

        for child in root.children:
            self.assertIsNotNone(child.win_value)

    def test_make_choice_returns_root_when_no_children(self):
        root = Node("root")
        mcts = TraditionalMCTS(
            root_node=root,
            policy=SimpleChildPolicy(),
            evaluator=_int_evaluator,
            exploration_weight=0.0,
        )
        self.assertEqual(mcts.make_choice(), root)

    def test_backpropagation_updates_ancestors(self):
        root = Node("root")
        child = Node("child", parent=root)
        root.add_child(child)

        mcts = TraditionalMCTS(
            root_node=root,
            policy=SimpleChildPolicy(),
            evaluator=_int_evaluator,
            exploration_weight=0.0,
        )

        child.win_value = 3.0
        mcts._backpropagate_node_win_value(child)

        self.assertEqual(root.win_value, 3.0)
        self.assertEqual(root.visits, 1)
        self.assertEqual(child.visits, 0)  # unchanged

    def test_final_decision_mode_maximize_visits(self):
        root = Node("root")
        mcts = TraditionalMCTS(
            root_node=None,
            policy=SimpleChildPolicy(),
            evaluator=_int_evaluator,
            final_decision_mode=FinalDecisionMode.MAXIMIZE_VISITS,
        )
        mcts.set_root_node(root)
        mcts.simulate(expansion_count=5)
        self.assertIsNotNone(mcts.best_answer)

        save_tree_to_txt(
            root_node=mcts.root_node,
            output_path="tests/outputs/traditional_mcts_maximize_visits.txt",
            selected_solution=mcts.best_answer,
        )

    def test_final_decision_mode_maximize_value(self):
        root = Node("root")
        mcts = TraditionalMCTS(
            root_node=None,
            policy=SimpleChildPolicy(),
            evaluator=_int_evaluator,
            final_decision_mode=FinalDecisionMode.MAXIMIZE_VALUE,
        )
        mcts.set_root_node(root)
        mcts.simulate(expansion_count=5)
        self.assertIsNotNone(mcts.best_answer)

        save_tree_to_txt(
            root_node=mcts.root_node,
            output_path="tests/outputs/traditional_mcts_maximize_value.txt",
            selected_solution=mcts.best_answer,
        )

    def test_remove_duplicate_children(self):
        root = Node("root")

        def policy(node, method):
            node.add_child(Node("DUP", parent=node))
            node.add_child(Node("DUP", parent=node))
            node.add_child(Node("UNIQ", parent=node))
            node._uniq_ref = node.children[-1]

        def evaluator(x, method):
            if isinstance(x, Node):
                return [0]
            return [5, 3]

        mcts = TraditionalMCTS(
            root_node=None,
            policy=policy,
            evaluator=evaluator,
        )
        mcts.set_root_node(root)
        mcts.simulate(expansion_count=1, remove_duplicate_children=True)

        # After dedup, 2 unique children remain
        self.assertEqual(len(root.children), 2)
        win_values = {c.win_value for c in root.children}
        self.assertEqual(win_values, {5, 3})

    def test_str_repr(self):
        mcts = TraditionalMCTS(
            root_node=Node("root"),
            policy=SimpleChildPolicy(),
            evaluator=_int_evaluator,
            exploration_weight=1.414,
        )
        s = str(mcts)
        self.assertIn("TraditionalMCTS", s)
        self.assertIn("exploration_weight", s)
        self.assertIn("rollout_max_tokens", s)

    def test_rollout_config_stored(self):
        """Verify rollout config params are stored when provided."""

        def stub_eval(nodes, method):
            return [1.0 for _ in nodes]

        mcts = TraditionalMCTS(
            root_node=Node("root"),
            policy=SimpleChildPolicy(),
            evaluator=_int_evaluator,
            rollout_evaluator=stub_eval,
            rollout_n=3,
            rollout_max_tokens=2048,
            rollout_temperature=0.5,
            rollout_top_p=0.9,
        )
        self.assertEqual(mcts.rollout_n, 3)
        self.assertEqual(mcts.rollout_max_tokens, 2048)
        self.assertEqual(mcts.rollout_temperature, 0.5)
        self.assertEqual(mcts.rollout_top_p, 0.9)
        self.assertIsNotNone(mcts._rollout_evaluator)


if __name__ == "__main__":
    logger.remove(0)
    logger.add(sys.stderr, level="TRACE")
    unittest.main()
