import sys
import tempfile
import unittest

from loguru import logger

from tests.common import RandomNodeEvaluator, SimpleChildExpander
from treethink.graph import (  # noqa
    extract_solution_from_graphviz,
    save_tree_to_txt,
)
from treethink.methods import MCTS, Node  # noqa


class TestMCTS(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".dot"
        )
        self.temp_file_path = self.temp_file.name
        self._global_child_counter = 0

    def test_exploration_zero(self):
        """Test when exploration_weight is set to 0."""

        root = Node("root")
        evaluator_func = RandomNodeEvaluator()
        expander_func = SimpleChildExpander()

        mcts = MCTS(
            root_node=None,
            expander=expander_func,
            evaluator=evaluator_func,
            exploration_weight=0.0,
        )
        mcts.set_root_node(root)
        mcts.simulate(expansion_count=3)
        solution = mcts.best_answer
        logger.debug(f"Solution: {solution}")

        save_tree_to_txt(
            root_node=mcts.root_node,
            output_path="tests/outputs/mcts_zero_exploration.txt",
            selected_solution=solution,
        )

    def test_final_decision_mode_maximize_visits(self):
        """Test final_decision_mode = maximize_visits"""

        root = Node("root")
        evaluator_func = RandomNodeEvaluator()
        expander_func = SimpleChildExpander()

        mcts = MCTS(
            root_node=None,
            expander=expander_func,
            evaluator=evaluator_func,
            final_decision_mode="maximize_visits",
        )
        mcts.set_root_node(root)
        mcts.simulate(expansion_count=5)
        solution = mcts.best_answer
        logger.debug(f"Solution: {solution}")

        save_tree_to_txt(
            root_node=mcts.root_node,
            output_path="tests/outputs/mcts_maximize_visits.txt",
            selected_solution=solution,
        )

    def test_final_decision_mode_maximize_value(self):
        """Test final_decision_mode = maximize_value"""

        root = Node("root")
        evaluator_func = RandomNodeEvaluator()
        expander_func = SimpleChildExpander()

        mcts = MCTS(
            root_node=None,
            expander=expander_func,
            evaluator=evaluator_func,
            final_decision_mode="maximize_value",
        )
        mcts.set_root_node(root)
        mcts.simulate(expansion_count=5)
        solution = mcts.best_answer
        logger.debug(f"Solution: {solution}")

        save_tree_to_txt(
            root_node=mcts.root_node,
            output_path="tests/outputs/mcts_maximize_value.txt",
            selected_solution=solution,
        )

    def test_simulate_remove_duplicate_children(self):
        """Test simulate with remove_duplicate_children=True uses deduped
        children for evaluation and multiplies win_value by duplicate count."""

        root = Node("root")

        # Create explicit children with duplicates so we can reference them
        def expander(node, method):
            # First duplicate
            dup1 = Node(text="DUP", parent=node)
            node.add_child(dup1)
            # Second duplicate (same text)
            dup2 = Node(text="DUP", parent=node)
            node.add_child(dup2)
            # Third duplicate (same text)
            dup3 = Node(text="DUP", parent=node)
            node.add_child(dup3)
            # Unique child
            uniq = Node(text="UNIQ", parent=node)
            node.add_child(uniq)

            # A child that itself has a grandchild (to check propagation)
            parent_with_child = Node(text="PARENT", parent=node)
            node.add_child(parent_with_child)
            gc = Node(text="GC", parent=parent_with_child)
            parent_with_child.add_child(gc)

            # Keep references on the parent for assertions
            node._test_refs = {
                "dup1": dup1,
                "dup2": dup2,
                "dup3": dup3,
                "uniq": uniq,
                "parent": parent_with_child,
                "gc": gc,
            }

        def evaluator(x, method):
            # Called once for root during set_root_node
            from treethink.methods.node import Node as _Node

            if isinstance(x, _Node):
                return [0]

            # x is the deduplicated list passed from expand
            # return win values in the same order (DUP, UNIQ, PARENT)
            return [5, 3, 2]

        mcts = MCTS(
            root_node=None,
            expander=expander,
            evaluator=evaluator,
            final_decision_mode="maximize_value",
        )
        mcts.set_root_node(root)

        # Run a single expansion with duplicate removal enabled
        mcts.simulate(expansion_count=1, remove_duplicate_children=True)

        dup1 = root._test_refs["dup1"]
        dup2 = root._test_refs["dup2"]
        dup3 = root._test_refs["dup3"]
        uniq = root._test_refs["uniq"]
        parent_with_child = root._test_refs["parent"]
        gc = root._test_refs["gc"]

        # After evaluation: DUP had 3 duplicates -> win_value should be 5 * 3 = 15
        self.assertEqual(dup1.win_value, 15)
        # The other duplicates weren't the evaluated representative, so they should stay unchanged
        self.assertEqual(dup2.win_value, 0)
        self.assertEqual(dup3.win_value, 0)
        # Unique child should have received its own score
        self.assertEqual(uniq.win_value, 3)
        # Parent node (with a grandchild) should have received its own score
        self.assertEqual(parent_with_child.win_value, 2)
        # Grandchild shouldn't have been evaluated directly
        self.assertEqual(gc.win_value, 0)

        # Root should have received all aggregated updates: 15 + 3 + 2 = 20
        self.assertEqual(root.win_value, 15 + 3 + 2)

        solution = mcts.best_answer

        save_tree_to_txt(
            root_node=mcts.root_node,
            output_path="tests/outputs/mcts_remove_duplicates.txt",
            selected_solution=solution,
        )


if __name__ == "__main__":
    logger.remove(0)
    logger.add(sys.stderr, level="TRACE")
    unittest.main()
