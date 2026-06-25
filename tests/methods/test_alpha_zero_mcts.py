import sys
import tempfile
import unittest

from loguru import logger

from tests.common import RandomNodeEvaluator, SimpleChildPolicy
from treethink.graph import (  # noqa
    extract_solution_from_graphviz,
    save_tree_to_txt,
)
from treethink.methods import AlphaZeroMCTS, Node  # noqa
from treethink.utils.enums import BestAnswerReason, FinalDecisionMode

logger.remove(0)
logger.add(sys.stderr, level="TRACE")


class TestAlphaZeroMCTS(unittest.TestCase):
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
        policy_func = SimpleChildPolicy()

        mcts = AlphaZeroMCTS(
            root_node=None,
            policy=policy_func,
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
        policy_func = SimpleChildPolicy()

        mcts = AlphaZeroMCTS(
            root_node=None,
            policy=policy_func,
            evaluator=evaluator_func,
            final_decision_mode=FinalDecisionMode.MAXIMIZE_VISITS,
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
        policy_func = SimpleChildPolicy()

        mcts = AlphaZeroMCTS(
            root_node=None,
            policy=policy_func,
            evaluator=evaluator_func,
            final_decision_mode=FinalDecisionMode.MAXIMIZE_VALUE,
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
        def policy(node, method):
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

        mcts = AlphaZeroMCTS(
            root_node=None,
            policy=policy,
            evaluator=evaluator,
            final_decision_mode=FinalDecisionMode.MAXIMIZE_VALUE,
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

        # After evaluation: DUP had 3 duplicates, remove two of them, preserve the win_value
        self.assertEqual(dup1.win_value, 5)
        # Unique child should have received its own score
        self.assertEqual(uniq.win_value, 3)
        # Parent node (with a grandchild) should have received its own score
        self.assertEqual(parent_with_child.win_value, 2)
        # Grandchild shouldn't have been evaluated directly
        self.assertEqual(gc.win_value, 0)

        # No backpropagation because MCTS starts evaluating from parent
        self.assertEqual(root.win_value, 0)

        solution = mcts.best_answer

        save_tree_to_txt(
            root_node=mcts.root_node,
            output_path="tests/outputs/mcts_remove_duplicates.txt",
            selected_solution=solution,
        )

    # ──────────────────────────────────────────────
    # Termination-related tests
    # ──────────────────────────────────────────────

    def test_simulate_termination_valid(self):
        """Simulate stops when a valid termination node is found.

        The policy creates one termination node (``text == termination_str``)
        and one regular child.  When the search selects the termination node,
        ``termination_encountered_fn`` returns a proof → search breaks
        immediately.
        """
        root = Node("root", termination_str="TERM")

        def policy(node, method):
            term = Node(
                text="TERM", parent=node, termination_str=node.termination_str
            )
            node.add_child(term)
            regular = Node(
                text="REG", parent=node, termination_str=node.termination_str
            )
            node.add_child(regular)

        def evaluator(x, method):
            return [1.0, 0.5]

        def termination_encountered_fn(node):
            return "rootTERM"

        mcts = AlphaZeroMCTS(
            root_node=None,
            policy=policy,
            evaluator=evaluator,
        )
        mcts.set_root_node(root)
        mcts.simulate(
            expansion_count=10,
            termination_encountered_fn=termination_encountered_fn,
        )

        self.assertEqual(mcts.best_answer, "rootTERM")
        self.assertEqual(
            mcts.best_answer_reason, BestAnswerReason.CHECKED_AND_TRUE
        )

        save_tree_to_txt(
            root_node=mcts.root_node,
            output_path="tests/outputs/mcts_termination_valid.txt",
            selected_solution=mcts.best_answer,
        )

    def test_simulate_termination_invalid(self):
        """Simulate continues when a termination node turns out wrong.

        ``termination_encountered_fn`` returns ``None`` → the node's
        ``win_value`` is set to ``-inf`` and the search proceeds to explore
        other branches.
        """
        root = Node("root", termination_str="WRONG")

        def policy(node, method):
            # Root level: one termination node + one regular child
            wrong = Node(
                text="WRONG",
                parent=node,
                termination_str=node.termination_str,
            )
            node.add_child(wrong)
            regular = Node(
                text="REG", parent=node, termination_str=node.termination_str
            )
            node.add_child(regular)
            # Deeper expansions: only regular children (no termination_str)
            # so we don't keep generating more termination nodes.
            deeper = Node(text="DEEPER", parent=node)
            node.add_child(deeper)

        def evaluator(x, method):
            return [0.5, 0.3, 0.1]

        call_count = 0

        def termination_encountered_fn(node):
            nonlocal call_count
            call_count += 1
            return None  # always invalid

        mcts = AlphaZeroMCTS(
            root_node=None,
            policy=policy,
            evaluator=evaluator,
        )
        mcts.set_root_node(root)
        mcts.simulate(
            expansion_count=5,
            termination_encountered_fn=termination_encountered_fn,
        )

        # The WRONG node should be marked with -inf to avoid reselection
        wrong_node = root.children[0]
        self.assertEqual(wrong_node.win_value, float("-inf"))
        self.assertGreater(wrong_node.visits, 0)

        # termination_encountered_fn was called at least once
        self.assertGreaterEqual(call_count, 1)

        # The search continued past the wrong termination —
        # best_answer was computed normally, not set by the callback
        self.assertIsNotNone(mcts.best_answer)
        self.assertEqual(mcts.best_answer_reason, BestAnswerReason.CALCULATED)

        save_tree_to_txt(
            root_node=mcts.root_node,
            output_path="tests/outputs/mcts_termination_invalid.txt",
            selected_solution=mcts.best_answer,
        )

    def test_simulate_termination_mixed(self):
        """First termination is invalid, second is valid — search recovers
        and stops at the correct one.

        The tree has an intermediate non-termination level so that
        ``termination_encountered_fn`` can demonstrate constructing the
        real proof from the tree (via ``traverse_to_root``) rather than
        returning a hard-coded placeholder.

        Tree structure::

            Root
            ├── A          (intermediate, higher score → selected first)
            │   ├── TERM   (invalid → rejected)
            │   └── TERM   (valid   → accepted, proof = "RootATERM")
            └── B          (never reached)
        """
        root = Node("Root", termination_str="TERM")

        def policy(node, method):
            if node is root:
                # Two intermediate branches — A gets the higher score
                # so UCT selects it first.
                path_a = Node(
                    "A", parent=node, termination_str=node.termination_str
                )
                node.add_child(path_a)
                path_b = Node(
                    "B", parent=node, termination_str=node.termination_str
                )
                node.add_child(path_b)
            else:
                # Deeper level: two termination children under the
                # selected intermediate node.
                wrong = Node(
                    text="TERM",
                    parent=node,
                    termination_str=node.termination_str,
                )
                node.add_child(wrong)
                valid = Node(
                    text="TERM",
                    parent=node,
                    termination_str=node.termination_str,
                )
                node.add_child(valid)

        def evaluator(x, method):
            # Higher score for the first child so UCT deterministically
            # picks the invalid termination before the valid one.
            return [0.9, 0.1]

        call_count = 0

        def termination_encountered_fn(node):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # first termination is invalid
            # Construct the real proof from the tree structure.
            return mcts.traverse_to_root(node)

        mcts = AlphaZeroMCTS(
            root_node=None,
            policy=policy,
            evaluator=evaluator,
        )
        mcts.set_root_node(root)
        mcts.simulate(
            expansion_count=10,
            termination_encountered_fn=termination_encountered_fn,
        )

        self.assertEqual(mcts.best_answer, "RootATERM")
        self.assertEqual(
            mcts.best_answer_reason, BestAnswerReason.CHECKED_AND_TRUE
        )
        self.assertEqual(call_count, 2)

        save_tree_to_txt(
            root_node=mcts.root_node,
            output_path="tests/outputs/mcts_termination_mixed.txt",
            selected_solution=mcts.best_answer,
        )


if __name__ == "__main__":
    unittest.main()
