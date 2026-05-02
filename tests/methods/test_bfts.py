import sys
import tempfile
import unittest

from loguru import logger

from tests.common import RandomNodeEvaluator, SimpleChildFinder
from treethink.graph import (  # noqa
    extract_solution_from_graphviz,
    save_tree_to_txt,
)
from treethink.methods import BFTS, Node  # noqa


class TestBFTS(unittest.TestCase):
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
        finder_func = SimpleChildFinder()

        bfts = BFTS(
            root_node=None,
            finder=finder_func,
            evaluator=evaluator_func,
        )
        bfts.set_root_node(root)
        bfts.simulate(expansion_count=10)
        solution = bfts.best_answer
        logger.debug(f"Solution: {solution}")

        save_tree_to_txt(
            root_node=bfts.root_node,
            output_path="tests/outputs/bfts_simple.txt",
            selected_solution=solution,
        )


if __name__ == "__main__":
    logger.remove(0)
    logger.add(sys.stderr, level="TRACE")
    unittest.main()
