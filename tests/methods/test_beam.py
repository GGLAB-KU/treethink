import sys
import tempfile
import unittest

from loguru import logger

from tests.common import RandomNodeEvaluator, SimpleChildExpander
from treethink.graph import (  # noqa
    extract_solution_from_graphviz,
    save_tree_to_txt,
)
from treethink.methods import BeamSearch, Node  # noqa


class TestBeamSearch(unittest.TestCase):
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

        beam = BeamSearch(
            root_node=None,
            expander=expander_func,
            evaluator=evaluator_func,
            beam_width=2,
        )
        beam.set_root_node(root)
        beam.simulate(expansion_count=2)
        solution = beam.best_answer
        logger.debug(f"Solution: {solution}")

        save_tree_to_txt(
            root_node=beam.root_node,
            output_path="tests/outputs/beam_simple.txt",
            selected_solution=solution,
        )


if __name__ == "__main__":
    logger.remove(0)
    logger.add(sys.stderr, level="TRACE")
    unittest.main()
