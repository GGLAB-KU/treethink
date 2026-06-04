import unittest
from pathlib import Path

from treethink.graph import load_graphviz_state, save_tree_to_txt
from treethink.methods import Node


class TestLoadingTreeState(unittest.TestCase):
    def _build_sample_tree(self) -> Node:
        root = Node(text="root", max_children=2)
        root.win_value = 1.0
        root.visits = 2

        child_a = Node(text="root\nchild_a", max_children=2, parent=root)
        child_a.win_value = 0.5
        child_a.visits = 1

        grandchild = Node(
            text="root\nchild_a\ngrandchild",
            max_children=2,
            parent=child_a,
        )
        grandchild.win_value = 0.1
        grandchild.visits = 0

        child_b = Node(text="root\nchild_b", max_children=2, parent=root)
        child_b.win_value = -0.2
        child_b.visits = 0

        root.add_child(child_a)
        root.add_child(child_b)
        child_a.add_child(grandchild)

        return root

    def test_create_and_load_tree_state(self):
        root = self._build_sample_tree()
        tree_path = Path(__file__).parent / "outputs/load_state_tree.txt"
        save_tree_to_txt(root, tree_path)

        metadata = {
            "treethink_args": {
                "method_name": "MCTS",
                "max_children": 2,
                "exploration_weight": 0.5,
            }
        }

        state = load_graphviz_state(tree_path, metadata)
        loaded_root = state["root_node"]

        self.assertEqual(loaded_root.text, "root")
        self.assertEqual(loaded_root.win_value, 1.0)
        self.assertEqual(loaded_root.visits, 2)
        self.assertEqual(loaded_root.max_children, 2)
        self.assertEqual(len(loaded_root.children), 2)

        child_texts = {child.text for child in loaded_root.children}
        self.assertEqual(child_texts, {"root\nchild_a", "root\nchild_b"})

        child_a = next(
            child
            for child in loaded_root.children
            if child.text.endswith("child_a")
        )
        self.assertEqual(len(child_a.children), 1)
        self.assertEqual(child_a.children[0].text, "root\nchild_a\ngrandchild")

    def load_hardcoded_state(self):
        # TODO(burak): maybe publish a sample tree and metadata file?
        graph_path = ""
        metadata_path = ""
        state = load_graphviz_state(graph_path, metadata=metadata_path)

        # NOTE(burak): I have also tested it by hand:
        # breakpoint()
        print(state["root_node"])
        return state

    def test_load_hardcoded_state(self):
        state = self.load_hardcoded_state()
        root = state["root_node"]

        self.assertEqual(root.text, "hardcoded_root")
        self.assertEqual(root.level, 0)
        self.assertGreater(len(root.children), 0)


if __name__ == "__main__":
    unittest.main()
