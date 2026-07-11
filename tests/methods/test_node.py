import os
import random
import tempfile
import unittest

from treethink.graph import extract_solution_from_graphviz, save_tree_to_txt
from treethink.methods import RFMCTS, Node


class TestNode(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".dot"
        )
        self.temp_file_path = self.temp_file.name

    def tearDown(self):
        """Clean up after each test method."""
        self.temp_file.close()
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)

    def test_root_node_with_simple_text(self):
        """Test printing a root node with simple text."""
        node = Node(text="Root node")
        node.win_value = 42

        with open(self.temp_file_path, "w") as f:
            node.print_node(f, 0, "root")

        with open(self.temp_file_path, "r") as f:
            content = f.read()

        self.assertIn("root [label=<<TABLE", content)
        self.assertIn("42", content)
        self.assertIn("Root node", content)
        self.assertIn("shape=box", content)

    def test_root_node_with_multiline_text(self):
        """Test printing a root node with multiline text."""
        node = Node(text="Line 1\nLine 2\nLine 3")
        node.win_value = 100

        with open(self.temp_file_path, "w") as f:
            node.print_node(f, 0, "root")

        with open(self.temp_file_path, "r") as f:
            content = f.read()

        self.assertIn("100", content)
        self.assertIn("Line 1", content)

    def test_node_with_special_characters(self):
        """Test printing node with special characters that need escaping."""
        node = Node(
            text='Text with "quotes", \\ backslashes, greater than: >, '
            + "smaller than: <, ampersand &, "
        )
        node.win_value = 10

        method = RFMCTS(None, None, None)
        method.root_node = node

        save_tree_to_txt(
            root_node=node,
            output_path="tests/outputs/complex_text.txt",
        )

    def test_node_with_zero_win_value(self):
        """Test printing node with win_value of 0."""
        node = Node(text="Zero value node")
        node.win_value = 0

        with open(self.temp_file_path, "w") as f:
            node.print_node(f, 0, "root")

        with open(self.temp_file_path, "r") as f:
            content = f.read()

        self.assertIn("0", content)

    def test_node_with_negative_win_value(self):
        """Test printing node with negative win_value."""
        node = Node(text="Negative value node")
        node.win_value = -15

        with open(self.temp_file_path, "w") as f:
            node.print_node(f, 0, "root")

        with open(self.temp_file_path, "r") as f:
            content = f.read()

        self.assertIn("-15", content)

    def test_node_with_float_win_value(self):
        """Test printing node with float win_value."""
        node = Node(text="Float value node")
        node.win_value = 3.14159

        with open(self.temp_file_path, "w") as f:
            node.print_node(f, 0, "root")

        with open(self.temp_file_path, "r") as f:
            content = f.read()

        self.assertIn(str(round(node.win_value, 3)), content)

    def test_tree_with_multiple_children(self):
        """Test printing a tree with multiple children and edges."""
        root = Node(text="Root")
        root.win_value = 100

        child1 = Node(text="Root\nChild 1", parent=root)
        child1.win_value = 50

        child2 = Node(text="Root\nChild 2", parent=root)
        child2.win_value = 30

        root.children = [child1, child2]

        with open(self.temp_file_path, "w") as f:
            root.print_node(f, 0, "root")

        with open(self.temp_file_path, "r") as f:
            content = f.read()

        # Check root
        self.assertIn("100", content)
        # Check children
        self.assertIn("50", content)
        self.assertIn("30", content)
        # Check edges
        self.assertIn("root -- root_0", content)
        self.assertIn("root -- root_1", content)

    def test_remove_duplicate_children(self):
        """Test that duplicate children (by text) are removed, preserving order."""
        root = Node(text="Root")

        child1 = Node(text="Choice A", parent=root)
        child2 = Node(text="Choice B", parent=root)
        # duplicate of child1
        child3 = Node(text="Choice A", parent=root)

        root.children = [child1, child2, child3]

        # Sanity check before removal
        self.assertEqual(
            [c.text for c in root.children],
            ["Choice A", "Choice B", "Choice A"],
        )

        root.remove_duplicate_children()

        # After removal we expect only the first occurrences to remain and order preserved
        self.assertEqual(
            [c.text for c in root.children], ["Choice A", "Choice B"]
        )

    def test_indentation_levels(self):
        """Test that indentation increases with depth."""
        root = Node(text="Root")
        root.win_value = 1

        child = Node(text="Root\nChild", parent=root)
        child.win_value = 2
        root.children = [child]

        grandchild = Node(text="Root\nChild\nGrandchild", parent=child)
        grandchild.win_value = 3
        child.children = [grandchild]

        with open(self.temp_file_path, "w") as f:
            root.print_node(f, 0, "root")

        with open(self.temp_file_path, "r") as f:
            content = f.read()

        lines = content.split("\n")

        # Check that deeper nodes have more indentation
        root_line = [
            l for l in lines if "root [label=" in l and "root_" not in l
        ][0]
        child_line = [l for l in lines if "root_0 [label=" in l][0]

        # Child should have more leading spaces than root
        self.assertTrue(
            len(child_line) - len(child_line.lstrip())
            > len(root_line) - len(root_line.lstrip())
        )

    def test_empty_text(self):
        """Test printing node with empty text."""
        node = Node(text="")
        node.win_value = 5

        with open(self.temp_file_path, "w") as f:
            node.print_node(f, 0, "root")

        with open(self.temp_file_path, "r") as f:
            content = f.read()

        self.assertIn("5", content)
        self.assertIn("root [label=<<TABLE", content)

    def test_simple_output(self):
        """Create a dummy Method and a proof tree and save the result as
        .txt."""

        root = Node(text="Root")
        root.win_value = random.random()

        child = Node(text="Root\nChild", parent=root)
        child.win_value = random.random()
        root.children = [child]

        grandchild = Node(text="Root\nChild\nGrandchild", parent=child)
        grandchild.win_value = random.random()
        child.children = [grandchild]

        method = RFMCTS(None, None, None)
        method.root_node = root

        save_tree_to_txt(
            root_node=root,
            output_path="tests/outputs/simple_graph.txt",
        )

    def test_long_text_output(self):
        """Create a dummy Method and a proof tree and save the result as
        .txt."""

        loremipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        root = Node(text="root-" + loremipsum)
        root.win_value = random.random()

        child = Node(text="child-" + loremipsum, parent=root)
        child.win_value = random.random()
        root.children = [child]

        grandchild = Node(text="grandchild-" + loremipsum, parent=child)
        grandchild.win_value = random.random()
        child.children = [grandchild]

        from treethink import RFMCTS

        method = RFMCTS(None, None, None)
        method.root_node = root

        save_tree_to_txt(
            root_node=root,
            output_path="tests/outputs/long_text.txt",
        )

    def test_colored_selected_proof_output(self):
        """Test selected proof color."""

        root = Node(text="Root")
        root.win_value = random.random()

        child1 = Node(text="Root\nChild1", parent=root)
        child1.win_value = random.random()
        child2 = Node(text="Root\nChild2", parent=root)
        child2.win_value = random.random()
        root.children = [child1, child2]

        grandchild = Node(text="Root\nChild\nGrandchild", parent=child1)
        grandchild.win_value = random.random()
        child1.children = [grandchild]

        method = RFMCTS(None, None, None)
        method.root_node = root

        solution = root.text + child1.text + grandchild.text

        save_tree_to_txt(
            root_node=root,
            output_path="tests/outputs/proof_colored.txt",
            selected_solution=solution,
        )

    def test_extract_solution(self):
        """Test method's capability of extracting proof from graphviz content."""

        root = Node(text="Root")
        root.win_value = random.random()

        child1 = Node(text="Child1", parent=root)
        child1.win_value = random.random()
        child2 = Node(text="Child2", parent=root)
        child2.win_value = random.random()
        root.children = [child1, child2]

        grandchild = Node(text="Grandchild", parent=child1)
        grandchild.win_value = random.random()
        child1.children = [grandchild]

        from treethink import RFMCTS

        method = RFMCTS(None, None, None)
        method.root_node = root

        solution = root.text + child1.text + grandchild.text

        save_tree_to_txt(
            root_node=root,
            output_path=self.temp_file_path,
            selected_solution=solution,
        )

        with open(self.temp_file_path, "r") as f:
            content = f.read()

        extracted_solution = extract_solution_from_graphviz(content)
        self.assertEqual(extracted_solution, solution)

    def test_proof_colored_same_level_error(self):
        """Test method's capability of extracting proof from graphviz content."""

        root = Node(text="Root")
        root.win_value = random.random()

        child1 = Node(text="Child1", parent=root)
        child1.win_value = random.random()
        child2 = Node(text="Child2", parent=root)
        child2.win_value = random.random()
        root.children = [child1, child2]

        grandchild1 = Node(text="Grandchild1", parent=child1)
        grandchild2 = Node(text="Grandchild1", parent=child2)
        grandchild3 = Node(text="Grandchild2", parent=child2)
        grandchild1.win_value = random.random()
        grandchild2.win_value = random.random()
        grandchild2.win_value = random.random()

        child1.children = [grandchild1]
        child2.children = [grandchild2, grandchild3]

        from treethink import RFMCTS

        method = RFMCTS(None, None, None)
        method.root_node = root

        solution = root.text + child1.text + grandchild1.text

        save_tree_to_txt(
            root_node=root,
            output_path="tests/outputs/proof_colored_same_level_error.txt",
            selected_solution=solution,
        )

        with open("tests/outputs/proof_colored_same_level_error.txt", "r") as f:
            content = f.read()

        extracted_solution = extract_solution_from_graphviz(content)
        self.assertEqual(extracted_solution, solution)

    def test_find_path_to_terminal(self):
        root = Node(text="Root")
        child1 = Node(text="Child1", parent=root)
        child2 = Node(text="Child2", parent=root)
        root.children = [child1, child2]
        grandchild1 = Node(text="Grandchild1", parent=child1)
        grandchild2 = Node(text="Grandchild2", parent=child2)
        grandchild3 = Node(text="Grandchild3", parent=child2)

        child1.children = [grandchild1]
        child2.children = [grandchild2, grandchild3]

        method = RFMCTS(None, None, None)
        self.assertEqual(
            method.traverse_to_root(grandchild2),
            "RootChild2Grandchild2",
        )

    def test_print_node_to_terminal(self):
        """Test printing a node to terminal instead of file."""
        node = Node(text="Terminal Node\nWith Multiple Lines\n~End")
        node.win_value = 20

        print(node)


if __name__ == "__main__":
    unittest.main()
