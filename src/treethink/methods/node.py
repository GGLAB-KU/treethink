"""Node implementation to be used with different inference time methods."""

import json
from typing import Iterable, List, Optional

from vllm import CompletionOutput


class Node:
    def __init__(
        self,
        text: Optional[str] = None,
        max_children: Optional[int] = None,
        level: int = 0,
        parent: Optional["Node"] = None,
        vllm_output: Optional[CompletionOutput] = None,
        termination_str: Optional[str] = None,
        exploration_weight: Optional[float] = None,
        *args,
        **kwargs,
    ):
        self.text = text
        self.max_children = max_children
        self.level: int = level if parent is None else parent.level + 1
        self.parent = parent
        self.vllm_output = vllm_output
        self.termination_str = termination_str
        self.exploration_weight = exploration_weight

        self.children: List["Node"] = []
        self.win_value = 0
        self.visits = 0
        self.id = id(self)
        self.is_widen_node = False

    def remove_duplicate_children(self) -> None:
        """
        Remove duplicate children based on their text property.
        Keeps the first occurrence of each unique text value.
        """
        if not self.children:
            return

        seen_texts = set()
        unique_children = []

        for child in self.children:
            if child.text not in seen_texts:
                seen_texts.add(child.text)
                unique_children.append(child)

        self.children = unique_children

    @property
    def is_termination_node(self) -> bool:
        return (
            self.termination_str == self.text if self.termination_str else False
        )

    @property
    def is_root_node(self) -> bool:
        return self.parent is None

    @property
    def is_expandable(self) -> bool:
        """Whether the node is ready for expansion.

        Checks both children_count < max_children and termination_str if it
        exists.
        """
        return len(self.children) == 0 and not self.is_termination_node

    def is_fully_expanded(self) -> bool:
        """
        Checks if the node is fully expanded, meaning it has the maximum number of children.

        Returns:
            bool: True if the node is fully expanded, False otherwise.
        """
        return len(self.children) >= self.max_children

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children: Iterable):
        for child in children:
            self.add_child(child)

    def _get_min_max_win_value(self):
        """
        Find min and max win_value in subtree rooted at this node.

        Returns:
            tuple: (min_value, max_value)
        """
        min_val = self.win_value
        max_val = self.win_value

        for child in self.children:
            child_min, child_max = child._get_min_max_win_value()
            min_val = min(min_val, child_min)
            max_val = max(max_val, child_max)

        return min_val, max_val

    def print_node(self, f, i, root, st, solution=None, prev_colored=False):
        def _escape(x):
            return json.dumps(x).strip('"')

        if self.parent is None:
            text_content = self.text
        else:
            diff = "\n".join(
                [
                    x
                    for x in self.text.split("\n")
                    if x not in self.parent.text.split("\n")
                ]
            )
            text_content = diff

        # Change special symbols to those that HTML like:
        text_content = (
            text_content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

        # Create HTML-like table with win value, visits and level of the node
        label = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            f'<TR><TD ROWSPAN="3">{_escape(text_content)}</TD>'
            f'<TD ALIGN="RIGHT"><FONT POINT-SIZE="10">W:{round(self.win_value, 3)}</FONT></TD></TR>'
            f'<TR><TD ALIGN="RIGHT"><FONT POINT-SIZE="10">V:{self.visits}</FONT></TD></TR>'
            f'<TR><TD ALIGN="RIGHT"><FONT POINT-SIZE="10">L:{self.level}</FONT></TD></TR>'
            "</TABLE>>"
        )

        # NOTE(burak): I have tried different colorings but single red color
        # seems to be sufficient. If we change it to something else, we need
        # to update default identifier arguments functions in graph.py
        color = "red"
        attr = "penwidth=2"
        if (
            solution is not None
            and solution[: len(self.text)] == self.text
            and prev_colored
        ):
            attr += f",color={color}"
            prev_colored = True
        else:
            prev_colored = False

        f.write((" " * i) + f"{st} [label={label},shape=box,{attr}]\n")

        num = 0
        for child in self.children:
            new_st = st + "_" + str(num)

            child.print_node(
                f,
                i + 2,
                root,
                new_st,
                solution=None
                if solution is None
                else solution[len(self.text) :],
                prev_colored=prev_colored,
            )

            f.write(" " * i + st + " -- " + new_st + "\n")
            num = num + 1

    def most_visited_child(self) -> "Node":
        """
        Selects the child node with the highest number of visits.

        Returns:
            Node: The most visited child node.
        """
        return max(self.children, key=lambda child: child.visits)

    def __str__(self) -> str:
        _escaped = self.text.replace("\n", "\\n")
        return f"Node(visits={self.visits},win_value={self.win_value},text={_escaped})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Node):
            raise NotImplementedError(
                f"Can not compare type {type(other)} and Node."
            )
        return (
            self.text == other.text
            and self.level == other.level
            and self.max_children == other.max_children
            and self.termination_str == other.termination_str
        )


def get_total_child_num(node: Node):
    """Get the number of total child nodes recursively."""
    if not node.children:
        return 1

    total = 1  # Count this node
    for child in node.children:
        total += get_total_child_num(child)

    return total
