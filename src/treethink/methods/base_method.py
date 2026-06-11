import inspect
import re
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Optional, Tuple, Union

from loguru import logger

from ..utils.enums import BestAnswerReason, FinalDecisionMode, coerce_enum
from .node import Node


class BaseMethod(ABC):
    """Abstract base class for all tree-search inference methods.

    Provides core utilities for tree search algorithms including node traversal,
    proof parsing, and best-answer selection.  All tree-search methods (MCTS,
    BFTS, BeamSearch, etc.) inherit from this class.

    Subclasses must implement ``simulate()`` for their specific search strategy.

    Key features:
    - ``traverse_to_root()`` — reconstruct the full proof path from node to root
    - ``parse_proof()`` — extract proof text from code-fence delimiters
    - ``best_answer`` / ``best_answer_reason`` — track selected answer and how
      it was determined (calculated, manually set, or REPL-verified)
    - ``final_decision_mode`` — strategy for picking the best node after search
      (native, maximize_visits, maximize_value, clear_frontier)
    - Expansion statistics (``stats_expansion_count``, etc.)
    """

    def _format_str_value(self, value):
        if callable(value):
            return getattr(value, "__name__", value.__class__.__name__)
        return repr(value)

    def _str_fields(self):
        return [
            ("root_node", getattr(self, "root_node", None)),
            ("policy", getattr(self, "policy", None)),
            ("evaluator", getattr(self, "evaluator", None)),
            (
                "final_decision_mode",
                getattr(self, "final_decision_mode", None),
            ),
            (
                "stats_expansion_count",
                getattr(self, "stats_expansion_count", None),
            ),
            (
                "stats_failed_expansion_count",
                getattr(self, "stats_failed_expansion_count", None),
            ),
        ]

    def __str__(self) -> str:
        fields = ", ".join(
            f"{name}={self._format_str_value(value)}"
            for name, value in self._str_fields()
        )
        return f"{self.__class__.__name__}({fields})"

    __repr__ = __str__

    def __init__(
        self,
        root_node: Optional[Union[Node, str]],
        policy: Callable,
        evaluator: Callable,
        final_decision_mode: FinalDecisionMode = FinalDecisionMode.NATIVE,
        *args,
        **kwargs,
    ):
        # Main functions
        self.policy = policy
        self.evaluator = evaluator

        # Final decision mode and its function, "native" for base method
        # Change _compute_best_answer to change best_answer computation in the
        # child class.
        self.final_decision_mode = coerce_enum(
            final_decision_mode, FinalDecisionMode
        )
        self._compute_best_answer = self._compute_native_best_answer

        # self._best_answer is defined in order for externally setting
        # best_answer property and its condition. Available ,
        self._best_answer: str = None
        self.best_answer_reason: BestAnswerReason = None

        # Expansion statistics
        self.stats_expansion_count = 0
        self.stats_failed_expansion_count = 0

        # Compile for faster use later
        self._re_parse = re.compile(r"```(?:\w+|\n)\s*((?:.|\n)*?)```")

        if root_node:
            self.root_node = self.set_root_node(root_node)

    def set_root_node(self, root_node: Union[Node, str]) -> Node:
        """Create the root node."""

        self.root_node = (
            root_node if isinstance(root_node, Node) else Node(root_node)
        )
        self.root_node.win_value = 0

        return self.root_node

    def reset(self, root_node: Optional[Node] = None):
        """Reset the inner variables so that the class can be used for another
        generation task without instantiating it again.
        """
        self.__init__(root_node, self.policy, self.evaluator)

    def get_widths(self):
        widths = [1]
        nodes = [self.root_node]
        while any([len(n.children) > 0 for n in nodes]):
            new_nodes = []
            for node in nodes:
                for child in node.children:
                    new_nodes.append(child)
            nodes = new_nodes
            widths.append(len(nodes))
        return widths

    def get_child_counts(self):
        counts = [1]
        nodes = [self.root_node]
        while any([len(n.children) > 0 for n in nodes]):
            new_nodes = []
            for node in nodes:
                for child in node.children:
                    new_nodes.append(child)
            nodes = new_nodes
            counts.extend([len(n.children) for n in nodes])
        return counts

    def get_values_and_visits(self):
        values = [self.root_node.win_value]
        visits = [self.root_node.visits]
        expected_values = [
            self.root_node.win_value / (self.root_node.visits or 1)
        ]
        nodes = [self.root_node]
        while any([len(n.children) > 0 for n in nodes]):
            new_nodes = []
            for node in nodes:
                for child in node.children:
                    new_nodes.append(child)
            nodes = new_nodes
            values.extend([n.win_value for n in nodes])
            visits.extend([n.visits for n in nodes])
            expected_values.extend(
                [n.win_value / (n.visits or 1) for n in nodes]
            )
        return values, visits, expected_values

    def get_widen_count(self):
        count = 0
        nodes = [self.root_node]
        while any([len(n.children) > 0 for n in nodes]):
            new_nodes = []
            for node in nodes:
                for child in node.children:
                    new_nodes.append(child)
            nodes = new_nodes
            count += len([n for n in nodes if n.is_widen_node])
        return count

    def get_stat_dict(self):
        stat = {}
        widths = self.get_widths()
        stat["width"] = max(widths)
        stat["depth"] = len(widths)
        stat["total_nodes"] = sum(widths)

        child_counts = self.get_child_counts()
        stat["mean_child_count"] = sum(child_counts) / len(child_counts)
        stat["max_child_count"] = max(child_counts)
        stat["leaf_node_count"] = len([1 for c in child_counts if c == 0])
        stat["widen_count"] = self.get_widen_count()

        values, visits, expected_values = self.get_values_and_visits()
        stat["mean_value"] = sum(values) / len(values)
        stat["max_value"] = max(values)
        stat["min_value"] = min(values)
        stat["mean_visits"] = sum(visits) / len(visits)
        stat["max_visits"] = max(visits)
        stat["min_visits"] = min(visits)
        stat["mean_expected_value"] = sum(expected_values) / len(
            expected_values
        )
        stat["max_expected_value"] = max(expected_values)
        stat["min_expected_value"] = min(expected_values)

        # if termination_str is defined for root it will be defined for all
        if self.root_node.termination_str:
            leaves = self.find_leaves(self.root_node)
            stat["termination_count"] = len(
                [1 for leaf in leaves if leaf.is_termination_node]
            )

        return stat

    def find_path_to_terminal(self, terminal_node):
        """
        Find the path from root to a terminal node.

        Args:
            terminal_node: The terminal node to trace back from

        Returns:
            List of states from root to terminal node
        """
        path = deque()
        current = terminal_node
        while current is not None:
            path.appendleft(current.text)
            current = current.parent
        return list(path)

    def traverse_to_root(self, node, include_root=True) -> str:
        if include_root:
            return "".join(self.find_path_to_terminal(node))
        else:
            path = self.find_path_to_terminal(node)
            if len(path) > 1:
                return "".join(path[1:])
            else:
                # Maybe unexpected behavior, putting it to trace:
                logger.trace(
                    "Even though include_root=False, you have provided root "
                    + "node itself. Returning empty string."
                )
                return ""

    def _compute_native_best_answer(self) -> str:
        final_node = self.make_choice()
        path = self.find_path_to_terminal(final_node)
        return "".join(path)

    @property
    def best_answer(self) -> str:
        if self._best_answer is None:
            self._best_answer = self._compute_best_answer()
            self.best_answer_reason = BestAnswerReason.CALCULATED

        return self._best_answer

    @best_answer.setter
    def best_answer(self, value: Union[str, Tuple[str, str]]):
        if isinstance(value, Node):
            logger.warning(
                "You are trying to set best_answer from a node, did you forget "
                + "to run find_path_to_terminal? Extracting node's text to be "
                + "best_answer for now..."
            )
            value = value.text

        if not isinstance(value, str):
            raise TypeError(
                f"Setting best_answer to be of type {type(value)} is not "
                + "supported."
            )

        if self._best_answer is not None:
            logger.debug(
                f"Overriding previous best answer:\n{self._best_answer}\n"
                + f"In favor of:\n{value}"
            )

        self._best_answer = value
        self.best_answer_reason = BestAnswerReason.SET

    def find_leaves(self, node):
        """Helper function to find the leaves in a tree."""
        queue = [node]
        visited = set()
        leaves = []

        while node:
            if queue:
                current_node = queue.pop(0)  # Remove from front (BFS)
            else:
                break

            # Avoid infinite loops in case of cycles (though shouldn't happen in trees)
            if id(current_node) in visited:
                continue
            visited.add(id(current_node))

            if current_node.children:
                for child in current_node.children:
                    queue.append(child)
            else:
                leaves.append(current_node)

        return leaves

    def parse_proof(self, proof: str, pattern=None):
        # TODO(burak): This should be moved elsewhere and maybe improved like the one I use in assessment scripts.
        pattern = self._re_parse if pattern is None else pattern
        matches = re.findall(pattern, proof)

        # If we fail to find anything, we should return
        if not matches:
            logger.debug(
                "Parsing yielded nothing, returning the original proof trajectory.",
            )
            return proof

        return matches[-1]

    def update_proof_path_visits(self, node: Node):
        """Update visits on the proof trajectory."""
        current = node
        while current is not None:
            current.visits += 1
            current = current.parent

    @abstractmethod
    def make_choice(self, node: Optional[Node] = None):
        pass

    @abstractmethod
    def make_exploratory_choice(self, node: Optional[Node] = None):
        pass

    @abstractmethod
    def simulate(
        self,
        expansion_count: int = 5,
        timeout: Optional[float] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ):
        pass

    def expand(self, node: Node):
        """Base expand method that calls policy and evaluator, then
        updates visits. May be overridden in child classes for additional
        functionality.

        Does not remove duplicate children, use expand_remove_duplicate_children
        for that.
        """
        self.stats_expansion_count += 1
        logger.trace(f"Node to expand: {node}")

        # Use policy to generate children
        self.policy(node, self)
        logger.trace(f"Found children: {node.children}")

        if not node.children:
            logger.warning(f"Failed to expand node: {node}.")
            self.stats_failed_expansion_count += 1
            return

        # Evaluate each child in a batched way
        children_win_values = self.evaluator(node.children, self)
        logger.trace(f"Found children win values: {children_win_values}")
        for i, win_val in enumerate(children_win_values):
            if win_val is not None:
                node.children[i].win_value = win_val

    async def async_expand(self, node: Node):
        """Async version of expand method that calls policy and evaluator.

        The evaluator is called asynchronously, which allows for concurrent
        evaluation of children nodes, significantly improving performance for I/O-bound
        operations like REPL verification and LLM-as-judge scoring.

        Does not remove duplicate children, use async_expand_rm_dupes for that.
        """
        self.stats_expansion_count += 1
        logger.trace(f"Node to async expand: {node}")

        # Use policy to generate children
        # Check if policy is async or sync
        if inspect.iscoroutinefunction(self.policy):
            await self.policy(node, self)
        else:
            self.policy(node, self)
        logger.trace(f"Found children: {node.children}")

        if not node.children:
            logger.warning(f"Failed to expand node: {node}.")
            self.stats_failed_expansion_count += 1
            return

        # Evaluate each child in a batched way (async)
        children_win_values = await self.evaluator(node.children, self)
        logger.trace(f"Found children win values: {children_win_values}")
        for i, win_val in enumerate(children_win_values):
            if win_val is not None:
                node.children[i].win_value = win_val

    def expand_rm_dupes(self, node: Node):
        """Base expand method that calls policy, removes duplicates, and
        calls evaluator, then updates visits. May be overridden in child
        classes for additional functionality.
        """
        self.stats_expansion_count += 1

        # Use policy to generate children
        self.policy(node, self)

        if not node.children:
            logger.warning(f"Failed to expand node: {node}.")
            self.stats_failed_expansion_count += 1
            return

        # Remove duplicate children based on their text property
        node.remove_duplicate_children()

        # Evaluate each child in a batched way
        children_win_values = self.evaluator(node.children, self)
        logger.trace(f"Found children win values: {children_win_values}")
        for i, win_val in enumerate(children_win_values):
            if win_val is not None:
                node.children[i].win_value = win_val

    async def async_expand_rm_dupes(self, node: Node):
        """Async version of expand_rm_dupes method.

        This method generates children using policy, removes duplicates,
        and then evaluates them asynchronously. The async evaluation allows for
        concurrent processing of children nodes.
        """
        self.stats_expansion_count += 1
        logger.trace(f"Expanding node: {node}")

        # Generate children
        logger.trace(f"Calling policy for node {id(node)}...")
        await self.policy(node, self)

        if not node.children:
            logger.warning(f"Failed to expand node: {node}.")
            self.stats_failed_expansion_count += 1
            return

        # Remove duplicate children based on their text property
        prev_child_count = len(node.children)
        logger.trace(f"Removing duplicate children: {node.children}")
        node.remove_duplicate_children()
        logger.debug(
            f"Removed duplicates: {prev_child_count - len(node.children)} duplicate(s) removed. {len(node.children)} unique child(ren) remain."
        )

        # Evaluate each child in a async and batched way
        children_win_values = await self.evaluator(node.children, self)
        logger.trace(f"Found children win values: {children_win_values}")
        for i, win_val in enumerate(children_win_values):
            if win_val is not None:
                node.children[i].win_value = win_val
