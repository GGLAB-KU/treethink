"""Best First Tree Search algorithm for inference time scaling techniques."""

import asyncio
import heapq
import random
import time
from typing import Callable, List, Literal, Optional

from loguru import logger

from .base_method import BaseMethod
from .node import Node


class BFTS(BaseMethod):
    def __init__(
        self,
        root_node: Optional[Node | str],
        policy: Callable,
        evaluator: Callable,
        final_decision_mode: Literal[
            "clear_frontier", "native"
        ] = "clear_frontier",
        *args,
        **kwargs,
    ):
        """Best First Tree Search implementation.

        Similar to GPT-f: https://arxiv.org/pdf/2009.03393

        Args:
            root_node (Node): root node.
            policy (Callable): policy method to use in expansion.
            evaluator (Callable): evaluator method to give the node
                a score.
        """

        # Priority queue for best-first search
        # Using negative values since heapq is a min-heap but we want max priority
        self.frontier = []

        # Keep track of nodes in frontier to avoid duplicates
        self.in_frontier = set()

        super().__init__(
            root_node=root_node,
            policy=policy,
            evaluator=evaluator,
            final_decision_mode=final_decision_mode,
        )

        if self.final_decision_mode == "clear_frontier":
            self._compute_best_answer = self._best_answer_clear_frontier
        elif self.final_decision_mode == "native":
            # already set in BaseMethod
            pass
        else:
            logger.warning(
                f"Given {self.final_decision_mode} is not supported, "
                + "falling back to `native` implementation."
            )

    def set_root_node(self, root_node):
        self.root_node = super().set_root_node(root_node)

        # Add root to frontier with its priority
        self.push_to_frontier(self.root_node)

    def push_to_frontier(self, node: Node):
        child_id = id(node)
        heapq.heappush(self.frontier, (-node.win_value, child_id, node))
        self.in_frontier.add(child_id)

    def pop_from_frontier(self):
        if self.frontier:
            win_value, node_id, node = heapq.heappop(self.frontier)
            self.in_frontier.discard(node_id)
        else:
            logger.warning("frontier is empty, returning root node")
            win_value, node_id, node = (
                self.root_node.win_value,
                id(self.root_node),
                self.root_node,
            )

        # we put win values as negatives, we should extract them as negative too
        return -win_value, node_id, node

    def make_choice(self, node: Optional[Node] = None):
        """Select the best child based on win_value using best-first search.

        Args:
            node (Optional[Node], optional): Node to start from. Defaults to None (uses root).

        Returns:
            Node: The child with the highest cumulative win_value, or None if no children.
        """

        # get the most promising node out
        _, node_id, node = self.pop_from_frontier()
        return node

    def make_exploratory_choice(self, node: Optional[Node] = None):
        """Select a child probabilistically based on expected values"""

        node = self.root_node if node is None else node

        if not node.children:
            return None

        # Calculate expected values for all children
        expected_values = []
        total_expected = 0

        for child in node.children:
            if child.visits > 0:
                expected = child.win_value / child.visits
            else:
                expected = 0
            expected_values.append(max(expected, 0))  # Ensure non-negative
            total_expected += expected

        if total_expected == 0:
            return random.choice(node.children)

        # Normalize to probabilities
        probabilities = [ev / total_expected for ev in expected_values]

        # Select based on probabilities
        random_probability = random.uniform(0, 1)
        cumulative_prob = 0.0

        for i, probability in enumerate(probabilities):
            cumulative_prob += probability
            if cumulative_prob >= random_probability:
                return node.children[i]

        return node.children[-1]  # Fallback

    def simulate(
        self,
        expansion_count: int = 5,
        timeout: Optional[float] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ):
        """
        Main search loop, expands nodes in best-first order.
        """
        i = 0
        start_time = time.time()

        logger.debug("Simulation started.")
        while (i < expansion_count) and self.frontier:
            logger.debug(f"Expansion no: {i}")
            logger.trace(f"Current frontier:\n{self.frontier}")
            i += 1

            # Get the node that has biggest priority
            _, node_id, current_node = self.pop_from_frontier()

            # Check if termination node is encountered and run termination fn
            if (
                termination_encountered_fn is not None
                and current_node.is_termination_node
            ):
                answer = termination_encountered_fn(current_node)
                if answer:
                    self.best_answer = answer
                    self.best_answer_reason = "checked_and_true"
                    break

            # Skip if node was already expanded
            if not current_node.is_expandable:
                continue

            # Expand the current node
            if remove_duplicate_children:
                self.expand_rm_dupes(current_node)
            else:
                self.expand(current_node)

            if timeout is not None:
                duration = time.time() - start_time
                if duration > timeout:
                    logger.debug(
                        f"Reached timelimit, stopping expansion on current node: {current_node}"
                    )
                    return

    def expand(self, node):
        """Additionally push children to frontier. See BaseMethod.expand for
        details.
        """
        # Run the usual expansion
        super().expand(node)

        # Add remaining children to frontier
        for child in node.children:
            if id(child) not in self.in_frontier:
                self.push_to_frontier(child)

    def expand_rm_dupes(self, node):
        """Additionally push children to frontier after removing duplicates.
        See BaseMethod.expand_remove_duplicate_children for details.
        """
        # Run the usual expansion
        super().expand_rm_dupes(node)

        # Add remaining children to frontier
        for child in node.children:
            if id(child) not in self.in_frontier:
                self.push_to_frontier(child)

    def get_frontier_size(self) -> int:
        """Get current size of the frontier"""
        return len(self.frontier)

    def get_frontier_priorities(self) -> List[float]:
        """Get current priorities in the frontier (for debugging)"""
        return [-priority for priority, _, _ in self.frontier]

    def get_stat_dict(self):
        stat = super().get_stat_dict()

        # Add BFTS-specific stats
        stat["frontier_size"] = self.get_frontier_size()

        return stat

    def _best_answer_clear_frontier(self):
        """Reset self.frontier and select max valued leaf node via
        self.make_choice, similar to `native` implementation.

        NOTE(burak): The reason why we do this is because if all leaf nodes
        are termination nodes, then self.frontier is empty. We need to calculate
        leaf nodes again."""
        leaves = self.find_leaves(self.root_node)

        # Reset frontier and push leaves to it
        self.frontier = []
        for n in leaves:
            self.push_to_frontier(n)

        return super()._compute_native_best_answer()


class AsyncBFTS(BFTS):
    """
    Async version of BFTS that supports asynchronous node expansion.

    This implementation leverages async_expand and async_expand_rm_dupes from
    BaseMethod to enable concurrent evaluation of children nodes during best-first
    tree search expansion.

    Key features:
    - Asynchronous expansion of nodes as they are popped from the frontier
    - Concurrent node evaluation for I/O-bound operations (REPL, LLM-as-judge)
    - Compatible with both sync and async node evaluators
    - Maintains the same priority queue semantics as BFTS

    Attributes:
        max_concurrent_expansions (int): Maximum number of nodes to expand concurrently
    """

    def __init__(
        self,
        root_node: Optional[Node | str],
        policy: Callable,
        evaluator: Callable,
        max_concurrent_expansions: int = 8,
        final_decision_mode: Literal[
            "clear_frontier", "native"
        ] = "clear_frontier",
        *args,
        **kwargs,
    ):
        """
        Initialize AsyncBFTS.

        Args:
            root_node: The root node of the search tree
            policy: Function to generate child nodes
            evaluator: Async function to evaluate nodes
            max_concurrent_expansions: Max number of nodes to expand in parallel
            final_decision_mode: How to compute the final answer
            *args, **kwargs: Additional arguments passed to BFTS
        """
        super().__init__(
            root_node=root_node,
            policy=policy,
            evaluator=evaluator,
            final_decision_mode=final_decision_mode,
            *args,
            **kwargs,
        )
        self.max_concurrent_expansions = max_concurrent_expansions
        self._expansion_semaphore = asyncio.Semaphore(max_concurrent_expansions)

    async def async_simulate(
        self,
        expansion_count: int = 5,
        timeout: Optional[float] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ):
        """
        Async version of simulate method that expands nodes asynchronously.

        This method processes nodes from the priority queue (frontier) and expands
        them asynchronously, allowing for significant speedup when using async
        node evaluators.

        Args:
            expansion_count: Number of nodes to expand
            timeout: Maximum time in seconds for the search
            remove_duplicate_children: Whether to remove duplicate children
            termination_encountered_fn: Optional callback when termination node is found
        """
        i = 0
        start_time = time.time()

        logger.debug("Async simulation started.")
        while (i < expansion_count) and self.frontier:
            logger.debug(f"Async expansion no: {i}")
            logger.trace(f"Current frontier:\n{self.frontier}")
            i += 1

            # Get the node that has biggest priority
            _, node_id, current_node = self.pop_from_frontier()

            # Check if termination node is encountered and run termination fn
            if (
                termination_encountered_fn is not None
                and current_node.is_termination_node
            ):
                # Check if async or sync termination function
                if asyncio.iscoroutinefunction(termination_encountered_fn):
                    answer = await termination_encountered_fn(current_node)
                else:
                    answer = termination_encountered_fn(current_node)

                if answer:
                    self.best_answer = answer
                    self.best_answer_reason = "checked_and_true"
                    break

            # Skip if node was already expanded
            if not current_node.is_expandable:
                continue

            # Expand the current node asynchronously
            try:
                if remove_duplicate_children:
                    await self.async_expand_rm_dupes(current_node)
                else:
                    await self.async_expand(current_node)
            except Exception as e:
                logger.error(f"Failed to expand node asynchronously: {e}")
                self.stats_failed_expansion_count += 1
                continue

            # Check timeout
            if timeout is not None:
                duration = time.time() - start_time
                if duration > timeout:
                    logger.warning(
                        f"Reached timelimit, stopping expansion on current node: {current_node}"
                    )
                    return

    async def async_expand(self, node):
        """Async version of expand that additionally pushes children to frontier.

        Uses the BaseMethod.async_expand for async node evaluation, then adds
        children to the frontier priority queue.
        """
        # Run the async expansion
        await super(BFTS, self).async_expand(node)

        # Add remaining children to frontier
        for child in node.children:
            if id(child) not in self.in_frontier:
                self.push_to_frontier(child)

    async def async_expand_rm_dupes(self, node):
        """Async version of expand_rm_dupes that pushes children to frontier.

        Uses the BaseMethod.async_expand_rm_dupes for async node evaluation
        with duplicate removal, then adds children to the frontier priority queue.
        """
        # Run the async expansion with duplicate removal
        await super(BFTS, self).async_expand_rm_dupes(node)

        # Add remaining children to frontier
        for child in node.children:
            if id(child) not in self.in_frontier:
                self.push_to_frontier(child)

    def simulate(
        self,
        expansion_count: int = 5,
        timeout: Optional[float] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ):
        """
        Synchronous wrapper for async simulate.

        This allows AsyncBFTS to be used with existing sync code by
        automatically running the async version in an event loop.

        Args:
            expansion_count: Number of nodes to expand
            timeout: Maximum time in seconds for the search
            remove_duplicate_children: Whether to remove duplicate children
            termination_encountered_fn: Optional callback when termination node is found
        """
        # Try to get running event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, create a task
            logger.warning(
                "AsyncBFTS.simulate() called from within async context. "
                "Consider using async_simulate() directly."
            )
            return loop.create_task(
                self.async_simulate(
                    expansion_count=expansion_count,
                    timeout=timeout,
                    remove_duplicate_children=remove_duplicate_children,
                    termination_encountered_fn=termination_encountered_fn,
                )
            )
        except RuntimeError:  # pragma: no cover
            # No event loop running, create new one
            asyncio.run(
                self.async_simulate(
                    expansion_count=expansion_count,
                    timeout=timeout,
                    remove_duplicate_children=remove_duplicate_children,
                    termination_encountered_fn=termination_encountered_fn,
                )
            )
