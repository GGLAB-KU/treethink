import asyncio
import time
from typing import Callable, List, Literal, Optional

import numpy as np
from loguru import logger

from .base_method import BaseMethod
from .node import Node


class MCTS(BaseMethod):
    """
    Implements the Monte Carlo Tree Search (MCTS) algorithm.

    This class orchestrates the search process by selecting, expanding, simulating,
    and backpropagating through the tree to find the best answer.

    Attributes:
        root_node (Node): The root node of the search tree.
        child_finder: Function to generate child nodes.
        node_evaluator: Function to evaluate node quality.
    """

    def __init__(
        self,
        root_node: Optional[Node | str],
        child_finder: Callable,
        node_evaluator: Callable,
        exploration_weight: float = 0.5,
        final_decision_mode: Literal[
            "maximize_visits", "maximize_value", "native"
        ] = "maximize_visits",
        *args,
        **kwargs,
    ):
        super().__init__(
            root_node=root_node,
            child_finder=child_finder,
            node_evaluator=node_evaluator,
            final_decision_mode=final_decision_mode,
        )
        # Exploration weight for UCT algorithm
        self.exploration_weight = exploration_weight

        # Set best_answer function based on the given final_decision_mode
        if self.final_decision_mode == "maximize_value":
            self._compute_best_answer = self._best_answer_maximize_value
        elif self.final_decision_mode == "maximize_visits":
            self._compute_best_answer = self._best_answer_maximize_visits
        elif self.final_decision_mode == "native":
            # already set in BaseMethod
            pass
        else:
            logger.warning(
                f"Given {self.final_decision_mode} is not supported, "
                + "falling back to `native` implementation."
            )

    def make_choice(self, node: Optional[Node] = None):
        """
        Selects a node to expand or simulate based on UCT algorithm.

        Args:
            node (Node): The node from which to begin selection. Default to
            self.root_node

        Returns:
            Node: The selected node for expansion or simulation.
        """
        node = self.root_node if node is None else node

        # Edge Case: a leaf or the root node
        if not node.children:
            return node

        while not node.is_expandable and node.children:
            node = self._best_child_uct(node)

        return node

    def make_exploratory_choice(self):
        logger.warning("Not implemented, falling back to self.make_choice...")
        return self.make_choice(self.root_node)

    def simulate(
        self,
        expansion_count: int = 1,
        timeout: Optional[int] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ) -> None:
        """
        Simulates the MCTS process for a given number of iterations.

        Args:
            expansion_count (int): Number of expansion iterations to perform.

        """

        i = 0
        start_time = time.time()

        termination_checked_nodes = []

        logger.debug(f"Simulation started.")
        while expansion_count is None or i < expansion_count:
            logger.debug(f"Expansion: {i}")
            i += 1

            # Check timeout
            if timeout is not None:
                curr_time = time.time()
                duration = curr_time - start_time
                if duration > timeout:
                    logger.warning(
                        "Reached time limit, stopping expansion on current node."
                    )
                    return

            # Select node for expansion
            current_node = self.make_choice(self.root_node)
            logger.trace(f"Current node: {current_node}")
            # Check if termination node is encountered and run termination fn
            if (
                termination_encountered_fn is not None
                and current_node.is_termination_node
                and id(current_node) not in termination_checked_nodes
            ):
                logger.trace(f"Termination node encountered: {current_node}")
                answer = termination_encountered_fn(current_node)
                termination_checked_nodes.append(id(current_node))
                if answer:
                    self.best_answer = answer
                    self.best_answer_reason = "checked_and_true"
                    break

                # If the proof is wrong, set nodes's win_value to -inf to avoid
                # selecting it again.
                # NOTE(burak): I previously thought about punishing the entire
                # path to the root, but that would be too harsh. Let's just
                # punish the node itself for now.
                current_node.win_value = float("-inf")

            # Expansion logic based on preferences on duplicate children
            # handling and backpropagation strategy
            if current_node.is_expandable:
                if remove_duplicate_children:
                    self.expand_rm_dupes(current_node)
                else:
                    self.expand(current_node)

                # Backpropagate parent's win_value if we expanded
                logger.trace("Backpropagating node win value...")
                self._backpropagate_node_win_value(current_node)
            else:
                logger.debug(f"Node not expandable: {current_node}")

    def _best_child_uct(self, node: Node) -> Node:
        """
        Selects the best child of the given node based on the UCT.

        Returns:
            Node: The child node with the highest computed weight.
        """
        choices_weights: List[float] = []
        for child in node.children:
            if child.visits == 0:
                weight: float = float("inf")  # explore unvisited nodes first
            else:
                exploitation_term: float = child.win_value / child.visits
                exploration_term: float = self.exploration_weight * np.sqrt(
                    np.log(node.visits) / child.visits
                )
                weight = exploitation_term + exploration_term

            choices_weights.append(weight)
        
        logger.trace(
            f"UCT weights for children of node {id(node)}: {choices_weights}"
        )
        return node.children[np.argmax(choices_weights)]

    def _best_answer_maximize_value(self):
        """Select best answer by following max win_value / visits.
        Overriding BaseMethod's best_answer to discard exploration_term in the
        UCT score.
        """
        if not self.root_node.children:
            logger.warning("Root is not expanded, did you call simulate()?")
            return self.root_node

        # Briefly modify exploration_weight to be 0.0
        old_exploration_weight = self.exploration_weight
        self.exploration_weight = 0.0

        # best_answer calls self.make_choice which uses exploration_weight
        _best_answer = super()._compute_native_best_answer()
        self.exploration_weight = old_exploration_weight

        return _best_answer

    def _best_answer_maximize_visits(self):
        """Select best answer by following max visits."""

        if not self.root_node.children:
            logger.warning("Root is not expanded, did you call simulate()?")
            return self.root_node

        # Start from top
        node = self.root_node

        # Always select max visits, if there is equality look win_value for
        # tie-breaking. TODO(burak): could there be a better approach?
        while node.children:
            node = max(node.children, key=lambda n: (n.visits, n.win_value))

        return self.traverse_to_root(node, include_root=True)

    def _backpropagate_node_win_value(self, node: Node):
        val = node.win_value

        while node.parent is not None:
            # start updating from parent as we don't want node to be updated
            node = node.parent
            node.win_value += val
            node.visits += 1


class AsyncMCTS(MCTS):
    """
    Async version of MCTS that supports asynchronous node expansion.
    
    This implementation leverages async_expand and async_expand_rm_dupes from
    BaseMethod to enable asynchronous evaluation of children nodes during MCTS
    tree search.
    
    Key features:
    - Asynchronous expansion of selected nodes in MCTS iterations
    - Async node evaluation for I/O-bound operations (REPL, LLM-as-judge)
    - Compatible with both sync and async node evaluators
    - Maintains the same UCT selection and backpropagation semantics as MCTS
    
    Note: MCTS is inherently sequential (select -> expand -> backpropagate), so
    the main benefit of async comes from concurrent evaluation of multiple children
    during the expansion phase, not from parallelizing MCTS iterations themselves.
    """
    
    def __init__(
        self,
        root_node: Optional[Node | str],
        child_finder: Callable,
        node_evaluator: Callable,
        exploration_weight: float = 0.5,
        final_decision_mode: Literal[
            "maximize_visits", "maximize_value", "native"
        ] = "maximize_visits",
        *args,
        **kwargs,
    ):
        """
        Initialize AsyncMCTS.
        
        Args:
            root_node: The root node of the search tree
            child_finder: Function to generate child nodes
            node_evaluator: Async function to evaluate nodes
            exploration_weight: Weight for exploration term in UCT
            final_decision_mode: How to compute the final answer
            *args, **kwargs: Additional arguments passed to MCTS
        """
        super().__init__(
            root_node=root_node,
            child_finder=child_finder,
            node_evaluator=node_evaluator,
            exploration_weight=exploration_weight,
            final_decision_mode=final_decision_mode,
            *args,
            **kwargs,
        )

    async def async_simulate(
        self,
        expansion_count: int = 1,
        timeout: Optional[int] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ) -> None:
        """
        Async version of simulate method that expands nodes asynchronously.
        
        This method follows the standard MCTS loop (select -> expand -> backpropagate)
        but uses async expansion for I/O-bound operations like node evaluation.
        
        Note: The MCTS iterations themselves remain sequential as each iteration
        depends on the backpropagated values from the previous iteration. The
        async benefit comes from concurrent evaluation of multiple children during
        each expansion.
        
        Args:
            expansion_count: Number of MCTS iterations to perform
            timeout: Maximum time in seconds for the search
            remove_duplicate_children: Whether to remove duplicate children
            termination_encountered_fn: Optional callback when termination node is found
        """
        i = 0
        start_time = time.time()

        termination_checked_nodes = []

        logger.debug("Async MCTS simulation started.")
        while expansion_count is None or i < expansion_count:
            logger.debug(f"Async MCTS expansion: {i}")
            i += 1

            # Check timeout
            if timeout is not None:
                curr_time = time.time()
                duration = curr_time - start_time
                if duration > timeout:
                    logger.warning(
                        "Reached time limit, stopping expansion on current node."
                    )
                    return

            # Select node for expansion using UCT
            current_node = self.make_choice(self.root_node)
            logger.trace(f"Current node selected: {current_node}")
            
            # Check if termination node is encountered and run termination fn
            if (
                termination_encountered_fn is not None
                and current_node.is_termination_node
                and id(current_node) not in termination_checked_nodes
            ):
                logger.trace(f"Termination node encountered: {current_node}")
                
                # Check if async or sync termination function
                if asyncio.iscoroutinefunction(termination_encountered_fn):
                    answer = await termination_encountered_fn(current_node)
                else:
                    answer = termination_encountered_fn(current_node)
                
                termination_checked_nodes.append(id(current_node))
                
                if answer:
                    self.best_answer = answer
                    self.best_answer_reason = "checked_and_true"
                    break

                # If the proof is wrong, set node's win_value to -inf to avoid
                # selecting it again.
                current_node.win_value = float("-inf")

            # Expansion logic with async node evaluation
            if current_node.is_expandable:
                try:
                    if remove_duplicate_children:
                        await self.async_expand_rm_dupes(current_node)
                    else:
                        await self.async_expand(current_node)

                    # Backpropagate parent's win_value if we expanded
                    logger.trace("Backpropagating node win value...")
                    self._backpropagate_node_win_value(current_node)
                    
                except Exception as e:
                    logger.error(f"Failed to expand node asynchronously: {e}")
                    self.stats_failed_expansion_count += 1
            else:
                logger.warning(f"Node not expandable: {current_node}")

    async def async_expand(self, node):
        """Async version of expand that uses async node evaluation.
        
        Uses the BaseMethod.async_expand for async node evaluation with
        concurrent evaluation of multiple children.
        """
        await super(MCTS, self).async_expand(node)

    async def async_expand_rm_dupes(self, node):
        """Async version of expand_rm_dupes with async node evaluation.
        
        Uses the BaseMethod.async_expand_rm_dupes for async node evaluation
        with duplicate removal and concurrent evaluation of children.
        """
        await super(MCTS, self).async_expand_rm_dupes(node)

    def simulate(
        self,
        expansion_count: int = 1,
        timeout: Optional[int] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ) -> None:
        """
        Synchronous wrapper for async simulate.
        
        This allows AsyncMCTS to be used with existing sync code by
        automatically running the async version in an event loop.
        
        Args:
            expansion_count: Number of MCTS iterations to perform
            timeout: Maximum time in seconds for the search
            remove_duplicate_children: Whether to remove duplicate children
            termination_encountered_fn: Optional callback when termination node is found
        """
        # Try to get running event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, create a task
            logger.warning(
                "AsyncMCTS.simulate() called from within async context. "
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
