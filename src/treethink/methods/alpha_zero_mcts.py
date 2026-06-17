"""AlphaZero-style Monte Carlo Tree Search with UCB1 selection.

No rollout phase — evaluation happens directly on expanded children
via a learned value function (evaluator).  Identical to the standard
AlphaZero approach.

Four phases per iteration:
1. **Select** — walk from root to a leaf using UCB1:
   UCB = win_value/visits + exploration_weight * sqrt(ln(N) / n_i)
2. **Expand** — call the policy on the selected leaf
3. **Evaluate** — score each child via the evaluator
4. **Backpropagate** — propagate scores up to the root

Attributes:
    root_node (Node): The root node of the search tree.
    policy: Function to generate child nodes.
    evaluator: Function to evaluate node quality.

Async variant: :class:`AsyncAlphaZeroMCTS`
"""

import asyncio
import time
from typing import Callable, Optional

from loguru import logger

from ..utils.enums import BestAnswerReason, FinalDecisionMode
from ._uct import best_child_uct
from .base_method import BaseMethod
from .node import Node


class AlphaZeroMCTS(BaseMethod):
    """AlphaZero-style Monte Carlo Tree Search — no rollout, direct evaluation.

    Four phases per iteration:
    1. **Select** — walk from root to a leaf using UCB1
    2. **Expand** — call the policy on the selected leaf
    3. **Evaluate** — score each child via the evaluator
    4. **Backpropagate** — propagate scores up to the root

    For traditional MCTS (with rollout + REPL evaluation), see
    :class:`~treethink.methods.traditional_mcts.TraditionalMCTS`.
    """

    def __init__(
        self,
        root_node: Optional[Node | str],
        policy: Callable,
        evaluator: Callable,
        exploration_weight: float = 0.5,
        final_decision_mode: FinalDecisionMode = FinalDecisionMode.MAXIMIZE_VISITS,
        *args,
        **kwargs,
    ):
        super().__init__(
            root_node=root_node,
            policy=policy,
            evaluator=evaluator,
            final_decision_mode=final_decision_mode,
        )
        # Exploration weight for UCT algorithm
        self.exploration_weight = exploration_weight

        # Set best_answer function based on the given final_decision_mode
        if self.final_decision_mode == FinalDecisionMode.MAXIMIZE_VALUE:
            self._compute_best_answer = self._best_answer_maximize_value
        elif self.final_decision_mode == FinalDecisionMode.MAXIMIZE_VISITS:
            self._compute_best_answer = self._best_answer_maximize_visits
        elif self.final_decision_mode == FinalDecisionMode.NATIVE:
            # already set in BaseMethod
            pass
        else:
            logger.warning(
                f"Given {self.final_decision_mode.value} is not supported, "
                + "falling back to `native` implementation."
            )

    def make_choice(self, node: Optional[Node] = None):
        """Select a node to expand using the UCT algorithm.

        Args:
            node: The node from which to begin selection. Defaults to
                self.root_node.

        Returns:
            Node: The selected node for expansion.
        """
        node = self.root_node if node is None else node

        # Edge Case: a leaf or the root node
        if not node.children:
            return node

        while not node.is_expandable and node.children:
            node = best_child_uct(node, self.exploration_weight)

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
        """Run the AlphaZero MCTS search for ``expansion_count`` iterations.

        Args:
            expansion_count: Number of expansion iterations to perform.
            timeout: Maximum seconds before stopping early.
            remove_duplicate_children: Whether to deduplicate by text.
            termination_encountered_fn: Callback for termination nodes.
        """
        i = 0
        start_time = time.time()

        termination_checked_nodes = []

        logger.debug("AlphaZero MCTS simulation started.")
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
                    self.best_answer_reason = BestAnswerReason.CHECKED_AND_TRUE
                    break

                # If the proof is wrong, set node's win_value to -inf to avoid
                # selecting it again.
                current_node.win_value = float("-inf")

            # Expansion logic
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

    def _best_answer_maximize_value(self):
        """Select best answer by following max win_value / visits.

        Temporarily sets exploration_weight to 0.0 so that the UCT
        selection greedily follows the highest expected value.
        """
        if not self.root_node.children:
            logger.warning("Root is not expanded, did you call simulate()?")
            return self.root_node

        old_exploration_weight = self.exploration_weight
        self.exploration_weight = 0.0

        _best_answer = super()._compute_native_best_answer()
        self.exploration_weight = old_exploration_weight

        return _best_answer

    def _best_answer_maximize_visits(self):
        """Select best answer by following max visits at each level.

        Tie-breaks on win_value when visits are equal.
        """
        if not self.root_node.children:
            logger.warning("Root is not expanded, did you call simulate()?")
            return self.root_node

        node = self.root_node
        while node.children:
            node = max(node.children, key=lambda n: (n.visits, n.win_value))

        return self.traverse_to_root(node, include_root=True)

    def _backpropagate_node_win_value(self, node: Node):
        """Propagate win_value and visits from *parent* of ``node`` to root."""
        val = node.win_value

        while node.parent is not None:
            # start updating from parent as we don't want node to be updated
            node = node.parent
            node.win_value += val
            node.visits += 1

    def _str_fields(self):
        return super()._str_fields() + [
            ("exploration_weight", self.exploration_weight),
        ]


class AsyncAlphaZeroMCTS(AlphaZeroMCTS):
    """Async variant of :class:`AlphaZeroMCTS`.

    Uses asynchronous node expansion (``async_expand`` and
    ``async_expand_rm_dupes`` from :class:`BaseMethod`) for concurrent
    evaluation of children during the expansion phase.

    Maintains the same UCT selection and backpropagation semantics as
    :class:`AlphaZeroMCTS`.
    """

    def __init__(
        self,
        root_node: Optional[Node | str],
        policy: Callable,
        evaluator: Callable,
        exploration_weight: float = 0.5,
        final_decision_mode: FinalDecisionMode = FinalDecisionMode.MAXIMIZE_VISITS,
        *args,
        **kwargs,
    ):
        super().__init__(
            root_node=root_node,
            policy=policy,
            evaluator=evaluator,
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
        """Async version of simulate — concurrent child evaluation.

        Each iteration selects via UCT, expands asynchronously, and
        backpropagates.  The main benefit comes from concurrent evaluation
        of multiple children during expansion, not from parallel iterations.
        """
        i = 0
        start_time = time.time()

        termination_checked_nodes = []

        logger.debug("Async AlphaZero MCTS simulation started.")
        while expansion_count is None or i < expansion_count:
            logger.debug(f"Async AlphaZero MCTS expansion: {i}")
            i += 1

            if timeout is not None:
                curr_time = time.time()
                duration = curr_time - start_time
                if duration > timeout:
                    logger.warning(
                        "Reached time limit, stopping expansion on current node."
                    )
                    return

            current_node = self.make_choice(self.root_node)
            logger.trace(f"Current node selected: {current_node}")

            if (
                termination_encountered_fn is not None
                and current_node.is_termination_node
                and id(current_node) not in termination_checked_nodes
            ):
                logger.trace(f"Termination node encountered: {current_node}")

                if asyncio.iscoroutinefunction(termination_encountered_fn):
                    answer = await termination_encountered_fn(current_node)
                else:
                    answer = termination_encountered_fn(current_node)

                termination_checked_nodes.append(id(current_node))

                if answer:
                    self.best_answer = answer
                    self.best_answer_reason = BestAnswerReason.CHECKED_AND_TRUE
                    break

                current_node.win_value = float("-inf")

            if current_node.is_expandable:
                try:
                    if remove_duplicate_children:
                        await self.async_expand_rm_dupes(current_node)
                    else:
                        await self.async_expand(current_node)

                    logger.trace("Backpropagating node win value...")
                    self._backpropagate_node_win_value(current_node)

                except Exception as e:
                    logger.error(f"Failed to expand node asynchronously: {e}")
                    self.stats_failed_expansion_count += 1
            else:
                logger.warning(f"Node not expandable: {current_node}")

    async def async_expand(self, node):
        """Async expand using ``BaseMethod.async_expand``."""
        await super(AlphaZeroMCTS, self).async_expand(node)

    async def async_expand_rm_dupes(self, node):
        """Async expand with dedup using ``BaseMethod.async_expand_rm_dupes``."""
        await super(AlphaZeroMCTS, self).async_expand_rm_dupes(node)

    def simulate(
        self,
        expansion_count: int = 1,
        timeout: Optional[int] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ) -> None:
        """Synchronous wrapper that delegates to ``async_simulate``."""
        try:
            loop = asyncio.get_running_loop()
            logger.warning(
                "AsyncAlphaZeroMCTS.simulate() called from within async "
                "context. Consider using async_simulate() directly."
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
            asyncio.run(
                self.async_simulate(
                    expansion_count=expansion_count,
                    timeout=timeout,
                    remove_duplicate_children=remove_duplicate_children,
                    termination_encountered_fn=termination_encountered_fn,
                )
            )

    def _str_fields(self):
        return super()._str_fields() + [
            (
                "max_concurrent_expansions",
                getattr(self, "max_concurrent_expansions", None),
            ),
        ]
