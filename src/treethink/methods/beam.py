import asyncio
import random
import time
from typing import Callable, List, Optional

from loguru import logger

from ..utils.enums import BestAnswerReason, FinalDecisionMode, TieBreaker
from .base_method import BaseMethod
from .node import Node


class BeamSearch(BaseMethod):
    """
    Beam Search implementation built on top of BaseMethod.
    """

    def __init__(
        self,
        root_node: Optional[Node | str],
        policy: Callable,
        evaluator: Callable,
        beam_width: int = 5,
        max_depth: Optional[int] = None,
        tie_breaker: TieBreaker = TieBreaker.RANDOM,
        final_decision_mode: FinalDecisionMode = FinalDecisionMode.NATIVE,
        *args,
        **kwargs,
    ):
        super().__init__(
            root_node=root_node,
            policy=policy,
            evaluator=evaluator,
            final_decision_mode=final_decision_mode,
        )

        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")

        self.beam_width = int(beam_width)
        self.max_depth = max_depth  # measured in number of expansions from root
        self._current_beam: List[Node] = [self.root_node] if root_node else []
        self._frontier_size: int = len(self._current_beam)
        self._last_depth: int = 0
        self._tie_breaker = tie_breaker

        if self.final_decision_mode == FinalDecisionMode.NATIVE:
            # already set in BaseMethod
            pass
        else:
            logger.warning(
                f"Given {self.final_decision_mode.value} is not supported, "
                + "falling back to `native` implementation."
            )

    def set_root_node(self, root_node):
        _root_node = super().set_root_node(root_node)
        self._current_beam = [_root_node]
        self._frontier_size = 1

    def _select_top_k(self, nodes: List[Node], k: int) -> List[Node]:
        """Select top-k nodes by score. Handles ties per self._tie_breaker."""
        if not nodes:
            return []

        scored = [(n.win_value, i, n, n.level) for i, n in enumerate(nodes)]

        # for reproducibility
        random.seed(42)

        # Sort descending by score; break ties either stably by index or randomly
        if self._tie_breaker == TieBreaker.RANDOM:
            # add small random jitter to break ties reproducibly per call
            jittered = [
                (s + random.random() * 1e-9, i, n, level)
                for (s, i, n, level) in scored
            ]
            jittered.sort(key=lambda x: x[0], reverse=True)
            selected = [n for (_, _, n, _) in jittered[:k]]
        elif self._tie_breaker == TieBreaker.DEEP:
            scored.sort(key=lambda x: (x[0], x[3]), reverse=True)
            selected = [n for (_, _, n, _) in scored[:k]]
        else:
            scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
            selected = [n for (_, _, n, _) in scored[:k]]
        return selected

    def simulate(
        self,
        expansion_count: int = 5,
        timeout: Optional[float] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ):
        """
        Run beam search for up to `expansion_count` layers (from the current
        beam), respecting an optional `timeout` in seconds. Maintains
        self._current_beam and self._frontier_size. Also, stops early if the
        beam yields no new children (all terminals).
        """
        if not self._current_beam:
            logger.warning("Beam is empty, nothing to simulate.")
            return

        layers_to_expand = expansion_count
        depth_limit = self.max_depth

        # If a max_depth is set, cap total depth including already expanded levels
        if depth_limit is not None:
            remaining_depth = max(0, depth_limit - self._last_depth)
            layers_to_expand = min(layers_to_expand, remaining_depth)

        logger.debug("Simulation started.")
        for layer in range(layers_to_expand):
            start_time = time.time()
            logger.debug(f"Expansion count: {layer}")
            if timeout is not None and (time.time() - start_time) >= timeout:
                break

            all_candidates: List[Node] = []
            progressed = False

            # Expand every node in the current beam
            for current_node in self._current_beam:
                if (
                    timeout is not None
                    and (time.time() - start_time) >= timeout
                ):
                    break

                # Check if termination node is encountered and run termination fn
                if (
                    termination_encountered_fn is not None
                    and current_node.is_termination_node
                ):
                    answer = termination_encountered_fn(current_node)
                    if answer:
                        self.best_answer = answer
                        self.best_answer_reason = (
                            BestAnswerReason.CHECKED_AND_TRUE
                        )
                        break

                if current_node.is_expandable:
                    if remove_duplicate_children:
                        self.expand_rm_dupes(current_node)
                    else:
                        self.expand(current_node)

                if current_node.children:
                    progressed = True
                    all_candidates.extend(current_node.children)

            # Safety: if something went wrong, keep the old beam
            if not all_candidates:
                break

            # If none produced children, we reached terminals; keep the best
            # terminals and stop
            new_beam = self._select_top_k(all_candidates, self.beam_width)
            self._current_beam = new_beam
            logger.trace(f"New beam: {self._current_beam}")
            self._frontier_size = len(self._current_beam)
            self._last_depth += 1

            # No new children — fully terminal layer
            if not progressed:
                break

    def make_choice(self, node: Optional[Node] = None) -> Node:
        """
        Choose the best node.
        """
        if not self._current_beam:
            raise RuntimeError(
                "No current beam to choose from. Did you call simulate()?"
            )

        best = self._select_top_k(self._current_beam, 1)
        return best[0]

    def make_exploratory_choice(self, node: Optional[Node] = None) -> Node:
        """
        A slightly exploratory pick: pick the 2nd-best from the current beam if
        available; otherwise best.
        """
        if not self._current_beam:
            raise RuntimeError(
                "No current beam to choose from. Did you call simulate()?"
            )

        top2 = self._select_top_k(
            self._current_beam, min(2, len(self._current_beam))
        )
        return top2[1] if len(top2) > 1 else top2[0]

    def __repr__(self) -> str:
        return self.__str__()

    def _str_fields(self):
        return super()._str_fields() + [
            ("beam_width", self.beam_width),
            ("max_depth", self.max_depth),
            ("tie_breaker", self._tie_breaker),
            ("frontier_size", self._frontier_size),
            ("depth_so_far", self._last_depth),
        ]

    def __str__(self) -> str:
        return super().__str__()


class AsyncBeamSearch(BeamSearch):
    """
    Async version of BeamSearch that supports asynchronous node expansion.

    This implementation leverages async_expand and async_expand_rm_dupes from
    BaseMethod to enable concurrent evaluation of children nodes during beam
    search expansion.

    Key features:
    - Parallel expansion of beam nodes at each depth level
    - Concurrent node evaluation for I/O-bound operations (REPL, LLM-as-judge)
    - Compatible with both sync and async node evaluators

    Attributes:
        max_concurrent_expansions (int): Maximum number of nodes to expand concurrently
    """

    def __init__(
        self,
        root_node: Optional[Node | str],
        policy: Callable,
        evaluator: Callable,
        max_concurrent_expansions: int = 8,
        *args,
        **kwargs,
    ):
        """
        Initialize AsyncBeamSearch.

        Args:
            root_node: The root node of the search tree
            policy: Function to generate child nodes
            evaluator: Async function to evaluate nodes
            max_concurrent_expansions: Max number of nodes to expand in parallel
            *args, **kwargs: Additional arguments passed to BeamSearch
        """
        super().__init__(
            root_node=root_node,
            policy=policy,
            evaluator=evaluator,
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
        Async version of simulate method that expands beam levels asynchronously.

        This method processes all nodes in each beam level concurrently, allowing
        for significant speedup when using async node evaluators.

        Args:
            expansion_count: Number of depth levels to expand
            timeout: Maximum time in seconds for the search
            remove_duplicate_children: Whether to remove duplicate children
            termination_encountered_fn: Optional callback when termination node is found
        """
        if not self._current_beam:
            logger.warning("Beam is empty, nothing to simulate.")
            return

        layers_to_expand = expansion_count
        depth_limit = self.max_depth

        # If a max_depth is set, cap total depth including already expanded levels
        if depth_limit is not None:
            remaining_depth = max(0, depth_limit - self._last_depth)
            layers_to_expand = min(layers_to_expand, remaining_depth)

        start_time = time.time()

        for layer in range(layers_to_expand):
            logger.debug(f"Async expansion layer: {layer}")

            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                logger.warning("Reached timeout, stopping expansion.")
                break

            # Check for termination nodes in current beam
            if termination_encountered_fn is not None:
                for current_node in self._current_beam:
                    if current_node.is_termination_node:
                        # Check if async or sync termination function
                        if asyncio.iscoroutinefunction(
                            termination_encountered_fn
                        ):
                            answer = await termination_encountered_fn(
                                current_node
                            )
                        else:
                            answer = termination_encountered_fn(current_node)

                        if answer:
                            self.best_answer = answer
                            self.best_answer_reason = (
                                BestAnswerReason.CHECKED_AND_TRUE
                            )
                            return

            # Expand all nodes in current beam concurrently
            expandable_nodes = [
                n for n in self._current_beam if n.is_expandable
            ]

            if expandable_nodes:
                await self._expand_beam_async(
                    expandable_nodes, remove_duplicate_children
                )

            # Collect all children from expanded nodes
            all_candidates: List[Node] = []
            for node in self._current_beam:
                if node.children:
                    all_candidates.extend(node.children)

            # Safety: if no candidates, stop
            if not all_candidates:
                logger.debug("No candidates found, stopping expansion.")
                break

            # Select top-k children for next beam
            new_beam = self._select_top_k(all_candidates, self.beam_width)
            self._current_beam = new_beam
            logger.debug(f"New beam size: {len(self._current_beam)}")
            self._frontier_size = len(self._current_beam)
            self._last_depth += 1

    async def _expand_beam_async(
        self, nodes: List[Node], remove_duplicate_children: bool = False
    ):
        """
        Expand all nodes in the beam concurrently.

        Args:
            nodes: List of nodes to expand
            remove_duplicate_children: Whether to remove duplicate children
        """
        # Create tasks for concurrent expansion
        tasks = []
        for node in nodes:
            task = self._expand_single_async(node, remove_duplicate_children)
            tasks.append(task)

        # Execute all expansions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to expand node {i}: {result}")

    async def _expand_single_async(
        self, node: Node, remove_duplicate_children: bool = False
    ):
        """
        Expand a single node asynchronously using semaphore for concurrency control.

        Args:
            node: The node to expand
            remove_duplicate_children: Whether to remove duplicate children
        """
        async with self._expansion_semaphore:
            try:
                # Use the async expand methods from BaseMethod
                if remove_duplicate_children:
                    await self.async_expand_rm_dupes(node)
                else:
                    await self.async_expand(node)

            except Exception as e:
                logger.exception(f"Failed to expand node asynchronously: {e}")
                self.stats_failed_expansion_count += 1

    def simulate(
        self,
        expansion_count: int = 5,
        timeout: Optional[float] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ):
        """
        Synchronous wrapper for async simulate.

        This allows AsyncBeamSearch to be used with existing sync code by
        automatically running the async version in an event loop.

        Args:
            expansion_count: Number of depth levels to expand
            timeout: Maximum time in seconds for the search
            remove_duplicate_children: Whether to remove duplicate children
            termination_encountered_fn: Optional callback when termination node is found
        """
        # Try to get running event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, create a task
            logger.warning(
                "AsyncBeamSearch.simulate() called from within async context. "
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

    def _str_fields(self):
        return super()._str_fields() + [
            (
                "max_concurrent_expansions",
                self.max_concurrent_expansions,
            ),
        ]
