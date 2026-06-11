"""Traditional Monte Carlo Tree Search with rollout and REPL evaluation.

Four phases per iteration:
1. **Select** — walk from root to a leaf using UCB1
2. **Expand** — call the policy on the selected leaf to generate children
3. **Rollout** — for each child, generate a complete formal proof via
   single-shot LLM call
4. **Evaluate** — score each complete proof via a separate rollout evaluator
   (typically a formal language REPL)
5. **Backpropagate** — propagate scores up to the root

This differs from :class:`~treethink.methods.alpha_zero_mcts.AlphaZeroMCTS`
in that the rollout phase generates a *complete* proof (not just a single
next step), which is then verified by a formal REPL.  The main evaluator
(e.g. logprob-based) can be different from the rollout evaluator (e.g.
Lean REPL).
"""

import asyncio
import time
from typing import Callable, List, Optional

from loguru import logger

from ..utils.enums import BestAnswerReason, FinalDecisionMode
from ._uct import best_child_uct
from .base_method import BaseMethod
from .node import Node


class TraditionalMCTS(BaseMethod):
    """Traditional MCTS with rollout + formal REPL evaluation.

    Each iteration:
    1. **Select** a leaf via UCB1
    2. **Expand** — generate candidate next-step children via policy
    3. **Rollout** — for each child, use the LLM to generate a complete
       formal proof (single-shot with high ``max_tokens``)
    4. **Evaluate** — score complete proofs via ``rollout_evaluator``
       (e.g. Lean REPL, Rocq)
    5. **Backpropagate** the scores

    Parameters
    ----------
    root_node : Node | str
    policy : Callable
        Policy for generating child nodes (next-step candidates).
    evaluator : Callable
        Main evaluator for node scoring (used when ``rollout_evaluator``
        is not provided, falling back to AlphaZero-style behavior).
    exploration_weight : float
        UCB1 exploration constant (often √2 ≈ 1.414).
    final_decision_mode : FinalDecisionMode
        How to pick the final answer after search.
    rollout_evaluator : Callable, optional
        A :class:`BaseEvaluator` (or async variant) that scores complete
        rollout proofs.  If ``None``, falls back to ``evaluator`` (behaves
        like AlphaZeroMCTS).
    rollout_max_tokens : int
        ``max_tokens`` for rollout generation.  Default 4096.
    rollout_n : int
        Number of independent rollouts per child.  Scores are averaged
        when > 1.  Default 1.
    rollout_temperature : float
        Sampling temperature for rollout generation.  Default 0.8.
    rollout_top_p : float
        ``top_p`` for rollout generation.  Default 1.0.
    """

    def __init__(
        self,
        root_node: Optional[Node | str],
        policy: Callable,
        evaluator: Callable,
        exploration_weight: float = 0.5,
        final_decision_mode: FinalDecisionMode = FinalDecisionMode.MAXIMIZE_VISITS,
        rollout_evaluator: Optional[Callable] = None,
        rollout_max_tokens: int = 4096,
        rollout_n: int = 1,
        rollout_temperature: float = 0.8,
        rollout_top_p: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            root_node=root_node,
            policy=policy,
            evaluator=evaluator,
            final_decision_mode=final_decision_mode,
        )
        self.exploration_weight = exploration_weight

        # Rollout evaluator — separate from main evaluator
        self._rollout_evaluator = rollout_evaluator

        # Rollout generation params
        self.rollout_max_tokens = rollout_max_tokens
        self.rollout_n = rollout_n
        self.rollout_temperature = rollout_temperature
        self.rollout_top_p = rollout_top_p

        # Determine best_answer strategy
        if self.final_decision_mode == FinalDecisionMode.MAXIMIZE_VALUE:
            self._compute_best_answer = self._best_answer_maximize_value
        elif self.final_decision_mode == FinalDecisionMode.MAXIMIZE_VISITS:
            self._compute_best_answer = self._best_answer_maximize_visits
        elif self.final_decision_mode != FinalDecisionMode.NATIVE:
            logger.warning(
                f"Given {self.final_decision_mode.value} is not supported, "
                + "falling back to `native` implementation."
            )

    # ------------------------------------------------------------------
    # Selection (same UCT as AlphaZeroMCTS)
    # ------------------------------------------------------------------

    def make_choice(self, node: Optional[Node] = None) -> Node:
        """Select a leaf node for expansion using UCB1."""
        node = self.root_node if node is None else node

        if not node.children:
            return node

        while not node.is_expandable and node.children:
            node = best_child_uct(node, self.exploration_weight)

        return node

    def make_exploratory_choice(self):
        logger.warning("Not implemented, falling back to self.make_choice...")
        return self.make_choice(self.root_node)

    # ------------------------------------------------------------------
    # Simulate — main search loop
    # ------------------------------------------------------------------

    def simulate(
        self,
        expansion_count: int = 1,
        timeout: Optional[int] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ) -> None:
        """Run traditional MCTS for ``expansion_count`` iterations.

        Each iteration: select → expand → rollout → evaluate → backpropagate.
        """
        i = 0
        start_time = time.time()
        termination_checked_nodes: List[int] = []

        logger.debug("Traditional MCTS simulation started.")
        while expansion_count is None or i < expansion_count:
            logger.debug(f"Expansion: {i}")
            i += 1

            # Timeout check
            if timeout is not None:
                if time.time() - start_time > timeout:
                    logger.warning("Reached time limit, stopping.")
                    return

            # ---- Select ----
            current_node = self.make_choice(self.root_node)
            logger.trace(f"Selected node: {id(current_node)}")

            # ---- Termination encountered ----
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
                current_node.win_value = float("-inf")

            # ---- Expand + Rollout + Evaluate ----
            if current_node.is_expandable:
                self._expand_with_rollout(
                    current_node,
                    remove_duplicate_children=remove_duplicate_children,
                )

                # ---- Backpropagate ----
                logger.trace("Backpropagating...")
                self._backpropagate_node_win_value(current_node)
            else:
                logger.debug(f"Node not expandable: {current_node}")

    # ------------------------------------------------------------------
    # Expansion + Rollout + Evaluation (combined)
    # ------------------------------------------------------------------

    def _expand_with_rollout(
        self,
        node: Node,
        remove_duplicate_children: bool = False,
    ) -> None:
        """Generate children, run rollouts, and evaluate.

        Steps:
        1. Call ``self.policy`` to generate candidate next-step children.
        2. For each child, generate a complete proof continuation via
           a single-shot LLM call (the *rollout*).
        3. Evaluate the complete proof via ``self._rollout_evaluator``,
           falling back to ``self.evaluator`` on the children themselves
           when no rollout evaluator is configured.
        """
        self.stats_expansion_count += 1

        # ---- Step 1: Generate children ----
        self.policy(node, self)
        logger.trace(f"Policy generated {len(node.children)} child(ren).")

        if not node.children:
            logger.warning(f"Failed to expand node: {node}.")
            self.stats_failed_expansion_count += 1
            return

        if remove_duplicate_children:
            node.remove_duplicate_children()

        # ---- Step 2: Rollout — generate complete proofs ----
        if self._has_rollout_policy():
            for child in node.children:
                rollout_texts = self._rollout_from_child(child)
                child.rollout_output = rollout_texts
        else:
            # No rollout capability → fall back to AlphaZero-style
            logger.debug(
                "No rollout support in policy, "
                "falling back to direct child evaluation."
            )
            children_win_values = self.evaluator(node.children, self)
            for i, win_val in enumerate(children_win_values):
                if win_val is not None:
                    node.children[i].win_value = win_val
            return

        # ---- Step 3: Evaluate rollout results ----
        if self._rollout_evaluator is not None:
            scores = self._rollout_evaluator(node.children, self)
            for i, score in enumerate(scores):
                if score is not None:
                    node.children[i].win_value = score
            logger.trace(f"Rollout scores: {scores}")
        else:
            # Fallback: evaluate children directly
            logger.debug(
                "No rollout evaluator configured, using main evaluator."
            )
            children_win_values = self.evaluator(node.children, self)
            for i, win_val in enumerate(children_win_values):
                if win_val is not None:
                    node.children[i].win_value = win_val

    # ------------------------------------------------------------------
    # Rollout generation
    # ------------------------------------------------------------------

    def _has_rollout_policy(self) -> bool:
        """Check whether the policy supports raw text generation.

        Requires the policy to expose ``model``, ``prompter``,
        ``system_prompt``, ``_tokenizer``, and ``_max_model_len``
        attributes (currently satisfied by :class:`VLLMPolicy`).
        """
        return all(
            hasattr(self.policy, attr)
            for attr in (
                "model",
                "prompter",
                "system_prompt",
                "_tokenizer",
                "_max_model_len",
            )
        )

    def _rollout_from_child(self, child: Node) -> List[str]:
        """Generate a complete proof continuation from a child node.

        Uses the same LLM as the main policy but with rollout-specific
        sampling parameters (higher ``max_tokens``, optional lower
        temperature).

        Returns a list of complete proof texts (length ``self.rollout_n``).
        """
        # Build the same chat prompt as the policy would
        proof_so_far = self.traverse_to_root(child, include_root=True)

        messages = [
            {"role": "system", "content": self.policy.system_prompt},
            {"role": "user", "content": self.root_node.text},
            {"role": "assistant", "content": proof_so_far},
        ]
        prompt = self.policy.prompter(messages)

        # Token budget
        prompt_tokens = len(self.policy._tokenizer.encode(prompt))
        safe_max_tokens = self.policy._max_model_len - prompt_tokens - 20

        if safe_max_tokens <= 0:
            logger.error(
                f"Rollout prompt too long ({prompt_tokens} tokens). "
                "Skipping rollout for this child."
            )
            return []

        effective_max_tokens = min(
            safe_max_tokens,
            self.rollout_max_tokens,
        )
        logger.trace(
            f"Rollout: prompt_tokens={prompt_tokens}, "
            f"max_tokens={effective_max_tokens}, n={self.rollout_n}"
        )

        # Build rollout sampling params
        sampling_params = self._build_rollout_sampling_params(
            max_tokens=effective_max_tokens,
        )

        # Generate (use policy's underlying model directly)
        try:
            output = self.policy.model.generate(
                [prompt],
                sampling_params,
                use_tqdm=False,
            )
        except Exception as e:
            logger.error(f"Rollout generation failed: {e}")
            return []

        # Extract generated texts
        rollout_texts: List[str] = []
        for response in output:
            for completion in response.outputs:
                rollout_texts.append(completion.text)

        return rollout_texts

    def _build_rollout_sampling_params(
        self,
        max_tokens: int,
    ):
        """Create ``SamplingParams`` for rollout generation."""
        import vllm

        return vllm.SamplingParams(
            max_tokens=max_tokens,
            n=self.rollout_n,
            temperature=self.rollout_temperature,
            top_p=self.rollout_top_p,
            include_stop_str_in_output=True,
        )

    # ------------------------------------------------------------------
    # Backpropagation (same as AlphaZeroMCTS)
    # ------------------------------------------------------------------

    def _backpropagate_node_win_value(self, node: Node):
        """Propagate ``win_value`` and ``visits`` from parent to root."""
        val = node.win_value
        while node.parent is not None:
            node = node.parent
            node.win_value += val
            node.visits += 1

    # ------------------------------------------------------------------
    # Best-answer selection
    # ------------------------------------------------------------------

    def _best_answer_maximize_value(self):
        """Select best answer by following max win_value / visits."""
        if not self.root_node.children:
            logger.warning("Root is not expanded, did you call simulate()?")
            return self.root_node

        old_exploration_weight = self.exploration_weight
        self.exploration_weight = 0.0
        _best_answer = super()._compute_native_best_answer()
        self.exploration_weight = old_exploration_weight
        return _best_answer

    def _best_answer_maximize_visits(self):
        """Select best answer by following max visits (win_value tiebreak)."""
        if not self.root_node.children:
            logger.warning("Root is not expanded, did you call simulate()?")
            return self.root_node

        node = self.root_node
        while node.children:
            node = max(node.children, key=lambda n: (n.visits, n.win_value))
        return self.traverse_to_root(node, include_root=True)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def _str_fields(self):
        return super()._str_fields() + [
            ("exploration_weight", self.exploration_weight),
            ("rollout_max_tokens", self.rollout_max_tokens),
            ("rollout_n", self.rollout_n),
            ("rollout_temperature", self.rollout_temperature),
            (
                "rollout_evaluator",
                self._rollout_evaluator,
            ),
        ]


class AsyncTraditionalMCTS(TraditionalMCTS):
    """Async variant of :class:`TraditionalMCTS`.

    Uses ``async_expand`` from :class:`BaseMethod` for concurrent
    child evaluation, plus async rollout generation when the policy
    supports it (e.g. :class:`AsyncVLLMPolicy`).
    """

    def __init__(
        self,
        root_node: Optional[Node | str],
        policy: Callable,
        evaluator: Callable,
        exploration_weight: float = 0.5,
        final_decision_mode: FinalDecisionMode = FinalDecisionMode.MAXIMIZE_VISITS,
        rollout_evaluator: Optional[Callable] = None,
        rollout_max_tokens: int = 4096,
        rollout_n: int = 1,
        rollout_temperature: float = 0.8,
        rollout_top_p: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            root_node=root_node,
            policy=policy,
            evaluator=evaluator,
            exploration_weight=exploration_weight,
            final_decision_mode=final_decision_mode,
            rollout_evaluator=rollout_evaluator,
            rollout_max_tokens=rollout_max_tokens,
            rollout_n=rollout_n,
            rollout_temperature=rollout_temperature,
            rollout_top_p=rollout_top_p,
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
        """Async version of :meth:`TraditionalMCTS.simulate`."""
        i = 0
        start_time = time.time()
        termination_checked_nodes: List[int] = []

        logger.debug("Async Traditional MCTS simulation started.")
        while expansion_count is None or i < expansion_count:
            logger.debug(f"Async Traditional MCTS expansion: {i}")
            i += 1

            if timeout is not None:
                if time.time() - start_time > timeout:
                    logger.warning("Reached time limit, stopping.")
                    return

            current_node = self.make_choice(self.root_node)
            logger.trace(f"Selected node: {id(current_node)}")

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
                    await self._async_expand_with_rollout(
                        current_node,
                        remove_duplicate_children=remove_duplicate_children,
                    )
                    self._backpropagate_node_win_value(current_node)
                except Exception as e:
                    logger.error(f"Failed to expand node asynchronously: {e}")
                    self.stats_failed_expansion_count += 1
            else:
                logger.warning(f"Node not expandable: {current_node}")

    async def _async_expand_with_rollout(
        self,
        node: Node,
        remove_duplicate_children: bool = False,
    ) -> None:
        """Async expansion with rollout generation and evaluation."""
        import inspect

        self.stats_expansion_count += 1

        # Step 1: Generate children (async or sync)
        if inspect.iscoroutinefunction(self.policy):
            await self.policy(node, self)
        else:
            self.policy(node, self)

        if not node.children:
            self.stats_failed_expansion_count += 1
            return

        if remove_duplicate_children:
            node.remove_duplicate_children()

        # Step 2: Rollout
        if self._has_rollout_policy():
            for child in node.children:
                rollout_texts = await self._async_rollout_from_child(child)
                child.rollout_output = rollout_texts
        else:
            scores = await self.evaluator(node.children, self)
            for i, win_val in enumerate(scores):
                if win_val is not None:
                    node.children[i].win_value = win_val
            return

        # Step 3: Evaluate rollout results
        if self._rollout_evaluator is not None:
            if inspect.iscoroutinefunction(self._rollout_evaluator):
                scores = await self._rollout_evaluator(node.children, self)
            else:
                scores = self._rollout_evaluator(node.children, self)

            for i, score in enumerate(scores):
                if score is not None:
                    node.children[i].win_value = score
        else:
            if inspect.iscoroutinefunction(self.evaluator):
                scores = await self.evaluator(node.children, self)
            else:
                scores = self.evaluator(node.children, self)
            for i, win_val in enumerate(scores):
                if win_val is not None:
                    node.children[i].win_value = win_val

    async def _async_rollout_from_child(self, child: Node) -> List[str]:
        """Async rollout generation using the policy's model."""
        proof_so_far = self.traverse_to_root(child, include_root=True)

        messages = [
            {"role": "system", "content": self.policy.system_prompt},
            {"role": "user", "content": self.root_node.text},
            {"role": "assistant", "content": proof_so_far},
        ]
        prompt = self.policy.prompter(messages)

        prompt_tokens = len(self.policy._tokenizer.encode(prompt))
        safe_max_tokens = self.policy._max_model_len - prompt_tokens - 20

        if safe_max_tokens <= 0:
            logger.error(f"Rollout prompt too long ({prompt_tokens} tokens).")
            return []

        effective_max_tokens = min(safe_max_tokens, self.rollout_max_tokens)

        sampling_params = self._build_rollout_sampling_params(
            max_tokens=effective_max_tokens,
        )

        try:
            # Use async engine if available; fall back to sync
            if hasattr(self.policy, "engine"):
                # AsyncLLMEngine path
                import uuid

                results: List[str] = []

                async def generate_one(idx: int):
                    request_id = f"rollout_{idx}_{uuid.uuid4().hex[:8]}"
                    final_text = ""
                    try:
                        async for out in self.policy.engine.generate(
                            prompt,
                            sampling_params,
                            request_id=request_id,
                        ):
                            if out.finished:
                                final_text = out.outputs[0].text
                                break
                    except Exception as e:
                        logger.error(f"Async rollout {idx} failed: {e}")
                    return final_text

                tasks = [generate_one(i) for i in range(self.rollout_n)]
                results = await asyncio.gather(*tasks)
                return results
            else:
                # Sync model — run in thread pool
                loop = asyncio.get_running_loop()
                output = await loop.run_in_executor(
                    None,
                    lambda: self.policy.model.generate(
                        [prompt],
                        sampling_params,
                        use_tqdm=False,
                    ),
                )
                rollout_texts: List[str] = []
                for response in output:
                    for completion in response.outputs:
                        rollout_texts.append(completion.text)
                return rollout_texts
        except Exception as e:
            logger.error(f"Async rollout generation failed: {e}")
            return []

    def simulate(
        self,
        expansion_count: int = 1,
        timeout: Optional[int] = None,
        remove_duplicate_children: bool = False,
        termination_encountered_fn: Optional[Callable[[Node], str]] = None,
    ) -> None:
        """Synchronous wrapper that delegates to :meth:`async_simulate`."""
        try:
            loop = asyncio.get_running_loop()
            logger.warning(
                "AsyncTraditionalMCTS.simulate() called from within async "
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
