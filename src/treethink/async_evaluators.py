"""
Async node evaluators for TreeThink.

This module provides async versions of node evaluators that can perform
evaluations in parallel, significantly improving performance for I/O-bound
operations like REPL verification and LLM-as-judge scoring.
"""

import asyncio
import math
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import vllm
from loguru import logger

try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine
except ImportError:
    AsyncEngineArgs = None
    AsyncLLMEngine = None

from treethink.client_factory import create_async_client
from treethink.clients.cache import ProofCache
from treethink.evaluators import (
    LLM_AS_JUDGE_SYSTEM_PROMPT,
    LLM_AS_JUDGE_SYSTEM_PROMPT_PAIRWISE,
    ProofLevelRewardEvaluator,
    StateLevelRewardEvaluator,
    _RMaxNoveltyTracker,
)
from treethink.methods import BaseMethod, Node
from treethink.utils import (
    ClientArgs,
    EvaluatorArgs,
    ModelArgs,
    SamplingArgs,
    calculate_logprobs,
    extract_result,
)
from treethink.utils.enums import FormalLanguage

# Language-agnostic regex: captures content inside any ```<lang>\n...\n``` fence
_RE_PROOF_FENCE = re.compile(r"```(?:\w+|\n)\s*((?:.|\n)*?)```")


class AsyncBaseEvaluator(ABC):
    """Abstract base for async node scoring evaluators.

    Like :class:`BaseEvaluator` but with ``async __call__`` for non-blocking
    evaluation.  Subclasses must implement ``async __call__(self, node, method)``
    returning a list of float scores.

    To create a custom async evaluator, subclass this, implement
    ``async __call__``, and register in the ``AsyncEvaluatorType`` enum.
    """

    def __init__(self, name: str, *args, **kwargs):
        self.name = name

    def _format_str_value(self, value):
        if callable(value):
            return getattr(value, "__name__", value.__class__.__name__)
        return repr(value)

    def _str_fields(self):
        return [("name", self.name)]

    def __str__(self) -> str:
        fields = ", ".join(
            f"{name}={self._format_str_value(value)}"
            for name, value in self._str_fields()
        )
        return f"{self.__class__.__name__}({fields})"

    __repr__ = __str__

    @abstractmethod
    async def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        pass


class AsyncLeanREPLEvaluator(AsyncBaseEvaluator):
    """Async version of REPL Node Evaluator.

    Uses :func:`create_async_client` to build the language-appropriate
    async client (defaults to Lean 4).  If *cache* is provided the client
    is wrapped with :class:`AsyncCachedClient`.
    """

    def __init__(
        self,
        client_args: Optional[ClientArgs] = None,
        cache: Optional[ProofCache] = None,
        language: FormalLanguage = FormalLanguage.LEAN4,
        *args,
        **kwargs,
    ):
        super().__init__(name="async_lean_repl_evaluator", *args, **kwargs)
        if client_args is None:
            client_args = ClientArgs()
        self.client_args = client_args
        self.language = language
        self.async_client = create_async_client(
            language, client_args, cache=cache
        )

    async def __call__(
        self, nodes: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(nodes, Node):
            nodes = [nodes]

        if not nodes:
            return []

        try:
            snips = [node.answer for node in nodes]

            response = await self.async_client.check(
                snips=snips,
                timeout=self.client_args.timeout,
                show_progress=False,
                batch_size=self.client_args.batch_size,
                max_workers=self.client_args.num_proc,
            )

            scores = []
            if response and getattr(response, "results", None):
                for result in response.results:
                    if result.error:
                        scores.append(0.0)
                        continue
                    scores.append(
                        1.0
                        if self.async_client.is_success_response(
                            result.response
                        )
                        else 0.0
                    )
            else:
                scores = [0.0] * len(nodes)

            return scores

        except Exception as e:
            logger.error(f"Async Lean REPL evaluation failed: {e}")
            return [0.0] * len(nodes)

    def _str_fields(self):
        return super()._str_fields() + [
            ("client_args", self.client_args),
            ("language", self.language),
        ]


class AsyncJudgeEvaluator(AsyncBaseEvaluator):
    """Async version of LLM-as-Judge Evaluator.

    Language-agnostic: uses :func:`create_async_client` to build the
    proof-assistant client (defaults to Lean 4).
    """

    def __init__(
        self,
        llm_as_judge_model: Union["AsyncLLMEngine", ModelArgs],
        llm_as_judge_sampling: Union[vllm.SamplingParams, SamplingArgs],
        client_args: Optional[ClientArgs] = None,
        cache: Optional[ProofCache] = None,
        llm_as_judge_system_prompt: str = LLM_AS_JUDGE_SYSTEM_PROMPT,
        prompter: Optional[Callable] = None,
        language: FormalLanguage = FormalLanguage.LEAN4,
        *args,
        **kwargs,
    ):
        super().__init__(name="async_judge_evaluator", *args, **kwargs)
        if client_args is None:
            client_args = ClientArgs()
        self.client_args = client_args
        self.language = language
        self.system_prompt = llm_as_judge_system_prompt

        # Language-agnostic async proof-assistant client
        self.async_client = create_async_client(
            language, client_args, cache=cache
        )

        # Initialize Model (AsyncLLMEngine)
        if AsyncLLMEngine is not None and isinstance(
            llm_as_judge_model, AsyncLLMEngine
        ):
            self.model = llm_as_judge_model
        elif isinstance(llm_as_judge_model, ModelArgs):
            visible_devices = kwargs.get("llm_as_judge_visible_devices", "1")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
            engine_args = AsyncEngineArgs(**llm_as_judge_model)
            self.model = AsyncLLMEngine.from_engine_args(engine_args)
        else:
            self.model = llm_as_judge_model

        # Sampling
        if isinstance(llm_as_judge_sampling, SamplingArgs):
            self.sampling_params = vllm.SamplingParams(**llm_as_judge_sampling)
        else:
            self.sampling_params = (
                llm_as_judge_sampling or vllm.SamplingParams()
            )

        # Prompter (Simplified)
        self.prompter = prompter

    def _prepare_judge_messages(
        self,
        proof_so_far: str,
        current_goals: Optional[str] = None,
        applied_tactic: Optional[str] = None,
        solved_goals: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        prompt = f"Here you can see the proof so far, the applied tactic, the open goals, and the solved goals.  Depends on these information, you should judge the quality of the tactic application. # Proof So Far:\n{proof_so_far}\n"
        prompt += f"# Applied Tactic:\n{applied_tactic}\n"
        prompt += f"# Open Goals:\n{current_goals}\n"
        prompt += f"# Solved Goals:\n{solved_goals}\n"
        prompt += "If goals are not provided, you should judge the quality of the tactic application based on the proof so far. Put your score in \\boxed{}. "
        if error_message:
            prompt += (
                f"# Error Message from the Proof Assistant:\n{error_message}\n"
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Apply prompter
        # If we don't have a prompter, do simple formatting
        if self.prompter:
            result = self.prompter(messages)
            return result
        else:
            result = "\n".join([m["content"] for m in messages])
            return result

    async def generate_judge_answers(self, prompts: List[str]) -> List[str]:
        """Generate answers for multiple prompts concurrently."""
        import time
        import uuid

        results = [None] * len(prompts)
        logger.debug(f"Generating judge answers for {len(prompts)} prompts.")

        async def generate_one(idx, prompt):
            request_id = f"judge_{idx}_{time.time()}_{uuid.uuid4().hex[:8]}"
            final_text = "ERROR: Failed"
            try:
                async for output in self.model.generate(
                    prompt, self.sampling_params, request_id=request_id
                ):
                    if output.finished:
                        final_text = output.outputs[0].text
                        break
            except Exception as e:
                logger.error(f"Judge generation failed (idx={idx}): {e}")

            results[idx] = final_text

        tasks = [generate_one(i, p) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)
        return results

    def parse_proof(self, proof: str, pattern=None) -> str:
        """Extract proof content from code-fence delimiters.

        Uses a language-agnostic regex that captures the content inside
        any ```<lang>\\n...\\n``` fence.  Falls back to the raw proof
        string if no fence is found.
        """
        pattern = pattern or _RE_PROOF_FENCE
        matches = re.findall(pattern, proof)
        if not matches:
            return proof
        return matches[-1]

    async def __call__(
        self, nodes: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(nodes, Node):
            nodes = [nodes]
        if not nodes:
            logger.warning(
                "No nodes provided for AsyncJudgeEvaluator, returning []."
            )
            return []

        # 1. REPL Check
        snips = []
        for node in nodes:
            proof = method.traverse_to_root(node, include_root=True)
            parsed_proof = self.parse_proof(proof)
            snips.append(parsed_proof)

        try:
            response = await self.async_client.check(
                snips=snips,
                timeout=self.client_args.timeout,
                show_progress=False,
            )
        except Exception as e:
            logger.error(f"Async REPL check failed: {e}")
            response = None

        # 2. Prepare Prompts
        prompts = []
        for i, node in enumerate(nodes):
            # Extract proof state via the language-agnostic client interface
            result_response = (
                response.results[i].response
                if response and getattr(response, "results", None)
                else None
            )
            proof_state = self.async_client.extract_proof_state(
                proof_string=snips[i],
                response=result_response,
            )

            current_goals = proof_state.open_goals
            applied_tactic = proof_state.applied_tactic
            solved_goals = proof_state.closed_goals
            error_msg = proof_state.error_message

            judge_prompt = self._prepare_judge_messages(
                snips[i],
                current_goals,
                applied_tactic,
                solved_goals,
                error_msg,
            )
            prompts.append(judge_prompt)

        # 3. Generate Judge Answers
        logger.debug("Generating judge answers...")
        answers = await self.generate_judge_answers(prompts)
        logger.trace(f"Judge answers: {answers}")

        # 4. Parse Scores
        scores = []
        for idx, ans in enumerate(answers):
            if ans.startswith("ERROR"):
                scores.append(10.0)
                logger.debug(
                    f"Answer {idx} indicates error, setting score 10.0"
                )
                logger.trace(f"Answer {idx}\nContent: {ans}")
                continue

            extracted = extract_result(ans)
            logger.debug(f"Extracted score from answer {idx}: {extracted}")
            if extracted.isnumeric():
                val = float(extracted)
                score_val = max(0.0, min(20.0, val))
                logger.debug(f"Numeric score for answer {idx}: {score_val}")
                scores.append(score_val)
            else:
                scores.append(10.0)
                logger.debug(
                    f"Could not extract numeric score for answer {idx}, using default 10.0"
                )

        final_scores = [score / 20.0 for score in scores]
        logger.debug(f"Final normalized scores: {final_scores}")

        return final_scores

    def _str_fields(self):
        return super()._str_fields() + [
            ("model", self.model.__class__.__name__),
            ("sampling_params", self.sampling_params),
            ("client_args", self.client_args),
            ("language", self.language),
            ("system_prompt", self.system_prompt),
        ]


class AsyncNormLenEvaluator(AsyncBaseEvaluator):
    """
    Async wrapper for NormalizedLengths.
    """

    def __init__(self, length_norm: float = 0.5, *args, **kwargs):
        super().__init__(
            name="async_normalized_lengths_evaluator", *args, **kwargs
        )
        self.length_norm = length_norm

    async def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        # Logic is identical to sync, no I/O involved.
        if isinstance(node, list):
            evaluations = []
            for n in node:
                # Recursive async call? No, just loop
                res = await self.__call__(n, method)
                evaluations.extend(res)
            return evaluations

        L = node.level
        whole_path_cumulative_logprobs = 0.0

        curr = node
        # Safety for root
        if not curr.parent:
            return [0.0]

        # Traverse back and sum the cumulative log probs.
        if hasattr(node.vllm_output, "cumulative_logprob"):
            while node.parent:
                whole_path_cumulative_logprobs += (
                    node.vllm_output.cumulative_logprob
                )
                node = node.parent
        # If cumulative_logprob is not available, calculate logprobs by hand.
        else:
            while node.parent:
                whole_path_cumulative_logprobs += calculate_logprobs([node])[0]
                node = node.parent

        scores = [whole_path_cumulative_logprobs / (L**self.length_norm)]
        logger.debug(
            f"Node level: {L}, whole path cumulative logprobs: {whole_path_cumulative_logprobs}, score: {scores}"
        )
        return scores

    def _str_fields(self):
        return super()._str_fields() + [("length_norm", self.length_norm)]


class AsyncCumulativeLogprobEvaluator(AsyncBaseEvaluator):
    """Async version of cumulative log-probability evaluator.

    Pure computation — no I/O. Returns cumulative logprobs from
    ``node.vllm_output.cumulative_logprob`` or computes them via
    ``calculate_logprobs()`` if unavailable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            name="async_cumulative_logprob_evaluator", *args, **kwargs
        )

    async def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(node, Node):
            node = [node]

        if all([n.vllm_output for n in node]):
            if hasattr(node[0].vllm_output, "cumulative_logprob"):
                return [n.vllm_output.cumulative_logprob for n in node]
            return calculate_logprobs(node)
        else:
            if node[0].parent:
                logger.warning(f"No vllm_output found in node(s): {node}")
            return [0.0] * len(node)

    def _str_fields(self):
        return super()._str_fields()


class AsyncNormLenProbEvaluator(AsyncBaseEvaluator):
    """Async Normalized Lengths evaluator using probabilities.

    Like :class:`AsyncNormLenEvaluator` but uses probabilities
    (exponentiated logprobs) instead of cumulative logprobs.
    Pure computation — no I/O.
    """

    def __init__(self, length_norm: float = 0.5, *args, **kwargs):
        self.length_norm = length_norm
        super().__init__(
            name="async_normalized_lengths_probs_evaluator", *args, **kwargs
        )

    async def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(node, list):
            evaluations = []
            for n in node:
                res = await self.__call__(n, method)
                evaluations.extend(res)
            return evaluations

        L = node.level
        whole_path_cumulative_probs = 0.0

        if not node.parent:
            return [0.0]

        curr = node
        if hasattr(curr.vllm_output, "cumulative_logprob"):
            while curr.parent:
                whole_path_cumulative_probs += math.exp(
                    curr.vllm_output.cumulative_logprob
                )
                curr = curr.parent
        else:
            while curr.parent:
                whole_path_cumulative_probs += math.exp(
                    calculate_logprobs([curr])[0]
                )
                curr = curr.parent

        return [whole_path_cumulative_probs / (L**self.length_norm)]

    def _str_fields(self):
        return super()._str_fields() + [("length_norm", self.length_norm)]


class AsyncTournamentEvaluator(AsyncBaseEvaluator):
    """Async single-elimination tournament evaluation between sibling nodes.

    Runs a bracket-style tournament where an async LLM judge picks winners
    pairwise among siblings.  Losers are scored by their elimination round,
    and the winner receives the highest score.  All scores normalized to [0, 1].

    Language-agnostic: uses :func:`create_async_client` to build the
    proof-assistant client (defaults to Lean 4).
    """

    def __init__(
        self,
        llm_as_judge_model: Union["AsyncLLMEngine", ModelArgs],
        llm_as_judge_sampling: Union[vllm.SamplingParams, SamplingArgs],
        client_args: Optional[ClientArgs] = None,
        cache: Optional[ProofCache] = None,
        llm_as_judge_system_prompt: str = LLM_AS_JUDGE_SYSTEM_PROMPT_PAIRWISE,
        prompter: Optional[Callable] = None,
        shuffle_bracket: bool = True,
        language: FormalLanguage = FormalLanguage.LEAN4,
        *args,
        **kwargs,
    ):
        super().__init__(
            name="async_pairwise_tournament_evaluator", *args, **kwargs
        )
        if client_args is None:
            client_args = ClientArgs()
        self.client_args = client_args
        self.shuffle_bracket = shuffle_bracket
        self.language = language
        self.system_prompt = llm_as_judge_system_prompt

        # Language-agnostic async proof-assistant client
        self.async_client = create_async_client(
            language, client_args, cache=cache
        )

        # Initialize Model (AsyncLLMEngine)
        if AsyncLLMEngine is not None and isinstance(
            llm_as_judge_model, AsyncLLMEngine
        ):
            self.model = llm_as_judge_model
        elif isinstance(llm_as_judge_model, ModelArgs):
            visible_devices = kwargs.get("llm_as_judge_visible_devices", "1")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
            engine_args = AsyncEngineArgs(**llm_as_judge_model)
            self.model = AsyncLLMEngine.from_engine_args(engine_args)
        else:
            self.model = llm_as_judge_model

        # Sampling
        if isinstance(llm_as_judge_sampling, SamplingArgs):
            self.sampling_params = vllm.SamplingParams(**llm_as_judge_sampling)
        else:
            self.sampling_params = (
                llm_as_judge_sampling or vllm.SamplingParams()
            )

        self.prompter = prompter

    def _prepare_pairwise_judge_messages(
        self,
        proof_a: str,
        proof_b: str,
        info_a: Optional[dict] = None,
        info_b: Optional[dict] = None,
    ):
        """Create a prompt for LLM to judge between two proofs."""
        lang_tag = self.language.value
        prompt = "You are comparing two proof attempts. Choose which one is better.\n\n"

        prompt += "# Proof A:\n"
        prompt += f"```{lang_tag}\n{proof_a}\n```\n"
        if info_a:
            if info_a.get("applied_tactic"):
                prompt += f"Applied Tactic: {info_a['applied_tactic']}\n"
            if info_a.get("current_goals"):
                prompt += f"Open Goals: {info_a['current_goals']}\n"
            if info_a.get("solved_goals"):
                prompt += f"Solved Goals: {info_a['solved_goals']}\n"
            if info_a.get("error_message"):
                prompt += f"Error: {info_a['error_message']}\n"

        prompt += "\n# Proof B:\n"
        prompt += f"```{lang_tag}\n{proof_b}\n```\n"
        if info_b:
            if info_b.get("applied_tactic"):
                prompt += f"Applied Tactic: {info_b['applied_tactic']}\n"
            if info_b.get("current_goals"):
                prompt += f"Open Goals: {info_b['current_goals']}\n"
            if info_b.get("solved_goals"):
                prompt += f"Solved Goals: {info_b['solved_goals']}\n"
            if info_b.get("error_message"):
                prompt += f"Error: {info_b['error_message']}\n"

        prompt += "\nWhich proof is better? Answer with either 'A' or 'B' in \\boxed{}."

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        if self.prompter:
            return self.prompter(messages)
        else:
            result = "\n".join([m["content"] for m in messages])
            return result

    def _extract_proof_info(
        self, snip: str, response, result_idx: int
    ) -> Optional[dict]:
        """Extract tactic and goal information from the proof-assistant response."""
        if not response or not getattr(response, "results", None):
            return None

        result_obj = response.results[result_idx]
        proof_state = self.async_client.extract_proof_state(
            proof_string=snip,
            response=result_obj.response,
        )

        info = {}
        if proof_state.applied_tactic is not None:
            info["applied_tactic"] = proof_state.applied_tactic
        if proof_state.open_goals is not None:
            info["current_goals"] = proof_state.open_goals
        if proof_state.closed_goals is not None:
            info["solved_goals"] = proof_state.closed_goals
        if proof_state.error_message is not None:
            info["error_message"] = proof_state.error_message

        return info if info else None

    def parse_proof(self, proof: str):
        start = proof.find("import Mathlib")
        end = proof.find("```", start)
        if end == -1 and start == -1:
            return proof
        elif end == -1:
            return proof[start:]
        return proof[start:end]

    async def _batch_compare_pairs(
        self,
        pairs: List[Tuple[int, int]],
        nodes: List[Node],
        method: BaseMethod,
        snips: List[str],
        proof_infos: List[Optional[dict]],
    ) -> List[int]:
        """Compare pairs of nodes in batch and return winner indices."""
        if not pairs:
            return []

        import time
        import uuid

        # Prepare messages for all pairs
        messages = []
        for idx_a, idx_b in pairs:
            message = self._prepare_pairwise_judge_messages(
                snips[idx_a],
                snips[idx_b],
                proof_infos[idx_a],
                proof_infos[idx_b],
            )
            messages.append(message)

        # Batch generate
        results = [None] * len(messages)

        async def generate_one(idx, prompt):
            request_id = (
                f"tournament_{idx}_{time.time()}_{uuid.uuid4().hex[:8]}"
            )
            final_text = "ERROR: Failed"
            try:
                async for output in self.model.generate(
                    prompt, self.sampling_params, request_id=request_id
                ):
                    if output.finished:
                        final_text = output.outputs[0].text
                        break
            except Exception as e:
                logger.error(
                    f"Tournament judge generation failed (idx={idx}): {e}"
                )

            results[idx] = final_text

        tasks = [generate_one(i, p) for i, p in enumerate(messages)]
        await asyncio.gather(*tasks)

        # Extract winners
        winners = []
        for i, (idx_a, idx_b) in enumerate(pairs):
            answer = results[i] if i < len(results) else "ERROR: Missing"
            if answer.startswith("ERROR"):
                logger.warning(
                    f"Judge error for pair ({idx_a}, {idx_b}), defaulting to A"
                )
                winners.append(idx_a)
                continue

            extracted = extract_result(answer)
            if extracted == "NO_BOXED_STRING_FOUND":
                logger.warning(
                    f"No boxed answer for pair ({idx_a}, {idx_b}), defaulting to A"
                )
                winners.append(idx_a)
            elif extracted.strip().upper() == "A":
                winners.append(idx_a)
            elif extracted.strip().upper() == "B":
                winners.append(idx_b)
            else:
                logger.warning(
                    f"Invalid answer '{extracted}' for pair ({idx_a}, {idx_b}), defaulting to A"
                )
                winners.append(idx_a)

        return winners

    async def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(node, Node):
            node = [node]

        n = len(node)
        if n == 1:
            return [1.0]

        # Prepare all proofs
        snips = []
        for i in range(n):
            proof_so_far = method.traverse_to_root(node[i], include_root=True)
            proof_so_far = self.parse_proof(proof=proof_so_far)
            snips.append(proof_so_far)

        # Get Lean info for all proofs in batch
        try:
            response = await self.async_client.check(
                snips=snips,
                timeout=self.client_args.timeout,
                show_progress=False,
            )
        except Exception as e:
            logger.error(f"Async REPL check failed: {e}")
            response = None

        # Extract proof-state info for all nodes
        proof_infos = []
        for i in range(n):
            info = self._extract_proof_info(snips[i], response, i)
            proof_infos.append(info)

        # Initialize bracket
        bracket_indices = list(range(n))
        if self.shuffle_bracket:
            import random

            random.shuffle(bracket_indices)
            logger.info(f"Shuffled bracket order: {bracket_indices}")

        # Pad to next power of 2 if needed
        n_padded = 2 ** math.ceil(math.log2(n))
        while len(bracket_indices) < n_padded:
            bracket_indices.append(-1)

        scores = [0.0] * n
        round_num = 1
        current_bracket = bracket_indices.copy()

        while len(current_bracket) > 1:
            pairs = []
            valid_pairs = []
            pair_to_valid_idx = {}

            for i in range(0, len(current_bracket), 2):
                idx_a = current_bracket[i]
                idx_b = current_bracket[i + 1]

                if idx_a == -1 and idx_b == -1:
                    pairs.append((-1, -1))
                elif idx_a == -1:
                    pairs.append((idx_a, idx_b))
                elif idx_b == -1:
                    pairs.append((idx_a, idx_b))
                else:
                    pair_to_valid_idx[len(pairs)] = len(valid_pairs)
                    valid_pairs.append((idx_a, idx_b))
                    pairs.append((idx_a, idx_b))

            if valid_pairs:
                winners_from_comparison = await self._batch_compare_pairs(
                    valid_pairs, node, method, snips, proof_infos
                )
            else:
                winners_from_comparison = []

            winners = []
            comparison_idx = 0

            for pair_idx, (idx_a, idx_b) in enumerate(pairs):
                if idx_a == -1 and idx_b == -1:
                    winners.append(-1)
                elif idx_a == -1:
                    winners.append(idx_b)
                elif idx_b == -1:
                    winners.append(idx_a)
                else:
                    winner_idx = winners_from_comparison[comparison_idx]
                    loser_idx = idx_b if winner_idx == idx_a else idx_a
                    scores[loser_idx] = float(round_num)
                    winners.append(winner_idx)
                    comparison_idx += 1

            current_bracket = winners
            round_num += 1

        winner_idx = current_bracket[0]
        if winner_idx != -1:
            scores[winner_idx] = float(round_num)

        max_score = float(round_num)
        normalized_scores = [s / max_score for s in scores]

        logger.info(
            f"Tournament complete. Final scores: {scores} -> normalized: {normalized_scores}"
        )

        return normalized_scores

    def _str_fields(self):
        return super()._str_fields() + [
            ("model", self.model.__class__.__name__),
            ("sampling_params", self.sampling_params),
            ("client_args", self.client_args),
            ("language", self.language),
            ("shuffle_bracket", self.shuffle_bracket),
            ("system_prompt", self.system_prompt),
        ]


class AsyncRocqEvaluator(AsyncBaseEvaluator):
    """Async evaluation of whole Rocq proofs via rocq-ml-server.

    Wraps the synchronous :class:`RocqClient` using ``asyncio.to_thread``.
    Returns 1.0 if the proof closes, 0.0 otherwise.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        timeout: Optional[float] = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

        from treethink.clients.coq.rocq import RocqClient

        self._client = RocqClient(
            host=self.host,
            port=self.port,
        )

        super().__init__(name="async_rocq_evaluator")

    async def __call__(
        self, code: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        snippets = [code] if isinstance(code, str) else code
        results: List[float] = []

        for snippet in snippets:
            response = await asyncio.to_thread(
                self._client.verify_whole_proof, snippet
            )
            results.append(1.0 if response.get("proof_finished") else 0.0)

        return results[0] if isinstance(code, str) else results

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AsyncRocqEvaluator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _str_fields(self):
        return super()._str_fields() + [
            ("host", self.host),
            ("port", self.port),
            ("timeout", self.timeout),
        ]


class AsyncRMaxTSEvaluator(AsyncBaseEvaluator):
    """Async twin of :class:`~treethink.evaluators.RMaxTSEvaluator`.

    Awards the RMax intrinsic-reward novelty signal (``1[new node]``).  The
    novelty bookkeeping is CPU-only set membership, so this simply shares the
    same :class:`_RMaxNoveltyTracker` behind an ``async __call__``.
    """

    def __init__(
        self,
        state_fn: Optional[Callable[[Node, BaseMethod], str]] = None,
        novel_reward: float = 1.0,
        seen_reward: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(name="async_rmaxts_evaluator", *args, **kwargs)
        self._tracker = _RMaxNoveltyTracker(
            state_fn=state_fn,
            novel_reward=novel_reward,
            seen_reward=seen_reward,
        )

    def reset(self) -> None:
        """Clear the seen-state set (call between proof attempts)."""
        self._tracker.reset()

    async def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(node, Node):
            node = [node]
        return self._tracker.rewards(node, method)

    def _str_fields(self):
        return super()._str_fields() + [
            ("novel_reward", self._tracker.novel_reward),
            ("seen_reward", self._tracker.seen_reward),
            ("state_fn", self._tracker.state_fn),
        ]


class _AsyncRewardModelEvaluator(AsyncBaseEvaluator):
    """Async wrapper around a sync reward-model evaluator.

    vLLM's pooling ``encode`` is synchronous, so the sync evaluator is run
    in a worker thread (``asyncio.to_thread``); the main benefit is not
    blocking the event loop while the reward model scores a batch.
    """

    _sync_cls = None  # set by subclasses

    def __init__(self, reward_model, name: str, *args, **kwargs):
        super().__init__(name=name)
        self._sync = self._sync_cls(reward_model, *args, **kwargs)

    async def __call__(self, node, method) -> List[float]:
        return await asyncio.to_thread(self._sync, node, method)

    def _str_fields(self):
        return self._sync._str_fields()


class AsyncProofLevelRewardEvaluator(_AsyncRewardModelEvaluator):
    """Async twin of :class:`~treethink.evaluators.ProofLevelRewardEvaluator`."""

    _sync_cls = ProofLevelRewardEvaluator

    def __init__(self, reward_model, *args, **kwargs):
        super().__init__(
            reward_model,
            name="async_proof_level_reward_evaluator",
            *args,
            **kwargs,
        )


class AsyncStateLevelRewardEvaluator(_AsyncRewardModelEvaluator):
    """Async twin of :class:`~treethink.evaluators.StateLevelRewardEvaluator`."""

    _sync_cls = StateLevelRewardEvaluator

    def __init__(self, reward_model, *args, **kwargs):
        super().__init__(
            reward_model,
            name="async_state_level_reward_evaluator",
            *args,
            **kwargs,
        )


class AsyncEvaluatorType(Enum):
    ASYNC_LEAN_REPL = AsyncLeanREPLEvaluator
    ASYNC_LLM_AS_JUDGE = AsyncJudgeEvaluator
    ASYNC_NORMALIZED_LENGTHS = AsyncNormLenEvaluator
    ASYNC_CUMULATIVE_LOGPROB = AsyncCumulativeLogprobEvaluator
    ASYNC_TOURNAMENT = AsyncTournamentEvaluator
    ASYNC_NORMALIZED_LENGTHS_PROBS = AsyncNormLenProbEvaluator
    ASYNC_ROCQ = AsyncRocqEvaluator
    ASYNC_RMAXTS = AsyncRMaxTSEvaluator
    ASYNC_PROOF_LEVEL_REWARD = AsyncProofLevelRewardEvaluator
    ASYNC_STATE_LEVEL_REWARD = AsyncStateLevelRewardEvaluator

    @classmethod
    def from_str(cls, name: str) -> "AsyncEvaluatorType":
        normalized = name.strip().lower().replace("-", "_")
        for suffix in ("_evaluator", "_policy"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
        for member in cls:
            if normalized == member.name.lower():
                return member
        valid_keys = [member.name.lower() for member in cls]
        raise ValueError(
            f"Unknown async evaluator '{name}'. Valid options: {valid_keys}"
        )

    def initialize(self, *args, **kwargs) -> Callable:
        return self.value(*args, **kwargs)


ASYNC_IMPLEMENTED_EVALUATORS = AsyncEvaluatorType
ASYNC_EVALUATORS = [member.name.lower() for member in AsyncEvaluatorType]


def get_async_evaluator_from_config(config: EvaluatorArgs, *args, **kwargs):
    func_name = config.func_name
    if not func_name.startswith("async_"):
        logger.warning(
            f"EvaluatorArgs func_name '{func_name}' does not start with 'async_', "
            + "but we're in async_evaluators. Prepending 'async_' to func_name."
        )
        func_name = "async_" + func_name

    return AsyncEvaluatorType.from_str(func_name).initialize(
        *args, **dict(vars(config)), **kwargs
    )
