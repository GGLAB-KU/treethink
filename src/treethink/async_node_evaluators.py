"""
Async node evaluators for TreeThink.

This module provides async versions of node evaluators that can perform
evaluations in parallel, significantly improving performance for I/O-bound
operations like REPL verification and LLM-as-judge scoring.
"""

import asyncio
from typing import List, Union, Callable, Optional
from abc import ABC, abstractmethod

from loguru import logger
from kimina_client import AsyncKiminaClient
from kimina_client.models import Infotree

import os

import vllm

try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine
except ImportError:
    AsyncEngineArgs = None
    AsyncLLMEngine = None

from treethink.grading import extract_data, split_proof_header
from treethink.methods import BaseMethod, Node
from treethink.node_evaluators import (
    BaseNodeEvaluator,
    LLMAsJudgeNodeEvaluator,
    REPLNodeEvaluator,
    LLM_AS_JUDGE_SYSTEM_PROMPT,
)
from treethink.utils import (
    LeanREPLArgs,
    extract_result,
    NodeEvaluatorArgs,
    ModelArgs,
    SamplingArgs,
)


class AsyncBaseNodeEvaluator(ABC):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name

    @abstractmethod
    async def __call__(
        self, node: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        pass


class AsyncREPLNodeEvaluator(AsyncBaseNodeEvaluator):
    """
    Async version of REPL Node Evaluator.

    Uses AsyncKiminaClient for parallel proof verification.
    """

    def __init__(self, repl_args: LeanREPLArgs, *args, **kwargs):
        super().__init__(name="async_repl_node_evaluator", *args, **kwargs)
        self.repl_args = repl_args
        self.async_client = AsyncKiminaClient(api_url=repl_args.lean_server_url)

    def _is_lean_success(self, entry: dict) -> bool:
        """Check if a Lean verification entry represents a successful verification."""
        if not isinstance(entry, dict):
            return False
        if entry.get("error") is not None:
            return False
        for msg in entry.get("messages", []):
            if msg.get("severity", "").lower() == "error":
                return False
        return True

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
                snips=snips, timeout=self.repl_args.timeout, show_progress=False
            )

            scores = []
            if response and getattr(response, "results", None):
                for result in response.results:
                    # Async client might return different structure than sync?
                    # Assuming similar structure: result has error, response fields
                    if result.error:
                        scores.append(0.0)
                        continue

                    is_successful = self._is_lean_success(result.response)
                    scores.append(1.0 if is_successful else 0.0)
            else:
                # Fallback if no results
                scores = [0.0] * len(nodes)

            return scores

        except Exception as e:
            logger.error(f"Async REPL evaluation failed: {e}")
            logger.debug("Returning default scores of 0.0 for all nodes due to error.")
            return [0.0] * len(nodes)


class AsyncLLMAsJudgeNodeEvaluator(AsyncBaseNodeEvaluator):
    """
    Async version of LLM-as-Judge Node Evaluator.
    """

    def __init__(
        self,
        llm_as_judge_model: Union["AsyncLLMEngine", ModelArgs],
        llm_as_judge_sampling: Union[vllm.SamplingParams, SamplingArgs],
        repl_args: LeanREPLArgs,
        llm_as_judge_system_prompt: str = LLM_AS_JUDGE_SYSTEM_PROMPT,
        prompter: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            name="async_llm_as_judge_node_evaluator", *args, **kwargs
        )
        self.repl_args = repl_args
        self.system_prompt = llm_as_judge_system_prompt

        # Async Lean Client
        self.async_lean_client = AsyncKiminaClient(
            api_url=self.repl_args.lean_server_url
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
        error_message_from_lean: Optional[str] = None,
    ):
        prompt = f"Here you can see the proof so far, the applied tactic, the open goals, and the solved goals.  Depends on these information, you should judge the quality of the tactic application. # Proof So Far:\n{proof_so_far}\n"
        prompt += f"# Applied Tactic:\n{applied_tactic}\n"
        prompt += f"# Open Goals:\n{current_goals}\n"
        prompt += f"# Solved Goals:\n{solved_goals}\n"
        prompt += "If goals are not provided, you should judge the quality of the tactic application based on the proof so far. Put your score in \\boxed{}. "
        if error_message_from_lean:
            prompt += f"# Error Message from Lean:\n{error_message_from_lean}\n"

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
        import uuid
        import time

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

    def parse_proof(self, proof: str):
        # Same as sync
        start = proof.find("import Mathlib")
        end = proof.find("```", start)
        if end == -1 and start == -1:
            res = proof
        elif end == -1:
            res = proof[start:]
        else:
            res = proof[start:end]
        return res

    async def __call__(
        self, nodes: Union[Node, List[Node]], method: BaseMethod
    ) -> List[float]:
        if isinstance(nodes, Node):
            nodes = [nodes]
        if not nodes:
            logger.warning("No nodes provided for AsyncLLMAsJudgeNodeEvaluator, returning [].")
            return []

        # 1. REPL Check
        snips = []
        for node in nodes:
            proof = method.traverse_to_root(node, include_root=True)
            parsed_proof = self.parse_proof(proof)
            snips.append(parsed_proof)


        try:
            response = await self.async_lean_client.check(
                snips=snips,
                timeout=self.repl_args.timeout,
                infotree=Infotree.original,
                show_progress=False,
            )
        except Exception as e:
            logger.error(f"Async Lean check failed: {e}")
            response = None

        # 2. Prepare Prompts
        prompts = []
        for i, node in enumerate(nodes):
            infotree = None
            result_obj = None

            if response and getattr(response, "results", None):
                result_obj = response.results[i]
                infotree = (
                    result_obj.response.get("infotree")
                    if result_obj.response
                    else None
                )

            if infotree:
                header, body = split_proof_header(snips[i])
                intervals = extract_data(infotree, body)
                current_goals = (
                    intervals[-1]["goalsAfter"] if intervals else None
                )
                applied_tactic = intervals[-1]["tactic"] if intervals else None
                solved_goals = (
                    intervals[-1]["goalsBefore"] if intervals else None
                )
                error_msg = None
            else:
                current_goals = None
                applied_tactic = None
                solved_goals = None
                error_msg = (
                    result_obj.response.get("error")
                    if result_obj and result_obj.response
                    else None
                )
                logger.warning(
                    f"No infotree for node {i}: {error_msg}"
                )

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
                logger.debug(f"Answer {idx} indicates error, setting score 10.0")
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
                logger.debug(f"Could not extract numeric score for answer {idx}, using default 10.0")

        final_scores = [score / 20.0 for score in scores]
        logger.debug(f"Final normalized scores: {final_scores}")

        return final_scores




class AsyncNormalizedLengthsNodeEvaluator(AsyncBaseNodeEvaluator):
    """
    Async wrapper for NormalizedLengths (CPU bound, so just wraps).
    """

    def __init__(self, length_norm: float = 0.5, *args, **kwargs):
        super().__init__(
            name="async_normalized_lengths_node_evaluator", *args, **kwargs
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

        while curr.parent:
            if hasattr(curr, "vllm_output") and curr.vllm_output:
                whole_path_cumulative_logprobs += (
                    curr.vllm_output.cumulative_logprob
                )
            curr = curr.parent

        scores = [whole_path_cumulative_logprobs / (L**self.length_norm)]
        logger.debug(
            f"Node level: {L}, whole path cumulative logprobs: {whole_path_cumulative_logprobs}, score: {scores}"
        )
        return scores


# Registry
ASYNC_IMPLEMENTED_ND = {
    "async_repl_node_evaluator": AsyncREPLNodeEvaluator,
    "async_llm_as_judge_node_evaluator": AsyncLLMAsJudgeNodeEvaluator,
    "async_normalized_lengths_node_evaluator": AsyncNormalizedLengthsNodeEvaluator,
}


def get_async_node_evaluator_from_config(
    config: NodeEvaluatorArgs, *args, **kwargs
):
    try:
        func_name = config.func_name
        if not func_name.startswith("async_"):
            logger.warning(
                f"NodeEvaluatorArgs func_name '{func_name}' does not start with 'async_', " 
                + "but we're in async_node_evaluators. Prepending 'async_' to func_name."
            )
            func_name = "async_" + func_name
        return ASYNC_IMPLEMENTED_ND[func_name](*args, **config, **kwargs)
    except KeyError:
        logger.error(f"Async node evaluator not found: {config.func_name}")
        return None
