"""
Fully Asynchronous Sampler for TreeThink.

This sampler leverages AsyncMCTS, AsyncChildExpander, and AsyncNodeEvaluator
to achieve high-throughput parallel inference on multiple datapoints.
"""

import asyncio
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from loguru import logger
from tqdm.asyncio import tqdm_asyncio

# Import async components
from treethink.expanders import (
    get_expander_from_config,  # Has async support
)

from treethink import (
    EvaluatorArgs,
    InferenceTimeArgs,
    PolicyArgs,
    TreeThink,
    get_inference_time_method,
)  # Wrapper class
from treethink.async_evaluators import get_async_evaluator_from_config

# Check for AsyncEngine
try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine
except ImportError:
    AsyncLLMEngine = None
    AsyncEngineArgs = None


class AsyncSampler:
    """
    High-throughput async sampler using pure async stack.
    """

    def __init__(
        self,
        expander_args: PolicyArgs,
        evaluator_args: EvaluatorArgs,
        inference_time_args: InferenceTimeArgs,
        prompter: Optional[Callable] = None,
        max_concurrent_datapoints: int = 16,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1,
        visible_devices: str = "0",
        task_name: str = "async_generate",
    ):
        self.expander_args = expander_args
        self.evaluator_args = evaluator_args
        self.inference_time_args = inference_time_args
        self.max_concurrent_datapoints = max_concurrent_datapoints
        self.prompter = prompter or self._default_prompter
        self.task_name = task_name
        # Initialize Shared Async Engine
        logger.info("Initializing Shared AsyncLLMEngine...")

        if AsyncLLMEngine is None:
            raise ImportError(
                "vLLM AsyncLLMEngine not found. Please upgrade vLLM."
            )

        # Initialize Shared Components
        # Use get_expander_from_config - it supports async expanders
        self.shared_expander = get_expander_from_config(
            expander_args, prompter=self.prompter
        )

        # For Judge, if it uses the same model, we can reuse the engine
        # If Judge is a different model, we would need a separate engine (and GPU memory!)
        # Here we assume Judge uses the SAME model for simplicity or it's handled externally.
        # If evaluator needs a model, we pass the SAME engine.
        self.shared_evaluator = get_async_evaluator_from_config(
            evaluator_args,
            # We try to pass model if the evaluator needs it (like LLMAsJudge)
            prompter=self.prompter,
        )

        logger.success(
            f"AsyncSampler initialized with max concurrency: {max_concurrent_datapoints} and task name: {self.task_name}"
        )

    def _default_prompter(self, messages: List[Dict]) -> str:
        return "\n".join([m["content"] for m in messages])

    def _message_creator(
        self,
        system_prompt: str,
        datapoint: dict,
        data_key: List[str],
        prompt_format: Optional[str] = None,
    ) -> List[dict]:
        try:
            values = [datapoint[k] for k in data_key]
            if prompt_format:
                content = prompt_format.format(*values)
            else:
                content = " ".join(str(v) for v in values)

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        except KeyError as e:
            raise KeyError(
                f"Key {e} not found in datapoint keys: {list(datapoint.keys())}"
            )

    def _check_if_processed(self, problem_id: str, graph_path: str) -> bool:
        if not graph_path or not problem_id:
            return False
        safe_id = re.sub(r"[^\w\-_]", "_", str(problem_id))
        search_path = Path(graph_path)
        if not search_path.exists():
            return False
        return any(search_path.glob(f"tree_{safe_id}_*.txt"))

    async def _process_single_datapoint(
        self,
        datapoint: dict,
        system_prompt: str,
        data_key: List[str],
        prompt_format: Optional[str],
        skip_if_exists: bool,
    ) -> dict:
        problem_id = (
            datapoint.get("id") or datapoint.get("problem_id") or "unknown"
        )

        try:
            # Skip check
            if skip_if_exists and self.inference_time_args.graph_path:
                if self._check_if_processed(
                    problem_id, self.inference_time_args.graph_path
                ):
                    logger.info(f"Skipping {problem_id}")
                    datapoint["skipped"] = True
                    return datapoint

            # Prepare Input
            messages = self._message_creator(
                system_prompt, datapoint, data_key, prompt_format
            )
            prompt = self.prompter(messages)

            logger.debug(f"Starting async_simulate for {problem_id}")

            method = get_inference_time_method(
                inference_time_config=self.inference_time_args,
                root_node=None,
                expander=self.shared_expander,
                evaluator=self.shared_evaluator,
            )

            wrapper = TreeThink(method, self.inference_time_args)

            # Use the wrapper's async generation which handles simulation,
            # REPL checks, and saving the tree safely.
            result = await wrapper.async_generate(prompt, problem_id=problem_id)

            # Extract output from TreeThinkOutputs
            if result.outputs:
                datapoint["output"] = [result.outputs[0].text]
            else:
                datapoint["output"] = [""]

            if result.graph_stats:
                datapoint["graph_stats"] = result.graph_stats

            return datapoint

        except Exception as e:
            import traceback

            logger.error(f"Error processing {problem_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            datapoint["error"] = str(e)
            datapoint["output"] = [f"ERROR: {e}"]
            return datapoint

    async def async_inference(
        self,
        data: List[dict],
        system_prompt: str,
        data_key: List[str] = ["problem"],
        prompt_format: Optional[str] = None,
        skip_if_exists: bool = True,
        show_progress: bool = True,
    ):
        semaphore = asyncio.Semaphore(self.max_concurrent_datapoints)

        async def worker(dp):
            async with semaphore:
                return await self._process_single_datapoint(
                    dp, system_prompt, data_key, prompt_format, skip_if_exists
                )

        tasks = [worker(dp) for dp in data]

        logger.info(f"Starting async inference on {len(data)} items...")
        if show_progress:
            results = await tqdm_asyncio.gather(*tasks, desc="Processing")
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def run(self, *args, **kwargs):
        """Sync entry point."""
        return asyncio.run(self.async_inference(*args, **kwargs))
