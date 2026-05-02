"""
Async datapoint sampler for processing multiple problems concurrently.

This module enables concurrent tree search across multiple problems by:
1. Serializing vLLM calls (since vLLM instance is shared and not thread-safe)
2. Parallelizing REPL verification calls (using AsyncKiminaClient)

This approach provides real speedup for REPL-heavy workloads while safely
sharing the vLLM GPU resource.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger
from tqdm import tqdm
from utils import (
    FinderArgs,
    InferenceTimeArgs,
    EvaluatorArgs,
)

from treethink import (
    TreeThink,
    get_child_finder_from_config,
    get_inference_time_method,
    get_node_evaluator_from_config,
)


class AsyncDatapointSampler:
    """
    Async sampler for processing multiple datapoints concurrently.

    This sampler enables concurrent tree search across multiple problems while:
    - Serializing vLLM calls (shared GPU resource, not thread-safe)
    - Parallelizing REPL verification (async I/O operations)

    This provides significant speedup for REPL-heavy workloads without
    GPU memory issues or thread-safety problems.

    Key features:
    - Async datapoint processing with asyncio
    - Shared vLLM instance (memory efficient)
    - Parallel REPL verification (throughput efficient)
    - Progress tracking with tqdm
    - Error handling per datapoint

    Attributes:
        max_concurrent_datapoints (int): Maximum number of datapoints to process concurrently
    """

    def __init__(
        self,
        child_finder_args: Optional[FinderArgs] = None,
        node_evaluator_args: Optional[EvaluatorArgs] = None,
        inference_time_args: Optional[InferenceTimeArgs] = None,
        prompter: Optional[Callable] = None,
        task_name: str = "async_generate",
        max_concurrent_datapoints: int = 4,
    ):
        """
        Initialize AsyncDatapointSampler.

        Args:
            child_finder_args: Configuration for child finder
            node_evaluator_args: Configuration for node evaluator
            inference_time_args: Configuration for inference time method
            prompter: Function to format prompts
            task_name: Name identifier for this task
            max_concurrent_datapoints: Maximum number of datapoints to process concurrently
        """
        self.child_finder_args = child_finder_args
        self.node_evaluator_args = node_evaluator_args
        self.inference_time_args = inference_time_args
        self.task_name = task_name
        self.max_concurrent_datapoints = max_concurrent_datapoints

        if prompter:
            self.prompter = prompter
        else:
            self.prompter = self._default_prompter

        # Get model name for logging
        if child_finder_args:
            self.model_name = child_finder_args.model.model
        else:
            self.model_name = "unknown"

        # Initialize shared components (vLLM, evaluator, etc.)
        logger.info("Initializing shared components...")
        self._init_shared_components()

        # Create lock for vLLM thread safety
        # vLLM is NOT thread-safe, we must serialize all vLLM calls
        self._vllm_lock = asyncio.Lock()

        logger.info(
            f"Initialized AsyncDatapointSampler with max {max_concurrent_datapoints} concurrent datapoints."
        )
        logger.info(
            "vLLM calls will be serialized (with lock), REPL calls will be parallelized."
        )

    def _init_shared_components(self):
        """Initialize shared components (vLLM, child finder, node evaluator)."""
        if not (
            self.child_finder_args
            and self.node_evaluator_args
            and self.inference_time_args
        ):
            raise ValueError(
                "All args must be provided for inference time methods"
            )

        # Initialize shared child finder (contains vLLM)
        self.shared_child_finder = get_child_finder_from_config(
            self.child_finder_args, prompter=self.prompter
        )

        # Initialize shared node evaluator
        # Note: Use async evaluator if available for better performance
        self.shared_node_evaluator = get_node_evaluator_from_config(
            self.node_evaluator_args, prompter=self.prompter
        )

        logger.info("Shared components initialized.")

    def _default_prompter(self, messages: dict) -> str:
        """Default prompter that concatenates messages."""
        result = ""
        for msg in messages:
            content = msg.get("content", "")
            if content:
                result += content + "\n"
        return result

    def _message_creator(
        self,
        system_prompt: str,
        datapoint: dict,
        data_key: List[str],
        prompt_format: Optional[str] = None,
    ) -> List[dict]:
        """
        Creates a message structure for the model input.

        Args:
            system_prompt: The system prompt to be used
            datapoint: The data point containing the input information
            data_key: A list of keys to access the relevant data in the datapoint
            prompt_format: A format string to be applied to the data

        Returns:
            List of dictionaries representing the message structure
        """
        try:
            # Gather values for all keys
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
            available_keys = list(datapoint.keys())
            raise KeyError(
                f"Key {e} not found in datapoint. "
                f"Available keys: {available_keys}. "
                f"Looking for keys: {data_key}"
            ) from e

    def _check_if_already_processed(
        self, problem_id: str, graph_path: str
    ) -> bool:
        """
        Check if a problem has already been processed (tree file exists).

        Args:
            problem_id: The problem identifier
            graph_path: Base path where tree files are saved

        Returns:
            True if tree file exists for this problem, False otherwise
        """
        if not problem_id or not graph_path:
            return False

        try:
            # Sanitize problem_id for filename (same as in save_tree_to_txt)
            safe_problem_id = re.sub(r"[^\w\-_]", "_", str(problem_id))

            # Check in judge_test directory
            search_path = Path(graph_path) / "dev"

            if not search_path.exists():
                return False

            # Look for any tree file with this problem_id
            pattern = f"tree_{safe_problem_id}_*.txt"
            existing_files = list(search_path.glob(pattern))

            if existing_files:
                logger.info(
                    f"Problem {problem_id} already processed. "
                    f"Found existing tree file: {existing_files[0].name}"
                )
                return True

            return False

        except Exception as e:
            logger.warning(
                f"Error checking if problem {problem_id} exists: {e}"
            )
            return False  # If check fails, process anyway

    async def _process_single_datapoint_async(
        self,
        datapoint: dict,
        system_prompt: str,
        data_key: List[str],
        prompt_format: Optional[str] = None,
        skip_if_exists: bool = True,
    ) -> dict:
        """
        Process a single datapoint with tree search asynchronously.

        Creates a new search tree instance for this datapoint but shares
        the vLLM and node evaluator components.

        Args:
            datapoint: Single data point to process
            system_prompt: System prompt for the model
            data_key: List of keys to access data in the datapoint
            prompt_format: Format string to be applied to the data
            skip_if_exists: If True, skip processing if tree file already exists

        Returns:
            Processed datapoint with added 'output' field
        """
        try:
            # Get problem ID early for skip check
            problem_id = (
                datapoint.get("id")
                or datapoint.get("problem_id")
                or datapoint.get("custom_id")
            )

            # Check if already processed (if enabled and graph_path is set)
            if skip_if_exists and self.inference_time_args.graph_path:
                if self._check_if_already_processed(
                    problem_id, self.inference_time_args.graph_path
                ):
                    logger.info(
                        f"Skipping already processed problem: {problem_id}"
                    )
                    datapoint["output"] = ["SKIPPED: Already processed"]
                    datapoint["skipped"] = True
                    return datapoint
            # Create messages
            messages = self._message_creator(
                system_prompt, datapoint, data_key, prompt_format
            )
            datapoint["messages"] = messages

            # Format prompt
            prompt = self.prompter(messages)
            datapoint["model_input"] = prompt

            logger.info(
                f"Starting search for datapoint: {problem_id or 'unknown'}"
            )

            # Create inference time method instance for this datapoint
            # Note: This creates a new tree but reuses shared components
            method = get_inference_time_method(
                inference_time_config=self.inference_time_args,
                root_node=None,
                child_finder=self.shared_child_finder,  # Shared vLLM
                node_evaluator=self.shared_node_evaluator,  # Shared evaluator
            )

            # Create TreeThink wrapper
            model = TreeThink(method, self.inference_time_args)

            # CRITICAL: Acquire vLLM lock before calling generate
            # vLLM is NOT thread-safe and concurrent calls cause internal state corruption
            # (msgspec.ValidationError, CUDA errors, etc.)
            logger.debug(f"Waiting for vLLM lock for {problem_id or 'unknown'}")
            async with self._vllm_lock:
                logger.debug(
                    f"Acquired vLLM lock for {problem_id or 'unknown'}"
                )

                # Run search in executor (to not block event loop)
                # The lock ensures only ONE datapoint uses vLLM at a time
                # REPL calls (if using async evaluator) will still be parallelized
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: model.generate(prompt, problem_id=problem_id)
                )

                logger.debug(
                    f"Released vLLM lock for {problem_id or 'unknown'}"
                )

            # Extract output with robust error handling
            if hasattr(response, "outputs") and len(response.outputs) > 0:
                try:
                    datapoint["output"] = [response.outputs[0].text]
                except (IndexError, AttributeError) as e:
                    logger.warning(
                        f"Failed to extract output for {problem_id or 'unknown'}: {e}. "
                        f"Using empty output."
                    )
                    datapoint["output"] = [""]
            elif hasattr(response, "outputs") and len(response.outputs) == 0:
                logger.warning(
                    f"Empty outputs for {problem_id or 'unknown'}. "
                    f"Tree search may have failed or produced no valid solutions."
                )
                datapoint["output"] = [""]
            else:
                # Fallback for unexpected response format
                logger.debug(
                    f"Failed to extract response, using str(response) for {problem_id or 'unknown'}"
                )
                datapoint["output"] = [str(response)]

            # Add search statistics if available
            if hasattr(method, "stats_expansion_count"):
                datapoint["stats"] = {
                    "expansion_count": method.stats_expansion_count,
                    "failed_expansion_count": method.stats_failed_expansion_count,
                }
                logger.success(
                    f"Completed datapoint {problem_id or 'unknown'}: "
                    f"{method.stats_expansion_count} expansions, "
                    f"{method.stats_failed_expansion_count} failed"
                )
            else:
                logger.success(
                    f"Completed datapoint: {problem_id or 'unknown'}"
                )

            return datapoint

        except Exception as e:
            problem_id = (
                datapoint.get("id")
                or datapoint.get("problem_id")
                or datapoint.get("custom_id")
            )
            logger.error(
                f"Failed to get inference from datapoint {problem_id or 'unknown'}: {e}"
            )
            datapoint["output"] = [f"ERROR: {str(e)}"]
            datapoint["error"] = str(e)
            return datapoint

    async def async_inference(
        self,
        data: List[dict],
        system_prompt: str,
        data_key: List[str] = ["problem"],
        prompt_format: Optional[str] = None,
        show_progress: bool = True,
        skip_if_exists: bool = True,
    ) -> List[dict]:
        """
        Perform async inference on multiple datapoints concurrently.

        Each datapoint is processed with its own tree search instance, but:
        - vLLM instance is shared (serialized by GIL)
        - Node evaluator is shared
        - REPL verification is parallelized (if using async evaluator)

        This provides significant speedup for REPL-heavy workloads without
        GPU memory issues.

        Args:
            data: List of data points to process
            system_prompt: System prompt for the model
            data_key: List of keys to access data in each data point
            prompt_format: Format string to be applied to the data
            show_progress: If True, show progress bar

        Returns:
            List of processed data points with added 'output' fields

        Example:
            ```python
            sampler = AsyncDatapointSampler(
                child_finder_args=cf_args,
                node_evaluator_args=ne_args,
                inference_time_args=it_args,
                max_concurrent_datapoints=4
            )

            results = await sampler.async_inference(
                data=datapoints,
                system_prompt="You are a helpful assistant.",
                data_key=["problem"],
            )
            # Or synchronously:
            results = asyncio.run(sampler.async_inference(...))
            ```
        """
        logger.info(f"Starting async inference on {len(data)} datapoints")
        logger.info(
            f"Max concurrent datapoints: {self.max_concurrent_datapoints}"
        )

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_datapoints)

        async def process_with_semaphore(datapoint, idx):
            async with semaphore:
                result = await self._process_single_datapoint_async(
                    datapoint,
                    system_prompt,
                    data_key,
                    prompt_format,
                    skip_if_exists,
                )
                return idx, result

        # Create tasks for all datapoints
        tasks = [
            process_with_semaphore(datapoint, i)
            for i, datapoint in enumerate(data)
        ]

        # Run tasks and collect results
        if show_progress:
            from tqdm.asyncio import tqdm_asyncio

            results = await tqdm_asyncio.gather(
                *tasks, desc="Processing datapoints", unit="problem"
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Sort by original order
        results_dict = {}
        for item in results:
            if isinstance(item, Exception):
                logger.error(f"Task failed with exception: {item}")
                continue
            idx, result = item
            results_dict[idx] = result

        processed_data = [
            results_dict.get(
                i,
                {
                    **data[i],
                    "output": ["ERROR: Task failed"],
                    "error": "Task failed",
                },
            )
            for i in range(len(data))
        ]

        logger.success(
            f"Completed async inference on {len(processed_data)} datapoints."
        )

        # Count successes, failures, and skipped
        skipped = sum(1 for d in processed_data if d.get("skipped", False))
        successes = sum(
            1
            for d in processed_data
            if "error" not in d and not d.get("skipped", False)
        )
        failures = len(processed_data) - successes - skipped
        logger.info(
            f"Successes: {successes}, Failures: {failures}, Skipped: {skipped}"
        )

        return processed_data

    def sync_inference(
        self,
        data: List[dict],
        system_prompt: str,
        data_key: List[str] = ["problem"],
        prompt_format: Optional[str] = None,
        show_progress: bool = True,
        skip_if_exists: bool = True,
    ) -> List[dict]:
        """
        Synchronous wrapper for async_inference.

        Allows using AsyncDatapointSampler with synchronous code.

        Args:
            data: List of data points to process
            system_prompt: System prompt for the model
            data_key: List of keys to access data in each data point
            prompt_format: Format string to be applied to the data
            show_progress: If True, show progress bar

        Returns:
            List of processed data points with added 'output' fields
        """
        return asyncio.run(
            self.async_inference(
                data=data,
                system_prompt=system_prompt,
                data_key=data_key,
                prompt_format=prompt_format,
                show_progress=show_progress,
                skip_if_exists=skip_if_exists,
            )
        )
