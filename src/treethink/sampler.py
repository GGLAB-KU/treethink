"""Sampler base classes for TreeThink.

Provides :class:`SamplerBase` (abstract batched inference),
:class:`TreeThinkSampler` (sync tree-search sampler), and
:class:`AsyncTreeThinkSampler` (async tree-search sampler).
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import vllm
from loguru import logger
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

from treethink.async_evaluators import get_async_evaluator_from_config
from treethink.async_policies import get_async_policy_from_config
from treethink.evaluators import get_evaluator_from_config
from treethink.methods import get_method
from treethink.policies import get_policy_from_config
from treethink.treethink import TreeThink
from treethink.utils.args import (
    EvaluatorArgs,
    PolicyArgs,
    SamplingArgs,
    TreeThinkArgs,
)


class SamplerBase:
    """Abstract base for batched-inference samplers.

    Handles tokenizer initialisation, sampling parameter setup, and prompt
    formatting.  Subclasses implement the actual inference loop.
    """

    def __init__(
        self,
        sample_params: Optional[
            Union[vllm.SamplingParams, SamplingArgs]
        ] = None,
        prompter: Optional[Callable] = None,
        task_name="generate",
        model_name: Optional[str] = None,
    ):
        self.task_name = task_name
        self.model_name = model_name

        if isinstance(sample_params, SamplingArgs):
            sample_params = vllm.SamplingParams(**sample_params)
        self.sample_params = sample_params

        if self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

        if prompter:
            self.prompter = prompter
        else:
            self.prompter = self._prompter

    def _format_text(self, text):
        text = text[text.find("<|end_header_id|>") :]
        return text

    def _prompter(self, messages: dict):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return prompt

    def _save_outputs(self, outputs, k, path="data/results"):
        model_name = (
            self.model_name.replace("-", "_").replace("/", "_")
            if self.model_name
            else "unknown_model"
        )
        file_path = path + f"/{self.task_name}_{model_name}_{k}.jsonl"
        with open(file_path, "w") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")

    def _message_creator(
        self, system_prompt, datapoint, data_key, prompt_format
    ):
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
            available_keys = list(datapoint.keys())
            raise KeyError(
                f"Key {e} not found in datapoint. Available keys: {available_keys}. Looking for keys: {data_key}"
            ) from e
        except IndexError:
            raise ValueError("data_key must be a non-empty list")

    def _generate_batch(self, batch_data, batch_messages):
        raise NotImplementedError("Subclasses must implement _generate_batch")

    def batched_inference(
        self,
        data: list,
        batch_size: int = None,
        data_key: list = ["problem"],
        system_prompt: str = None,
        start_from_kth_batch: int = 0,
        save_after_k_batches: int = None,
        save_final_outputs: bool = False,
        prompt_format: str = None,
        lora_path: Optional[str] = None,
    ) -> list:
        """Run batched inference over *data*.

        Args:
            data: List of datapoint dicts.
            batch_size: Number of datapoints per batch.
            data_key: Keys to extract from each datapoint for the prompt.
            system_prompt: System prompt to prepend.
            start_from_kth_batch: Resume from a specific batch index.
            save_after_k_batches: Save intermediate results after every K batches.
            save_final_outputs: Save results even if an error occurs.
            prompt_format: Format string for the prompt (uses ``str.format``).
            lora_path: Optional LoRA adapter path.

        Returns:
            List of processed datapoints.
        """
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if start_from_kth_batch * batch_size >= len(data):
            raise IndexError("start_from_kth_batch is out of range")

        processed_data = []
        try:
            for i in range(
                batch_size * start_from_kth_batch, len(data), batch_size
            ):
                batch_data = data[i : i + batch_size]
                batch_messages = []

                for datapoint in batch_data:
                    messages = self._message_creator(
                        system_prompt, datapoint, data_key, prompt_format
                    )
                    datapoint["messages"] = messages
                    messages = self.prompter(messages)
                    batch_messages.append(messages)
                    datapoint["model_input"] = messages

                batch_responses = self._generate_batch(
                    batch_data, batch_messages
                )

                for j, resp in enumerate(batch_responses):
                    datapoint = batch_data[j].copy()

                    # TreeThinkOutputs or direct string outputs
                    if hasattr(resp, "solution"):
                        datapoint["output"] = [resp.solution]
                    # For vllm outputs
                    elif hasattr(resp, "outputs"):
                        datapoint["output"] = [
                            resp.outputs[idx].text
                            for idx in range(len(resp.outputs))
                        ]
                    else:
                        datapoint["output"] = [resp]

                    if (
                        hasattr(self, "treethink_args")
                        and self.treethink_args
                        and self.treethink_args.store_graph_stats
                    ):
                        if hasattr(resp, "graph_stats"):
                            datapoint["graph_stats"] = resp.graph_stats

                    if hasattr(resp, "checked_and_true"):
                        datapoint["checked_and_true"] = resp.checked_and_true

                    processed_data.append(datapoint)

                if (
                    save_after_k_batches is not None
                    and (i + batch_size) % (save_after_k_batches * batch_size)
                    == 0
                ):
                    self._save_outputs(
                        processed_data,
                        k=(i + batch_size)
                        // (save_after_k_batches * batch_size),
                    )
        except Exception as e:
            logger.error(f"Error occurred while processing batch: {e}")
            if save_final_outputs:
                self._save_outputs(processed_data, "error")

        return processed_data


class TreeThinkSampler(SamplerBase):
    """Sync batched tree-search sampler.

    Processes problems one at a time.  For each problem, creates a
    ``TreeThink`` instance and calls ``generate()``.
    """

    def __init__(
        self,
        policy_args: PolicyArgs,
        evaluator_args: EvaluatorArgs,
        treethink_args: TreeThinkArgs,
        sample_params: Optional[
            Union[vllm.SamplingParams, SamplingArgs]
        ] = None,
        prompter: Optional[Callable] = None,
        task_name="generate",
        lora_path: Optional[str] = None,
    ):
        super().__init__(
            sample_params=sample_params,
            prompter=prompter,
            task_name=task_name,
            model_name=policy_args.model.model,
        )
        self.policy_args = policy_args
        self.evaluator_args = evaluator_args
        self.treethink_args = treethink_args
        self.enable_lora = policy_args.model.enable_lora
        self.lora_path = lora_path

        self._init_treethink_method()

    def _init_treethink_method(self):
        logger.info(
            f"Instantiating selected method: {self.treethink_args.method_name}"
        )
        self.policy = get_policy_from_config(
            self.policy_args,
            prompter=self.prompter,
            lora_path=self.lora_path,
            parse_tag=self.treethink_args.parse_tag,
        )
        self.evaluator = get_evaluator_from_config(
            self.evaluator_args,
            prompter=self.prompter,
            lora_path=self.lora_path,
            language=self.treethink_args.language,  # unified language
            client_args=self.evaluator_args.client_args,
        self.evaluator = get_evaluator_from_config(
            self.evaluator_args,
            prompter=self.prompter,
            lora_path=self.lora_path,
            language=self.treethink_args.language,  # unified language
        )
        self.method = get_method(
            treethink_config=self.treethink_args,
            root_node=None,
            policy=self.policy,
            evaluator=self.evaluator,
        )
        self.model = TreeThink(self.method, self.treethink_args)

    def _set_lora_path(self, lora_path: Optional[str]):
        if lora_path is None:
            return
        self.lora_path = lora_path
        if hasattr(self, "policy"):
            self.policy.lora_path = lora_path
        if hasattr(self, "evaluator"):
            self.evaluator.lora_path = lora_path

    def _generate_batch(self, batch_data, batch_messages):
        batch_responses = []
        for datapoint, prompt in zip(batch_data, batch_messages):
            problem_id = (
                datapoint.get("id")
                or datapoint.get("problem_id")
                or datapoint.get("custom_id")
            )
            response = self.model.generate(
                prompts=prompt, problem_id=problem_id
            )
            batch_responses.append(response)
        return batch_responses

    def batched_inference(self, *args, **kwargs):
        self._set_lora_path(kwargs.pop("lora_path", None))
        return super().batched_inference(*args, **kwargs)


class AsyncTreeThinkSampler:
    """High-throughput async tree-search sampler.

    Uses :class:`AsyncMCTS` (or other async methods),
    :class:`AsyncChildPolicy`, and :class:`AsyncNodeEvaluator`
    to process multiple datapoints concurrently.

    Args:
        policy_args: Policy configuration.
        evaluator_args: Evaluator configuration.
        treethink_args: TreeThink configuration.
        prompter: Callable that converts a message list into a prompt string.
        max_concurrent_datapoints: Maximum number of datapoints to process in parallel.
        task_name: Identifier for the current task.
    """

    def __init__(
        self,
        policy_args: PolicyArgs,
        evaluator_args: EvaluatorArgs,
        treethink_args: TreeThinkArgs,
        prompter: Optional[Callable] = None,
        max_concurrent_datapoints: int = 16,
        task_name: str = "async_generate",
    ):
        self.policy_args = policy_args
        self.evaluator_args = evaluator_args
        self.treethink_args = treethink_args
        self.max_concurrent_datapoints = max_concurrent_datapoints
        self.prompter = prompter or self._default_prompter
        self.task_name = task_name

        # Initialize shared async components
        logger.info(
            "Initializing async components for AsyncTreeThinkSampler..."
        )

        self.shared_policy = get_async_policy_from_config(
            policy_args,
            prompter=self.prompter,
            parse_tag=self.treethink_args.parse_tag,
        )

        self.shared_evaluator = get_async_evaluator_from_config(
            evaluator_args,
            prompter=self.prompter,
            language=self.treethink_args.language,
            client_args=self.evaluator_args.client_args,
        )

        logger.success(
            f"AsyncTreeThinkSampler initialized with max concurrency: "
            f"{max_concurrent_datapoints} and task name: {self.task_name}"
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
            if skip_if_exists and self.treethink_args.graph_path:
                if self._check_if_processed(
                    problem_id, self.treethink_args.graph_path
                ):
                    logger.info(f"Skipping {problem_id}")
                    datapoint["skipped"] = True
                    return datapoint

            # Prepare input
            messages = self._message_creator(
                system_prompt, datapoint, data_key, prompt_format
            )
            prompt = self.prompter(messages)

            logger.debug(f"Starting async_simulate for {problem_id}")

            method = get_method(
                treethink_config=self.treethink_args,
                root_node=None,
                policy=self.shared_policy,
                evaluator=self.shared_evaluator,
            )

            wrapper = TreeThink(method, self.treethink_args)

            result = await wrapper.async_generate(prompt, problem_id=problem_id)

            # Extract output from TreeThinkOutputs
            if result.outputs:
                datapoint["output"] = [result.solution]
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
        """Run async inference on all datapoints concurrently."""
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
        """Synchronous entry point (wraps :meth:`async_inference`)."""
        return asyncio.run(self.async_inference(*args, **kwargs))
