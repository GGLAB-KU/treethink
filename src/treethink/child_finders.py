import asyncio
import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Optional, TypeVar, Union

import vllm
from loguru import logger
from vllm import AsyncEngineArgs, AsyncLLMEngine

from .methods import Node
from .utils import FinderArgs, ModelArgs, SamplingArgs


class BaseFinder(ABC):
    def __init__(
        self,
        name: str,
        *args,
        **kwargs,
    ):
        self.name = name

    @abstractmethod
    def __call__(self, node, method):
        pass

    def set_sampling_params(
        self, sampling_params: Union[SamplingArgs, vllm.SamplingParams]
    ):
        if isinstance(sampling_params, SamplingArgs):
            sampling_params = vllm.SamplingParams(**sampling_params)
            logger.trace("Converted SamplingArgs to vllm.SamplingParams.")
        sampling_params.include_stop_str_in_output = True
        self.sampling_params = sampling_params
        return self.sampling_params

    def init_model(self, model_args: ModelArgs, visible_devices: str = "0"):
        logger.info(f"Instantiating model: {model_args.model} for {self.name}.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
        return vllm.LLM(**model_args)


class VLLMFinder(BaseFinder):
    def __init__(
        self,
        model: Union[vllm.LLM, ModelArgs],
        sampling: Optional[Union[vllm.SamplingParams, SamplingArgs]] = None,
        system_prompt: str = "You are a helpful math assistant.",
        prompter: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        logger.debug("Initializing VLLMFinder.")
        super().__init__(name="vllm_finder")
        if isinstance(model, ModelArgs):
            logger.trace("ModelArgs is given, using init_model()")
            self.model = self.init_model(
                model, kwargs.get("visible_devices", "0")
            )
        else:
            logger.trace("Model is pre-initialized LLM.")
            self.model = model

        self.sampling_params = (
            self.set_sampling_params(sampling)
            if sampling
            else vllm.SamplingParams(include_stop_str_in_output=True)
        )

        self.system_prompt = system_prompt

        if isinstance(prompter, Callable):
            logger.debug("Using custom prompter.")
            self.prompter = prompter
        else:
            logger.debug("Using default chat template prompter.")
            self.prompter = partial(
                self.model.get_tokenizer().apply_chat_template,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
                enable_thinking=False,
            )
        self._max_model_len = self.model.llm_engine.model_config.max_model_len
        self._tokenizer = self.model.get_tokenizer()
        logger.debug(f"Max model length: {self._max_model_len}")
        logger.info("VLLMFinder is initialized.")

    def __call__(self, node: Node, method):
        proof_so_far = method.traverse_to_root(node, include_root=False)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": method.root_node.text},
            {"role": "assistant", "content": proof_so_far},
        ]

        messages = self.prompter(messages)

        prompt_tokens = len(self._tokenizer.encode(messages))

        # hardcoded buffer for safety
        safe_max_tokens = self._max_model_len - prompt_tokens - 20

        if safe_max_tokens <= 0:
            logger.error(
                f"Prompt too long: {prompt_tokens} tokens is greater than max "
                + f"model length {self._max_model_len}. Cannot generate children."
            )
            return

        self.sampling_params.max_tokens = min(
            safe_max_tokens, self.sampling_params.max_tokens or float("inf")
        )
        logger.debug(
            f"Set max_tokens for sampling: {self.sampling_params.max_tokens}"
        )

        output = self.model.generate(
            messages, self.sampling_params, use_tqdm=False
        )

        children = []

        for response in output:
            for i in range(len(response.outputs)):
                logger.trace(f"Model response: {response.outputs[i].text}")
                _child_node = Node(
                    text=response.outputs[i].text,
                    max_children=node.max_children,
                    parent=node,
                    vllm_output=response.outputs[i],
                    termination_str=node.termination_str,
                )

                children.append(_child_node)
        node.add_children(children=children)
        logger.debug(f"Added {len(children)} children to node {node}.")


class DynamicFinder(BaseFinder):
    def __init__(
        self,
        model: Union[vllm.LLM, ModelArgs],
        sampling: Optional[Union[vllm.SamplingParams, SamplingArgs]] = None,
        system_prompt: str = "You are a helpful math assistant.",
        prompter: Optional[Callable] = None,
        param_modifier: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        logger.debug("Initializing DynamicFinder.")
        super().__init__(name="dynamic_sampling_finder")
        if isinstance(model, ModelArgs):
            logger.trace("Model is ModelArgs, using init_model()")
            self.model = self.init_model(
                model, kwargs.get("visible_devices", "0")
            )
        else:
            logger.trace("Model is pre-initialized LLM.")
            self.model = model

        self.sampling_params = (
            self.set_sampling_params(sampling)
            if sampling
            else vllm.SamplingParams(include_stop_str_in_output=True)
        )

        self.system_prompt = system_prompt

        if isinstance(prompter, Callable):
            logger.debug("Using custom prompter for DynamicFinder.")
            self.prompter = prompter
        else:
            logger.debug("Using default prompter for DynamicFinder.")
            self.prompter = partial(
                self.model.get_tokenizer().apply_chat_template,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
                enable_thinking=False,
            )

        if isinstance(param_modifier, Callable):
            logger.debug("Using custom param_modifier function.")
            self.param_modifier = param_modifier
        else:
            logger.debug("Using default param_modifier function.")
            self.param_modifier = self._default_param_modifier

        logger.info("DynamicFinder initialized.")

    def _default_param_modifier(self, node: Node) -> vllm.SamplingParams:
        new_params = self.sampling_params.clone()
        initial = 1.1
        alpha = 0.01
        new_params.temperature = max(0.1, initial - node.level * alpha)
        logger.debug(f"New sampling temperature: {new_params.temperature}")
        return new_params

    def param_modifier_experimental(self, node: Node) -> vllm.SamplingParams:
        new_params = self.sampling_params.clone()

        if node.parent:
            child_num = node.parent.children.index(node)
        else:
            child_num = 0

        initial_temperature = 1.1
        initial_top_p = 0.95
        alpha = 0.01

        new_params.temperature = max(
            0.1,
            initial_temperature
            - node.level * alpha / 2
            - child_num * alpha / 2,
        )
        new_params.top_p = max(
            0.5,
            initial_top_p - node.level * alpha / 4 - child_num * alpha / 2,
        )
        logger.debug(
            f"New sampling temperature: {new_params.temperature}, top_p: {new_params.top_p}"
        )
        return new_params

    def __call__(self, node: Node, method):
        proof_so_far = method.traverse_to_root(node, include_root=False)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": method.root_node.text},
            {"role": "assistant", "content": proof_so_far},
        ]

        messages = self.prompter(messages)

        sampling_params = self.param_modifier(node)

        output = self.model.generate(messages, sampling_params, use_tqdm=False)

        children = []
        for response in output:
            for i in range(len(response.outputs)):
                logger.trace(f"Model response: {response.outputs[i].text}")
                _child_node = Node(
                    text=response.outputs[i].text,
                    max_children=node.max_children,
                    exploration_weight=node.exploration_weight,
                    parent=node,
                    vllm_output=response.outputs[i],
                    termination_str=node.termination_str,
                )

                children.append(_child_node)

        node.add_children(children=children)
        logger.debug(f"Added {len(children)} children to node {node}.")


class AsyncVLLMFinder(BaseFinder):
    """Async vLLM based inference for node expansion with batch processing support."""

    def __init__(
        self,
        model: Union[ModelArgs, dict],
        sampling: Optional[Union[vllm.SamplingParams, SamplingArgs]] = None,
        system_prompt: str = "You are a helpful math assistant.",
        prompter: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        logger.debug("Initializing AsyncVLLMFinder.")
        super().__init__(name="async_vllm_finder")

        if isinstance(model, ModelArgs):
            engine_args = AsyncEngineArgs(
                model=model.model,
                tensor_parallel_size=getattr(model, "tensor_parallel_size", 1),
                gpu_memory_utilization=getattr(
                    model, "gpu_memory_utilization", 0.9
                ),
                max_model_len=getattr(model, "max_model_len", None),
                trust_remote_code=getattr(model, "trust_remote_code", True),
                download_dir=getattr(model, "download_dir", None),
                enable_prefix_caching=getattr(
                    model, "enable_prefix_caching", True
                ),
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("AsyncLLMEngine initialized.")
            self._max_model_len = (
                model.max_model_len if hasattr(model, "max_model_len") else 4096
            )
        else:
            raise ValueError(
                "model must be ModelArgs instance for AsyncVLLMFinder."
            )

        self.sampling_params = (
            self.set_sampling_params(sampling)
            if sampling
            else vllm.SamplingParams(include_stop_str_in_output=True)
        )
        self.system_prompt = system_prompt
        self._tokenizer = None
        self._init_tokenizer_task = None

        if isinstance(prompter, Callable):
            logger.debug("Using provided prompter for AsyncVLLMFinder.")
            self.prompter = prompter
        else:
            self.prompter = self._default_prompter

        logger.info("AsyncVLLMFinder initialized.")

    async def _async_init_tokenizer(self):
        """Initialize tokenizer asynchronously."""
        try:
            from transformers import AutoTokenizer

            model_config = await self.engine.get_model_config()
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_config.model, trust_remote_code=True
            )
            logger.debug("Tokenizer loaded successfully in AsyncVLLMFinder.")
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer: {e}. Token counting disabled."
            )
            self._tokenizer = None

    def _default_prompter(self, messages):
        result = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content:
                result += f"{role}: {content}\n"
        return result

    async def _ensure_tokenizer(self):
        if self._tokenizer is None:
            if self._init_tokenizer_task is None:
                logger.debug("Creating tokenizer initialization async task.")
                self._init_tokenizer_task = asyncio.create_task(
                    self._async_init_tokenizer()
                )
            logger.debug("Awaiting tokenizer initialization.")
            await self._init_tokenizer_task

    async def __call__(self, node: Node, method):
        proof_so_far = method.traverse_to_root(node, include_root=False)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": method.root_node.text},
            {"role": "assistant", "content": proof_so_far},
        ]

        prompt = self.prompter(messages)

        try:
            # Get number of children to generate (n parameter from sampling_params)
            n_generations = getattr(self.sampling_params, "n", 1)

            # Create n separate generation tasks with n=1 each for better diversity
            # This is more effective than a single generation with n=N
            tasks = []
            base_seed = getattr(self.sampling_params, "seed", None)

            for i in range(n_generations):
                # Clone sampling params and set n=1 for each generation
                sampling_params_single = self.sampling_params.clone()
                sampling_params_single.n = 1

                # Use different seeds for diversity if seed is set
                if base_seed is not None:
                    sampling_params_single.seed = base_seed + i * 1000

                request_id = f"node_{id(node)}_{i}"
                task = self._generate_single(
                    prompt, sampling_params_single, request_id
                )
                tasks.append(task)

            # Run all generations in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            children = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Generation {i} failed: {result}")
                    continue
                if result is not None:
                    logger.trace(f"Model response: {result.text}")
                    _child_node = Node(
                        text=result.text,
                        max_children=node.max_children,
                        parent=node,
                        vllm_output=result,
                        termination_str=node.termination_str,
                    )
                    children.append(_child_node)

            node.add_children(children=children)
            logger.debug(f"Added {len(children)} children to node {id(node)}.")

        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            logger.error(f"Current prompt: {prompt}")

    async def _generate_single(self, prompt, sampling_params, request_id):
        """Generate a single completion asynchronously."""
        try:
            results = self.engine.generate(
                prompt, sampling_params, request_id=request_id
            )
            final_output = None
            async for request_output in results:
                final_output = request_output

            if final_output and final_output.outputs:
                return final_output.outputs[0]
            return None
        except Exception as e:
            logger.error(f"Single generation failed for {request_id}: {e}")


class AsyncBatchVLLMFinder(BaseFinder):
    """
    Async vLLM child finder with smart batching across multiple nodes.

    This finder can batch multiple node expansions into a single vLLM call,
    significantly improving throughput.
    """

    def __init__(
        self,
        model: Union[ModelArgs, dict],
        sampling: Optional[Union[vllm.SamplingParams, SamplingArgs]] = None,
        system_prompt: str = "You are a helpful math assistant.",
        prompter: Optional[Callable] = None,
        max_batch_size: int = 8,
        *args,
        **kwargs,
    ):
        logger.debug("Initializing AsyncBatchVLLMFinder.")
        super().__init__(name="async_batch_vllm_finder")

        if isinstance(model, ModelArgs):
            engine_args = AsyncEngineArgs(
                model=model.model,
                tensor_parallel_size=getattr(model, "tensor_parallel_size", 1),
                gpu_memory_utilization=getattr(
                    model, "gpu_memory_utilization", 0.9
                ),
                max_model_len=getattr(model, "max_model_len", None),
                trust_remote_code=getattr(model, "trust_remote_code", True),
                dtype=getattr(model, "dtype", "auto"),
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.debug("AsyncBatchVLLMFinder AsyncLLMEngine initialized.")
            self._max_model_len = (
                model.max_model_len if hasattr(model, "max_model_len") else 4096
            )
        else:
            raise ValueError("model must be ModelArgs instance")

        self.sampling_params = (
            self.set_sampling_params(sampling)
            if sampling
            else vllm.SamplingParams(include_stop_str_in_output=True)
        )
        self.system_prompt = system_prompt
        self.max_batch_size = max_batch_size

        self._tokenizer = None
        self._init_tokenizer_task = asyncio.create_task(
            self._async_init_tokenizer()
        )
        logger.debug("Tokenizer async initialization task created.")

        if isinstance(prompter, Callable):
            logger.debug("Using custom prompter for AsyncBatchVLLMFinder.")
            self.prompter = prompter
        else:
            self.prompter = self._default_prompter

        self._pending_nodes = []
        self._batch_lock = asyncio.Lock()

        logger.debug("AsyncBatchVLLMFinder initialized.")

    async def _async_init_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            model_config = await self.engine.get_model_config()
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_config.model, trust_remote_code=True
            )
            logger.debug(
                "Tokenizer loaded successfully for AsyncBatchVLLMFinder."
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            self._tokenizer = None

    def _default_prompter(self, messages):
        result = ""
        for msg in messages:
            content = msg.get("content", "")
            if content:
                result += content + "\n"
        return result

    async def _ensure_tokenizer(self):
        if self._tokenizer is None and self._init_tokenizer_task:
            await self._init_tokenizer_task

    async def __call__(self, node: Node, method):
        return await self._expand_single_node(node, method)

    async def _expand_single_node(self, node: Node, method):
        proof_so_far = method.traverse_to_root(node, include_root=False)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": method.root_node.text},
            {"role": "assistant", "content": proof_so_far},
        ]

        prompt = self.prompter(messages)

        await self._ensure_tokenizer()

        # Prepare base sampling params with token limit
        if self._tokenizer:
            prompt_tokens = len(self._tokenizer.encode(prompt))
            safe_max_tokens = self._max_model_len - prompt_tokens - 20

            if safe_max_tokens <= 0:
                logger.error(
                    f"Prompt too long, skipping expansion. See prompt:\n {prompt}"
                )
                return

            base_sampling_params = self.sampling_params.clone()
            base_sampling_params.max_tokens = min(
                safe_max_tokens, self.sampling_params.max_tokens or float("inf")
            )
            logger.debug(
                f"Set max_tokens for sampling: {base_sampling_params.max_tokens}"
            )
        else:
            base_sampling_params = self.sampling_params

        try:
            # Get number of children to generate (n parameter from sampling_params)
            n_generations = getattr(base_sampling_params, "n", 1)

            # Create n separate generation tasks with n=1 each for better diversity
            tasks = []
            base_seed = getattr(base_sampling_params, "seed", None)

            for i in range(n_generations):
                # Clone sampling params and set n=1 for each generation
                sampling_params_single = base_sampling_params.clone()
                sampling_params_single.n = 1

                # Use different seeds for diversity if seed is set
                if base_seed is not None:
                    sampling_params_single.seed = base_seed + i * 1000

                request_id = f"node_{id(node)}_{i}"
                task = self._generate_single_for_node(
                    prompt, sampling_params_single, request_id
                )
                tasks.append(task)

            # Run all generations in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            children = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Generation {i} failed: {result}")
                    continue
                if result is not None:
                    logger.trace(f"Model response: {result.text}")
                    _child_node = Node(
                        text=result.text,
                        max_children=node.max_children,
                        parent=node,
                        vllm_output=result,
                        termination_str=node.termination_str,
                    )
                    children.append(_child_node)

            node.add_children(children=children)
            logger.debug(f"Added {len(children)} children to node {id(node)}.")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def _generate_single_for_node(
        self, prompt, sampling_params, request_id
    ):
        """Generate a single completion asynchronously."""
        try:
            results = self.engine.generate(
                prompt, sampling_params, request_id=request_id
            )
            final_output = None
            async for request_output in results:
                final_output = request_output

            if final_output and final_output.outputs:
                return final_output.outputs[0]
            return None
        except Exception as e:
            logger.error(f"Single generation failed for {request_id}: {e}")
            raise

    async def batch_expand(self, nodes: list, method):
        if not nodes:
            logger.warning("Empty node list provided to batch_expand.")
            return []
        logger.debug(f"Batch expanding {len(nodes)} nodes.")

        prompts = []
        node_map = {}

        await self._ensure_tokenizer()

        for i, node in enumerate(nodes):
            proof_so_far = method.traverse_to_root(node, include_root=False)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": method.root_node.text},
                {"role": "assistant", "content": proof_so_far},
            ]

            prompt = self.prompter(messages)
            request_id = f"batch_node_{i}_{id(node)}"

            prompts.append((prompt, request_id, node))
            node_map[request_id] = node

        tasks = []
        for prompt, request_id, node in prompts:
            sampling_params = self.sampling_params.clone()

            if self._tokenizer:
                prompt_tokens = len(self._tokenizer.encode(prompt))
                safe_max_tokens = self._max_model_len - prompt_tokens - 20
                if safe_max_tokens > 0:
                    sampling_params.max_tokens = min(
                        safe_max_tokens,
                        self.sampling_params.max_tokens or float("inf"),
                    )
                else:
                    logger.warning(
                        f"Skipping node {request_id}. prompt too long. See prompt:\n {prompt}"
                    )
                    continue
            task = self._generate_for_node(
                prompt, sampling_params, request_id, node
            )
            tasks.append(task)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch generation failed: {result}")
            else:
                success_count += 1

        logger.debug(
            f"Batch expansion complete: {success_count}/{len(nodes)} succeeded."
        )
        return nodes

    async def _generate_for_node(
        self, prompt, sampling_params, request_id, node
    ):
        try:
            results = self.engine.generate(
                prompt, sampling_params, request_id=request_id
            )

            final_output = None
            async for request_output in results:
                final_output = request_output

            if final_output is None:
                logger.warning(f"No output for request_id: {request_id}")
                return

            children = []
            for output in final_output.outputs:
                logger.trace(f"Model response for {request_id}: {output.text}")
                _child_node = Node(
                    text=output.text,
                    max_children=node.max_children,
                    parent=node,
                    vllm_output=output,
                    termination_str=node.termination_str,
                )
                children.append(_child_node)

            node.add_children(children=children)
            logger.info(
                f"Batch node {request_id} was expanded with {len(children)} children."
            )

        except Exception as e:
            logger.error(f"Generation failed for {request_id}: {e}")
            raise


# Constants
IMPLEMENTED_CF = {
    "vllm_finder": VLLMFinder,
    "dynamic_sampling_finder": DynamicFinder,
    "async_batch_vllm_finder": AsyncBatchVLLMFinder,
    "async_vllm_finder": AsyncVLLMFinder,
}
CHILD_FINDERS = list(IMPLEMENTED_CF.keys())
CHILD_FINDER_TYPE = TypeVar("CHILD_FINDER_TYPE", bound=BaseFinder)


def get_finder(func_name, *args, **kwargs) -> BaseFinder:
    try:
        logger.info(f"Instantiating child finder: {func_name}")
        return IMPLEMENTED_CF[func_name](*args, **kwargs)
    except KeyError:
        logger.error(
            f"Could not initialize child finder: {func_name}"
            + f"Available child finder: {list(IMPLEMENTED_CF.keys())}"
        )


def get_finder_from_config(
    config: FinderArgs, *args, **kwargs
) -> CHILD_FINDER_TYPE:
    try:
        logger.info(
            f"Instantiating child finder from config: {config.func_name}"
        )
        return IMPLEMENTED_CF[config.func_name](*args, **config, **kwargs)
    except KeyError:
        logger.error(
            f"Could not initialize child finder: {config.func_name}"
            + f"Available child finder: {list(IMPLEMENTED_CF.keys())}"
        )
