import asyncio
from enum import Enum
from typing import Callable, Optional, TypeVar, Union

import openai
import vllm
from loguru import logger
from vllm import AsyncEngineArgs, AsyncLLMEngine

from .methods import Node
from .policies import BasePolicy
from .utils import ModelArgs, PolicyArgs, SamplingArgs, ServerArgs


class AsyncVLLMPolicy(BasePolicy):
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
        logger.debug("Initializing AsyncVLLMPolicy.")
        super().__init__(name="async_vllm_policy")

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
                "model must be ModelArgs instance for AsyncVLLMPolicy."
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
            logger.debug("Using provided prompter for AsyncVLLMPolicy.")
            self.prompter = prompter
        else:
            self.prompter = self._default_prompter

        logger.info("AsyncVLLMPolicy initialized.")

    async def _async_init_tokenizer(self):
        """Initialize tokenizer asynchronously."""
        try:
            from transformers import AutoTokenizer

            model_config = await self.engine.get_model_config()
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_config.model, trust_remote_code=True
            )
            logger.debug("Tokenizer loaded successfully in AsyncVLLMPolicy.")
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


class AsyncBatchVLLMPolicy(BasePolicy):
    """
    Async vLLM policy with smart batching across multiple nodes.

    This policy can batch multiple node expansions into a single vLLM call,
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
        logger.debug("Initializing AsyncBatchVLLMPolicy.")
        super().__init__(name="async_batch_vllm_policy")

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
            logger.debug("AsyncBatchVLLMPolicy AsyncLLMEngine initialized.")
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
            logger.debug("Using custom prompter for AsyncBatchVLLMPolicy.")
            self.prompter = prompter
        else:
            self.prompter = self._default_prompter

        self._pending_nodes = []
        self._batch_lock = asyncio.Lock()

        logger.debug("AsyncBatchVLLMPolicy initialized.")

    async def _async_init_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            model_config = await self.engine.get_model_config()
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_config.model, trust_remote_code=True
            )
            logger.debug(
                "Tokenizer loaded successfully for AsyncBatchVLLMPolicy."
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


class AsyncVLLMServerPolicy(BasePolicy):
    """Async vLLM Server based inference for node expansion."""

    def __init__(
        self,
        model: Union[str, ModelArgs],
        sampling: Optional[Union[vllm.SamplingParams, SamplingArgs]] = None,
        server: Optional[ServerArgs] = None,
        system_prompt: str = "You are a helpful math assistant.",
        *args,
        **kwargs,
    ):
        logger.debug("Initializing AsyncVLLMServerPolicy.")
        super().__init__(name="async_vllm_server_policy")

        self.server_args = server if server else ServerArgs()
        self.client = openai.AsyncOpenAI(
            base_url=self.server_args.base_url,
            api_key=self.server_args.api_key,
            timeout=self.server_args.timeout,
        )

        if isinstance(model, ModelArgs):
            self.model_name = model.model
        elif hasattr(model, "llm_engine"):
            self.model_name = model.llm_engine.model_config.model
        else:
            self.model_name = model or "default"

        if isinstance(sampling, SamplingArgs) or isinstance(
            sampling, vllm.SamplingParams
        ):
            self.sampling_dict = sampling.__dict__
        else:
            self.sampling_dict = {
                "max_tokens": 2048,
                "temperature": 1.0,
                "n": 1,
                "logprobs": 1,
                "top_k": 50,  # hack, just to remove...
            }

        # Remove top_k if it exists, as OpenAI API does not support it
        self.sampling_dict.pop("top_k", None)

        self.system_prompt = system_prompt
        logger.info("AsyncVLLMServerPolicy initialized.")

    async def __call__(self, node: Node, method):
        proof_so_far = method.traverse_to_root(node, include_root=False)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": method.root_node.text},
        ]
        if proof_so_far.strip():
            messages.append({"role": "assistant", "content": proof_so_far})

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.sampling_dict,
            )

            children = []
            for choice in response.choices:
                text = choice.message.content
                logger.trace(f"Model response: {text}")
                _child_node = Node(
                    text=text,
                    max_children=node.max_children,
                    parent=node,
                    vllm_output=choice,
                    termination_str=node.termination_str,
                )
                children.append(_child_node)

            node.add_children(children=children)
            logger.debug(f"Added {len(children)} children to node {node}.")
        except Exception as e:
            logger.error(f"AsyncVLLMServerPolicy generation failed: {e}")


class AsyncPolicyType(Enum):
    ASYNC_BATCH_VLLM = AsyncBatchVLLMPolicy
    ASYNC_VLLM = AsyncVLLMPolicy
    ASYNC_VLLM_SERVER = AsyncVLLMServerPolicy

    @classmethod
    def from_str(cls, name: str) -> "AsyncPolicyType":
        normalized = name.strip().lower().replace("-", "_")
        for suffix in ("_policy", "_evaluator"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
        for member in cls:
            if normalized == member.name.lower():
                return member
        valid_keys = [member.name.lower() for member in cls]
        raise ValueError(
            f"Unknown async policy '{name}'. Valid options: {valid_keys}"
        )

    def initialize(self, *args, **kwargs) -> BasePolicy:
        return self.value(*args, **kwargs)


IMPLEMENTED_ASYNC_POLICIES = AsyncPolicyType
ASYNC_POLICIES = [member.name.lower() for member in AsyncPolicyType]
ASYNC_POLICY_TYPE = TypeVar("ASYNC_POLICY_TYPE", bound=BasePolicy)


def get_async_policy(func_name, *args, **kwargs) -> BasePolicy:
    logger.info(f"Instantiating async policy: {func_name}")
    return AsyncPolicyType.from_str(func_name).initialize(*args, **kwargs)


def get_async_policy_from_config(
    config: PolicyArgs, *args, **kwargs
) -> ASYNC_POLICY_TYPE:
    logger.info(f"Instantiating async policy from config: {config.func_name}")
    return AsyncPolicyType.from_str(config.func_name).initialize(
        *args, **dict(vars(config)), **kwargs
    )
