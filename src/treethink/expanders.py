import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Optional, TypeVar, Union

import vllm
from loguru import logger

from .methods import Node
from .utils import ExpanderArgs, ModelArgs, SamplingArgs


class BaseExpander(ABC):
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


class VLLMExpander(BaseExpander):
    def __init__(
        self,
        model: Union[vllm.LLM, ModelArgs],
        sampling: Optional[Union[vllm.SamplingParams, SamplingArgs]] = None,
        system_prompt: str = "You are a helpful math assistant.",
        prompter: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        logger.debug("Initializing VLLMExpander.")
        super().__init__(name="vllm_expander")
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
        logger.info("VLLMExpander is initialized.")

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


class DynamicExpander(BaseExpander):
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
        logger.debug("Initializing DynamicExpander.")
        super().__init__(name="dynamic_expander")
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
            logger.debug("Using custom prompter for DynamicExpander.")
            self.prompter = prompter
        else:
            logger.debug("Using default prompter for DynamicExpander.")
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

        logger.info("DynamicExpander initialized.")

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


# Constants
IMPLEMENTED_EXPANDERS = {
    "vllm_expander": VLLMExpander,
    "dynamic_expander": DynamicExpander,
}
EXPANDERS = list(IMPLEMENTED_EXPANDERS.keys())
EXPANDER_TYPE = TypeVar("EXPANDER_TYPE", bound=BaseExpander)


def get_expander(func_name, *args, **kwargs) -> BaseExpander:
    try:
        logger.info(f"Instantiating expander: {func_name}")
        return IMPLEMENTED_EXPANDERS[func_name](*args, **kwargs)
    except KeyError:
        logger.error(
            f"Could not initialize expander: {func_name}"
            + f"Available expanders: {list(IMPLEMENTED_EXPANDERS.keys())}"
        )


def get_expander_from_config(
    config: ExpanderArgs, *args, **kwargs
) -> EXPANDER_TYPE:
    try:
        logger.info(f"Instantiating expander from config: {config.func_name}")
        return IMPLEMENTED_EXPANDERS[config.func_name](
            *args, **config, **kwargs
        )
    except KeyError:
        logger.error(
            f"Could not initialize expander: {config.func_name}"
            + f"Available expander: {list(IMPLEMENTED_EXPANDERS.keys())}"
        )
