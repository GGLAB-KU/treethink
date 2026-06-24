import os
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Callable, Optional, TypeVar, Union

import openai
import vllm
from loguru import logger
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RepetitionDetectionParams

from .methods import Node
from .utils import ModelArgs, PolicyArgs, SamplingArgs, ServerArgs


class BasePolicy(ABC):
    """Abstract base for all child-generation policies.

    A policy wraps an LLM to produce candidate next proof steps (child
    nodes) from a given parent node.  Subclasses must implement
    ``__call__(self, node, method)`` which generates completions and
    attaches them to the node via ``node.add_children()``.

    To create a custom policy, subclass this and implement ``__call__``,
    then register in the ``PolicyType`` enum.
    """

    def __init__(
        self,
        name: str,
        parse_tag: str = "\n",
        *args,
        **kwargs,
    ):
        self.name = name
        self.parse_tag = parse_tag

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
    def __call__(self, node, method):
        pass

    def _derive_closing_tag(self) -> str:
        """Derive the closing XML tag from the opening tag.

        For example, ``"<reasoning>"`` → ``"</reasoning>"``.
        """
        return f"</{self.parse_tag[1:]}"

    def _clean_text(self, text: str) -> str:
        """Strip the opening and closing tags from generated text.

        When ``parse_tag == "\n"`` (the default), the text is returned
        unchanged so that newlines are preserved in ``Node.text``.

        Otherwise, a leading opening tag and trailing closing tag are
        removed, tolerating surrounding whitespace.
        """
        if self.parse_tag == "\n":
            return text
        closing_tag = self._derive_closing_tag()
        cleaned = text.strip()
        if cleaned.startswith(self.parse_tag):
            cleaned = cleaned[len(self.parse_tag) :].lstrip()
        if cleaned.endswith(closing_tag):
            cleaned = cleaned[: -len(closing_tag)].rstrip()
        return cleaned

    def set_sampling_params(
        self, sampling_params: Union[SamplingArgs, vllm.SamplingParams]
    ):
        if isinstance(sampling_params, SamplingArgs):
            sampling_params = vllm.SamplingParams(**sampling_params)
            logger.trace("Converted SamplingArgs to vllm.SamplingParams.")
        sampling_params.include_stop_str_in_output = True

        # NOTE(burak): hardcoding repetition_detection to finish of nonsensical
        # outputs early. I believe this will help working with small LMs.
        # Parameters here are out of intuition, and I think it'll suffice.
        sampling_params.repetition_detection = RepetitionDetectionParams(
            max_pattern_size=12,
            min_pattern_size=6,
            min_count=3,
        )

        # When a non-newline parse_tag is configured, replace the stop
        # tokens with the derived closing tag so generation halts at the
        # expected delimiter.
        if self.parse_tag != "\n":
            sampling_params.stop = [self._derive_closing_tag()]

        self.sampling_params = sampling_params
        return self.sampling_params

    def init_model(self, model_args: ModelArgs, visible_devices: str = "0"):
        logger.info(f"Instantiating model: {model_args.model} for {self.name}.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
        return vllm.LLM(**model_args)


class VLLMPolicy(BasePolicy):
    """Standard vLLM-based child-generation policy.

    Uses a local vLLM model to generate ``n`` completions from a chat-style
    prompt (system + problem + proof so far).  Automatically clips
    ``max_tokens`` to stay within the model's context window.

    Supports LoRA adapters via the ``lora_path`` parameter.

    Typical YAML config::

        policy:
          func_name: "vllm_policy"
          model:
            model: "internlm/internlm2-7b"
            tensor_parallel_size: 1
          sampling:
            max_tokens: 2048
            temperature: 1.0
            n: 4
            stop: ["\\n"]
    """

    def __init__(
        self,
        model: Union[vllm.LLM, ModelArgs],
        sampling: Optional[Union[vllm.SamplingParams, SamplingArgs]] = None,
        system_prompt: str = "You are a helpful math assistant.",
        prompter: Optional[Callable] = None,
        lora_path: Optional[str] = None,
        parse_tag: str = "\n",
        *args,
        **kwargs,
    ):
        logger.debug("Initializing VLLMPolicy.")
        super().__init__(name="vllm_policy", parse_tag=parse_tag)
        if isinstance(model, ModelArgs):
            logger.trace("ModelArgs is given, using init_model()")
            self.model = self.init_model(
                model, kwargs.get("visible_devices", "0")
            )
            self.enable_lora = model.enable_lora
        else:
            logger.trace("Model is pre-initialized LLM.")
            self.model = model
            self.enable_lora = bool(kwargs.get("enable_lora", False))

        self.lora_path = lora_path
        if self.lora_path and not self.enable_lora:
            logger.warning(
                "lora_path provided but enable_lora is False. Ignoring lora_path."
            )
            self.lora_path = None

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
        logger.info("VLLMPolicy is initialized.")

    def _get_lora_request(self) -> Optional[LoRARequest]:
        if self.enable_lora and self.lora_path:
            return LoRARequest("lora_adapter", 1, self.lora_path)
        return None

    def _str_fields(self):
        return super()._str_fields() + [
            (
                "model",
                getattr(self.model, "__class__", type(self.model)).__name__,
            ),
            ("enable_lora", self.enable_lora),
            ("lora_path", self.lora_path),
            ("sampling_params", self.sampling_params),
            ("system_prompt", self.system_prompt),
        ]

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

        lora_request = self._get_lora_request()
        if lora_request:
            output = self.model.generate(
                messages,
                self.sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
        else:
            output = self.model.generate(
                messages, self.sampling_params, use_tqdm=False
            )

        children = []

        for response in output:
            for i in range(len(response.outputs)):
                raw_text = response.outputs[i].text
                logger.trace(f"Model response: {raw_text}")
                _child_node = Node(
                    text=self._clean_text(raw_text),
                    max_children=node.max_children,
                    parent=node,
                    vllm_output=response.outputs[i],
                    termination_str=node.termination_str,
                )

                children.append(_child_node)
        node.add_children(children=children)
        logger.debug(f"Added {len(children)} children to node {node}.")


class DynamicPolicy(BasePolicy):
    """vLLM policy with dynamically adjustable sampling parameters.

    Like :class:`VLLMPolicy`, but accepts a ``param_modifier`` callback
    that can adjust sampling parameters (temperature, top_p, etc.)
    per-node based on tree depth or sibling index.

    The default modifier linearly decreases temperature with depth::

        temperature = max(0.1, 1.1 - node.level * 0.01)

    An experimental modifier (``param_modifier_experimental``) also adjusts
    ``top_p`` based on depth and child position.

    Typical YAML config::

        policy:
          func_name: "dynamic_policy"
          # All other fields same as vllm_policy
    """

    def __init__(
        self,
        model: Union[vllm.LLM, ModelArgs],
        sampling: Optional[Union[vllm.SamplingParams, SamplingArgs]] = None,
        system_prompt: str = "You are a helpful math assistant.",
        prompter: Optional[Callable] = None,
        param_modifier: Optional[Callable] = None,
        lora_path: Optional[str] = None,
        parse_tag: str = "\n",
        *args,
        **kwargs,
    ):
        logger.debug("Initializing DynamicPolicy.")
        super().__init__(name="dynamic_policy", parse_tag=parse_tag)
        if isinstance(model, ModelArgs):
            logger.trace("Model is ModelArgs, using init_model()")
            self.model = self.init_model(
                model, kwargs.get("visible_devices", "0")
            )
            self.enable_lora = model.enable_lora
        else:
            logger.trace("Model is pre-initialized LLM.")
            self.model = model
            self.enable_lora = bool(kwargs.get("enable_lora", False))

        self.lora_path = lora_path
        if self.lora_path and not self.enable_lora:
            logger.warning(
                "lora_path provided but enable_lora is False. Ignoring lora_path."
            )
            self.lora_path = None

        self.sampling_params = (
            self.set_sampling_params(sampling)
            if sampling
            else vllm.SamplingParams(include_stop_str_in_output=True)
        )

        self.system_prompt = system_prompt

        if isinstance(prompter, Callable):
            logger.debug("Using custom prompter for DynamicPolicy.")
            self.prompter = prompter
        else:
            logger.debug("Using default prompter for DynamicPolicy.")
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

        logger.info("DynamicPolicy initialized.")

    def _get_lora_request(self) -> Optional[LoRARequest]:
        if self.enable_lora and self.lora_path:
            return LoRARequest("lora_adapter", 1, self.lora_path)
        return None

    def _str_fields(self):
        return super()._str_fields() + [
            (
                "model",
                getattr(self.model, "__class__", type(self.model)).__name__,
            ),
            ("enable_lora", self.enable_lora),
            ("lora_path", self.lora_path),
            ("sampling_params", self.sampling_params),
            ("system_prompt", self.system_prompt),
            (
                "param_modifier",
                getattr(self.param_modifier, "__name__", None),
            ),
        ]

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

        lora_request = self._get_lora_request()
        if lora_request:
            output = self.model.generate(
                messages,
                sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
        else:
            output = self.model.generate(
                messages, sampling_params, use_tqdm=False
            )

        children = []
        for response in output:
            for i in range(len(response.outputs)):
                raw_text = response.outputs[i].text
                logger.trace(f"Model response: {raw_text}")
                _child_node = Node(
                    text=self._clean_text(raw_text),
                    max_children=node.max_children,
                    exploration_weight=node.exploration_weight,
                    parent=node,
                    vllm_output=response.outputs[i],
                    termination_str=node.termination_str,
                )

                children.append(_child_node)

        node.add_children(children=children)
        logger.debug(f"Added {len(children)} children to node {node}.")


class VLLMServerPolicy(BasePolicy):
    """vLLM child-generation policy using the OpenAI-compatible API.

    Connects to an **external vLLM server** via the OpenAI chat endpoint
    instead of loading the model locally.  Useful when the model is hosted
    on a separate machine or when sharing a GPU across processes.

    Uses ``openai.OpenAI`` client under the hood.

    Typical YAML config::

        policy:
          func_name: "vllm_server_policy"
          model:
            model: "internlm/internlm2-7b"
          sampling:
            max_tokens: 2048
            temperature: 1.0
            n: 4
          server:
            base_url: "http://localhost:8000/v1"
            api_key: "token-abc123"
    """

    def __init__(
        self,
        model: Union[str, ModelArgs],
        sampling: Optional[Union[vllm.SamplingParams, SamplingArgs]] = None,
        server: Optional[ServerArgs] = None,
        system_prompt: str = "You are a helpful math assistant.",
        parse_tag: str = "\n",
        *args,
        **kwargs,
    ):
        logger.debug("Initializing VLLMServerPolicy.")
        super().__init__(name="vllm_server_policy", parse_tag=parse_tag)

        self.server_args = server if server else ServerArgs()
        self.client = openai.OpenAI(
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

        self.sampling_params = (
            self.set_sampling_params(sampling)
            if sampling
            else vllm.SamplingParams(include_stop_str_in_output=True)
        )

        self.system_prompt = system_prompt
        logger.info("VLLMServerPolicy initialized.")

    def __call__(self, node: Node, method):
        proof_so_far = method.traverse_to_root(node, include_root=False)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": method.root_node.text},
        ]
        if proof_so_far.strip():
            messages.append({"role": "assistant", "content": proof_so_far})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.sampling_params.max_tokens,
                temperature=self.sampling_params.temperature,
                top_p=self.sampling_params.top_p,
                seed=self.sampling_params.seed,
                stop=self.sampling_params.stop,
                n=self.sampling_params.n,
                # In the Chat API, logprobs is a boolean and top_logprobs specifies the count
                logprobs=True,
                top_logprobs=self.sampling_params.logprobs,
            )

            children = []
            for choice in response.choices:
                raw_text = choice.message.content
                logger.trace(f"Model response: {raw_text}")
                _child_node = Node(
                    text=self._clean_text(raw_text),
                    max_children=node.max_children,
                    parent=node,
                    vllm_output=choice,
                    termination_str=node.termination_str,
                )
                children.append(_child_node)

            node.add_children(children=children)
            logger.debug(f"Added {len(children)} children to node {node}.")
        except Exception as e:
            logger.error(f"VLLMServerPolicy generation failed: {e}")

    def _str_fields(self):
        return super()._str_fields() + [
            ("model_name", self.model_name),
            ("server_args", self.server_args),
            ("sampling_params", self.sampling_params),
            ("system_prompt", self.system_prompt),
        ]


class PolicyType(Enum):
    """Enum mapping policy config names to their implementation classes.

    Members are accessed via ``from_str()`` which normalises the config
    ``func_name`` (e.g. ``"vllm_policy"`` → ``PolicyType.VLLM``).

    To add a new policy, add a member here and ensure the value is a
    :class:`BasePolicy` subclass.
    """

    VLLM = VLLMPolicy
    DYNAMIC = DynamicPolicy
    VLLM_SERVER = VLLMServerPolicy

    @classmethod
    def from_str(cls, name: str) -> "PolicyType":
        normalized = name.strip().lower().replace("-", "_")
        for suffix in ("_policy", "_evaluator"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
        for member in cls:
            if normalized == member.name.lower():
                return member
        valid_keys = [member.name.lower() for member in cls]
        raise ValueError(
            f"Unknown policy '{name}'. Valid options: {valid_keys}"
        )

    def initialize(self, *args, **kwargs) -> BasePolicy:
        return self.value(*args, **kwargs)


IMPLEMENTED_POLICIES = PolicyType
POLICIES = [member.name.lower() for member in PolicyType]
POLICY_TYPE = TypeVar("POLICY_TYPE", bound=BasePolicy)


def get_policy(func_name, *args, **kwargs) -> BasePolicy:
    logger.info(f"Instantiating policy: {func_name}")
    return PolicyType.from_str(func_name).initialize(*args, **kwargs)


def get_policy_from_config(config: PolicyArgs, *args, **kwargs) -> POLICY_TYPE:
    logger.info(f"Instantiating policy from config: {config.func_name}")
    return PolicyType.from_str(config.func_name).initialize(
        *args, **dict(vars(config)), **kwargs
    )
