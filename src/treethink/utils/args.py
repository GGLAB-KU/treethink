from dataclasses import dataclass, field, fields
from typing import List, Optional

from .enums import FinalDecisionMode, TieBreaker


class BaseArgs:
    def __iter__(self):
        """Enable unpacking with * operator by yielding field values"""
        for dataclass_field in fields(self):
            yield getattr(self, dataclass_field.name)

    def keys(self):
        return [dataclass_field.name for dataclass_field in fields(self)]

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class LeanREPLArgs(BaseArgs):
    """
    Args:
        lean_server_url (str): The URL of the Lean server. Defaults to
            "http://localhost:8000".
        batch_size (int): The batch size for processing. Defaults to 8.
        num_proc (int): The number of processes to use. Defaults to 4.
        timeout (int): The timeout in seconds. Defaults to 400.
    """

    lean_server_url: str = "http://localhost:8000"
    batch_size: int = 8
    num_proc: int = 4
    timeout: int = 400


@dataclass
class ReplStrategyArgs(BaseArgs):
    enabled: bool = False
    backend_name: str = "kimina"
    backend_args: dict = field(default_factory=dict)
    repl_args: Optional[LeanREPLArgs] = None
    max_repl: int = 16
    sync_fn_name: str = "repl_encountered_termination"
    async_fn_name: str = "async_repl_encountered_termination"


@dataclass
class TreeThinkArgs(BaseArgs):
    """
    Args:
        method_name (str): The name of the inference time method to use.
            Defaults to "MCTS".
        max_children (int): The maximum number of children for each node.
            Defaults to 4.
        expansion_count (int): The number of times to expand the tree.
            Defaults to 128.
        timeout (int): The timeout in seconds for the inference process.
            Defaults to None.
        graph_path (str): The path to save the generated graph.
            Defaults to None.
        termination_str (str): The string that indicates the termination of a
            proof path. Used when repl_terminating_paths=True. Set None to
            disable. Defaults to None.
        store_method_class (bool): Whether to store the method class under the
            generation. Can be quite memory-intensive as method class contains
            the whole proof tree. Defaults to False.
        store_graph_stats (bool): Whether to store the graph related statistics
            under the generation. Defaults to False.
        remove_duplicate_children (bool): Whether to remove duplicate children
            in the generation process. Defaults to False.
        repl_args (LeanREPLArgs): The arguments for the Lean REPL server, if
            repl_terminated_paths=True. Defaults to None.
        max_repl (int): The maximum number of REPL calls, if
            repl_terminated_paths=True. Defaults to 16.
        repl_terminated_paths (bool): Whether to use REPL on terminated paths
            after the generation ends. This is done in a batched way, therefore,
            the time penalty is not significant. Defaults to False.
        repl_encountered_termination (bool): Whether to use REPL when a
            termination node is encountered in expansion. Defaults to False. TODO(burak): make this batched
        beam_width (int): The beam width for beam search. Defaults to None.
        exploration_weight (float): The exploration weight for MCTS.
            Defaults to None.
        final_decision_mode (FinalDecisionMode): See methods for information regarding
            the final-answer selection mode. This parameter is not applicable
            for all methods.
    """

    method_name: str = "MCTS"
    max_children: int = 4
    expansion_count: int = 128
    timeout: int = None
    graph_path: str = None
    termination_str: str = None
    store_method_class: bool = False
    store_graph_stats: bool = True
    remove_duplicate_children: bool = False

    # REPL Proof Paths
    repl_args: LeanREPLArgs = None
    max_repl: int = 16
    repl_terminated_paths: bool = False
    repl_encountered_termination: bool = False
    repl_terminated_paths_args: ReplStrategyArgs = field(
        default_factory=lambda: ReplStrategyArgs(
            sync_fn_name="repl_terminated_paths",
            async_fn_name="async_repl_terminated_paths",
        )
    )
    repl_encountered_termination_args: ReplStrategyArgs = field(
        default_factory=lambda: ReplStrategyArgs()
    )

    # Method Special
    beam_width: int = None
    exploration_weight: float = None
    final_decision_mode: FinalDecisionMode = FinalDecisionMode.NATIVE
    max_concurrent_expansions: int = 8
    tie_breaker: TieBreaker = TieBreaker.RANDOM

    def build_repl_runtime(self):
        from treethink.repl_runtime import ReplRuntime

        return ReplRuntime.from_treethink_args(
            self, termination_str=self.termination_str
        )


@dataclass
class SamplingArgs(BaseArgs):
    """
    Args:
        max_tokens (int): The maximum number of tokens to generate.
            Defaults to 8192.
        temperature (float): The temperature for sampling. Defaults to 1.0.
        top_k (int): The top-k sampling parameter. Defaults to -1.
        top_p (float): The top-p sampling parameter. Defaults to 0.9.
        seed (int): The random seed. Defaults to 1337.
        stop (List[str]): The list of stop strings. Set ["\\n"] for next tactic
            generation. Defaults to None.
        n (int): The number of samples to generate. This should be the same as
            max_children in TreeThinkArgs. Defaults to 1.
        logprobs (int): The number of log probabilities to return. When using
            cumulative_logprob_evaluator, set this to 1. Defaults to None.
    """

    max_tokens: int = 8192
    temperature: float = 1.0
    top_k: int = -1
    top_p: int = 0.9
    seed: int = 1337
    stop: List[str] = None
    n: int = 1
    logprobs: int = None


@dataclass
class ModelArgs(BaseArgs):
    """
    Args:
        model (str): The name of the model to use. Defaults to None.
        tensor_parallel_size (int): The tensor parallel size for vllm.
            Defaults to 1.
        gpu_memory_utilization (float): The GPU memory utilization for vllm.
            Defaults to 0.95.
        enable_lora (bool): Whether to enable LoRA. Defaults to False.
        dtype (str): The data type for the model. Defaults to "float16".
        max_model_len (int): The maximum length of the model. Defaults to None.
        download_dir (str): The directory to download the model. Defaults to None.
        enable_prefix_caching (bool): Whether to enable prefix caching for vllm.
            Defaults to False.
    """

    model: str = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.95
    enable_lora: bool = False
    dtype: str = "float16"
    max_model_len: int = None
    download_dir: str = None
    enable_prefix_caching: bool = True


@dataclass
class ServerArgs(BaseArgs):
    """
    Args:
        base_url (str): The base URL of the OpenAI compatible server (vLLM server).
            Defaults to "http://localhost:8000/v1".
        api_key (str): The API key for the server. Defaults to "EMPTY".
        timeout (int): The timeout for the server connection. Defaults to 600.
    """

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    timeout: int = 600


@dataclass
class PolicyArgs(BaseArgs):
    """
    Args:
        func_name (str): The name of the function to use for finding children.
        model (ModelArgs): The model arguments. See ModelArgs.
        sampling (SamplingArgs): The sampling arguments. See SamplingArgs.
        server (ServerArgs): Server arguments for remote API access. Defaults to None.
        system_prompt (str): The system prompt to use. Defaults to
            "You are a helpful math assistant.".
        visible_devices (str): The visible devices for the model. While using
            multiple GPUs (say 4) set it like this: "0,1,2,3". Defaults to "0".
        store_method_class (bool): whether to store method class used in
            generation to access additional functionality that method class
            offers. Default to False, as it is memory-heavy in large sampling
            scenarios.
    """

    func_name: str
    model: ModelArgs
    sampling: SamplingArgs
    server: ServerArgs = None
    system_prompt: str = "You are a helpful math assistant."
    visible_devices: str = "0"


@dataclass
class EvaluatorArgs(BaseArgs):
    """
    Args:
        func_name (str): The name of the function to use for evaluating nodes.
            Defaults to None.
        repl_args (LeanREPLArgs): The arguments for the Lean REPL server when
            using repl_evaluator or llm_as_judge_evaluator. Defaults
            to None.
        llm_as_judge_model (ModelArgs): The model to use as a judge. Defaults
            to None.
        llm_as_judge_sampling (SamplingArgs): The sampling arguments for the
            judge model. Defaults to None.
        llm_as_judge_system_prompt (str): The system prompt for the judge model.
            Defaults to "You are a helpful math assistant who judges the given
            problem and score it out of 20.".
        llm_as_judge_visible_devices (str): The visible devices for the judge
            model. While using multiple GPUs (say 4) set it like this: "0,1,2,3".
            Also see: see: https://discuss.vllm.ai/t/run-multiple-models/1181.
            Defaults to "1".
    """

    func_name: str = None
    repl_args: LeanREPLArgs = None
    length_norm: float = 0.5
    llm_as_judge_model: ModelArgs = None
    llm_as_judge_sampling: SamplingArgs = None
    llm_as_judge_system_prompt: str = "You are a helpful math assistant who judges the given problem and score it out of 20."
    llm_as_judge_visible_devices: str = "1"
