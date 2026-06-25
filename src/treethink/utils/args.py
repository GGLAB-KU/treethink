from dataclasses import dataclass, field, fields
from typing import List, Optional

from .enums import (
    FinalDecisionMode,
    FormalLanguage,
    PoolingTask,
    ScoreReduction,
    TieBreaker,
)


class BaseArgs:
    """Base dataclass with iteration and dict-like access.

    Provides ``__iter__``, ``keys()``, and ``__getitem__`` so config
    dataclasses can be unpacked with ``*`` and accessed like dicts.
    """

    def __iter__(self):
        """Enable unpacking with * operator by yielding field values"""
        for dataclass_field in fields(self):
            yield getattr(self, dataclass_field.name)

    def keys(self):
        return [dataclass_field.name for dataclass_field in fields(self)]

    def __getitem__(self, key):
        return getattr(self, key)


# ---------------------------------------------------------------------------
# Client-level arguments (unified across formal languages)
# ---------------------------------------------------------------------------


@dataclass
class ClientArgs(BaseArgs):
    """Arguments for any proof-assistant REPL client.

    Only the fields relevant to the selected :class:`FormalLanguage` are
    used at runtime; the rest are quietly ignored.

    Common fields:
        batch_size (int): Batch size for verification requests. Default 8.
        num_proc (int): Number of parallel workers. Default 4.
        timeout (int): Per-request timeout in seconds. Default 400.

    Cache:
        enable_cache (bool): Whether to use an in-memory LRU cache for
            proof-snippet verification results. Default ``True``.
        cache_maxsize (int): Maximum number of entries in the LRU cache.
            Default 4096.

    Lean 4:
        lean_server_url (str): Kimina Lean server URL.
            Default ``"http://localhost:8000"``.

    Rocq:
        host (str): Rocq ML server host. Default ``"127.0.0.1"``.
        port (int): Rocq ML server port. Default 5000.

    Isabelle:
        isabelle_session (str): Isabelle session/logic to start.
            Default ``"HOL"``.
        isabelle_imports (str): Imports clause for each generated theory.
            Default ``"Main"``.
        isabelle_server_log (str | None): Optional Isabelle server log path.
    """

    # Common
    batch_size: int = 8
    num_proc: int = 4
    timeout: int = 400

    # Cache
    enable_cache: bool = True
    cache_maxsize: int = 4096

    # Lean 4
    lean_server_url: str | None = None

    # Rocq
    host: str | None = None
    port: int | None = None

    # Isabelle
    isabelle_session: str | None = None
    isabelle_imports: str | None = None
    isabelle_server_log: str | None = None


# ---------------------------------------------------------------------------
# Termination-check configuration
# ---------------------------------------------------------------------------


@dataclass
class TerminationOnEncounterConfig(BaseArgs):
    """Settings for checking a termination node as soon as it is found.

    Args:
        enabled (bool): Whether to perform the check. Default ``True``.
        batch_size (int): Number of termination nodes to accumulate before
            sending to the REPL in a single batch. ``1`` (default) sends
            immediately.
    """

    enabled: bool = True
    batch_size: int = 1


@dataclass
class TerminationOnPathsConfig(BaseArgs):
    """Settings for batch-checking terminated leaves after search.

    Args:
        enabled (bool): Whether to perform the check. Default ``True``.
        max_repl (int): Maximum number of terminated leaves to verify.
            Default 16.
    """

    enabled: bool = True
    max_repl: int = 16


# ---------------------------------------------------------------------------
# TreeThink configuration
# ---------------------------------------------------------------------------


@dataclass
class TreeThinkArgs(BaseArgs):
    """Top-level configuration for tree-search inference.

    Controls the search method, expansion budget, REPL-based termination
    checking, and output behaviour.

    Args:
        method_name:
            One of the registered method names — ``"AlphaZeroMCTS"``,
            ``"TraditionalMCTS"``, ``"BFTS"``, ``"BeamSearch"`` (or their
            ``Async*`` counterparts when using ``--async``).
            Defaults to ``"AlphaZeroMCTS"``.
        max_children:
            Maximum branching factor (children per node).  Should match
            ``sampling.n`` in the policy config.  Defaults to 4.
        expansion_count:
            Total number of tree expansions (node expansions) to perform
            during search.  Higher values explore more branches but take
            longer.  Defaults to 128.
        timeout:
            Hard timeout in seconds for the entire search.  ``None`` means
            no timeout.  Defaults to ``None``.
        graph_path:
            Directory (or file path) where tree-graph ``.txt`` files are
            saved.  If a directory, files are named
            ``tree_{problem_id}_{timestamp}.txt``.  Set to ``None`` to
            disable graph saving.
        termination_str:
            String that marks a node as a *termination candidate* — when a
            generated child node's text exactly equals this string, the
            search treats is as a potential proof end.  Typical values are
            ``"```"`` (closing code fence) or ``"\\boxed{}"``.  Set to
            ``None`` to disable REPL-based termination checking.
        store_method_class:
            If ``True``, the method object (with full tree state) is stored
            in the output.  Memory-heavy; only enable for debugging.
            Defaults to ``False``.
        store_graph_stats:
            If ``True``, collect tree statistics (width, depth, visits,
            values, termination count, ...) after search and include them
            in the output.  Defaults to ``True``.
        remove_duplicate_children:
            If ``True``, deduplicate child nodes by their text content
            after each expansion.  Defaults to ``False``.
        parse_tag:
            XML opening tag used to delimit generated content (e.g.
            ``"<reasoning>"``).  When set to something other than ``"\\n"``,
            the system:
            - Replaces the stop tokens with the derived closing tag
              (e.g. ``"</reasoning>"``), so generation stops at the
              closing tag.
            - Strips both the opening and closing tags from
              :attr:`Node.text` so that downstream code sees only the
              content between the tags.

            Defaults to ``"\\n"`` (no XML parsing — the newline character
            is kept in the text).

            Note that when using this feature, give your model a proper system
            prompt as the generation relies entirely on the model emitting the
            closing tag. In other words, newline character is not kept when you
            provide a `parse_tag`.
        language:
            The formal proof language to use for REPL verification.
            One of :class:`~treethink.utils.enums.FormalLanguage`.
            Defaults to ``FormalLanguage.LEAN4``.
        client_args:
            Arguments for the language-specific REPL client
            (server URL, batch size, timeouts, etc.).  See
            :class:`ClientArgs`.
        termination_on_encounter:
            Settings for immediate verification when a termination node is
            first encountered.  See :class:`TerminationOnEncounterConfig`.
        termination_on_paths:
            Settings for batch-verifying all terminated leaves after search
            completes.  See :class:`TerminationOnPathsConfig`.
        beam_width:
            Beam width for the ``BeamSearch`` method.  ``None`` means
            ``max_children`` is used.  Defaults to ``None``.
        exploration_weight:
            Exploration constant for the UCB formula used by
            ``AlphaZeroMCTS`` and ``TraditionalMCTS``.  Typical value
            is ``sqrt(2) ≈ 1.414``.  ``None`` means the method default
            is used.  Defaults to ``None``.
        final_decision_mode:
            How the final answer is selected from the tree.
            See :class:`~treethink.utils.enums.FinalDecisionMode`.
            Options: ``"native"``, ``"maximize_visits"``,
            ``"maximize_value"``, ``"clear_frontier"``.
            Defaults to ``FinalDecisionMode.NATIVE``.
        max_concurrent_expansions:
            Maximum number of concurrent expansions (used in async methods).
            Defaults to 8.
        tie_breaker:
            Strategy for breaking ties when multiple children have the same
            score.  See :class:`~treethink.utils.enums.TieBreaker`.
            Options: ``"random"``, ``"deep"``, ``"stable"``.
            Defaults to ``TieBreaker.RANDOM``.
        rollout_evaluator_args:
            Configuration for the rollout evaluator used by
            :class:`~treethink.methods.traditional_mcts.TraditionalMCTS`
            to score complete rollout proofs.  If ``None`` (default), no
            rollout is performed and the main ``evaluator`` is used instead
            (AlphaZero-style behaviour).  See :class:`EvaluatorArgs`.
        rollout_max_tokens:
            Maximum number of tokens for rollout generation.
            Defaults to 4096.
        rollout_n:
            Number of independent rollouts per child.  Scores are averaged
            when > 1.  Defaults to 1.
        rollout_temperature:
            Sampling temperature for rollout generation.
            Defaults to 0.8.
        rollout_top_p:
            ``top_p`` for rollout generation.  Defaults to 1.0.
    """

    method_name: str = "AlphaZeroMCTS"
    max_children: int = 4
    expansion_count: int = 128
    timeout: int = None
    graph_path: str = None
    termination_str: str = None
    store_method_class: bool = False
    store_graph_stats: bool = True
    remove_duplicate_children: bool = False
    parse_tag: str = "\n"

    # REPL / termination
    language: FormalLanguage = FormalLanguage.LEAN4
    client_args: ClientArgs = field(default_factory=ClientArgs)
    termination_on_encounter: TerminationOnEncounterConfig = field(
        default_factory=TerminationOnEncounterConfig,
    )
    termination_on_paths: TerminationOnPathsConfig = field(
        default_factory=TerminationOnPathsConfig,
    )

    # Method Special
    beam_width: int = None
    exploration_weight: float = None
    final_decision_mode: FinalDecisionMode = FinalDecisionMode.NATIVE
    max_concurrent_expansions: int = 8
    tie_breaker: TieBreaker = TieBreaker.RANDOM

    # Rollout (TraditionalMCTS)
    rollout_evaluator_args: Optional["EvaluatorArgs"] = None
    rollout_max_tokens: int = 4096
    rollout_n: int = 1
    rollout_temperature: Optional[float] = None
    rollout_top_p: Optional[float] = None

    def build_repl_runtime(self):
        from treethink.repl_runtime import ReplRuntime

        return ReplRuntime.from_treethink_args(
            self, termination_str=self.termination_str
        )


@dataclass
class SamplingArgs(BaseArgs):
    """vLLM sampling parameters for node expansion.

    These are passed directly to :class:`vllm.SamplingParams` and control
    how the LLM generates child nodes.

    Args:
        max_tokens:
            Maximum number of tokens to generate per child.
            Defaults to 8192.
        temperature:
            Sampling temperature.  Lower values (e.g. 0.1) make output more
            deterministic; higher values (e.g. 1.0) increase diversity.
            Defaults to 1.0.
        top_k:
            Top-k sampling: only the *k* most likely tokens are considered.
            ``-1`` disables top-k filtering.  Defaults to -1.
        top_p:
            Top-p (nucleus) sampling: tokens with cumulative probability
            *p* are considered.  Defaults to 0.9.
        seed:
            Random seed for reproducible generation.  Defaults to 1337.
        stop:
            List of stop strings.  Generation stops when any of these
            strings is produced.  For tactic-by-tactic generation, set
            ``["\\n"]``.  Defaults to ``None``.
        n:
            Number of samples (children) to generate per prompt.  This
            should match ``max_children`` in :class:`TreeThinkArgs`.
            Defaults to 1.
        logprobs:
            Number of token-level log probabilities to return.  When using
            :class:`~treethink.evaluators.LogprobEvaluator`, set this to 1
            to enable cumulative-logprob scoring.  Defaults to ``None``.
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
    """vLLM model initialisation arguments.

    These are unpacked as keyword arguments to :class:`vllm.LLM`
    (or :class:`vllm.AsyncEngineArgs` for async mode).

    Args:
        model:
            HuggingFace model name or path (e.g.
            ``"internlm/internlm2-7b"``).  Defaults to ``None``.
        tensor_parallel_size:
            Number of GPUs to use for tensor parallelism.
            Defaults to 1.
        gpu_memory_utilization:
            Fraction of GPU memory to reserve for the model.
            Defaults to 0.95.
        enable_lora:
            Whether to enable LoRA adapter support.  Defaults to ``False``.
        dtype:
            Model dtype (``"float16"``, ``"bfloat16"``, ``"auto"``, etc.).
            Defaults to ``"float16"``.
        max_model_len:
            Maximum sequence length the model can handle.  ``None`` uses
            the model's default.  Defaults to ``None``.
        download_dir:
            Directory to cache/download the model weights.  ``None`` uses
            the HuggingFace default cache.  Defaults to ``None``.
        enable_prefix_caching:
            Whether to enable vLLM's automatic prefix caching (reuses KV
            cache across requests with common prefixes).  Defaults to
            ``True``.
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
    """vLLM OpenAI-compatible server connection parameters.

    Args:
        base_url: Server URL (default: ``"http://localhost:8000/v1"``)
        api_key: API key (default: ``"EMPTY"``)
        timeout: Connection timeout in seconds (default: 600)
    """

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    timeout: int = 600


@dataclass
class PolicyArgs(BaseArgs):
    """Configuration for the child-node generation policy.

    The policy wraps an LLM to generate candidate children from a parent
    node during tree search.

    Args:
        func_name:
            Name of the policy implementation.  One of the keys in
            :data:`IMPLEMENTED_POLICIES` (sync) or
            :data:`IMPLEMENTED_ASYNC_POLICIES` (async).
            Common values: ``"vllm_policy"``, ``"dynamic_policy"``,
            ``"vllm_server_policy"``.
            When using ``--async``, sync names are auto-converted
            (e.g. ``"vllm_policy"`` → ``"async_vllm_policy"``).
        model:
            Model configuration.  See :class:`ModelArgs`.
        sampling:
            Sampling configuration.  See :class:`SamplingArgs`.
            The ``n`` field should match ``max_children`` in
            :class:`TreeThinkArgs`.
        server:
            Server configuration for remote API access (used with
            ``vllm_server_policy``).  Defaults to ``None``.
        system_prompt:
            System prompt prepended to every generation request.
            Defaults to ``"You are a helpful math assistant."``.
        visible_devices:
            CUDA_VISIBLE_DEVICES string for GPU selection.
            E.g. ``"0"`` for a single GPU, ``"0,1,2,3"`` for four GPUs.
            Defaults to ``"0"``.
    """

    func_name: str
    model: ModelArgs
    sampling: SamplingArgs
    server: ServerArgs = None
    system_prompt: str = "You are a helpful math assistant."
    visible_devices: str = "0"


@dataclass
class EvaluatorArgs(BaseArgs):
    """Configuration for node evaluation (scoring).

    The evaluator assigns a numeric score to each node, guiding the search
    toward promising branches.

    Args:
        func_name:
            Name of the evaluator implementation.  One of the keys in
            :data:`IMPLEMENTED_EVALUATORS` (sync) or
            :data:`ASYNC_IMPLEMENTED_EVALUATORS` (async).
            Common values: ``"cumulative_logprob_evaluator"``,
            ``"lean_repl_evaluator"``, ``"llm_as_judge_evaluator"``,
            ``"norm_len_evaluator"``.
            When using ``--async``, sync names are auto-converted
            (e.g. ``"lean_repl_evaluator"`` →
            ``"async_lean_repl_evaluator"``).
        client_args:
            Arguments for the proof-assistant REPL client.  Required when
            *func_name* involves REPL verification (``lean_repl_evaluator``,
            ``llm_as_judge_evaluator``).  See :class:`ClientArgs`.
        length_norm:
            Exponent for length-normalised scoring.  The raw score is
            divided by ``(length ** length_norm)``.  Only used by
            ``norm_len_evaluator``.  Defaults to 0.5.
        llm_as_judge_model:
            Model configuration for the judge LLM.  Required when
            *func_name* is ``"llm_as_judge_evaluator"``.
            See :class:`ModelArgs`.
        llm_as_judge_sampling:
            Sampling configuration for the judge LLM.
            See :class:`SamplingArgs`.
        llm_as_judge_system_prompt:
            System prompt for the judge LLM.
            Defaults to a prompt asking the judge to score solutions
            out of 20.
        llm_as_judge_visible_devices:
            CUDA_VISIBLE_DEVICES for the judge model (separate GPU from
            the policy model).  Defaults to ``"1"``.
        reward_model:
            Model configuration for the reward (pooling) model.  Required
            when *func_name* is ``"proof_level_reward_evaluator"`` or
            ``"state_level_reward_evaluator"``.  See :class:`ModelArgs`.
        reward_pooling_task:
            vLLM pooling task — see :class:`PoolingTask`
            (``CLASSIFY`` (default) for sequence reward models,
            ``TOKEN_CLASSIFY`` for process reward models).
        reward_score_reduction:
            Reduce a reward vector to a scalar — see
            :class:`ScoreReduction` (``LAST`` (default), ``MEAN``, ``FIRST``).
        reward_system_prompt:
            Optional system prompt prepended to the scored conversation.
        reward_visible_devices:
            CUDA_VISIBLE_DEVICES for the reward model.  Defaults to ``"1"``.
    """

    func_name: str = None
    client_args: ClientArgs = None
    length_norm: float = 0.5
    llm_as_judge_model: ModelArgs = None
    llm_as_judge_sampling: SamplingArgs = None
    llm_as_judge_system_prompt: str = "You are a helpful math assistant who judges the given problem and score it out of 20."
    llm_as_judge_visible_devices: str = "1"

    # Reward-model value functions (proof_level_reward / state_level_reward)
    reward_model: ModelArgs = None
    reward_pooling_task: PoolingTask = PoolingTask.CLASSIFY
    reward_score_reduction: ScoreReduction = ScoreReduction.LAST
    reward_system_prompt: str = None
    reward_visible_devices: str = "1"
