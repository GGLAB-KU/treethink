"""``treethink help`` — comprehensive documentation for all components."""

from typing import Optional

import typer

from treethink.async_evaluators import (
    ASYNC_EVALUATORS,
    AsyncEvaluatorType,
)
from treethink.async_policies import (
    ASYNC_POLICIES,
    AsyncPolicyType,
)
from treethink.evaluators import (
    EVALUATORS,
    EvaluatorType,
)
from treethink.methods import IMPLEMENTED_METHODS, METHODS
from treethink.policies import (
    POLICIES,
    PolicyType,
)

# ── Help topic data ─────────────────────────────────────────────────────

_HELP_TOPICS: dict = {}


def _register(topic: str, title: str, content: str):
    _HELP_TOPICS[topic] = {"title": title, "content": content}


_register(
    "overview",
    "TreeThink — Overview",
    """
TreeThink is a library for formal mathematical reasoning with LLMs using
tree-search methods (MCTS, BFTS, BeamSearch).  It supports:

  • Tree-search algorithms for proof exploration
  • Formal proof verification via REPL clients (Lean 4, Rocq, Isabelle)
  • Sync and async execution modes
  • LoRA adapters and vLLM-based inference
  • Graph visualization and statistics

Key concepts: Methods (search strategy), Policies (child generation),
Evaluators (node scoring), Termination (proof verification).
    """,
)

_register(
    "methods",
    "Methods — Search Strategies",
    f"""
Methods determine how the search tree is explored.  Each method inherits
from :class:`~treethink.methods.base_method.BaseMethod`.

Registered methods ({len(IMPLEMENTED_METHODS)}):
""",
)

# Build the methods detail
_methods_details = ""
for name in METHODS:
    cls = IMPLEMENTED_METHODS.get(name)
    if cls is None:
        continue
    doc = (cls.__doc__ or "").strip()
    _methods_details += f"\n  • **{name}**"
    if doc:
        # Take just the first line/sentence
        first_line = doc.split("\n")[0].split(".")[0]
        if first_line:
            _methods_details += f": {first_line}."

async_names = [n for n in METHODS if n.startswith("Async")]
sync_names = [n for n in METHODS if not n.startswith("Async")]

_methods_details += f"""

  ─ Sync ──────────────────────────────────────
  {", ".join(sorted(sync_names))}

  ─ Async ─────────────────────────────────────
  {", ".join(sorted(async_names))}

Usage in YAML config:
  treethink:
    method_name: "MCTS"        # or "BFTS", "BeamSearch"
    expansion_count: 64        # number of tree expansions
    max_children: 4            # branching factor
    exploration_weight: 1.414  # sqrt(2), for MCTS UCB formula

Use ``--async`` with the CLI to auto-convert to the async variant.
"""
_HELP_TOPICS["methods"]["content"] += _methods_details

_register(
    "policies",
    "Policies — Child Generation",
    f"""
Policies define how new child nodes are generated from a parent node.
They wrap an LLM (via vLLM) to produce candidate next steps.

Registered policies ({len(POLICIES)} sync, {len(ASYNC_POLICIES)} async):

  ─ Sync Policies ────────────────────────────
""",
)

_policies_details = ""
for name in sorted(POLICIES):
    # Look up the class via the PolicyType enum
    try:
        pt = PolicyType.from_str(name)
        cls = pt.value
    except ValueError:
        cls = None
    if cls is None:
        continue
    doc = (cls.__doc__ or "").strip()
    _policies_details += f"\n  • **{name}**"
    if doc:
        first_line = doc.split("\n")[0].split(".")[0]
        if first_line:
            _policies_details += f": {first_line}."

_policies_details += "\n\n  ─ Async Policies ─────────────────────────"
for name in sorted(ASYNC_POLICIES):
    try:
        pt = AsyncPolicyType.from_str(name)
        cls = pt.value
    except ValueError:
        cls = None
    if cls is None:
        continue
    doc = (cls.__doc__ or "").strip()
    _policies_details += f"\n  • **{name}**"
    if doc:
        first_line = doc.split("\n")[0].split(".")[0]
        if first_line:
            _policies_details += f": {first_line}."

_policies_details += """

Usage in YAML config:
  policy:
    func_name: "vllm_policy"       # or "dynamic_policy", "vllm_server_policy"
    model:
      model: "internlm/internlm2-7b"
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.95
    sampling:
      max_tokens: 2048
      temperature: 1.0
      n: 4                         # should match max_children
      stop: ["\\\\n"]
"""
_HELP_TOPICS["policies"]["content"] += _policies_details

_register(
    "evaluators",
    "Evaluators — Node Scoring",
    f"""
Evaluators assign a score to each node, guiding the tree search toward
promising branches.

Registered evaluators ({len(EVALUATORS)} sync, {len(ASYNC_EVALUATORS)} async):

  ─ Sync Evaluators ──────────────────────────
""",
)

_evaluators_details = ""
for name in sorted(EVALUATORS):
    try:
        et = EvaluatorType.from_str(name)
        cls = et.value
    except ValueError:
        cls = None
    if cls is None:
        continue
    doc = (cls.__doc__ or "").strip()
    _evaluators_details += f"\n  • **{name}**"
    if doc:
        first_line = doc.split("\n")[0].split(".")[0]
        if first_line:
            _evaluators_details += f": {first_line}."

_evaluators_details += "\n\n  ─ Async Evaluators ──────────────────────"
for name in sorted(ASYNC_EVALUATORS):
    try:
        aet = AsyncEvaluatorType.from_str(name)
        cls = aet.value
    except ValueError:
        cls = None
    if cls is None:
        continue
    doc = (cls.__doc__ or "").strip()
    _evaluators_details += f"\n  • **{name}**"
    if doc:
        first_line = doc.split("\n")[0].split(".")[0]
        if first_line:
            _evaluators_details += f": {first_line}."

_evaluators_details += """

Common evaluator types:
  - cumulative_logprob_evaluator: Score = sum of token log-probabilities.
  - lean_repl_evaluator: Score = 1.0 if the proof snippet passes the Lean REPL.
  - llm_as_judge_evaluator: Score from a secondary LLM judge.
  - norm_len_evaluator: Normalised-by-length version of logprob.
"""
_HELP_TOPICS["evaluators"]["content"] += _evaluators_details

_register(
    "termination",
    "Termination — Proof Verification",
    """
Termination determines when a branch of the search tree has found a valid
proof.  It works in two complementary phases:

1. **Termination on encounter** (``termination_on_encounter``)
   - When a node's text exactly matches ``termination_str`` (e.g. ``"```"``),
     the tree search pauses to verify that proof path via the REPL client.
   - If verified, the solution is recorded immediately.
   - Controlled by ``TerminationOnEncounterConfig.enabled``.

2. **Termination on paths** (``termination_on_paths``)
   - After the tree search completes, all terminated leaves (nodes where
     the path ends with ``termination_str``) are batch-verified via the REPL.
   - Controlled by ``TerminationOnPathsConfig.enabled`` and
     ``TerminationOnPathsConfig.max_repl``.

The ``ReplRuntime`` coordinator wires this up automatically.

Supported formal languages (``FormalLanguage`` enum):
  - ``LEAN4``: Lean 4 via Kimina server (default)
  - ``ROCQ``: Rocq via ``rocq-ml-server``
  - ``ISABELLE``: Isabelle (experimental)

YAML configuration:
  treethink:
    termination_str: "```"
    language: "lean4"
    repl_args:
      lean_server_url: "http://localhost:8000"
      batch_size: 8
      num_proc: 4
      timeout: 400
    termination_on_encounter:
      enabled: true
    termination_on_paths:
      enabled: true
      max_repl: 64
    """,
)

_register(
    "config",
    "Configuration Reference",
    """
The generation YAML config file has three sections (for tree-search):

### ``treethink`` section (TreeThinkArgs)
  - ``method_name``: "MCTS", "BFTS", "BeamSearch"
  - ``max_children``: Branching factor (default: 4)
  - ``expansion_count``: Number of tree expansions (default: 128)
  - ``timeout``: Total timeout in seconds
  - ``graph_path``: Directory to save tree graph files
  - ``termination_str``: String marking a terminated proof
  - ``language``: "lean4", "rocq", "isabelle"
  - ``exploration_weight``: sqrt(2) for MCTS UCB
  - ``final_decision_mode``: "native", "maximize_visits", "maximize_value",
    "clear_frontier"
  - ``tie_breaker``: "random", "deep", "stable"
  - ``store_graph_stats``: Enable graph statistics collection (default: True)

### ``policy`` section (PolicyArgs)
  - ``func_name``: Policy type ("vllm_policy", "dynamic_policy", etc.)
  - ``model``: Sub-section with model path, TP size, GPU memory, etc.
  - ``sampling``: Sub-section with max_tokens, temperature, n, stop, etc.
  - ``system_prompt``: System prompt for the LLM
  - ``visible_devices``: CUDA_VISIBLE_DEVICES setting

### ``evaluator`` section (EvaluatorArgs)
  - ``func_name``: Evaluator type
  - ``client_args``: REPL client arguments (for repl/lean evaluators)
  - ``llm_as_judge_model``: Judge model (for llm_as_judge_evaluator)
  - ``llm_as_judge_sampling``: Judge sampling parameters
  - ``llm_as_judge_system_prompt``: Judge system prompt
  - ``length_norm``: Length normalisation exponent

For normal (non-tree-search) inference, the YAML has ``model`` and
``sampling`` sections (see ``ModelArgs`` and ``SamplingArgs``).
    """,
)

_register(
    "datasets",
    "Dataset Configuration Reference",
    """
Dataset configurations are stored in TOML files with the following structure:

  [configs.my_config_name]
  name = "my_config_name"
  path_or_name = "path/to/data.jsonl"    # or HuggingFace dataset name
  dataset_split = "train"
  prompt_format = "Solve: {}"            # {} is replaced with data_keys values
  system_prompt = "You are a math assistant."
  data_keys = ["problem"]                # column names in the dataset
  format_type = "jsonl"                  # or "huggingface", "json", etc.
  renamed_data_keys = ["problem"]        # optional mapping to output keys

The CLI loads these via ``--data-config-path <file> --data-config-name <name>``.

Supported dataset formats: huggingface, json, jsonl, csv, tsv, parquet,
pickle, excel.
    """,
)

_register(
    "async",
    "Async Mode",
    """
TreeThink supports fully asynchronous tree search for maximum throughput.

When ``--async`` is passed to ``treethink run``:

1. The **method** is auto-converted (MCTS → AsyncMCTS, BFTS → AsyncBFTS,
   BeamSearch → AsyncBeamSearch).
2. The **policy** is auto-converted (vllm_policy → async_vllm_policy).
3. The **evaluator** is auto-converted (lean_repl_evaluator →
   async_lean_repl_evaluator).
4. The **sampler** is switched to AsyncTreeThinkSampler, which manages
   concurrent processing of multiple datapoints.

Async mode uses ``vllm.AsyncLLMEngine`` under the hood, which can serve
multiple generation requests concurrently without blocking.

Key benefits:
  - Higher throughput for I/O-bound operations (REPL verification)
  - Concurrent processing of multiple problems
  - Non-blocking tree expansion and evaluation

Limitations:
  - Not all method/policy/evaluator combinations have async equivalents
  - Rocq and Isabelle async clients are not yet implemented
  - Higher memory usage due to concurrent in-flight requests
    """,
)


def show_help(
    ctx: typer.Context,
    topic: Optional[str] = typer.Argument(
        None,
        help="Topic to show help for. One of: "
        + ", ".join(sorted(_HELP_TOPICS.keys()))
        + ". If omitted, shows the overview.",
    ),
):
    """Show comprehensive help about TreeThink components.

    Use ``treethink help <topic>`` to get detailed information about
    methods, policies, evaluators, termination, configuration, datasets,
    or async mode.

    Example::

        treethink help methods
        treethink help evaluators
        treethink help termination
    """
    if topic and topic not in _HELP_TOPICS:
        # Try case-insensitive match
        matches = [k for k in _HELP_TOPICS if k.lower() == topic.lower()]
        if matches:
            topic = matches[0]
        else:
            available = ", ".join(sorted(_HELP_TOPICS.keys()))
            print(f"Unknown topic: '{topic}'")
            print(f"Available topics: {available}")
            raise typer.Exit(code=1)

    if topic is None:
        topic = "overview"

    data = _HELP_TOPICS[topic]

    # Rich output via print — works in HPC environments
    separator = "═" * 72
    print()
    print(separator)
    print(f"  {data['title']}")
    print(separator)
    print()
    print(data["content"].strip())
    print()
    print(separator)
    print()
