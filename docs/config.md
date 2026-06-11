# Configuration Reference

TreeThink uses **YAML configuration files** to define the tree-search setup,
policy, evaluator, and termination behaviour.

---

## Config File Structure

The main YAML config has three top-level sections:

```yaml
treethink:
  # Tree-search method configuration (TreeThinkArgs)

policy:
  # LLM / child-generation configuration (PolicyArgs)

evaluator:
  # Node scoring configuration (EvaluatorArgs)
```

---

## `treethink:` Section (TreeThinkArgs)

**Dataclass:** `TreeThinkArgs` in `src/treethink/utils/args.py`

| Field | Default | Description |
|-------|---------|-------------|
| `method_name` | `"AlphaZeroMCTS"` | `"AlphaZeroMCTS"`, `"TraditionalMCTS"`, `"BFTS"`, or `"BeamSearch"` |
| `max_children` | `4` | Maximum branching factor |
| `expansion_count` | `128` | Number of tree expansions |
| `timeout` | — | Total timeout in seconds |
| `graph_path` | — | Directory to save tree graph files |
| `termination_str` | — | String marking a terminated proof (e.g. `` ``` ``) |
| `language` | `"lean4"` | `"lean4"`, `"rocq"`, or `"isabelle"` |
| `exploration_weight` | `1.414` | $\sqrt{2}$ — UCB exploration constant (AlphaZeroMCTS and TraditionalMCTS) |
| `final_decision_mode` | `"native"` | How to pick the best answer |
| `tie_breaker` | `"random"` | How to break ties |
| `store_graph_stats` | `true` | Collect graph statistics |
| `store_method_class` | `false` | Serialise the method object |
| `remove_duplicate_children` | `true` | Deduplicate children by text |
| `beam_width` | — | Beam width (BeamSearch only) |
| `max_repl` | `64` | Max leaf proofs to verify |
| `repl_args` | — | `ClientArgs` sub-section |
| `termination_on_encounter` | — | `TerminationOnEncounterConfig` sub-section |
| `termination_on_paths` | — | `TerminationOnPathsConfig` sub-section |

### `repl_args` sub-section (ClientArgs)

| Field | Default | Description |
|-------|---------|-------------|
| `lean_server_url` | `"http://localhost:8000"` | Kimina server URL |
| `batch_size` | `8` | Batch size for verification |
| `num_proc` | `4` | Number of parallel workers |
| `timeout` | `400` | Per-request timeout (seconds) |
| `enable_cache` | `true` | Enable proof verification cache |
| `cache_maxsize` | `4096` | Cache size |
| `host` | `"127.0.0.1"` | Rocq server host |
| `port` | `5000` | Rocq server port |
| `workspace_dir` | `"."` | Rocq workspace directory |
| `theorem_name` | `"__eval"` | Rocq wrapper theorem name |
| `statement` | `"True"` | Rocq theorem statement |
| `prelude` | — | Rocq prelude code |

---

## `policy:` Section (PolicyArgs)

**Dataclass:** `PolicyArgs` in `src/treethink/utils/args.py`

| Field | Default | Description |
|-------|---------|-------------|
| `func_name` | — | `"vllm_policy"`, `"dynamic_policy"`, `"vllm_server_policy"` |
| `model` | — | Sub-section with `model`, `tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`, `trust_remote_code`, `download_dir`, `enable_lora` |
| `sampling` | — | Sub-section with `max_tokens`, `temperature`, `top_k`, `top_p`, `seed`, `stop`, `n`, `logprobs` |
| `system_prompt` | — | System prompt for the LLM |
| `visible_devices` | `"0"` | `CUDA_VISIBLE_DEVICES` setting |
| `server` | — | Server args (for `vllm_server_policy`) |

---

## `evaluator:` Section (EvaluatorArgs)

**Dataclass:** `EvaluatorArgs` in `src/treethink/utils/args.py`

| Field | Default | Description |
|-------|---------|-------------|
| `func_name` | — | Evaluator type name |
| `repl_args` | — | `ClientArgs` sub-section |
| `llm_as_judge_model` | — | Judge model config (for LLM-as-judge) |
| `llm_as_judge_sampling` | — | Judge sampling params |
| `llm_as_judge_system_prompt` | — | Judge system prompt |
| `llm_as_judge_visible_devices` | `"1"` | Judge GPU devices |
| `length_norm` | `0.5` | $\alpha$ for normalised-lengths evaluators |
| `shuffle_bracket` | `true` | Shuffle tournament bracket |

---

## Normal Inference Config

For non-tree-search (standard) inference, the YAML has a simpler structure:

```yaml
model:
  model: "internlm/internlm2-7b"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.95

sampling:
  max_tokens: 2048
  temperature: 1.0
  n: 1
  stop: ["\n"]
```

---

## Dataset Config (TOML)

Dataset configurations use TOML files.  See [datasets.md](datasets.md) for
details.

---

## Configuration Dataclasses

All configuration dataclasses are in `src/treethink/utils/args.py`:

- `BaseArgs` — base with iteration and dict access
- `ClientArgs` — REPL client settings
- `TerminationOnEncounterConfig` — on-encounter termination
- `TerminationOnPathsConfig` — on-paths termination
- `TreeThinkArgs` — main tree-search config
- `PolicyArgs` — policy config
- `EvaluatorArgs` — evaluator config
- `ModelArgs` — model loading config
- `SamplingArgs` — sampling parameters
- `ServerArgs` — vLLM server connection

For the full field listing, refer to the source at
`src/treethink/utils/args.py`.
