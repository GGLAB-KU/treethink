# Policies — Child Node Generation

Policies are responsible for **generating child nodes** from a parent node.
They wrap an LLM (typically via vLLM) to produce candidate next proof steps.

---

## BasePolicy (ABC)

**File:** `src/treethink/policies.py`

All policies inherit from `BasePolicy`.  The core interface is:

```python
class BasePolicy(ABC):
    @abstractmethod
    def __call__(self, node: Node, method) -> None:
        """Generate children for `node` and attach them via `node.add_children()`."""
        ...
```

The policy receives the current node and the method (to traverse the proof
path so far), generates LLM completions, wraps each in a `Node`, and attaches
them to the parent.

### Shared Utilities

- **`init_model(model_args, visible_devices)`** — initialises a vLLM LLM from
  `ModelArgs`.
- **`set_sampling_params(sampling_params)`** — configures sampling parameters
  (temperature, top_p, n, stop tokens, etc.).

---

## Implemented Policies

### VLLMPolicy

**Class:** `VLLMPolicy` in `src/treethink/policies.py` (name: `vllm_policy`)

The standard policy.  Uses a local vLLM model to generate children.

- Builds a chat-format prompt: system prompt + problem + proof so far.
- Generates `n = sampling.n` completions, each becoming a child node.
- Automatically adjusts `max_tokens` to stay within the model's context
  window.
- Supports **LoRA adapters** via `lora_path`.

```yaml
policy:
  func_name: "vllm_policy"
  model:
    model: "internlm/internlm2-7b"
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.95
  sampling:
    max_tokens: 2048
    temperature: 1.0
    n: 4
    stop: ["\n"]
```

---

### DynamicPolicy

**Class:** `DynamicPolicy` in `src/treethink/policies.py` (name: `dynamic_policy`)

Like `VLLMPolicy`, but accepts a **`param_modifier`** callback that can
dynamically adjust sampling parameters per node (e.g. decrease temperature
as depth increases).

**Default modifier** — linearly decreases temperature with tree depth:
```python
temperature = max(0.1, 1.1 - node.level * 0.01)
```

An **experimental modifier** (`param_modifier_experimental`) also adjusts
`top_p` based on both depth and sibling index.

```yaml
policy:
  func_name: "dynamic_policy"
  # All other fields same as vllm_policy
```

---

### VLLMServerPolicy

**Class:** `VLLMServerPolicy` in `src/treethink/policies.py` (name: `vllm_server_policy`)

Uses an **external vLLM server** via the OpenAI-compatible API instead of
loading the model locally.  Useful when the model is hosted on a separate
machine or when sharing a GPU across multiple processes.

```yaml
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
    timeout: 120
```

---

## Async Variants

Async policies live in `src/treethink/async_policies.py`.  When `--async` is
passed to the CLI, the policy is auto-converted:

| Sync | Async |
|------|-------|
| `vllm_policy` | `async_vllm_policy` |
| `vllm_server_policy` | `async_vllm_server_policy` |

Async policies use `AsyncLLMEngine` for non-blocking generation, enabling
higher throughput when combined with async methods and evaluators.

---

## Adding a New Policy

1. Subclass `BasePolicy` (in `src/treethink/policies.py`) or
   `AsyncBasePolicy` (in `src/treethink/async_policies.py`).
2. Implement `__call__(self, node, method)`.
3. Register it in the **`PolicyType`** enum in `src/treethink/policies.py`.
4. For async variants, also register in `AsyncPolicyType` in
   `src/treethink/async_policies.py`.

See [extending.md](extending.md) for more detail.

For the full API, refer to the source at `src/treethink/policies.py` and
`src/treethink/async_policies.py`.
