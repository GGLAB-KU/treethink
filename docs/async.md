# Async Mode — Asynchronous Tree Search

TreeThink supports **fully asynchronous tree search** for maximum throughput.
When `--async` is passed to `treethink run`, all major components are
auto-converted to their async equivalents.

---

## Auto-Conversion

| Component | Sync → Async |
|-----------|-------------|
| **Method** | `AlphaZeroMCTS` → `AsyncAlphaZeroMCTS` |
| | `TraditionalMCTS` → `AsyncTraditionalMCTS` |
| | `BFTS` → `AsyncBFTS` |
| | `BeamSearch` → `AsyncBeamSearch` |
| **Policy** | `vllm_policy` → `async_vllm_policy` |
| | `vllm_server_policy` → `async_vllm_server_policy` |
| **Evaluator** | `cumulative_logprob_evaluator` → `async_cumulative_logprob_evaluator` |
| | `repl_evaluator` → `async_repl_evaluator` |
| | `llm_as_judge_evaluator` → `async_judge_evaluator` |
| | `norm_len_evaluator` → `async_norm_len_evaluator` |
| | `rocq_evaluator` → `async_rocq_evaluator` |
| **Sampler** | `TreeThinkSampler` → `AsyncTreeThinkSampler` |

The conversion is handled by `convert_to_async()` in
`src/treethink/cli/helpers/config.py` using a simple prefix convention
(prepending `"Async"` to method names and `"async_"` to policy/evaluator
function names).

---

## How It Works

1. **AsyncLLMEngine** — instead of `vllm.LLM`, async policies use
   `vllm.AsyncLLMEngine`, which can serve multiple generation requests
   concurrently without blocking.
2. **Async methods** — `AsyncAlphaZeroMCTS.simulate()`, `AsyncTraditionalMCTS.simulate()`, `AsyncBFTS.simulate()`, etc.
   use `asyncio` for non-blocking tree expansion and evaluation.
3. **Async evaluators** — I/O-bound operations (REPL verification,
   LLM-as-judge scoring) run concurrently via `asyncio`.
4. **AsyncTreeThinkSampler** — processes multiple problems concurrently using
   `tqdm_asyncio` for progress tracking.

---

## Benefits

- **Higher throughput** for I/O-bound operations (REPL verification, LLM calls).
- **Concurrent processing** of multiple problems in the same batch.
- **Non-blocking** tree expansion and evaluation — while one node is being
  evaluated, the search can continue expanding other nodes.

---

## Limitations

- **Not all combinations** have async equivalents — check the conversion
  tables above.
- **Rocq async client** is not yet implemented.
- **Isabelle async client** is not yet implemented.
- **Higher memory usage** due to concurrent in-flight requests.

---

## CLI Usage

```bash
# Sync mode (default)
treethink run --config config.yaml

# Async mode
treethink run --config config.yaml --async
```

---

## Implementing Async Components

Async variants follow a simple pattern:

1. **Policies:** subclass `AsyncBasePolicy` (in
   `src/treethink/async_policies.py`) and implement `__call__` using
   `AsyncLLMEngine.generate()`.
2. **Evaluators:** subclass `AsyncBaseEvaluator` (in
   `src/treethink/async_evaluators.py`) and implement `__call__` with
   `async def`.
3. **Methods:** subclass the sync method (e.g. `AlphaZeroMCTS`) and override
   `simulate()` to use `async def` with async callbacks.

Register async variants in their respective `*Type` enums.  The async
conversion uses a simple prefix convention (see
`src/treethink/cli/helpers/config.py`).

For the full API, refer to the source at `src/treethink/async_policies.py`
and `src/treethink/async_evaluators.py`.
