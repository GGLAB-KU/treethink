# Sampler — Batch Processing

The sampler manages **batch processing of multiple problems** through the
tree-search pipeline.  It handles dataset loading, per-problem inference,
and output collection.

---

## SamplerBase

**File:** `src/treethink/sampler.py`

Abstract base class for batched-inference samplers.  Provides:

- Tokenizer initialisation from the model name.
- Sampling parameters via `vllm.SamplingParams`.
- Prompter function for converting messages to prompt strings.

```python
class SamplerBase:
    def __init__(self, sample_params, prompter, task_name, model_name):
        ...
```

---

## TreeThinkSampler

**Class:** `TreeThinkSampler` in `src/treethink/sampler.py`

The **sync** sampler.  Processes problems one at a time:

1. For each problem, creates a `TreeThink` instance.
2. Calls `TreeThink.generate()` which runs the sync method's `simulate()`.
3. Collects outputs and saves them.

---

## AsyncTreeThinkSampler

**Class:** `AsyncTreeThinkSampler` in `src/treethink/sampler.py`

The **async** sampler.  Processes multiple problems concurrently:

1. Creates `TreeThink` instances for all problems.
2. Calls `TreeThink.async_generate()` for each, running them concurrently
   via `asyncio`.
3. Uses `tqdm_asyncio` for progress tracking.
4. Collects and saves outputs.

Used when `--async` is passed to `treethink run`.

---

## Output Handling

Both samplers produce output files (typically JSONL) containing the generated
solutions, method metadata, and graph statistics.  The output format is
controlled by the configuration (e.g. `graph_path` in `treethink:` section).

---

## Relationship to TreeThink

The samplers are the **entry point for batch inference**.  They orchestrate:

1. Loading dataset (via `prepare_datapoints()`)
2. Initialising the LLM (via policy)
3. Creating method and evaluator instances
4. Running `TreeThink.generate()` or `TreeThink.async_generate()` per problem
5. Collecting and saving results

For single-problem inference or debugging, use `TreeThink` directly instead
of the sampler.

For the full API, refer to the source at `src/treethink/sampler.py`.
