# Evaluators — Node Scoring

Evaluators **assign a score to each node**, guiding the tree search toward
promising branches.  Scores are typically in `[0, 1]` or unbounded, depending
on the evaluator.

---

## BaseEvaluator (ABC)

**File:** `src/treethink/evaluators.py`

All evaluators inherit from `BaseEvaluator`.  The core interface is:

```python
class BaseEvaluator(ABC):
    @abstractmethod
    def __call__(
        self, node: Node | list[Node], method: BaseMethod
    ) -> list[float]:
        """Score one or more nodes.  Returns a list of floats."""
        ...
```

Evaluators receive the node(s) and the method (for traversing the proof path)
and return a numerical score for each.

---

## Implemented Evaluators

### LogprobEvaluator

**Name:** `cumulative_logprob_evaluator`

Returns the **cumulative log-probability** of the node's text from the LLM.
The vLLM output stores `cumulative_logprob` directly; if unavailable, it is
computed manually.

- **Range:** $(-\infty, 0]$
- **Use case:** Simple likelihood-based selection.

```yaml
evaluator:
  func_name: "cumulative_logprob_evaluator"
```

---

### ProbEvaluator

**Name:** `cumulative_prob_evaluator`

Like `LogprobEvaluator`, but exponentiates the log-probability to return a
**probability** in $[0, 1]$.

```yaml
evaluator:
  func_name: "cumulative_prob_evaluator"
```

---

### REPLEvaluator

**Name:** `repl_evaluator`

Verifies the proof snippet via a corresponding formal language server.  Returns
`1.0` if the proof passes the REPL, `0.0` otherwise.

- Requires a running formal proof server:
  - `kimina-lean-server` for lean4.
  - `rocq-ml-server` for Rocq.
  - `isabelle-server` for isabelle.

- Optionally wraps the client with `CachedClient` for memoisation.

```yaml
evaluator:
  func_name: "repl_evaluator"
  client_args:
    lean_server_url: "http://localhost:8000"
    batch_size: 8
    num_proc: 4
    timeout: 400
    ...
```

---

### JudgeEvaluator

**Name:** `llm_as_judge_evaluator`

Uses a **secondary LLM judge** to score the proof step quality.  The judge
receives the proof so far, the applied tactic, open/solved goals (via the
Lean REPL's `infotree`), and returns a score out of 20.

- Supports **LoRA adapters** for the judge model.
- Falls back to a neutral score ($10/20$) if parsing fails.

```yaml
evaluator:
  func_name: "llm_as_judge_evaluator"
  llm_as_judge_model:
    model: "internlm/internlm2-7b"
    tensor_parallel_size: 1
  llm_as_judge_sampling:
    max_tokens: 2048
    temperature: 0.0
    n: 1
```

---

### TournamentEvaluator

**Name:** `pairwise_tournament_evaluator`

Runs a **single-elimination tournament** between sibling nodes.  At each
round, pairs of nodes are compared by a secondary LLM judge, which picks
the better proof step.  Winners advance; losers receive a score proportional
to the round they were eliminated in.

- Scores are normalised to $[0, 1]$.
- Optionally shuffles the bracket for fairness.

```yaml
evaluator:
  func_name: "pairwise_tournament_evaluator"
  # Same LLM judge config as llm_as_judge_evaluator
  shuffle_bracket: true
```

---

### NormLenEvaluator

**Name:** `normalized_lengths_evaluator`

Implements the **normalised-lengths** scoring from the BFS-Prover paper
([arXiv:2502.03438](https://arxiv.org/pdf/2502.03438)).

Score = $`\frac{\text{cumulative\_logprob}}{L^\alpha}`$ where $L$ is the
node depth and $\alpha$ (`length_norm`) controls the length penalty.

```yaml
evaluator:
  func_name: "normalized_lengths_evaluator"
  length_norm: 0.5
```

---

### NormLenProbEvaluator

**Name:** `normalized_lengths_probs_evaluator`

Same as `NormLenEvaluator` but uses **probabilities** (exponentiated
logprobs) instead of raw log-probabilities.

```yaml
evaluator:
  func_name: "normalized_lengths_probs_evaluator"
  length_norm: 0.5
```

---

## Async Variants

Async evaluators live in `src/treethink/async_evaluators.py`.  When `--async`
is passed, the evaluator is auto-converted:

| Sync | Async |
|------|-------|
| `cumulative_logprob_evaluator` | `async_cumulative_logprob_evaluator` |
| `repl_evaluator` | `async_repl_evaluator` |
| `llm_as_judge_evaluator` | `async_judge_evaluator` |
| `norm_len_evaluator` | `async_norm_len_evaluator` |
| `rocq_evaluator` | `async_rocq_evaluator` |

---

## Adding a New Evaluator

1. Subclass `BaseEvaluator` (in `src/treethink/evaluators.py`) or
   `AsyncBaseEvaluator` (in `src/treethink/async_evaluators.py`).
2. Implement `__call__(self, node, method)`.
3. Register it in the **`EvaluatorType`** enum.
4. For async variants, register in **`AsyncEvaluatorType`**.

See [extending.md](extending.md) for more detail.

For the full API, refer to the source at `src/treethink/evaluators.py` and
`src/treethink/async_evaluators.py`.
