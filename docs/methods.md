# Methods — Tree Search Strategies

Methods determine **how the search tree is explored**.  Each method inherits
from `BaseMethod` in `src/treethink/methods/base_method.py` and implements a
`simulate()` loop that drives the expansion, evaluation, and selection of
nodes.

---

## BaseMethod (ABC)

`BaseMethod` provides the shared utilities all methods rely on:

- **`traverse_to_root(node, include_root)`** — walks from a node up to the
  root, concatenating node texts to build a complete proof path.
- **`parse_proof(proof_path)`** — extracts the proof between code fences.
- **`simulate()`** — abstract; each subclass implements its own search loop.
- **Best answer tracking** — `best_answer` property lazily computed via
  `_compute_best_answer()`, with `BestAnswerReason` indicating how it was
  determined (calculated, set, or REPL-verified).

### Final Decision Mode

Controls how the best answer is selected from the tree after search:

| Mode | Behaviour |
|------|-----------|
| `native` | Method-specific default (e.g. root's best child for AlphaZeroMCTS) |
| `maximize_visits` | Pick the most-visited leaf |
| `maximize_value` | Pick the highest-valued leaf |
| `clear_frontier` | Re-score all frontier nodes and pick the best |

Configured via `final_decision_mode` in the YAML config.

### Tie Breaker

When multiple nodes have equal scores, the tie breaker (`tie_breaker`) decides:

| Option | Behaviour |
|--------|-----------|
| `random` | Pick randomly |
| `deep` | Pick the deeper node |
| `stable` | Pick the first encountered |

---

## Implemented Methods

### AlphaZeroMCTS — AlphaZero-Style Monte Carlo Tree Search

**Class:** `AlphaZeroMCTS` in `src/treethink/methods/alpha_zero_mcts.py`

AlphaZero-style MCTS with four phases per iteration — **no rollout** phase.
Evaluation happens directly on expanded children via a learned value function
(evaluator), mirroring the AlphaZero approach.

1. **Select** — walk from root to a leaf using UCB1:  
   $`\text{UCB} = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}}`$  
   where $c$ is `exploration_weight` (default: $\sqrt{2}$).
2. **Expand** — call the policy on the selected leaf to generate children.
3. **Evaluate** — score each child via the evaluator.
4. **Backpropagate** — propagate scores up to the root.

**Async variant:** `AsyncAlphaZeroMCTS` — same logic but with async callbacks.

**YAML:**
```yaml
treethink:
  method_name: "AlphaZeroMCTS"
  exploration_weight: 1.414  # sqrt(2)
```

---

### TraditionalMCTS — Traditional Monte Carlo Tree Search with Rollout

**Class:** `TraditionalMCTS` in `src/treethink/methods/traditional_mcts.py`

Traditional MCTS with a **rollout phase** that generates a complete formal
proof from each child via the LLM, then evaluates the complete proof using
a formal language REPL (Lean 4 or Rocq).

Five phases per iteration:

1. **Select** — walk from root to a leaf using UCB1 (same UCT formula as
   `AlphaZeroMCTS`).
2. **Expand** — generate candidate next-step children via the policy.
3. **Rollout** (new) — for each child, use the LLM to generate a **complete
   formal proof** in a single-shot call with high `max_tokens`.
4. **Evaluate** — score each complete proof via a **separate rollout
   evaluator** (e.g. `lean_repl_evaluator` for Lean 4, `rocq_evaluator` for
   Rocq).  This can be different from the main search evaluator — for
   example, use `cumulative_logprob_evaluator` for fast in-tree decisions
   and `lean_repl_evaluator` for accurate rollout verification.
5. **Backpropagate** — propagate scores up to the root.

**Async variant:** `AsyncTraditionalMCTS`

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exploration_weight` | 1.414 | UCB exploration constant |
| `rollout_evaluator` | — | Evaluator for complete proofs (e.g. `"lean_repl_evaluator"`) |
| `rollout_max_tokens` | 4096 | Max tokens for rollout generation |
| `rollout_n` | 1 | Number of rollouts per child (scores averaged when > 1) |
| `rollout_temperature` | 0.8 | Sampling temperature for rollouts |

**YAML:**
```yaml
treethink:
  method_name: "TraditionalMCTS"
  exploration_weight: 1.414
  rollout_evaluator:
    func_name: "lean_repl_evaluator"
    repl_args:
      lean_server_url: "http://localhost:12336"
  rollout_max_tokens: 4096
  rollout_n: 1
  rollout_temperature: 0.8
```

---

### BFTS — Breadth-First Tree Search

**Class:** `BFTS` in `src/treethink/methods/bfts.py`

Explores the tree level by level.  At each level, all nodes are expanded
before moving deeper.  Useful for exhaustive search in shallow trees.

**Async variant:** `AsyncBFTS`

**YAML:**
```yaml
treethink:
  method_name: "BFTS"
```

---

### Beam Search

**Class:** `BeamSearch` in `src/treethink/methods/beam.py`

Maintains a fixed-size beam (set) of the most promising nodes.  At each step,
all beam nodes are expanded, their children scored, and the top-k children
form the new beam.

| Parameter | Description |
|-----------|-------------|
| `beam_width` | Number of nodes kept per level (default: `max_children`) |

**Async variant:** `AsyncBeamSearch`

**YAML:**
```yaml
treethink:
  method_name: "BeamSearch"
  beam_width: 8  # optional, defaults to max_children
```

---

## Common Parameters

These parameters apply to all methods (set under the `treethink:` YAML key):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method_name` | — | `"AlphaZeroMCTS"`, `"TraditionalMCTS"`, `"BFTS"`, or `"BeamSearch"` |
| `expansion_count` | 128 | Number of tree expansions |
| `max_children` | 4 | Maximum branching factor |
| `timeout` | — | Total timeout in seconds |
| `final_decision_mode` | `"native"` | See above |
| `tie_breaker` | `"random"` | See above |
| `exploration_weight` | 1.414 | UCB exploration constant (AlphaZeroMCTS and TraditionalMCTS) |
| `store_graph_stats` | true | Enable graph statistics |

---

## Adding a New Method

1. Create a new file in `src/treethink/methods/` (e.g. `my_method.py`).
2. Inherit from `BaseMethod` and implement `simulate()`.
3. Register the class in `src/treethink/methods/__init__.py` by adding it to
   the `MethodType` enum.
4. See [extending.md](extending.md) for more detail.

For the full API, refer to the source code at `src/treethink/methods/`.
