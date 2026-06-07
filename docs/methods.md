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
| `native` | Method-specific default (e.g. root's best child for MCTS) |
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

### MCTS — Monte Carlo Tree Search

**Class:** `MCTS` in `src/treethink/methods/mcts.py`

Standard UCB-based MCTS with four phases per iteration:

1. **Select** — walk from root to a leaf using UCB1:  
   $`\text{UCB} = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}}`$  
   where $c$ is `exploration_weight` (default: $\sqrt{2}$).
2. **Expand** — call the policy on the selected leaf to generate children.
3. **Evaluate** — score each child via the evaluator.
4. **Backpropagate** — propagate scores up to the root.

**Async variant:** `AsyncMCTS` — same logic but with async callbacks.

**YAML:**
```yaml
treethink:
  method_name: "MCTS"
  exploration_weight: 1.414  # sqrt(2)
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
| `method_name` | — | `"MCTS"`, `"BFTS"`, or `"BeamSearch"` |
| `expansion_count` | 128 | Number of tree expansions |
| `max_children` | 4 | Maximum branching factor |
| `timeout` | — | Total timeout in seconds |
| `final_decision_mode` | `"native"` | See above |
| `tie_breaker` | `"random"` | See above |
| `exploration_weight` | 1.414 | MCTS UCB exploration constant |
| `store_graph_stats` | true | Enable graph statistics |

---

## Adding a New Method

1. Create a new file in `src/treethink/methods/` (e.g. `my_method.py`).
2. Inherit from `BaseMethod` and implement `simulate()`.
3. Register the class in `src/treethink/methods/__init__.py` by adding it to
   `IMPLEMENTED_METHODS`.
4. See [extending.md](extending.md) for more detail.

For the full API, refer to the source code at `src/treethink/methods/`.
