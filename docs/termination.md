# Termination — Proof Verification

The termination system determines **when a branch of the search tree has
found a valid proof** and handles the verification of that proof against the
formal proof assistant.

---

## Two-Phase Verification

Termination operates in two complementary phases:

### 1. Termination on Encounter

When a node's text exactly matches `termination_str` (e.g. `` ``` ``), the
search pauses **immediately** to verify that proof path via the REPL client.

```python
check_termination_encountered(method, node, client, ...)
```

- Traverses from node to root, concatenating the proof.
- Sends the parsed proof snippet to the REPL.
- If verified → `best_answer` is set with `BestAnswerReason.CHECKED_AND_TRUE`.
- **Fast path** — catches correct proofs during search, no need to wait.

### 2. Termination on Paths

After the tree search completes, **all terminated leaves** (nodes whose path
ends with `termination_str`) are batch-verified via the REPL.

```python
check_terminated_paths(method, root, client, ...)
```

- Collects all leaves where `is_termination_node == True`.
- Limits to `max_repl` nodes, prioritised by `win_value`.
- Batch-verifies proofs.
- Returns the first successful proof.

**Fallback path** — catches proofs missed by on-encounter (e.g. if the REPL
was temporarily unavailable during search).

---

## ReplRuntime

**File:** `src/treethink/repl_runtime.py`

`ReplRuntime` is the lightweight coordinator that wires everything together:

- Owns the proof-assistant clients (sync + async).
- Exposes `build_termination_callback()` consumed by `TreeThink.generate()`.
- Handles cache sharing between sync and async clients.

```python
runtime = ReplRuntime.from_treethink_args(args, termination_str="```")
callback = runtime.build_termination_callback()
method.simulate(..., termination_encountered_fn=callback)
```

---

## Best Answer Tracking

The `BestAnswerReason` enum (`src/treethink/utils/enums.py`) records how the
best answer was determined:

| Reason | Meaning |
|--------|---------|
| `CALCULATED` | Computed via final decision mode |
| `SET` | Manually set via callback |
| `CHECKED_AND_TRUE` | Verified by REPL as correct |

If a proof was already `CHECKED_AND_TRUE` during on-encounter, the
on-paths phase skips redundant REPL checks.

---

## Configuration

```yaml
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
    batch_size: 1
  termination_on_paths:
    enabled: true
    max_repl: 64
```

### TerminationOnEncounterConfig

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable on-encounter verification |
| `batch_size` | `1` | Batch size for encountered verification |

### TerminationOnPathsConfig

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable post-search batch verification |
| `max_repl` | `64` | Max number of leaf proofs to check |

---

## Supported Formal Languages

| Language | Enum Value | Status |
|----------|------------|--------|
| Lean 4 | `ProofLanguage.LEAN4` | Stable |
| Rocq | `ProofLanguage.ROCQ` | Stable (sync) |
| Isabelle | `ProofLanguage.ISABELLE` | not supported |

---

## Architecture Diagram

```
 TreeThink.generate()
        │
        ├─ method.simulate(termination_encountered_fn=callback)
        │       │
        │       └─ On termination node found:
        │             callback(node)
        │               → check_termination_encountered()
        │               → REPL verifies → best_answer = proof
        │
        └─ After simulate:
              runtime.check_terminated_paths()
                → check_terminated_paths()
                → batch REPL verify all termination leaves
                → best_answer = first verified proof
```

For the full API, refer to the source at `src/treethink/termination.py` and
`src/treethink/repl_runtime.py`.
