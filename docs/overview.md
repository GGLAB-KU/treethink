# TreeThink — Overview

TreeThink is a library for **formal mathematical reasoning** with large language
models (LLMs) using **tree-search methods**.  It enables LLMs to explore
multiple proof paths in parallel, verify candidates against formal proof
assistants (Lean 4, Rocq, Isabelle), and select the most promising solutions.

---

## Core Concept

At its heart, TreeThink wraps an LLM-powered generation loop inside a tree
search.  Starting from a problem statement (root node), the system repeatedly:

1. **Expands** promising nodes by asking the LLM to generate candidate next
   proof steps (→ [Policies](policies.md))
2. **Scores** each new node to estimate its quality (→ [Evaluators](evaluators.md))
3. **Selects** the best node(s) for further expansion based on the search
   strategy (→ [Methods](methods.md))
4. **Terminates** branches when a complete proof is found or the branch is
   exhausted (→ [Termination](termination.md))

```
        ┌──────────────┐
        │     Root     │
        └──────┬───────┘
               │
        ┌──────┴───────┐
        │  Tree Search │  (AlphaZeroMCTS / TraditionalMCTS / BFTS / BeamSearch)
        │  Method      │
        └──────┬───────┘
               │
     ┌─────────┼─────────┐
     │         │         │
  ┌──┴───┐  ┌──┴───┐  ┌──┴───┐
  │Policy│  │ Eval │  │ Term │
  │(LLM) │  │(Score│  │(Proof│
  │      │  │ Node)│  │Check)│
  └──────┘  └──────┘  └──────┘
```

---

## Main Components

| Component | Responsibility | Source |
|-----------|---------------|--------|
| **TreeThink** | Top-level orchestrator — wires method, policy, evaluator, and termination together | `src/treethink/treethink.py` |
| **Methods** | Search strategy (AlphaZeroMCTS, TraditionalMCTS, BFTS, BeamSearch) | `src/treethink/methods/` |
| **Policies** | LLM-driven child node generation | `src/treethink/policies.py` |
| **Evaluators** | Node scoring / fitness estimation | `src/treethink/evaluators.py` |
| **Termination** | Proof verification via REPL clients | `src/treethink/termination.py` |
| **REPL Clients** | Language-specific proof-assistant backends | `src/treethink/clients/` |
| **Sampler** | Batch processing of multiple problems | `src/treethink/sampler.py` |

---

## Execution Flow

1. **Config loading** — YAML config defines method, policy, evaluator, and
   termination settings (see [Configuration](config.md)).
2. **TreeThink.generate()** or **TreeThink.async_generate()** is called per
   problem.
3. The **method's `simulate()`** loop runs:
   - **Select** a node (strategy-specific)
   - **Expand** it via the policy (LLM generates children)
   - **Evaluate** children via the evaluator (score each node)
   - **Check termination** — if a node matches `termination_str`, the REPL
     client verifies the proof (see [Termination](termination.md)).
4. After search, **terminated paths** are batch-verified via the REPL.
5. The **best answer** is returned (highest score, or REPL-verified proof).

---

## Async Mode

TreeThink supports fully asynchronous execution for maximum throughput:

- Methods auto-convert (AlphaZeroMCTS → AsyncAlphaZeroMCTS, BFTS → AsyncBFTS, etc.)
- Policies and evaluators switch to async equivalents
- The sampler uses `AsyncTreeThinkSampler` for concurrent problem processing

See [Async Mode](async.md) for details.

---

## Supported Proof Assistants

| Language | Client | Status |
|----------|--------|--------|
| Lean 4 | Kimina server (`kimina-client`) | Stable |
| Rocq (Coq 8.20) | `rocq-ml-server` | Stable (sync) |
| Isabelle | Custom client | not supported |

See [Clients](clients.md) for details.

---

## Next Steps

- **Configuration reference**: [config.md](config.md)
- **Adding new methods / policies / evaluators**: [extending.md](extending.md)
- **CLI usage**: [cli.md](cli.md)
