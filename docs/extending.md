# Extending TreeThink

TreeThink is designed to be **easily extended** with new methods, policies,
evaluators, and REPL clients.  This document directs you to the relevant
source files for each extension point.

> **Note:** This is not a full API reference.  For detailed class and method
> signatures, refer to the source code directly.

---

## Adding a New Search Method

New methods go in `src/treethink/methods/` and must inherit from `BaseMethod`.

**Steps:**

1. Create a new file (e.g. `src/treethink/methods/my_method.py`).
2. Subclass `BaseMethod` (from `src/treethink/methods/base_method.py`).
3. Implement the `simulate()` abstract method.
4. Register the class in `src/treethink/methods/__init__.py` by adding it to
   the `MethodType` enum.

```python
# src/treethink/methods/__init__.py
class MethodType(Enum):
    RF_MCTS = "RFMCTS"
    BFTS = "BFTS"
    BEAM_SEARCH = "BeamSearch"
    TRADITIONAL_MCTS = "TraditionalMCTS"
    MY_METHOD = "MyMethod"  # ← add yours
```

For async variants, subclass the sync method and override `simulate()` with
`async def`.  Add an async member to the `MethodType` enum as well
(e.g. `ASYNC_MY_METHOD = "AsyncMyMethod"`).

**Reference:** `src/treethink/methods/`
**Example:** `src/treethink/methods/rf_mcts.py`, `traditional_mcts.py`, `bfts.py`, `beam.py`

---

## Adding a New Policy

New policies go in `src/treethink/policies.py` and must inherit from
`BasePolicy`.

**Steps:**

1. Subclass `BasePolicy` (in `src/treethink/policies.py`).
2. Implement `__call__(self, node, method)` to generate children and attach
   them via `node.add_children()`.
3. Register in the `PolicyType` enum.

```python
# src/treethink/policies.py
class PolicyType(Enum):
    VLLM = VLLMPolicy
    DYNAMIC = DynamicPolicy
    VLLM_SERVER = VLLMServerPolicy
    MY_POLICY = MyPolicy  # ← add yours
```

For async policies, subclass `AsyncBasePolicy` in
`src/treethink/async_policies.py` and register in `AsyncPolicyType`.

**Reference:** `src/treethink/policies.py`, `src/treethink/async_policies.py`

---

## Adding a New Evaluator

New evaluators go in `src/treethink/evaluators.py` and must inherit from
`BaseEvaluator`.

**Steps:**

1. Subclass `BaseEvaluator` (in `src/treethink/evaluators.py`).
2. Implement `__call__(self, node, method)` returning a list of float scores.
3. Register in the `EvaluatorType` enum.

```python
# src/treethink/evaluators.py
class EvaluatorType(Enum):
    CUMULATIVE_LOGPROB = LogprobEvaluator
    REPL = LeanREPLEvaluator
    LLM_AS_JUDGE = JudgeEvaluator
    TOURNAMENT = TournamentEvaluator
    NORMALIZED_LENGTHS = NormLenEvaluator
    NORMALIZED_LENGTHS_PROBS = NormLenProbEvaluator
    ROCQ = RocqEvaluator
    MY_EVALUATOR = MyEvaluator  # ← add yours
```

For async evaluators, subclass `AsyncBaseEvaluator` in
`src/treethink/async_evaluators.py` and register in `AsyncEvaluatorType`.

**Reference:** `src/treethink/evaluators.py`, `src/treethink/async_evaluators.py`

---

## Adding a New REPL Client

New clients go in a new directory under `src/treethink/clients/` and must
implement the ABCs in `src/treethink/clients/base.py`.

**Steps:**

1. Create a new directory `src/treethink/clients/my_lang/`.
2. Implement `ProofAssistantClient` (and optionally
   `AsyncProofAssistantClient`) from `src/treethink/clients/base.py`.
3. Add the language to the `ProofLanguage` enum in
   `src/treethink/utils/enums.py`.
4. Add match arms in `_create_raw_client()` and
   `_create_raw_async_client()` in `src/treethink/client_factory.py`.

```python
# src/treethink/utils/enums.py
class ProofLanguage(Enum):
    LEAN4 = "lean4"
    ROCQ = "rocq"
    ISABELLE = "isabelle"
    MY_LANG = "my_lang"  # ← add yours

# src/treethink/client_factory.py
case ProofLanguage.MY_LANG:
    return MyLangClient(...)
```

**Reference:** `src/treethink/clients/base.py`, `src/treethink/client_factory.py`
**Example:** `src/treethink/clients/lean/`, `coq/rocq.py`, `isabelle/client.py`

---

## Summary: Registration Points

| Component | Register in | File |
|-----------|-------------|------|
| Sync method | `IMPLEMENTED_METHODS` dict | `src/treethink/methods/__init__.py` |
| Sync policy | `PolicyType` enum | `src/treethink/policies.py` |
| Async policy | `AsyncPolicyType` enum | `src/treethink/async_policies.py` |
| Sync evaluator | `EvaluatorType` enum | `src/treethink/evaluators.py` |
| Async evaluator | `AsyncEvaluatorType` enum | `src/treethink/async_evaluators.py` |
| Language client | `ProofLanguage` enum + `client_factory.py` | `src/treethink/utils/enums.py`, `src/treethink/client_factory.py` |

For the actual class signatures and method contracts, always refer to the
source code — it is the authoritative API reference.
