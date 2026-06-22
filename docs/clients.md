# REPL Clients — Proof Assistant Backends

REPL clients are the **language-specific backends** that verify proof snippets
against formal proof assistants (Lean 4, Rocq, Isabelle).

> **Dependencies:** Each client requires its own extra. Install with
> `treethink[lean]`, `treethink[rocq]`, or `treethink[full]`.
> See [README → Installation](../README.md#installation) for details.

---

## Architecture

All clients implement one of two ABCs in `src/treethink/clients/base.py`:

```python
class ProofAssistantClient(ABC):       # Sync interface
    def check(self, *, snips, timeout, ...) -> Any
    def is_success_response(self, response) -> bool
    def close(self)

class AsyncProofAssistantClient(ABC):  # Async interface
    async def check(self, *, snips, timeout, ...) -> Any
    def is_success_response(self, response) -> bool
    def close(self)
```

This abstraction lets the termination layer and evaluators verify proof
snippets **without knowing which formal language is in use**.

---

## ClientFactory

**File:** `src/treethink/client_factory.py`

`create_client()` and `create_async_client()` are factory functions that build
the appropriate client based on the `FormalLanguage` enum:

```python
from treethink.client_factory import create_client
from treethink.utils.enums import FormalLanguage

client = create_client(
    language=FormalLanguage.LEAN4,
    client_args=my_client_args,
    cache=my_cache,  # optional
)
```

---

## Implemented Clients

### Lean 4 (Kimina)

**Client:** `LeanClientAdapter` / `AsyncLeanClientAdapter` in
`src/treethink/clients/lean/adapter.py`

- Communicates with a **Kimina Lean server** (typically
  `http://localhost:8000`).
- The adapter wraps the external `KiminaClient` library.
- Both sync and async variants are fully supported.

```yaml
treethink:
  language: "lean4"
  repl_args:
    lean_server_url: "http://localhost:8000"
    batch_size: 8
    num_proc: 4
    timeout: 400
```

> **Note:** We are pinned to a specific Kimina Lean Server commit because
> backward compatibility was dropped in later versions. Do NOT update
> `kimina-client` indiscriminately.

---

### Rocq (Coq 8.20)

**Client:** `RocqClient` in `src/treethink/clients/coq/rocq.py`

- Communicates with a **`rocq-ml-server`** (typically `localhost:5000`).
- Sync-only — async Rocq client is **not yet implemented**.

```yaml
treethink:
  language: "rocq"
  repl_args:
    host: "127.0.0.1"
    port: 5000
```

---

### Isabelle

**Client:** `IsabelleClient` in `src/treethink/clients/isabelle/client.py`

- **Experimental** — minimal implementation.
- Async variant is **not yet implemented**.

---

## ProofCache

**File:** `src/treethink/clients/cache.py`

An in-memory **LRU cache** for proof-snippet verification results.  When
enabled, previously verified snippets are served from the cache instead of
making network calls to the REPL server.

- Keyed by `sha256(proof_snippet)`.
- Configurable `maxsize` (default: 4096 entries).
- Wraps clients via `CachedClient` / `AsyncCachedClient`.
- Tracks hits/misses for diagnostics.

```yaml
treethink:
  repl_args:
    enable_cache: true
    cache_maxsize: 4096
```

---

## Adding a New Language Client

1. Implement `ProofAssistantClient` (and optionally
   `AsyncProofAssistantClient`) in a new directory under
   `src/treethink/clients/`.
2. Add the language to the `FormalLanguage` enum in
   `src/treethink/utils/enums.py`.
3. Add the match arm in `_create_raw_client()` and
   `_create_raw_async_client()` in `src/treethink/client_factory.py`.

See [extending.md](extending.md) for more detail.

For the full API, refer to the source at `src/treethink/clients/`.
