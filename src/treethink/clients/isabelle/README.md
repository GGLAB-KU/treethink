# Isabelle Language Support via isabelle-client

Isabelle/HOL is supported via the [isabelle-client](https://pypi.org/project/isabelle-client/) library, which talks to Isabelle's built-in `isabelle server`. It was chosen over ISA-REPL and PISA: it is documented, maintained, and handles concurrent verification on a single session (~linear speedup; ~2.1s per check after a ~3.8s one-time `HOL` session start).

Install Isabelle (provides the `isabelle` binary + server) and the Python extra:

```bash
brew install --cask isabelle        # macOS; or download from isabelle.in.tum.de
uv pip install -e ".[isabelle]"
```

On macOS the bundle is not notarized and the server spawns many nested binaries, so clear the quarantine flag on the whole app (GUI "Open Anyway" is not enough):

```bash
xattr -dr com.apple.quarantine /Applications/Isabelle2025-2.app
```
