"""Natural-language "proof assistant" client — answer extraction + GT check.

Plugs into treethink like a formal-language client (Lean / Rocq / Isabelle),
but instead of REPL verification it (1) **extracts** the final answer from an
LLM's natural-language output and (2) **compares** it against a ground-truth
answer loaded from a dataset.  This lets tree search verify / score NL
problems (e.g. GSM8K / MATH-style) through the same
:class:`~treethink.clients.base.ProofAssistantClient` interface.

Extraction and comparison are **pluggable** (``extract_fn`` / ``compare_fn``)
so project-specific utility scripts can drop in; sensible defaults are
provided.  The ground truth is fetched directly from the dataset at
``client_args.nl_dataset_path``.
"""

import re
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from ..base import (
    AsyncProofAssistantClient,
    CheckResponse,
    ProofAssistantClient,
    SnippetResult,
)

# ---------------------------------------------------------------------------
# Default answer extraction / comparison (override via extract_fn/compare_fn)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_RE = re.compile(
    r"(?:final answer|the answer is|answer)\s*[:=]?\s*\$?([^\n$]+)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def default_extract_answer(text: str) -> Optional[str]:
    """Best-effort final-answer extraction from a natural-language solution.

    Tries, in order: the last ``\\boxed{...}``, then a trailing
    ``"Answer: ..."`` phrase, then the last number in the text.  Returns
    ``None`` when nothing is found.
    """
    if not text:
        return None
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    phrased = _ANSWER_RE.findall(text)
    if phrased:
        return phrased[-1].strip()
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def _normalize_answer(value: str) -> str:
    """Loose normalisation for comparing answers."""
    value = value.strip().lower()
    for token in ("$", " ", ","):
        value = value.replace(token, "")
    return value.rstrip(".")


def default_compare(predicted: str, ground_truth: str) -> bool:
    """Normalised string equality (case / spaces / ``$`` / commas / dot)."""
    return _normalize_answer(predicted) == _normalize_answer(ground_truth)


class NLClient(ProofAssistantClient):
    """Verify natural-language answers against ground truth from a dataset.

    Parameters
    ----------
    dataset_path : str, optional
        Path to a JSON / JSONL dataset with problem + answer fields.  When
        omitted, ground truth must be supplied via :meth:`set_ground_truth`.
    problem_key, answer_key : str
        Field names for the problem statement and the ground-truth answer.
    extract_fn : Callable[[str], Optional[str]], optional
        Extracts the predicted answer from an LLM output.  Defaults to
        :func:`default_extract_answer`.
    compare_fn : Callable[[str, str], bool], optional
        Compares predicted vs. ground-truth answers.  Defaults to
        :func:`default_compare`.

    Each :meth:`check` result's ``response`` is a dict
    ``{"backend", "correct", "extracted", "ground_truth"}``.
    """

    backend_name = "nl"

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        problem_key: str = "problem",
        answer_key: str = "answer",
        extract_fn: Optional[Callable[[str], Optional[str]]] = None,
        compare_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.problem_key = problem_key
        self.answer_key = answer_key
        self.extract_fn = extract_fn or default_extract_answer
        self.compare_fn = compare_fn or default_compare

        # Ground truth keyed by the normalised problem statement.
        self._gt_by_problem: Dict[str, str] = {}
        # Optional explicit override for the current problem's ground truth
        # (takes precedence over dataset lookup).
        self.current_ground_truth: Optional[str] = None

        if dataset_path is not None:
            self._load_dataset(dataset_path)

    # ------------------------------------------------------------------
    # Ground-truth loading / resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _problem_key(problem: str) -> str:
        """Whitespace/case-normalised key for robust problem matching."""
        return " ".join(problem.split()).strip().lower()

    def _load_dataset(self, path: str) -> None:
        import json
        from pathlib import Path

        file_path = Path(path)
        try:
            text = file_path.read_text()
        except Exception as exc:
            logger.error(f"NLClient failed to read dataset '{path}': {exc}")
            return

        try:
            if file_path.suffix in (".jsonl", ".ndjson"):
                rows = [
                    json.loads(line)
                    for line in text.splitlines()
                    if line.strip()
                ]
            else:  # .json (a list of rows, or {"data": [...]})
                data = json.loads(text)
                rows = data if isinstance(data, list) else data.get("data", [])
        except Exception as exc:
            logger.error(f"NLClient failed to parse dataset '{path}': {exc}")
            return

        count = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            problem = row.get(self.problem_key)
            answer = row.get(self.answer_key)
            if problem is None or answer is None:
                continue
            self._gt_by_problem[self._problem_key(str(problem))] = str(answer)
            count += 1
        logger.info(
            f"NLClient loaded {count} ground-truth answers from '{path}'."
        )

    def set_ground_truth(self, answer: Optional[str]) -> None:
        """Set the current problem's ground truth (overrides dataset lookup).

        Useful when the harness already knows the ground truth for the
        problem currently being searched.
        """
        self.current_ground_truth = answer

    def _resolve_ground_truth(self, snip: str) -> Optional[str]:
        if self.current_ground_truth is not None:
            return self.current_ground_truth
        if not self._gt_by_problem:
            return None
        # The problem statement is a prefix of the snippet (the root node's
        # text).  Pick the longest known problem that prefixes the snippet.
        normalized = self._problem_key(snip)
        best_key: Optional[str] = None
        for problem_key in self._gt_by_problem:
            if problem_key and normalized.startswith(problem_key):
                if best_key is None or len(problem_key) > len(best_key):
                    best_key = problem_key
        return self._gt_by_problem[best_key] if best_key else None

    # ------------------------------------------------------------------
    # ProofAssistantClient interface
    # ------------------------------------------------------------------

    def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> CheckResponse:
        """Extract each answer and compare it to the ground truth."""
        results: List[SnippetResult] = []
        for snip in snips:
            predicted = self.extract_fn(snip)
            ground_truth = self._resolve_ground_truth(snip)
            correct = bool(
                predicted is not None
                and ground_truth is not None
                and self.compare_fn(predicted, ground_truth)
            )
            results.append(
                SnippetResult(
                    response={
                        "backend": self.backend_name,
                        "correct": correct,
                        "extracted": predicted,
                        "ground_truth": ground_truth,
                    }
                )
            )
        return CheckResponse(results=results)

    def is_success_response(self, response: Any) -> bool:
        """``True`` when the extracted answer matches the ground truth."""
        if isinstance(response, dict):
            return bool(response.get("correct"))
        return bool(getattr(response, "correct", False))

    def close(self) -> None:
        return None


class AsyncNLClient(AsyncProofAssistantClient):
    """Async wrapper around :class:`NLClient`.

    NL checking is CPU-only and fast, so this simply delegates to a shared
    sync :class:`NLClient`; it exists so Rocq/Lean-style async pipelines can
    treat NL identically.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._sync = NLClient(*args, **kwargs)

    async def check(
        self,
        *,
        snips: List[str],
        timeout: int | None = None,
        show_progress: bool = False,
        batch_size: int = 8,
        max_workers: int = 4,
    ) -> CheckResponse:
        return self._sync.check(
            snips=snips,
            timeout=timeout,
            show_progress=show_progress,
            batch_size=batch_size,
            max_workers=max_workers,
        )

    def is_success_response(self, response: Any) -> bool:
        return self._sync.is_success_response(response)

    def set_ground_truth(self, answer: Optional[str]) -> None:
        self._sync.set_ground_truth(answer)

    async def close(self) -> None:
        self._sync.close()


__all__ = [
    "NLClient",
    "AsyncNLClient",
    "default_extract_answer",
    "default_compare",
]
