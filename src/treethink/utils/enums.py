from enum import Enum
from typing import Type, TypeVar


class FinalDecisionMode(Enum):
    """How the best answer is selected after search completes.

    Options:
        NATIVE: Use the method's native best-node selection.
        MAXIMIZE_VISITS: Select the most-visited node.
        MAXIMIZE_VALUE: Select the highest-valued node.
        CLEAR_FRONTIER: Re-score frontier nodes and pick the best.
    """

    NATIVE = "native"
    MAXIMIZE_VISITS = "maximize_visits"
    MAXIMIZE_VALUE = "maximize_value"
    CLEAR_FRONTIER = "clear_frontier"


class BestAnswerReason(Enum):
    """How the best answer was determined.

    Options:
        CALCULATED: Selected by the method's scoring/scoring function.
        SET: Manually set via callback.
        CHECKED_AND_TRUE: Verified as correct by the REPL client.
    """

    CALCULATED = "calculated"
    SET = "set"
    CHECKED_AND_TRUE = "checked_and_true"


class TieBreaker(Enum):
    """Tie-breaking strategy for nodes with equal scores.

    Options:
        RANDOM: Pick randomly among tied nodes.
        DEEP: Prefer the deeper node.
        STABLE: Deterministic — pick the first encountered.
    """

    RANDOM = "random"
    DEEP = "deep"
    STABLE = "stable"


class FormalLanguage(Enum):
    """Languages/backends for termination / REPL(-style) verification.

    ``NL`` is not a formal language: it verifies natural-language answers by
    extraction + ground-truth comparison, reusing the same client interface.
    """

    LEAN4 = "lean4"
    ROCQ = "rocq"
    ISABELLE = "isabelle"
    NL = "nl"


class PoolingTask(Enum):
    """vLLM pooling task for reward-model evaluators.

    Options:
        CLASSIFY: Sequence reward models (one score per sequence).
        TOKEN_CLASSIFY: Token / process reward models (per-token scores).
    """

    CLASSIFY = "classify"
    TOKEN_CLASSIFY = "token_classify"


class ScoreReduction(Enum):
    """How to reduce a reward vector to a single scalar score.

    Options:
        LAST: Use the last value (default for value-head rewards).
        MEAN: Average all values.
        FIRST: Use the first value.
    """

    LAST = "last"
    MEAN = "mean"
    FIRST = "first"


EnumType = TypeVar("EnumType", bound=Enum)


def coerce_enum(value, enum_type: Type[EnumType]) -> EnumType:
    if value is None or isinstance(value, enum_type):
        return value

    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{value!r} is not a valid {enum_type.__name__}"
        ) from exc
