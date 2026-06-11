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
    """Supported formal proof assistants for termination / REPL clients."""

    LEAN4 = "lean4"
    ROCQ = "rocq"
    ISABELLE = "isabelle"


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
