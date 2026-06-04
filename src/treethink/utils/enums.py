from enum import Enum
from typing import Type, TypeVar


class FinalDecisionMode(Enum):
    NATIVE = "native"
    MAXIMIZE_VISITS = "maximize_visits"
    MAXIMIZE_VALUE = "maximize_value"
    CLEAR_FRONTIER = "clear_frontier"


class BestAnswerReason(Enum):
    CALCULATED = "calculated"
    SET = "set"
    CHECKED_AND_TRUE = "checked_and_true"


class TieBreaker(Enum):
    RANDOM = "random"
    DEEP = "deep"
    STABLE = "stable"


class FormalLanguage(Enum):
    """Supported formal proof assistants for termination / REPL clients."""

    LEAN4 = "lean4"
    RCOQ = "rcoq"
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
