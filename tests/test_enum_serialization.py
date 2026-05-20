import importlib.util
import unittest
from pathlib import Path

from treethink.graph import _dataclass_from_dict
from treethink.utils.args import TreeThinkArgs
from treethink.utils.enums import FinalDecisionMode, TieBreaker

_PARSER_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "utils" / "parser.py"
)
_PARSER_SPEC = importlib.util.spec_from_file_location(
    "enum_test_parser", _PARSER_PATH
)
assert _PARSER_SPEC is not None and _PARSER_SPEC.loader is not None
_PARSER_MODULE = importlib.util.module_from_spec(_PARSER_SPEC)
_PARSER_SPEC.loader.exec_module(_PARSER_MODULE)

_parse_config_to_dataclass = _PARSER_MODULE._parse_config_to_dataclass
serialize_args = _PARSER_MODULE.serialize_args


class TestEnumSerialization(unittest.TestCase):
    def test_serialize_args_converts_enums_to_strings(self):
        args = TreeThinkArgs(
            final_decision_mode=FinalDecisionMode.CLEAR_FRONTIER,
            tie_breaker=TieBreaker.STABLE,
        )

        serialized = serialize_args(args)

        self.assertEqual(serialized["final_decision_mode"], "clear_frontier")
        self.assertEqual(serialized["tie_breaker"], "stable")

    def test_parse_config_coerces_enum_values(self):
        parsed = _parse_config_to_dataclass(
            {
                "final_decision_mode": "maximize_value",
                "tie_breaker": "deep",
            },
            TreeThinkArgs,
        )

        self.assertIsInstance(parsed.final_decision_mode, FinalDecisionMode)
        self.assertEqual(
            parsed.final_decision_mode, FinalDecisionMode.MAXIMIZE_VALUE
        )
        self.assertIsInstance(parsed.tie_breaker, TieBreaker)
        self.assertEqual(parsed.tie_breaker, TieBreaker.DEEP)

    def test_dataclass_from_dict_coerces_enum_values(self):
        reconstructed = _dataclass_from_dict(
            TreeThinkArgs,
            {
                "final_decision_mode": "maximize_visits",
                "tie_breaker": "random",
            },
        )

        self.assertEqual(
            reconstructed.final_decision_mode,
            FinalDecisionMode.MAXIMIZE_VISITS,
        )
        self.assertEqual(reconstructed.tie_breaker, TieBreaker.RANDOM)


if __name__ == "__main__":
    unittest.main()
