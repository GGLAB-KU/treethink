import unittest

from treethink.graph import _dataclass_from_dict
from treethink.utils.args import TreeThinkArgs
from treethink.utils.enums import FinalDecisionMode, TieBreaker
from treethink.utils.parser import _parse_config_to_dataclass, serialize_args


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
