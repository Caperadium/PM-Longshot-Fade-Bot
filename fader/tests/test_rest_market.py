"""tests/test_rest_market.py

Unit tests for marketdata.rest_market.parse_market_outcomes (Phase 6, item 3
of temp/implementation-plan.md): centralizes the three previously-duplicated
Gamma clobTokenIds/outcomes parse sites (discover_new_rungs,
discover_series_markets, execution.provider.LiveProvider.resolve_no_token).

Run: python -m pytest fader/tests/test_rest_market.py -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from marketdata.rest_market import parse_market_outcomes, OUTCOME_NO


class TestParseMarketOutcomes(unittest.TestCase):
    def test_valid_already_decoded_lists(self):
        mkt = {
            "clobTokenIds": ["tok-yes", "tok-no"],
            "outcomes": ["Yes", "No"],
        }
        parsed = parse_market_outcomes(mkt)
        self.assertEqual(parsed["token_ids"], ["tok-yes", "tok-no"])
        self.assertEqual(parsed["outcomes"], ["Yes", "No"])
        self.assertEqual(parsed["no_index"], 1)
        self.assertEqual(parsed["no_token_id"], "tok-no")
        self.assertEqual(parsed["no_outcome"], "No")

    def test_json_string_fields(self):
        mkt = {
            "clobTokenIds": json.dumps(["tok-yes", "tok-no"]),
            "outcomes": json.dumps(["Yes", "No"]),
        }
        parsed = parse_market_outcomes(mkt)
        self.assertEqual(parsed["token_ids"], ["tok-yes", "tok-no"])
        self.assertEqual(parsed["no_index"], 1)
        self.assertEqual(parsed["no_token_id"], "tok-no")

    def test_case_insensitive_and_whitespace(self):
        mkt = {
            "clobTokenIds": ["tok-yes", "tok-no"],
            "outcomes": ["Yes", "  NO  "],
        }
        parsed = parse_market_outcomes(mkt)
        self.assertEqual(parsed["no_index"], 1)
        self.assertEqual(parsed["no_token_id"], "tok-no")

    def test_missing_outcomes_field(self):
        mkt = {"clobTokenIds": ["tok-yes", "tok-no"]}
        parsed = parse_market_outcomes(mkt)
        self.assertEqual(parsed["outcomes"], [])
        self.assertIsNone(parsed["no_index"])
        self.assertIsNone(parsed["no_token_id"])
        self.assertIsNone(parsed["no_outcome"])

    def test_missing_token_ids_field(self):
        mkt = {"outcomes": ["Yes", "No"]}
        parsed = parse_market_outcomes(mkt)
        self.assertEqual(parsed["token_ids"], [])
        # no_index requires i < len(token_ids); with an empty token list
        # there's no valid index, matching the pre-refactor loop guard.
        self.assertIsNone(parsed["no_index"])

    def test_no_no_outcome(self):
        mkt = {
            "clobTokenIds": ["tok-a", "tok-b"],
            "outcomes": ["Above", "Below"],
        }
        parsed = parse_market_outcomes(mkt)
        self.assertIsNone(parsed["no_index"])
        self.assertIsNone(parsed["no_token_id"])
        self.assertIsNone(parsed["no_outcome"])

    def test_malformed_json_string_falls_back_to_empty(self):
        mkt = {
            "clobTokenIds": "{not valid json",
            "outcomes": "{not valid json",
        }
        parsed = parse_market_outcomes(mkt)
        self.assertEqual(parsed["token_ids"], [])
        self.assertEqual(parsed["outcomes"], [])
        self.assertIsNone(parsed["no_index"])

    def test_empty_market_dict(self):
        parsed = parse_market_outcomes({})
        self.assertEqual(parsed["token_ids"], [])
        self.assertEqual(parsed["outcomes"], [])
        self.assertIsNone(parsed["no_index"])
        self.assertIsNone(parsed["no_token_id"])
        self.assertIsNone(parsed["no_outcome"])

    def test_outcome_no_constant_value(self):
        self.assertEqual(OUTCOME_NO, "no")


if __name__ == "__main__":
    unittest.main()
