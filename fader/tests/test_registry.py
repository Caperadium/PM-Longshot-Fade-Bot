"""tests/test_registry.py

Unit tests for engine/registry.py's MarketRegistry (Phase 3 of the
architecture refactor, temp/implementation-plan.md -- Single-owner state).
MarketRegistry replaces the former bare {slug: MarketInfo} `token_map`
dict shared across strategy_loop/pollers/ws callbacks/main with a single
owner exposing get/add/mark_resolved/active_items/slugs.

Run: python -m pytest fader/tests/test_registry.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))


def _make_market_info(slug="s1", token_id="tok-1", active=True, closed=False):
    from execution.provider import MarketInfo
    return MarketInfo(
        slug=slug, condition_id="0xCOND", token_id=token_id,
        outcome="No", outcome_index=0, question="Q?",
        end_date_iso="2030-01-01T00:00:00+00:00",
        active=active, closed=closed,
    )


class TestMarketRegistryBasics(unittest.TestCase):
    def test_get_missing_slug_returns_none(self):
        from engine.registry import MarketRegistry
        reg = MarketRegistry()
        self.assertIsNone(reg.get("nope"))

    def test_add_then_get_round_trips(self):
        from engine.registry import MarketRegistry
        reg = MarketRegistry()
        mi = _make_market_info()
        reg.add("s1", mi)
        self.assertIs(reg.get("s1"), mi)

    def test_add_replaces_existing_slug(self):
        from engine.registry import MarketRegistry
        reg = MarketRegistry()
        mi1 = _make_market_info(token_id="tok-1")
        mi2 = _make_market_info(token_id="tok-2")
        reg.add("s1", mi1)
        reg.add("s1", mi2)
        self.assertIs(reg.get("s1"), mi2)

    def test_contains(self):
        from engine.registry import MarketRegistry
        reg = MarketRegistry()
        reg.add("s1", _make_market_info())
        self.assertIn("s1", reg)
        self.assertNotIn("s2", reg)

    def test_len(self):
        from engine.registry import MarketRegistry
        reg = MarketRegistry()
        self.assertEqual(len(reg), 0)
        reg.add("s1", _make_market_info(slug="s1", token_id="t1"))
        reg.add("s2", _make_market_info(slug="s2", token_id="t2"))
        self.assertEqual(len(reg), 2)

    def test_slugs_returns_snapshot_list(self):
        from engine.registry import MarketRegistry
        reg = MarketRegistry()
        reg.add("s1", _make_market_info(slug="s1", token_id="t1"))
        reg.add("s2", _make_market_info(slug="s2", token_id="t2"))
        slugs = reg.slugs()
        self.assertEqual(sorted(slugs), ["s1", "s2"])
        # Mutating the returned list must not affect the registry.
        slugs.append("s3")
        self.assertEqual(sorted(reg.slugs()), ["s1", "s2"])

    def test_active_items_returns_snapshot_copy(self):
        """active_items() must be a copy -- mutating the registry while
        a caller holds an old snapshot must not change that snapshot
        (strategy_loop iterates while pollers' discovery loop may add)."""
        from engine.registry import MarketRegistry
        reg = MarketRegistry()
        reg.add("s1", _make_market_info(slug="s1", token_id="t1"))
        snapshot = reg.active_items()
        reg.add("s2", _make_market_info(slug="s2", token_id="t2"))
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(reg.active_items()), 2)


class TestMarkResolved(unittest.TestCase):
    def test_mark_resolved_matching_token_sets_closed_and_inactive(self):
        from engine.registry import MarketRegistry
        reg = MarketRegistry()
        mi = _make_market_info(slug="s1", token_id="tok-1")
        reg.add("s1", mi)

        slug = reg.mark_resolved("tok-1")

        self.assertEqual(slug, "s1")
        self.assertTrue(mi.closed)
        self.assertFalse(mi.active)

    def test_mark_resolved_no_match_returns_none_and_no_mutation(self):
        from engine.registry import MarketRegistry
        reg = MarketRegistry()
        mi = _make_market_info(slug="s1", token_id="tok-1")
        reg.add("s1", mi)

        result = reg.mark_resolved("tok-does-not-exist")

        self.assertIsNone(result)
        self.assertFalse(mi.closed)
        self.assertTrue(mi.active)

    def test_mark_resolved_matches_by_token_id_not_slug(self):
        from engine.registry import MarketRegistry
        reg = MarketRegistry()
        mi1 = _make_market_info(slug="s1", token_id="tok-1")
        mi2 = _make_market_info(slug="s2", token_id="tok-2")
        reg.add("s1", mi1)
        reg.add("s2", mi2)

        slug = reg.mark_resolved("tok-2")

        self.assertEqual(slug, "s2")
        self.assertTrue(mi2.closed)
        self.assertFalse(mi1.closed)  # unaffected


if __name__ == "__main__":
    unittest.main()
