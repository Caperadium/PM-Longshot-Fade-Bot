"""tests/test_model_pricer.py

Unit tests for strategy/model_pricer.py -- the FIGARCH model-pricer
integration. All tests run against injected seams (inline submit_fn, fake
compute_fn, controlled clock, fake file_age_fn); the real pricing engine
is never imported.

Covered:
  - BTC slug parsing (numeric, k-suffixed, non-BTC rejects)
  - evaluate() fail-open paths (disabled, non-BTC, missing dte, no cache)
  - edge math and cache behavior (TTL, new-strike recompute, stale result)
  - data-staleness gate (no model opinion on old BTC data)
  - compute-failure retry backoff (no hot-loop resubmit)
  - should_veto() logic (log-only default, veto threshold, None verdict)

Run: python -m pytest fader/tests/test_model_pricer.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Add fader root to path
_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from config.config_loader import PricerConfig
from strategy import model_pricer as mp
from strategy.model_pricer import ModelPricer, parse_btc_market


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _make_pricer(cfg=None, compute_fn=None, clock=None, file_age=100.0):
    """ModelPricer with an inline (synchronous) submit_fn and fakes."""
    cfg = cfg or PricerConfig()
    clock = clock or _Clock()

    calls = []

    def default_compute(strikes, hours):
        calls.append((tuple(strikes), hours))
        return {s: 0.15 for s in strikes}

    pricer = ModelPricer(
        cfg,
        submit_fn=lambda fn, *a: fn(*a),  # run worker inline
        compute_fn=compute_fn or default_compute,
        refresh_fn=lambda: None,
        file_age_fn=lambda path: file_age,
        now_fn=clock,
    )
    return pricer, calls, clock, cfg


class TestParseBtcMarket(unittest.TestCase):
    def test_numeric_strike(self):
        self.assertEqual(
            parse_btc_market("bitcoin-above-107000-on-july-22"),
            (107000.0, "july-22"),
        )

    def test_year_suffixed_expiry(self):
        self.assertEqual(
            parse_btc_market("bitcoin-above-64000-on-july-22-2026"),
            (64000.0, "july-22-2026"),
        )

    def test_k_suffix(self):
        self.assertEqual(
            parse_btc_market("bitcoin-above-100k-on-january-1"),
            (100000.0, "january-1"),
        )

    def test_decimal_k_suffix(self):
        self.assertEqual(
            parse_btc_market("bitcoin-above-112.5k-on-march-3"),
            (112500.0, "march-3"),
        )

    def test_rejects_non_btc(self):
        self.assertIsNone(parse_btc_market("ethereum-above-4000-on-july-22"))
        self.assertIsNone(
            parse_btc_market("highest-temperature-in-seoul-on-july-22")
        )
        self.assertIsNone(parse_btc_market("bitcoin-above-on-july-22"))


class TestEvaluate(unittest.TestCase):
    SLUG = "bitcoin-above-64000-on-july-22"

    def test_disabled_returns_none_and_never_computes(self):
        pricer, calls, _, _ = _make_pricer(cfg=PricerConfig(enabled=False))
        self.assertIsNone(pricer.evaluate(self.SLUG, 0.80, 4.0))
        self.assertEqual(calls, [])

    def test_non_btc_returns_none(self):
        pricer, calls, _, _ = _make_pricer()
        self.assertIsNone(
            pricer.evaluate("ethereum-above-4000-on-july-22", 0.80, 4.0)
        )
        self.assertEqual(calls, [])

    def test_missing_dte_returns_none(self):
        pricer, calls, _, _ = _make_pricer()
        self.assertIsNone(pricer.evaluate(self.SLUG, 0.80, None))
        self.assertEqual(calls, [])

    def test_first_call_none_then_verdict_with_edge_math(self):
        pricer, calls, _, _ = _make_pricer()
        # First call: no cache yet -> None, but compute was scheduled and
        # (inline submit) has already run.
        self.assertIsNone(pricer.evaluate(self.SLUG, 0.80, 4.0))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ((64000.0,), 96.0))
        # Second call: cache hit. p_yes=0.15 -> p_no=0.85, ask 0.80 -> +0.05.
        v = pricer.evaluate(self.SLUG, 0.80, 4.0)
        self.assertIsNotNone(v)
        self.assertAlmostEqual(v["model_p_yes"], 0.15)
        self.assertAlmostEqual(v["model_edge_no"], 0.05)
        self.assertEqual(len(calls), 1)  # no recompute on cache hit

    def test_cache_ttl_recompute(self):
        pricer, calls, clock, cfg = _make_pricer()
        pricer.evaluate(self.SLUG, 0.80, 4.0)
        self.assertEqual(len(calls), 1)
        clock.t += cfg.reprice_s + 1
        pricer.evaluate(self.SLUG, 0.80, 4.0)
        self.assertEqual(len(calls), 2)

    def test_new_strike_triggers_recompute_covering_it(self):
        pricer, calls, _, _ = _make_pricer()
        pricer.evaluate(self.SLUG, 0.80, 4.0)
        other = "bitcoin-above-70000-on-july-22"
        # New strike not in cached ladder: schedules a recompute that now
        # includes both strikes, and (inline submit) serves it immediately
        # on the next call.
        first = pricer.evaluate(other, 0.90, 4.0)
        self.assertIsNone(first)  # verdict served only from pre-schedule cache
        self.assertEqual(calls[-1], ((64000.0, 70000.0), 96.0))
        v = pricer.evaluate(other, 0.90, 4.0)
        self.assertIsNotNone(v)
        self.assertAlmostEqual(v["model_edge_no"], (1.0 - 0.15) - 0.90)

    def test_expiries_cached_independently(self):
        pricer, calls, _, _ = _make_pricer()
        pricer.evaluate(self.SLUG, 0.80, 4.0)
        pricer.evaluate("bitcoin-above-64000-on-july-23", 0.80, 5.0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], ((64000.0,), 120.0))

    def test_stale_result_returns_none(self):
        pricer, calls, clock, cfg = _make_pricer()
        pricer.evaluate(self.SLUG, 0.80, 4.0)
        self.assertIsNotNone(pricer.evaluate(self.SLUG, 0.80, 4.0))
        # Age the cached ladder past 3x reprice_s and make every recompute
        # fail: old probabilities must not be served as current.
        clock.t += 3 * cfg.reprice_s + 1

        def failing(strikes, hours):
            raise RuntimeError("engine down")

        pricer._compute_fn = failing
        self.assertIsNone(pricer.evaluate(self.SLUG, 0.80, 4.0))

    def test_stale_btc_data_suspends_model(self):
        # file_age_fn -> None (missing file) counts as stale
        pricer, calls, _, _ = _make_pricer(file_age=None)
        self.assertIsNone(pricer.evaluate(self.SLUG, 0.80, 4.0))
        self.assertEqual(calls, [])  # worker bailed before compute
        # file older than data_max_age_s counts as stale
        pricer2, calls2, _, _ = _make_pricer(
            file_age=PricerConfig().data_max_age_s + 1
        )
        self.assertIsNone(pricer2.evaluate(self.SLUG, 0.80, 4.0))
        self.assertEqual(calls2, [])

    def test_compute_failure_backoff_no_hot_loop(self):
        attempts = []

        def failing(strikes, hours):
            attempts.append(1)
            raise RuntimeError("boom")

        pricer, _, clock, _ = _make_pricer(compute_fn=failing)
        self.assertIsNone(pricer.evaluate(self.SLUG, 0.80, 4.0))
        self.assertEqual(len(attempts), 1)
        # Immediate retry suppressed by backoff
        self.assertIsNone(pricer.evaluate(self.SLUG, 0.80, 4.0))
        self.assertEqual(len(attempts), 1)
        # After backoff elapses, one more attempt
        clock.t += mp._RETRY_BACKOFF_S + 1
        self.assertIsNone(pricer.evaluate(self.SLUG, 0.80, 4.0))
        self.assertEqual(len(attempts), 2)


class TestShouldVeto(unittest.TestCase):
    def test_log_only_default_never_vetoes(self):
        pricer, _, _, _ = _make_pricer()  # veto=False default
        self.assertFalse(
            pricer.should_veto({"model_edge_no": -0.50, "model_p_yes": 0.9})
        )

    def test_none_verdict_never_vetoes(self):
        pricer, _, _, _ = _make_pricer(
            cfg=PricerConfig(veto=True, min_edge=0.02)
        )
        self.assertFalse(pricer.should_veto(None))

    def test_veto_threshold(self):
        pricer, _, _, _ = _make_pricer(
            cfg=PricerConfig(veto=True, min_edge=0.02)
        )
        self.assertTrue(pricer.should_veto({"model_edge_no": 0.019}))
        self.assertFalse(pricer.should_veto({"model_edge_no": 0.020}))

    def test_hot_reload_flip(self):
        # ModelPricer holds the live PricerConfig object; mutating it in
        # place (what ConfigWatcher._apply_hot does) changes behavior with
        # no restart.
        cfg = PricerConfig(veto=False, min_edge=0.02)
        pricer, _, _, _ = _make_pricer(cfg=cfg)
        v = {"model_edge_no": -0.10}
        self.assertFalse(pricer.should_veto(v))
        cfg.veto = True
        self.assertTrue(pricer.should_veto(v))


if __name__ == "__main__":
    unittest.main()
