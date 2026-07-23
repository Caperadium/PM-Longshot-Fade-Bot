"""strategy/model_pricer.py

FIGARCH model-pricer integration for the anti-longshot entry stack.

Wraps the vendored V2 pricing engine (repo-root ``core/pricing/``, see
PRICER_README.md) as an optional per-contract model-edge signal for BTC
daily binaries. The strategy loop consults it AFTER filters 1-8 pass:

  - log-only mode (``pricer.veto: false``, the default): the model's YES
    probability and NO-side edge are attached to every entered decision's
    filters dict, so live evidence accumulates in the decisions table
    without changing behavior.
  - veto mode (``pricer.veto: true``): entries whose model NO edge is
    below ``pricer.min_edge`` are rejected with reason ``model_edge_low``.

Edge convention: buying NO at the NO best ask,
    model_edge_no = (1 - model_p_yes) - no_ask
so a positive edge means the model thinks NO is underpriced at the ask.

Fail-open by design: any condition that prevents a model opinion (pricer
disabled, non-BTC market, unparseable slug, no cached ladder yet, BTC data
stale, engine import/compute failure) makes ``evaluate`` return None and
the entry proceeds exactly as the naive strategy would. The model can only
ever narrow the naive strategy, never widen it, and never block it by
being broken.

Concurrency model: ``evaluate`` is synchronous and cheap (dict lookups) --
safe to call from the 1s asyncio strategy tick. Ladder computes (GARCH/
FIGARCH MLE + Monte Carlo, seconds to tens of seconds) run on the injected
executor via ``submit_fn``; one ladder per expiry per ``reprice_s``, with
a global serialization lock so two MLE fits never run concurrently, and a
per-expiry in-flight flag plus retry backoff so a failing compute cannot
hot-loop. BTC data refresh shells out to the vendored stdlib
``core/data/data_fetcher.py`` (atomic writes, incremental) on its own
cadence; set ``data_refresh_s: 0`` when an external timer owns the fetch.

All engine imports happen lazily inside the worker thread -- constructing
a ModelPricer never imports numpy/arch, so the engine starts fast and a
missing dependency degrades to log-only-None instead of a crash.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

# fader/strategy/model_pricer.py -> fader/strategy -> fader -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Matches bitcoin-above-107000-on-july-22[-2026] and bitcoin-above-100k-on-...
_BTC_SLUG_RE = re.compile(
    r"^bitcoin-above-(\d+(?:\.\d+)?)(k)?-on-(.+)$"
)

_RETRY_BACKOFF_S = 60.0        # min gap between compute attempts after a failure
_STALE_WARN_EVERY_S = 600.0    # throttle for the data-stale warning log


def parse_btc_market(slug: str):
    """Parse a BTC daily binary slug into (strike, expiry_key).

    Returns (float, str) or None for anything that is not a bitcoin-above
    market (ETH, Seoul temp, malformed). Handles both the plain-number form
    (bitcoin-above-107000-on-july-22) and the k-suffixed form
    (bitcoin-above-100k-on-january-1).
    """
    m = _BTC_SLUG_RE.match(slug)
    if not m:
        return None
    try:
        strike = float(m.group(1))
    except ValueError:
        return None
    if m.group(2):
        strike *= 1000.0
    if strike <= 0:
        return None
    return strike, m.group(3)


class ModelPricer:
    """Cached FIGARCH ladder pricer with a fail-open evaluate() front end.

    pricer_cfg is the live PricerConfig object from AppConfig -- the config
    hot-reloader mutates its fields in place, so enabled/veto/min_edge etc.
    are re-read on every call and need no restart.
    """

    def __init__(
        self,
        pricer_cfg,
        submit_fn: Callable[..., Any],
        compute_fn: Optional[Callable] = None,
        refresh_fn: Optional[Callable[[], None]] = None,
        file_age_fn: Optional[Callable[[Path], Optional[float]]] = None,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = pricer_cfg
        self._submit = submit_fn
        self._compute_fn = compute_fn      # test seam; None = real engine
        self._refresh_fn = refresh_fn      # test seam; None = subprocess fetch
        self._file_age_fn = file_age_fn    # test seam; None = mtime stat
        self._now = now_fn

        self._lock = threading.Lock()
        self._compute_serial = threading.Lock()  # one MLE at a time
        # expiry_key -> {"probs": {strike: p_yes}, "ts": monotonic}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._strikes: Dict[str, Set[float]] = {}
        self._hours: Dict[str, float] = {}
        self._inflight: Set[str] = set()
        self._last_fail: Dict[str, float] = {}

        self._garch_cache: dict = {}
        self._garch_ts: Optional[float] = None
        self._jump_params: Optional[dict] = None
        self._last_refresh: Optional[float] = None
        self._last_stale_warn: Optional[float] = None

    # ------------------------------------------------------------------
    # Front end (called from the strategy loop, every tick)
    # ------------------------------------------------------------------

    def evaluate(
        self, slug: str, no_ask: float, dte_days: Optional[float]
    ) -> Optional[Dict[str, Any]]:
        """Model verdict for one contract, or None (= no opinion, fail open).

        Registers the contract's strike under its expiry and schedules a
        background ladder compute when the cache is missing, expired, or
        does not cover this strike yet. Never blocks.
        """
        if not self._cfg.enabled:
            return None
        if dte_days is None:
            return None
        parsed = parse_btc_market(slug)
        if parsed is None:
            return None
        strike, expiry_key = parsed
        hours = max(0.1, dte_days * 24.0)

        now = self._now()
        with self._lock:
            self._strikes.setdefault(expiry_key, set()).add(strike)
            self._hours[expiry_key] = hours
            entry = self._cache.get(expiry_key)
            needs_compute = (
                entry is None
                or (now - entry["ts"]) > self._cfg.reprice_s
                or strike not in entry["probs"]
            )
        if needs_compute:
            self._schedule(expiry_key)
        if entry is None:
            return None
        age = now - entry["ts"]
        # A persistently failing/stale compute must not serve arbitrarily
        # old probabilities as if current.
        if age > 3.0 * self._cfg.reprice_s:
            return None
        p_yes = entry["probs"].get(strike)
        if p_yes is None:
            return None
        return {
            "model_p_yes": round(p_yes, 4),
            "model_edge_no": round((1.0 - p_yes) - no_ask, 4),
            "model_age_s": round(age, 1),
        }

    def should_veto(self, verdict: Optional[Dict[str, Any]]) -> bool:
        """True when veto mode is on and the model edge is below min_edge.

        A None verdict never vetoes (fail-open).
        """
        if verdict is None or not self._cfg.veto:
            return False
        return verdict["model_edge_no"] < self._cfg.min_edge

    @property
    def min_edge(self) -> float:
        return self._cfg.min_edge

    # ------------------------------------------------------------------
    # Background compute
    # ------------------------------------------------------------------

    def _schedule(self, expiry_key: str) -> None:
        now = self._now()
        with self._lock:
            if expiry_key in self._inflight:
                return
            last_fail = self._last_fail.get(expiry_key)
            if last_fail is not None and (now - last_fail) < _RETRY_BACKOFF_S:
                return
            self._inflight.add(expiry_key)
        try:
            self._submit(self._worker, expiry_key)
        except Exception:
            with self._lock:
                self._inflight.discard(expiry_key)
            raise

    def _worker(self, expiry_key: str) -> None:
        try:
            self._maybe_refresh_data()
            if self._data_stale():
                self._warn_stale()
                with self._lock:
                    self._last_fail[expiry_key] = self._now()
                return
            with self._lock:
                strikes = sorted(self._strikes.get(expiry_key, ()))
                hours = self._hours.get(expiry_key)
            if not strikes or hours is None:
                return
            with self._compute_serial:
                probs = self._run_compute(strikes, hours)
            with self._lock:
                self._cache[expiry_key] = {"probs": probs, "ts": self._now()}
                self._last_fail.pop(expiry_key, None)
            logger.info(
                f"Model pricer: {expiry_key} ladder priced "
                f"({len(probs)} strikes, {hours:.1f}h)"
            )
        except Exception as e:
            with self._lock:
                self._last_fail[expiry_key] = self._now()
            logger.warning(f"Model pricer compute failed for {expiry_key}: {e}")
        finally:
            with self._lock:
                self._inflight.discard(expiry_key)

    def _run_compute(self, strikes, hours) -> Dict[float, float]:
        if self._compute_fn is not None:
            return self._compute_fn(strikes, hours)
        return self._real_compute(strikes, hours)

    # ------------------------------------------------------------------
    # Real engine plumbing (lazy imports, worker thread only)
    # ------------------------------------------------------------------

    def _data_dir(self) -> Path:
        configured = getattr(self._cfg, "data_dir", "") or ""
        return Path(configured) if configured else (REPO_ROOT / "DATA")

    def _real_compute(self, strikes, hours) -> Dict[float, float]:
        root = str(REPO_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from core.pricing.btc_pricing_engine import calculate_probabilities

        self._maybe_refit_shared_state()
        data = self._data_dir()
        result = calculate_probabilities(
            strikes=list(strikes),
            hours_to_expiry=float(hours),
            hourly_csv=str(data / "btc_hourly.csv"),
            intraday_csv=str(data / "btc_intraday_1m.csv"),
            n_sims=int(self._cfg.n_sims),
            jump_params=self._jump_params,
            use_svcj=True,
            use_skewed_t=True,
            use_figarch=True,
            garch_cache=self._garch_cache,
        )
        result.pop("_meta", None)
        return {float(k): float(v) for k, v in result.items()}

    def _maybe_refit_shared_state(self) -> None:
        """Reload calibrated jump params and clear the GARCH cache on the
        garch_refit_s cadence, so the MLE from hours ago does not price
        forever. Same pattern as the V2 market-maker's CachedEngine."""
        now = self._now()
        if (
            self._garch_ts is not None
            and (now - self._garch_ts) < self._cfg.garch_refit_s
        ):
            return
        self._garch_cache.clear()
        self._garch_ts = now
        self._jump_params = self._load_jump_params()

    def _load_jump_params(self) -> Optional[dict]:
        """Bipower-calibrated Kou jump params, keyed for the engine.

        load_calibrated_jumps returns 'lam'/'p_crash' keys; the engine
        expects 'lambda'/'crash_prob' -- passing the raw dict through would
        silently drop the calibrated values back to module defaults
        (mirrors V2 shadow_runner.load_jump_params_for_engine). None on any
        failure = engine defaults; pricing must not die on calibration.
        """
        try:
            from core.pricing.btc_pricing_engine import load_calibrated_jumps

            data = self._data_dir()
            cal = load_calibrated_jumps(
                hourly_csv=str(data / "btc_hourly.csv"),
                cache_path=str(data / "jump_calibration.csv"),
            )
            if not cal.get("fit_converged"):
                logger.warning(
                    "Jump calibration not converged; engine default jumps"
                )
                return None
            return {
                "lambda": cal["lam"], "crash_prob": cal["p_crash"],
                "eta_up": cal["eta_up"], "eta_down": cal["eta_down"],
                "mu_v": cal["mu_v"], "rho_J": cal["rho_J"],
                "rho_j_slope": cal.get("rho_j_slope", 0.0),
            }
        except Exception:
            logger.warning(
                "Jump calibration load failed; engine default jumps",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # BTC data freshness
    # ------------------------------------------------------------------

    def _intraday_age_s(self) -> Optional[float]:
        path = self._data_dir() / "btc_intraday_1m.csv"
        if self._file_age_fn is not None:
            return self._file_age_fn(path)
        try:
            return max(0.0, time.time() - path.stat().st_mtime)
        except OSError:
            return None

    def _data_stale(self) -> bool:
        age = self._intraday_age_s()
        return age is None or age > self._cfg.data_max_age_s

    def _warn_stale(self) -> None:
        now = self._now()
        if (
            self._last_stale_warn is not None
            and (now - self._last_stale_warn) < _STALE_WARN_EVERY_S
        ):
            return
        self._last_stale_warn = now
        age = self._intraday_age_s()
        logger.warning(
            f"Model pricer: BTC intraday data stale "
            f"(age={age if age is None else round(age)}s, "
            f"max={self._cfg.data_max_age_s}s); model opinions suspended"
        )

    def _maybe_refresh_data(self) -> None:
        """Run the vendored stdlib data fetcher when the refresh cadence is
        due. data_refresh_s <= 0 disables (external cron/timer owns the
        fetch). Failures are logged and swallowed -- the staleness gate is
        the actual guard."""
        if self._cfg.data_refresh_s <= 0:
            return
        now = self._now()
        if (
            self._last_refresh is not None
            and (now - self._last_refresh) < self._cfg.data_refresh_s
        ):
            return
        self._last_refresh = now
        try:
            if self._refresh_fn is not None:
                self._refresh_fn()
            else:
                script = REPO_ROOT / "core" / "data" / "data_fetcher.py"
                subprocess.run(
                    [sys.executable, str(script)],
                    cwd=str(REPO_ROOT),
                    timeout=600,
                    capture_output=True,
                )
        except Exception as e:
            logger.warning(f"Model pricer: BTC data refresh failed: {e}")
