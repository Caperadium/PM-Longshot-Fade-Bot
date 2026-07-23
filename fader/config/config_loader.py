"""config/config_loader.py

Load, validate, and provide hot-reload access to config.yaml + slugs.csv.

Hot-reloadable params: band, DTE, min_time_in_band_s, volumes, depth, sizing,
  risk caps, breaker %, poll/decision intervals.
Cold params (restart required): ws_url, auth, db path, telegram credentials.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent

# Single authoritative default for the CLOB market-data websocket URL
# (Phase 6, item 2). config.yaml's feed.ws_url is the user-facing override;
# ws_client.py no longer keeps its own module-level copy of this default.
DEFAULT_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

COLD_PARAMS = frozenset({
    "feed.ws_url",
    "auth.mode",
    "telegram.token",
    "telegram.chat_id",
})


@dataclass
class StrategyConfig:
    band_low: float = 0.80
    band_high: float = 0.95
    min_dte: int = 0
    max_dte: int = 365
    min_time_in_band_s: int = 600
    order_notional_usd: float = 10.0
    alpha: float = 0.0  # tilt ∈ [-1, +1]; 0=uniform, -1=low-price, +1=high-price


@dataclass
class FiltersConfig:
    min_24h_volume: float = 1000.0
    min_total_volume: float = 10000.0
    min_book_depth: float = 0.0


@dataclass
class FeedConfig:
    decision_interval_s: float = 1.0
    max_staleness_seconds: int = 30
    gap_halt_seconds: int = 60
    ws_url: str = DEFAULT_WS_URL
    ws_force_reconnect_s: int = 90   # force WS reconnect if no feed data this long
    ws_ping_interval_s: int = 10     # app-level PING cadence
    ws_pong_timeout_s: int = 25      # close socket if no PONG within this (1a only)
    ws_expect_pong: bool = False     # enable pong-timeout close (1a); default off
    resync_concurrency: int = 8      # bounded-concurrent REST /book resync on reconnect (COLD)
    executor_workers: int = 16       # sized ThreadPoolExecutor for blocking REST calls


@dataclass
class PollingConfig:
    bankroll_s: int = 30
    resolution_s: int = 60
    discovery_s: int = 300
    control_poll_s: float = 1.0
    calibration_fetch_s: int = 21600  # 6h; <= 0 disables the calibration-data poller


@dataclass
class OrdersConfig:
    spread_market_threshold_c: float = 1.0
    requote_move_c: float = 0.5
    limit_ttl_s: int = 300


@dataclass
class RiskConfig:
    daily_loss_breaker_pct: float = 5.0
    max_deployed_pct: float = 100.0
    per_market_cap_pct: float = 5.0
    matic_min_balance: float = 0.5


@dataclass
class TelegramConfig:
    heartbeat_minutes: int = 15
    enabled: bool = True


@dataclass
class RateLimitConfig:
    write_per_s: float = 10.0
    write_burst: int = 20
    read_per_s: float = 5.0
    read_burst: int = 10


@dataclass
class AuthConfig:
    mode: str = "eoa"


@dataclass
class BankrollConfig:
    basis: str = "cash"
    paper_bankroll_usdc: float = 0.0


@dataclass
class PricerConfig:
    """FIGARCH model-pricer integration (strategy/model_pricer.py).

    veto=False (default) = log-only: model probability/edge are attached to
    entered decisions but never reject. veto=True rejects entries whose
    model NO edge is below min_edge. All fields hot-reloadable."""
    enabled: bool = True
    veto: bool = False
    min_edge: float = 0.0        # NO-side model edge threshold (veto mode)
    reprice_s: int = 900         # per-expiry ladder cache TTL
    n_sims: int = 15000          # Monte Carlo paths per ladder
    garch_refit_s: int = 21600   # clear GARCH cache + reload jump params (6h)
    data_max_age_s: int = 7200   # BTC intraday CSV older than this = no model opinion
    data_refresh_s: int = 1800   # background BTC data fetch cadence; 0 = external timer owns it
    data_dir: str = ""           # empty = <repo root>/DATA


@dataclass
class SlugRow:
    slug: str
    enabled: bool
    market_kind: str  # binary | ladder | series
    series_from_date: str = ""
    series_filter: str = ""
    band_low: Optional[float] = None
    band_high: Optional[float] = None
    size_override: Optional[float] = None
    added_at: str = ""
    notes: str = ""


@dataclass
class AppConfig:
    mode: str = "paper"
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    feed: FeedConfig = field(default_factory=FeedConfig)
    polling: PollingConfig = field(default_factory=PollingConfig)
    orders: OrdersConfig = field(default_factory=OrdersConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    ratelimit: RateLimitConfig = field(default_factory=RateLimitConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    bankroll: BankrollConfig = field(default_factory=BankrollConfig)
    pricer: PricerConfig = field(default_factory=PricerConfig)
    slugs: List[SlugRow] = field(default_factory=list)

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def enabled_slugs(self) -> List[SlugRow]:
        return [s for s in self.slugs if s.enabled]

    def band_for_slug(self, slug: str):
        for s in self.slugs:
            if s.slug == slug:
                low = s.band_low if s.band_low is not None else self.strategy.band_low
                high = s.band_high if s.band_high is not None else self.strategy.band_high
                return low, high
        return self.strategy.band_low, self.strategy.band_high

    def size_for_slug(self, slug: str) -> float:
        for s in self.slugs:
            if s.slug == slug and s.size_override is not None:
                return s.size_override
        return self.strategy.order_notional_usd


def _parse_optional_float(val: str) -> Optional[float]:
    v = val.strip() if val else ""
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _load_slugs(path: Path) -> List[SlugRow]:
    rows = []
    if not path.exists():
        logger.warning(f"slugs.csv not found at {path}")
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get("slug", "").strip()
            if not slug:
                continue
            enabled_val = row.get("enabled", "1").strip()
            enabled = enabled_val not in ("0", "false", "False", "no", "")
            rows.append(SlugRow(
                slug=slug,
                enabled=enabled,
                market_kind=row.get("market_kind", "binary").strip() or "binary",
                series_from_date=row.get("series_from_date", "").strip(),
                series_filter=row.get("series_filter", "").strip(),
                band_low=_parse_optional_float(row.get("band_low", "")),
                band_high=_parse_optional_float(row.get("band_high", "")),
                size_override=_parse_optional_float(row.get("size_override", "")),
                added_at=row.get("added_at", "").strip(),
                notes=row.get("notes", "").strip(),
            ))
    return rows


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_section(dataclass_obj, raw: dict, section: str):
    section_data = raw.get(section, {}) or {}
    for key, val in section_data.items():
        if hasattr(dataclass_obj, key):
            setattr(dataclass_obj, key, val)


def load_config(
    config_path: Optional[Path] = None,
    slugs_path: Optional[Path] = None,
) -> AppConfig:
    cp = config_path or (_CONFIG_DIR / "config.yaml")
    sp = slugs_path or (_CONFIG_DIR / "slugs.csv")

    raw = _load_yaml(cp)
    cfg = AppConfig()
    cfg.mode = raw.get("mode", "paper")

    for section_name, cls_attr in [
        ("strategy", "strategy"),
        ("filters", "filters"),
        ("feed", "feed"),
        ("polling", "polling"),
        ("orders", "orders"),
        ("risk", "risk"),
        ("telegram", "telegram"),
        ("ratelimit", "ratelimit"),
        ("auth", "auth"),
        ("bankroll", "bankroll"),
        ("pricer", "pricer"),
    ]:
        _apply_section(getattr(cfg, cls_attr), raw, section_name)

    cfg.slugs = _load_slugs(sp)
    _validate(cfg)
    return cfg


def _validate(cfg: AppConfig) -> None:
    s = cfg.strategy
    assert 0 < s.band_low < s.band_high < 1, \
        f"Invalid band: {s.band_low} - {s.band_high}"
    assert s.order_notional_usd > 0, "order_notional_usd must be > 0"
    assert -1.0 <= s.alpha <= 1.0, f"alpha must be in [-1, 1], got {s.alpha}"
    assert cfg.risk.daily_loss_breaker_pct > 0, "daily_loss_breaker_pct must be > 0"
    assert cfg.mode in ("paper", "live"), f"mode must be paper|live, got {cfg.mode!r}"
    p = cfg.pricer
    assert p.reprice_s > 0, "pricer.reprice_s must be > 0"
    assert p.n_sims > 0, "pricer.n_sims must be > 0"
    assert -1.0 <= p.min_edge <= 1.0, \
        f"pricer.min_edge must be in [-1, 1], got {p.min_edge}"


class ConfigWatcher:
    """Watches config.yaml for changes; applies hot-reloadable params to existing AppConfig."""

    def __init__(
        self,
        config: AppConfig,
        config_path: Optional[Path] = None,
        slugs_path: Optional[Path] = None,
        poll_s: float = 5.0,
    ):
        self._cfg = config
        self._cp = config_path or (_CONFIG_DIR / "config.yaml")
        self._sp = slugs_path or (_CONFIG_DIR / "slugs.csv")
        self._poll_s = poll_s
        self._last_mtime_cfg = self._cp.stat().st_mtime if self._cp.exists() else 0.0
        self._last_mtime_slugs = self._sp.stat().st_mtime if self._sp.exists() else 0.0

    def check_and_reload(self) -> bool:
        """Return True if a hot-reload was applied."""
        cfg_mtime = self._cp.stat().st_mtime if self._cp.exists() else 0.0
        slugs_mtime = self._sp.stat().st_mtime if self._sp.exists() else 0.0

        changed = (cfg_mtime != self._last_mtime_cfg or
                   slugs_mtime != self._last_mtime_slugs)
        if not changed:
            return False

        try:
            new = load_config(self._cp, self._sp)
            self._apply_hot(new)
            # KV overlay always wins over yaml. Phase 6, item 6: surface
            # which YAML keys are currently shadowed by a config_kv row --
            # otherwise a stale dashboard override silently masks an
            # intentional config.yaml edit with no visible trace.
            shadowed = apply_config_kv_overrides(self._cfg)
            if shadowed:
                logger.info(
                    f"Config hot-reload: {len(shadowed)} key(s) shadowed by "
                    f"config_kv overrides: {', '.join(sorted(shadowed))}"
                )
            _publish_active_overrides(shadowed)
            self._last_mtime_cfg = cfg_mtime
            self._last_mtime_slugs = slugs_mtime
            logger.info("Config hot-reloaded")
            return True
        except Exception as e:
            logger.error(f"Config reload failed, keeping old values: {e}")
            return False

    def _apply_hot(self, new: AppConfig) -> None:
        c = self._cfg
        n = new
        # Strategy (all hot)
        c.strategy.band_low = n.strategy.band_low
        c.strategy.band_high = n.strategy.band_high
        c.strategy.min_dte = n.strategy.min_dte
        c.strategy.max_dte = n.strategy.max_dte
        c.strategy.min_time_in_band_s = n.strategy.min_time_in_band_s
        c.strategy.order_notional_usd = n.strategy.order_notional_usd
        c.strategy.alpha = n.strategy.alpha
        # Filters (all hot)
        c.filters.min_24h_volume = n.filters.min_24h_volume
        c.filters.min_total_volume = n.filters.min_total_volume
        c.filters.min_book_depth = n.filters.min_book_depth
        # Feed (decision_interval hot; ws_url cold)
        c.feed.decision_interval_s = n.feed.decision_interval_s
        c.feed.max_staleness_seconds = n.feed.max_staleness_seconds
        c.feed.gap_halt_seconds = n.feed.gap_halt_seconds
        c.feed.ws_force_reconnect_s = n.feed.ws_force_reconnect_s
        c.feed.ws_ping_interval_s = n.feed.ws_ping_interval_s
        c.feed.ws_pong_timeout_s = n.feed.ws_pong_timeout_s
        c.feed.ws_expect_pong = n.feed.ws_expect_pong
        # NOTE: resync_concurrency and executor_workers are COLD (constructor-only) —
        # do NOT add here.
        # Polling (all hot)
        c.polling.bankroll_s = n.polling.bankroll_s
        c.polling.resolution_s = n.polling.resolution_s
        c.polling.discovery_s = n.polling.discovery_s
        c.polling.control_poll_s = n.polling.control_poll_s
        c.polling.calibration_fetch_s = n.polling.calibration_fetch_s
        # Orders (all hot)
        c.orders.spread_market_threshold_c = n.orders.spread_market_threshold_c
        c.orders.requote_move_c = n.orders.requote_move_c
        c.orders.limit_ttl_s = n.orders.limit_ttl_s
        # Risk (all hot)
        c.risk.daily_loss_breaker_pct = n.risk.daily_loss_breaker_pct
        c.risk.max_deployed_pct = n.risk.max_deployed_pct
        c.risk.per_market_cap_pct = n.risk.per_market_cap_pct
        c.risk.matic_min_balance = n.risk.matic_min_balance
        # Pricer (all hot -- ModelPricer re-reads its live PricerConfig
        # object on every call, so mutating fields in place is sufficient)
        c.pricer.enabled = n.pricer.enabled
        c.pricer.veto = n.pricer.veto
        c.pricer.min_edge = n.pricer.min_edge
        c.pricer.reprice_s = n.pricer.reprice_s
        c.pricer.n_sims = n.pricer.n_sims
        c.pricer.garch_refit_s = n.pricer.garch_refit_s
        c.pricer.data_max_age_s = n.pricer.data_max_age_s
        c.pricer.data_refresh_s = n.pricer.data_refresh_s
        c.pricer.data_dir = n.pricer.data_dir
        # Slugs hot (triggers resubscribe via flag)
        c.slugs = n.slugs


# ---------------------------------------------------------------------------
# Config KV override application
# ---------------------------------------------------------------------------

# Mapping of dashboard keys to AppConfig attribute paths, with type coercion.
_KV_MAP: Dict[str, str] = {
    "strategy.band_low":               "strategy.band_low",
    "strategy.band_high":              "strategy.band_high",
    "strategy.min_dte":                "strategy.min_dte",
    "strategy.max_dte":                "strategy.max_dte",
    "strategy.min_time_in_band_s":     "strategy.min_time_in_band_s",
    "strategy.order_notional_usd":     "strategy.order_notional_usd",
    "strategy.alpha":                  "strategy.alpha",
    "filters.min_24h_volume":          "filters.min_24h_volume",
    "filters.min_total_volume":        "filters.min_total_volume",
    "filters.min_book_depth":          "filters.min_book_depth",
    "feed.max_staleness_seconds":      "feed.max_staleness_seconds",
    "feed.gap_halt_seconds":           "feed.gap_halt_seconds",
    "feed.decision_interval_s":        "feed.decision_interval_s",
    "polling.bankroll_s":              "polling.bankroll_s",
    "polling.resolution_s":            "polling.resolution_s",
    "polling.discovery_s":             "polling.discovery_s",
    "polling.control_poll_s":          "polling.control_poll_s",
    "orders.spread_market_threshold_c": "orders.spread_market_threshold_c",
    "orders.requote_move_c":           "orders.requote_move_c",
    "orders.limit_ttl_s":              "orders.limit_ttl_s",
    "risk.daily_loss_breaker_pct":     "risk.daily_loss_breaker_pct",
    "risk.max_deployed_pct":           "risk.max_deployed_pct",
    "risk.per_market_cap_pct":         "risk.per_market_cap_pct",
    "risk.matic_min_balance":          "risk.matic_min_balance",
}


def apply_config_kv_overrides(cfg: AppConfig) -> List[str]:
    """Read config_kv table and overlay dashboard-written overrides onto cfg.

    Safe to call at startup (after load_config) and on every config_reload.
    Only applies keys that exist in _KV_MAP; ignores unknown keys silently.

    Returns the list of keys actually applied (i.e. the YAML keys currently
    shadowed by a config_kv row) -- Phase 6, item 6: ConfigWatcher.check_and_
    reload logs and publishes this list as engine_state's active_overrides
    so it's visible on the dashboard, not just inferable from config_kv
    directly. Existing callers (engine/main.py, engine/control.py) ignore
    the return value -- this is an additive signature change.
    """
    import json

    applied: List[str] = []
    try:
        from persistence.repos import config_kv_repo

        rows = config_kv_repo.get_keys(list(_KV_MAP.keys()))

        if not rows:
            return applied

        for row in rows:
            key = row["key"]
            attr_path = _KV_MAP.get(key)
            if attr_path is None:
                continue
            try:
                raw = json.loads(row["value"])
                _set_config_attr(cfg, attr_path, raw)
                applied.append(key)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning(f"config_kv override '{key}' invalid: {e}")

    except Exception as e:
        logger.warning(f"Failed to apply config_kv overrides: {e}")

    return applied


def _publish_active_overrides(shadowed: List[str]) -> None:
    """Publish the list of config_kv-shadowed YAML keys to engine_state
    (Phase 6, item 6) so the dashboard sidebar can render them. Best-effort
    -- a publish failure must not take down the config watch loop."""
    try:
        from persistence.repos import engine_state_repo
        engine_state_repo.publish("active_overrides", shadowed)
    except Exception as e:
        logger.warning(f"Failed to publish active_overrides: {e}")


def _set_config_attr(cfg: AppConfig, attr_path: str, value: Any) -> None:
    """Set a dotted-path attribute on cfg (e.g. 'strategy.band_low')."""
    parts = attr_path.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
