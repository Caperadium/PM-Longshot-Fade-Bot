"""tests/test_config_overrides.py

Phase 6, item 6 (temp/implementation-plan.md): config override visibility.
  - apply_config_kv_overrides now returns the list of YAML keys currently
    shadowed by a config_kv row (was None).
  - ConfigWatcher.check_and_reload logs the shadowed keys and publishes
    them to engine_state under "active_overrides".

Run: python -m pytest fader/tests/test_config_overrides.py -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))


class _DbTestCase(unittest.TestCase):
    db_name = "test_fader_config_overrides.db"

    def setUp(self):
        os.environ["POLYMARKET_USER_ADDRESS"] = "0xTEST_USER"
        self.db_path = _FADER_ROOT / "tests" / self.db_name
        from infra.db import set_db_path, init_db
        set_db_path(self.db_path)
        if self.db_path.exists():
            self.db_path.unlink()
        init_db()

    def tearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()

    def _get_engine_state(self, key):
        from infra.db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT value_json FROM engine_state WHERE key=?", (key,)
            ).fetchone()
            return json.loads(row["value_json"]) if row else None
        finally:
            conn.close()


class TestApplyConfigKvOverridesReturnValue(_DbTestCase):
    def test_returns_empty_list_when_no_overrides(self):
        from config.config_loader import apply_config_kv_overrides, AppConfig
        cfg = AppConfig()
        result = apply_config_kv_overrides(cfg)
        self.assertEqual(result, [])

    def test_returns_shadowed_keys_and_applies_value(self):
        from config.config_loader import apply_config_kv_overrides, AppConfig
        from persistence.repos import config_kv_repo

        config_kv_repo.set("strategy.band_low", 0.77)
        config_kv_repo.set("risk.max_deployed_pct", 42.0)

        cfg = AppConfig()
        result = apply_config_kv_overrides(cfg)

        self.assertEqual(
            sorted(result), ["risk.max_deployed_pct", "strategy.band_low"]
        )
        self.assertEqual(cfg.strategy.band_low, 0.77)
        self.assertEqual(cfg.risk.max_deployed_pct, 42.0)

    def test_unknown_key_ignored_and_not_in_shadowed_list(self):
        from config.config_loader import apply_config_kv_overrides, AppConfig
        from persistence.repos import config_kv_repo

        config_kv_repo.set("not_a_real_key", 1)

        cfg = AppConfig()
        result = apply_config_kv_overrides(cfg)
        self.assertEqual(result, [])


class TestCheckAndReloadPublishesActiveOverrides(_DbTestCase):
    def _make_watcher(self, tmp_path):
        from config.config_loader import load_config, ConfigWatcher
        cfg = load_config()
        return ConfigWatcher(cfg), cfg

    def test_publishes_empty_list_when_no_overrides(self):
        from config.config_loader import load_config, ConfigWatcher
        import time

        cfg = load_config()
        cfg_path = _FADER_ROOT / "config" / "config.yaml"
        slugs_path = _FADER_ROOT / "config" / "slugs.csv"
        watcher = ConfigWatcher(cfg, cfg_path, slugs_path)
        # Force a reload by rewinding the recorded mtimes.
        watcher._last_mtime_cfg = 0.0

        reloaded = watcher.check_and_reload()
        self.assertTrue(reloaded)
        self.assertEqual(self._get_engine_state("active_overrides"), [])

    def test_publishes_shadowed_keys_on_reload(self):
        from config.config_loader import load_config, ConfigWatcher
        from persistence.repos import config_kv_repo

        config_kv_repo.set("strategy.alpha", 0.5)

        cfg = load_config()
        cfg_path = _FADER_ROOT / "config" / "config.yaml"
        slugs_path = _FADER_ROOT / "config" / "slugs.csv"
        watcher = ConfigWatcher(cfg, cfg_path, slugs_path)
        watcher._last_mtime_cfg = 0.0

        reloaded = watcher.check_and_reload()
        self.assertTrue(reloaded)
        self.assertEqual(
            self._get_engine_state("active_overrides"), ["strategy.alpha"]
        )
        self.assertEqual(cfg.strategy.alpha, 0.5)


if __name__ == "__main__":
    unittest.main()
