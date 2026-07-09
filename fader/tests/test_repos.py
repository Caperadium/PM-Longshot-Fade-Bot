"""tests/test_repos.py

Unit tests for persistence/repos.py (Phase 1 of the architecture refactor,
see temp/implementation-plan.md). Each repo method against a throwaway
per-test SQLite DB (set_db_path + init_db, mirrors the pattern used by
test_live_readiness.py / test_paper_resolution.py / test_vps_review_fixes.py).

Covers: PositionsRepo, OrdersRepo, BreakerRepo, DecisionsRepo, ControlRepo,
EngineStateRepo, ConfigKVRepo, and Db.transaction() commit/rollback.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import unittest
from pathlib import Path

_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))


class _RepoTestCase(unittest.TestCase):
    db_name = "test_fader_repos.db"

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

    def _raw_conn(self) -> sqlite3.Connection:
        from infra.db import get_connection
        return get_connection()


class TestPositionsRepo(_RepoTestCase):
    def test_insert_open_and_has_open(self):
        from persistence.repos import PositionsRepo
        repo = PositionsRepo()
        self.assertFalse(repo.has_open("0xTOK1"))
        repo.insert_open({
            "position_id": "p1", "slug": "s1", "condition_id": "c1",
            "token_id": "0xTOK1", "entry_price": 0.85, "size": 10.0,
            "notional": 8.5, "opened_at": "2026-01-01T00:00:00Z",
            "entry_order_id": "o1", "entry_decision_id": "ik1",
        })
        self.assertTrue(repo.has_open("0xTOK1"))

    def test_insert_open_ignores_duplicate_position_id(self):
        from persistence.repos import PositionsRepo
        repo = PositionsRepo()
        row = {
            "position_id": "p-dup", "slug": "s1", "condition_id": "c1",
            "token_id": "0xTOK1", "entry_price": 0.85, "size": 10.0,
            "notional": 8.5, "opened_at": "2026-01-01T00:00:00Z",
            "entry_order_id": "o1", "entry_decision_id": "ik1",
        }
        repo.insert_open(row)
        repo.insert_open(row)  # INSERT OR IGNORE -- must not raise
        self.assertEqual(repo.open_count(), 1)

    def test_open_positions_and_deployed_total(self):
        from persistence.repos import PositionsRepo
        repo = PositionsRepo()
        repo.insert_open({
            "position_id": "p1", "slug": "s1", "condition_id": "c1",
            "token_id": "0xTOK1", "entry_price": 0.85, "size": 10.0,
            "notional": 8.5, "opened_at": "2026-01-01T00:00:00Z",
            "entry_order_id": "o1", "entry_decision_id": "ik1",
        })
        repo.insert_open({
            "position_id": "p2", "slug": "s1", "condition_id": "c1",
            "token_id": "0xTOK2", "entry_price": 0.90, "size": 5.0,
            "notional": 4.5, "opened_at": "2026-01-01T00:00:00Z",
            "entry_order_id": "o2", "entry_decision_id": "ik2",
        })
        repo.insert_open({
            "position_id": "p3", "slug": "s2", "condition_id": "c2",
            "token_id": "0xTOK3", "entry_price": 0.80, "size": 2.0,
            "notional": 1.6, "opened_at": "2026-01-01T00:00:00Z",
            "entry_order_id": "o3", "entry_decision_id": "ik3",
        })
        total, by_slug = repo.deployed_total()
        self.assertAlmostEqual(total, 8.5 + 4.5 + 1.6)
        self.assertAlmostEqual(by_slug["s1"], 8.5 + 4.5)
        self.assertAlmostEqual(by_slug["s2"], 1.6)
        self.assertEqual(repo.deployed_by_slug(), by_slug)
        self.assertEqual(repo.open_count(), 3)
        self.assertEqual(len(repo.open_positions()), 3)

    def test_close(self):
        from persistence.repos import PositionsRepo
        repo = PositionsRepo()
        repo.insert_open({
            "position_id": "p1", "slug": "s1", "condition_id": "c1",
            "token_id": "0xTOK1", "entry_price": 0.85, "size": 10.0,
            "notional": 8.5, "opened_at": "2026-01-01T00:00:00Z",
            "entry_order_id": "o1", "entry_decision_id": "ik1",
        })
        n = repo.close("p1", realized_pnl=1.5, resolved_at="2026-01-02T00:00:00Z")
        self.assertEqual(n, 1)
        self.assertFalse(repo.has_open("0xTOK1"))
        self.assertEqual(repo.open_count(), 0)
        # Closing again (not OPEN anymore) is a no-op, not an error
        n2 = repo.close("p1", realized_pnl=9.9, resolved_at="2026-01-03T00:00:00Z")
        self.assertEqual(n2, 0)

    def test_bulk_close_paper(self):
        from persistence.repos import PositionsRepo
        repo = PositionsRepo()
        repo.insert_open({
            "position_id": "p1", "slug": "s1", "condition_id": "c1",
            "token_id": "0xTOK1", "entry_price": 0.85, "size": 10.0,
            "notional": 8.5, "opened_at": "2026-01-01T00:00:00Z",
            "entry_order_id": "o1", "entry_decision_id": "ik1",
        })
        repo.insert_open({
            "position_id": "p2", "slug": "s2", "condition_id": "c2",
            "token_id": "0xTOK2", "entry_price": 0.85, "size": 10.0,
            "notional": 8.5, "opened_at": "2026-01-01T00:00:00Z",
            "entry_order_id": "o2", "entry_decision_id": "ik2",
        })
        n = repo.bulk_close_paper()
        self.assertEqual(n, 2)
        self.assertEqual(repo.open_count(), 0)
        conn = self._raw_conn()
        try:
            rows = conn.execute(
                "SELECT realized_pnl, status FROM positions"
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            self.assertEqual(row["status"], "CLOSED")
            self.assertEqual(row["realized_pnl"], 0.0)


class TestOrdersRepo(_RepoTestCase):
    def test_insert_and_by_idem_key(self):
        from persistence.repos import OrdersRepo
        repo = OrdersRepo()
        self.assertIsNone(repo.by_idem_key("ik1"))
        repo.insert("o1", "ik1", "s1", "0xTOK1", 0.85, 10.0, "MARKET", status="FILLED")
        row = repo.by_idem_key("ik1")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "FILLED")

    def test_insert_rejects_invalid_status(self):
        from persistence.repos import OrdersRepo
        repo = OrdersRepo()
        with self.assertRaises(ValueError):
            repo.insert("o1", "ik1", "s1", "0xTOK1", 0.85, 10.0, "MARKET", status="BOGUS")

    def test_update_status(self):
        from persistence.repos import OrdersRepo
        repo = OrdersRepo()
        repo.insert("o1", "ik1", "s1", "0xTOK1", 0.85, 10.0, "LIMIT", status="PENDING")
        repo.update_status("o1", "CANCELLED", "requote")
        conn = self._raw_conn()
        try:
            row = conn.execute(
                "SELECT status, cancel_reason FROM orders WHERE order_id='o1'"
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(row["status"], "CANCELLED")
        self.assertEqual(row["cancel_reason"], "requote")

    def test_pending_limit_orders_newest_first(self):
        from persistence.repos import OrdersRepo
        repo = OrdersRepo()
        conn = self._raw_conn()
        try:
            conn.execute(
                "INSERT INTO orders (order_id, idempotency_key, slug, token_id, side, "
                "type, price, size, status, created_at) VALUES "
                "('o1','ik1','s1','0xT1','BUY','LIMIT',0.8,1,'PENDING','2026-01-01T00:00:00')"
            )
            conn.execute(
                "INSERT INTO orders (order_id, idempotency_key, slug, token_id, side, "
                "type, price, size, status, created_at) VALUES "
                "('o2','ik2','s1','0xT2','BUY','LIMIT',0.8,1,'PENDING','2026-01-02T00:00:00')"
            )
            conn.commit()
        finally:
            conn.close()
        rows = repo.pending_limit_orders()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["order_id"], "o2")  # newest first

    def test_not_terminal_excludes_terminal_statuses(self):
        from persistence.repos import OrdersRepo
        repo = OrdersRepo()
        repo.insert("o1", "ik1", "s1", "0xTOK1", 0.85, 10.0, "MARKET", status="PENDING")
        repo.insert("o2", "ik2", "s1", "0xTOK2", 0.85, 10.0, "MARKET", status="FILLED")
        rows = repo.not_terminal()
        ids = {r["order_id"] for r in rows}
        self.assertEqual(ids, {"o1"})

    def test_pending_count(self):
        from persistence.repos import OrdersRepo
        repo = OrdersRepo()
        repo.insert("o1", "ik1", "s1", "0xTOK1", 0.85, 10.0, "MARKET", status="PENDING")
        repo.insert("o2", "ik2", "s1", "0xTOK2", 0.85, 10.0, "MARKET", status="FILLED")
        repo.insert("o3", "ik3", "s1", "0xTOK3", 0.85, 10.0, "MARKET", status="PENDING")
        self.assertEqual(repo.pending_count(), 2)

    def test_set_status_updates_status_only(self):
        """set_status must not touch cancel_reason -- distinct from
        update_status (reconciler's narrower UPDATE, no cancel_reason)."""
        from persistence.repos import OrdersRepo
        repo = OrdersRepo()
        repo.insert("o1", "ik1", "s1", "0xTOK1", 0.85, 10.0, "MARKET", status="PENDING")
        repo.update_status("o1", "UNKNOWN", "cancel_failed:requote")
        repo.set_status("o1", "FILLED")
        conn = self._raw_conn()
        try:
            row = conn.execute(
                "SELECT status, cancel_reason FROM orders WHERE order_id='o1'"
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(row["status"], "FILLED")
        self.assertEqual(row["cancel_reason"], "cancel_failed:requote")

    def test_reap_stale_unknown(self):
        from persistence.repos import OrdersRepo
        repo = OrdersRepo()
        conn = self._raw_conn()
        try:
            conn.execute(
                "INSERT INTO orders (order_id, idempotency_key, slug, token_id, side, "
                "type, price, size, status, created_at) VALUES "
                "('o1','ik1','s1','0xT1','BUY','MARKET',0.8,1,'UNKNOWN','2020-01-01T00:00:00)')"
                .replace(")')", "')")
            )
            conn.commit()
        finally:
            conn.close()
        repo.reap_stale_unknown("2026-01-01T00:00:00")
        conn = self._raw_conn()
        try:
            row = conn.execute("SELECT status FROM orders WHERE order_id='o1'").fetchone()
        finally:
            conn.close()
        self.assertEqual(row["status"], "CANCELLED")


class TestBreakerRepo(_RepoTestCase):
    def test_day_state_none_when_absent(self):
        from persistence.repos import BreakerRepo
        repo = BreakerRepo()
        self.assertIsNone(repo.day_state("2026-01-01"))

    def test_record_pnl_event_accumulates(self):
        from persistence.repos import BreakerRepo
        repo = BreakerRepo()
        repo.record_pnl_event("2026-01-01", -1.0)
        repo.record_pnl_event("2026-01-01", -2.5)
        row = repo.day_state("2026-01-01")
        self.assertAlmostEqual(row["realized_pnl"], -3.5)
        self.assertEqual(row["tripped"], 0)

    def test_trip_and_reset(self):
        from persistence.repos import BreakerRepo
        repo = BreakerRepo()
        repo.trip("2026-01-01")
        row = repo.day_state("2026-01-01")
        self.assertEqual(row["tripped"], 1)
        repo.reset("2026-01-01")
        row2 = repo.day_state("2026-01-01")
        self.assertEqual(row2["tripped"], 0)


class TestDecisionsRepo(_RepoTestCase):
    def test_append_returns_true_and_persists(self):
        from persistence.repos import DecisionsRepo
        repo = DecisionsRepo()
        ok = repo.append("s1", "0xTOK1", "REJECTED", "ask_out_of_band", {"a": 1})
        self.assertTrue(ok)
        conn = self._raw_conn()
        try:
            row = conn.execute("SELECT * FROM decisions").fetchone()
        finally:
            conn.close()
        self.assertEqual(row["slug"], "s1")
        self.assertEqual(row["reason"], "ask_out_of_band")

    def test_prune_removes_old_rows(self):
        from persistence.repos import DecisionsRepo
        repo = DecisionsRepo()
        conn = self._raw_conn()
        try:
            conn.execute(
                "INSERT INTO decisions (ts, slug, decision, reason) VALUES "
                "('2000-01-01T00:00:00', 's1', 'REJECTED', 'x')"
            )
            conn.commit()
        finally:
            conn.close()
        repo.append("s1", None, "REJECTED", "y", {})
        n = repo.prune(14)
        self.assertEqual(n, 1)
        conn = self._raw_conn()
        try:
            remaining = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(remaining, 1)


class TestControlRepo(_RepoTestCase):
    def test_issue_pending_mark_done(self):
        from persistence.repos import ControlRepo
        repo = ControlRepo()
        repo.issue("stop", {"reason": "test"})
        rows = repo.pending()
        self.assertEqual(len(rows), 1)
        cmd_id = rows[0]["id"]
        self.assertEqual(rows[0]["command"], "stop")
        repo.mark_done(cmd_id, "ok")
        self.assertEqual(repo.pending(), [])

    def test_prune_removes_old_non_pending(self):
        from persistence.repos import ControlRepo
        repo = ControlRepo()
        conn = self._raw_conn()
        try:
            conn.execute(
                "INSERT INTO control_commands (ts, command, status) VALUES "
                "('2000-01-01T00:00:00', 'stop', 'DONE')"
            )
            conn.commit()
        finally:
            conn.close()
        n = repo.prune(14)
        self.assertEqual(n, 1)


class TestEngineStateRepo(_RepoTestCase):
    def test_publish_and_get(self):
        from persistence.repos import EngineStateRepo
        repo = EngineStateRepo()
        self.assertIsNone(repo.get("bankroll"))
        repo.publish("bankroll", 123.45)
        self.assertEqual(repo.get("bankroll"), 123.45)

    def test_publish_on_passed_conn_does_not_commit(self):
        """When conn is passed, publish() must not commit; caller owns
        the transaction lifecycle."""
        from persistence.repos import EngineStateRepo
        repo = EngineStateRepo()
        conn = self._raw_conn()
        try:
            repo.publish("k", "v", conn=conn)
            # Not committed yet on this connection -- a second connection
            # (WAL, separate handle) should not see it before commit.
            conn2 = self._raw_conn()
            try:
                row = conn2.execute(
                    "SELECT value_json FROM engine_state WHERE key='k'"
                ).fetchone()
                self.assertIsNone(row)
            finally:
                conn2.close()
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(repo.get("k"), "v")


class TestConfigKVRepo(_RepoTestCase):
    def test_set_and_get(self):
        from persistence.repos import ConfigKVRepo
        repo = ConfigKVRepo()
        self.assertEqual(repo.get("strategy.band_low", 0.8), 0.8)
        repo.set("strategy.band_low", 0.75)
        self.assertEqual(repo.get("strategy.band_low", 0.8), 0.75)

    def test_all_items(self):
        from persistence.repos import ConfigKVRepo
        repo = ConfigKVRepo()
        repo.set("a", 1)
        repo.set("b", 2)
        rows = repo.all_items()
        keys = {r["key"] for r in rows}
        self.assertEqual(keys, {"a", "b"})

    def test_get_keys_filters_to_allowlist(self):
        from persistence.repos import ConfigKVRepo
        repo = ConfigKVRepo()
        repo.set("a", 1)
        repo.set("b", 2)
        repo.set("c", 3)
        rows = repo.get_keys(["a", "c", "not_present"])
        keys = {r["key"] for r in rows}
        self.assertEqual(keys, {"a", "c"})

    def test_get_keys_empty_list_returns_empty(self):
        from persistence.repos import ConfigKVRepo
        repo = ConfigKVRepo()
        self.assertEqual(repo.get_keys([]), [])


class TestDbConnect(_RepoTestCase):
    def test_connect_returns_usable_plain_connection(self):
        """Db.connect() is a plain connection factory (no transaction
        semantics) -- used where a caller needs to share one connection
        across repo calls without commit-on-success (e.g. reconciler's
        pre-existing no-commit reaper UPDATE, preserved verbatim)."""
        from persistence.repos import Db, OrdersRepo
        db = Db()
        orders = OrdersRepo()
        conn = db.connect()
        try:
            orders.insert("o1", "ik1", "s1", "0xTOK1", 0.85, 10.0, "MARKET", conn=conn)
            # Not committed by connect()/insert(conn=...) -- caller owns it.
            row = conn.execute("SELECT * FROM orders WHERE order_id='o1'").fetchone()
            self.assertIsNotNone(row)
            conn.commit()
        finally:
            conn.close()
        self.assertIsNotNone(orders.by_idem_key("ik1"))


class TestDbTransaction(_RepoTestCase):
    def test_transaction_commits_on_success(self):
        from persistence.repos import Db, PositionsRepo, OrdersRepo
        db = Db()
        positions = PositionsRepo()
        orders = OrdersRepo()
        with db.transaction() as conn:
            orders.insert(
                "o1", "ik1", "s1", "0xTOK1", 0.85, 10.0, "MARKET",
                status="PENDING", conn=conn,
            )
            positions.insert_open({
                "position_id": "p1", "slug": "s1", "condition_id": "c1",
                "token_id": "0xTOK1", "entry_price": 0.85, "size": 10.0,
                "notional": 8.5, "opened_at": "2026-01-01T00:00:00Z",
                "entry_order_id": "o1", "entry_decision_id": "ik1",
            }, conn=conn)
        self.assertTrue(positions.has_open("0xTOK1"))
        self.assertIsNotNone(orders.by_idem_key("ik1"))

    def test_transaction_rolls_back_on_exception(self):
        from persistence.repos import Db, PositionsRepo, OrdersRepo
        db = Db()
        positions = PositionsRepo()
        orders = OrdersRepo()
        with self.assertRaises(RuntimeError):
            with db.transaction() as conn:
                orders.insert(
                    "o1", "ik1", "s1", "0xTOK1", 0.85, 10.0, "MARKET",
                    status="PENDING", conn=conn,
                )
                positions.insert_open({
                    "position_id": "p1", "slug": "s1", "condition_id": "c1",
                    "token_id": "0xTOK1", "entry_price": 0.85, "size": 10.0,
                    "notional": 8.5, "opened_at": "2026-01-01T00:00:00Z",
                    "entry_order_id": "o1", "entry_decision_id": "ik1",
                }, conn=conn)
                raise RuntimeError("boom")
        # Neither statement should have been persisted
        self.assertFalse(positions.has_open("0xTOK1"))
        self.assertIsNone(orders.by_idem_key("ik1"))

    def test_transaction_conn_not_closed_prematurely_by_repo_methods(self):
        """Repo methods must not close a passed conn -- verify the conn
        is still usable after multiple repo calls within one transaction."""
        from persistence.repos import Db, EngineStateRepo
        db = Db()
        state = EngineStateRepo()
        with db.transaction() as conn:
            state.publish("k1", "v1", conn=conn)
            state.publish("k2", "v2", conn=conn)
            # conn must still be open/usable here
            row = conn.execute("SELECT COUNT(*) FROM engine_state").fetchone()
            self.assertEqual(row[0], 2)


class TestOrderManagerAtomicFillBookkeeping(_RepoTestCase):
    """Phase 1's intentional behavior change: order status update + position
    insert on a market/paper-limit fill now happen in one Db.transaction()
    (execution/order_manager.py._save_order_and_insert_position). Verify
    both writes commit together, and if the position insert half fails,
    the order write it was paired with rolls back too (no more window
    where an order is FILLED/PENDING with no matching position row)."""

    def test_atomic_helper_commits_both_writes(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config
        from unittest.mock import MagicMock

        cfg = load_config()
        om = OrderManager(cfg=cfg, provider=MagicMock())
        om._save_order_and_insert_position(
            "order-1", "idem-1", "slug-1", "0xTOK1", 0.85, 10.0, "MARKET",
            "FILLED", 8.5, None,
        )

        from persistence.repos import OrdersRepo, PositionsRepo
        self.assertIsNotNone(OrdersRepo().by_idem_key("idem-1"))
        self.assertTrue(PositionsRepo().has_open("0xTOK1"))

    def test_atomic_helper_rolls_back_order_write_if_position_insert_fails(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config
        from unittest.mock import MagicMock, patch

        cfg = load_config()
        om = OrderManager(cfg=cfg, provider=MagicMock())

        with patch(
            "execution.order_manager.positions_repo.insert_open",
            side_effect=RuntimeError("simulated failure"),
        ):
            # _save_order_and_insert_position swallows the exception and
            # logs (matches the old _save_order/_insert_position's
            # individual try/except-log behavior) -- must not raise.
            om._save_order_and_insert_position(
                "order-2", "idem-2", "slug-2", "0xTOK2", 0.85, 10.0, "MARKET",
                "FILLED", 8.5, None,
            )

        from persistence.repos import OrdersRepo, PositionsRepo
        # Neither write persisted -- the order insert rolled back with the
        # failed position insert (this is the atomicity guarantee; before
        # this phase the order row would have committed on its own).
        self.assertIsNone(OrdersRepo().by_idem_key("idem-2"))
        self.assertFalse(PositionsRepo().has_open("0xTOK2"))


if __name__ == "__main__":
    unittest.main()
