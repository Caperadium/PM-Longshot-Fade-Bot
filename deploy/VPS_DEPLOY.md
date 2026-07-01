# VPS deployment checklist (live trading)

## 1. Layout

```
/opt/fader-bot/            # repo checkout
/opt/fader-bot/.venv/      # python -m venv .venv && pip install -r requirements
/opt/fader-bot/fader/.env  # secrets (chmod 600, owned by the fader user)
```

Create a dedicated non-root user: `sudo useradd -r -m -d /opt/fader-bot fader`.

## 2. .env — required for live

| Var | Why |
|---|---|
| `POLYMARKET_PRIVATE_KEY` | order signing |
| `POLYMARKET_USER_ADDRESS` | balances, positions |
| `POLYGON_RPC_URL` | **required** — the public default (polygon-rpc.com) rate-limits/401s, which zeroes balance reads and halts entries. Use Alchemy/Infura/QuickNode. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | the only way you hear about breaker trips / WS loss while headless |

Set `mode: live` in `fader/config/config.yaml` only after a paper run on the VPS itself.

## 3. Process supervision

Install both systemd units from this directory (see file headers). Key points:

- `Restart=always` + `RestartSec=5`: crash or OOM → auto-restart; startup then
  runs full reconcile + `rehydrate_resting()`, so state recovers from the API.
- VPS reboot: `systemctl enable` starts the engine on boot.
- `TimeoutStopSec=30` gives graceful shutdown time to cancel resting orders.
- Unclean shutdown (power loss / SIGKILL) is safe by design: orders carry
  idempotency keys, SQLite is WAL, and startup reconciles against the API as
  ground truth. Resting limit orders left on the exchange are re-adopted by
  `rehydrate_resting()` on the next boot.

Logs: `journalctl -u fader-engine -f` plus rotating `fader/engine.log` (10MB x5).

## 4. Disk / DB

- Engine prunes `decisions` and processed `control_commands` older than 14
  days automatically (hourly task).
- Optional nightly DB backup:
  `0 4 * * * sqlite3 /opt/fader-bot/fader/fader.db ".backup /opt/fader-bot/backups/fader-$(date +\%u).db"`
  (7-day rotation; the API remains ground truth, so backups are convenience).
- Keep >2GB free; monitor with any lightweight disk alert.

## 5. Clock

Risk-day boundaries and DTE use UTC. Enable NTP: `timedatectl set-ntp true`.
A drifted clock skews the daily breaker day-roll and DTE filters.

## 6. First-day watchlist

- Telegram: startup alert, 15-min heartbeats arriving.
- Dashboard (`ssh -L 8501:localhost:8501`): `ws_connected=true`, bankroll
  matches your wallet, `gap_halted=false`.
- `journalctl -u fader-engine | grep -i "breaker\|reject\|failed"` after the
  first fills.
- Confirm USDC.e allowance for the CTF Exchange is set (BUY orders spend
  USDC.e; startup logs the allowance and warns when < $1).
