# VPS Setup — Fader Bot on Debian 13

Deploy and run the Anti-Longshot Polymarket Bot on a Debian 13 VPS with SSH-only
access. Covers clone, dependencies, secrets, a systemd service that survives
reboot and SSH disconnect, and monitoring without the Streamlit dashboard.

The repo is public, so no git authentication is needed on the VPS.

---

## 1. Install system packages

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

Debian 13 ships Python 3.13 with PEP 668 (externally-managed environment).
System-wide `pip install` is blocked, so a virtualenv is mandatory (step 3).

## 2. Clone the repo

```bash
git clone https://github.com/Caperadium/PM-Longshot-Fade-Bot.git
cd PM-Longshot-Fade-Bot
```

## 3. Create virtualenv and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r fader/requirements.txt
```

## 4. Configure secrets (`.env`)

`.env` is not in the repo. Either copy it from your local machine or create it
on the VPS from the example.

**Option A — copy from local machine** (run on your local machine, not the VPS):

```bash
scp "fader/.env" user@VPS_IP:~/PM-Longshot-Fade-Bot/fader/.env
```

**Option B — create on VPS from the example:**

```bash
cp fader/.env.example fader/.env
nano fader/.env
```

Keys:

| Key | Required | Notes |
|---|---|---|
| `POLYMARKET_PRIVATE_KEY` | Yes (live) | EOA private key for signing orders |
| `POLYMARKET_USER_ADDRESS` | Yes | Wallet address for balance reads |
| `TELEGRAM_BOT_TOKEN` | Optional | Enables push alerts (recommended, see monitoring) |
| `TELEGRAM_CHAT_ID` | Optional | Telegram chat to alert |
| `POLYGON_RPC_URL` | Recommended (live) | Use a keyed provider (Alchemy/Infura). Public default is rate-limited / returns 401, which silently zeroes balance reads and stops trading. |

## 5. Test run

```bash
source venv/bin/activate
python fader/run_engine.py
```

A fresh `fader/fader.db` (SQLite WAL) is created automatically on first run.
Confirm startup is clean, then stop with `Ctrl+C` and move to the systemd service.

---

## 6. Run as a systemd service (survives reboot + SSH disconnect)

Without this the bot dies when you log out. Create the unit file:

```bash
sudo nano /etc/systemd/system/fader.service
```

Paste (replace `YOUR_USER` with your VPS username):

```ini
[Unit]
Description=Fader Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/PM-Longshot-Fade-Bot
ExecStart=/home/YOUR_USER/PM-Longshot-Fade-Bot/venv/bin/python fader/run_engine.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fader
```

Common commands:

```bash
sudo systemctl status fader     # current state
sudo systemctl restart fader    # restart after config/code change
sudo systemctl stop fader       # stop
```

---

## 7. Monitoring (SSH-only, no dashboard)

The Streamlit dashboard needs a browser. Two SSH-friendly options:

**A. journald logs (primary):**

```bash
journalctl -u fader -f          # live tail
journalctl -u fader --since "1 hour ago"
journalctl -u fader -p err      # errors only
```

**B. Telegram alerts (passive push):** set `TELEGRAM_BOT_TOKEN` +
`TELEGRAM_CHAT_ID` in `.env`. Sends heartbeat, breaker-trip, and error alerts
(`fader/infra/telegram.py`) so you do not need an open SSH session to know the
bot is alive or tripped.

**C. Optional — dashboard over SSH tunnel** when you do want the full UI. Run
from your local machine:

```bash
ssh -L 8501:localhost:8501 user@VPS_IP
```

Then on the VPS: `streamlit run fader/run_dashboard.py`, and open
`http://localhost:8501` in your local browser.

---

## Updating the bot later

```bash
cd ~/PM-Longshot-Fade-Bot
git pull
source venv/bin/activate
pip install -r fader/requirements.txt   # only if deps changed
sudo systemctl restart fader
```
