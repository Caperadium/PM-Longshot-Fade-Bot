# Changes

## Crypto parameter sweep (BTC, ETH, SOL, XRP) with walk-forward OOS validation

Extends `fader/backtest/crypto_sweep.py` (per-market grid sweep over band_low x alpha x DTE) to the requested ranges: alpha -0.50 to 0.50, band_low 0.50-0.90, band_high fixed 0.95, min_dte 0-7, max_dte 1-7 (all in 0.05/1-day increments), bankroll $1,000. Adds walk-forward OOS validation (`backtest/walkforward.py`, 4 contiguous calendar windows, no re-optimization) for the top 3 configs per market to flag overfitting via stability grade / Sortino CV / PnL concentration. Report expanded from top-5 to top-10 configs per market. Output: `DATA/crypto_sweep/summary.md` (+ per-market `*_results.csv` / `*_top30.csv`).

## VPS deployment guide

Adds `VPS_SETUP.md`: Debian 13 deployment guide covering git clone (public repo, no auth), venv setup under PEP 668, `.env` secrets, a systemd service so the bot survives reboot and SSH disconnect, and SSH-only monitoring via journald + Telegram alerts + optional Streamlit SSH tunnel.
