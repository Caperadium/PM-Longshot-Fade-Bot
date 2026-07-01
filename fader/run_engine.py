"""run_engine.py — Launch the fader trading engine.

Usage:
    python fader/run_engine.py

Requires .env with:
    POLYMARKET_PRIVATE_KEY=0x...
    POLYMARKET_USER_ADDRESS=0x...
    TELEGRAM_BOT_TOKEN=...   (optional)
    TELEGRAM_CHAT_ID=...     (optional)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.main import main

if __name__ == "__main__":
    main()
