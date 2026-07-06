"""run_engine_supervised.py — supervisor wrapper for the fader engine.

Relaunches the engine on the cold-restart sentinel (exit 42) issued by the
dashboard "KILL + COLD RESTART" button. Any other exit code stops the loop.

This is the cross-platform equivalent of systemd `Restart=always`:
  - VPS (Debian): use the systemd units in deploy/ — you do NOT need this.
  - Windows / bare terminal: launch the engine through this wrapper so the
    dashboard restart button actually cold-starts the process.

Usage:
    python fader/run_engine_supervised.py

Each relaunch is a full cold start: a fresh process re-runs load_dotenv,
telegram.configure and full_reconcile — so edits to .env take effect.
"""
import subprocess
import sys
import time
from pathlib import Path

RESTART_EXIT_CODE = 42  # keep in sync with engine/main.py sentinel
_HERE = Path(__file__).resolve().parent
_ENGINE = str(_HERE / "run_engine.py")


def main() -> int:
    while True:
        proc = subprocess.run([sys.executable, _ENGINE], cwd=str(_HERE.parent))
        if proc.returncode == RESTART_EXIT_CODE:
            print("[supervisor] cold-restart requested — relaunching engine...")
            time.sleep(2)  # let the OS release WS / DB handles
            continue
        print(f"[supervisor] engine exited ({proc.returncode}); stopping.")
        return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
