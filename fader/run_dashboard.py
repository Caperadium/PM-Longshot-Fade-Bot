"""run_dashboard.py — Launch the fader Streamlit dashboard.

Usage:
    streamlit run fader/run_dashboard.py

Or use the dedicated page:
    streamlit run fader/dashboard/app.py
    streamlit run fader/dashboard/backtest_page.py
"""
import runpy
import sys
from pathlib import Path

_FADER_ROOT = Path(__file__).resolve().parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

# Streamlit re-executes this file on each interaction. A plain
# `from dashboard.app import *` only runs the app on the FIRST rerun —
# Python caches the module in sys.modules, so every later rerun renders
# nothing and the dashboard goes blank. runpy executes the source fresh
# each time, matching `streamlit run dashboard/app.py` behavior.
runpy.run_path(str(_FADER_ROOT / "dashboard" / "app.py"), run_name="__main__")
