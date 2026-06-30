"""run_dashboard.py — Launch the fader Streamlit dashboard.

Usage:
    streamlit run fader/run_dashboard.py

Or use the dedicated page:
    streamlit run fader/dashboard/app.py
    streamlit run fader/dashboard/backtest_page.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Streamlit re-executes this file on each interaction; we simply import the app.
from dashboard.app import *  # noqa: F401, F403
