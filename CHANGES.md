# Changes

- Make all dashboard gradient tables matplotlib-safe and add matplotlib to requirements. New `fader/dashboard/table_style.py` `styled_dataframe()` (RdYlGn gradient when matplotlib importable, plain formatted table otherwise) replaces calibration_page's local fallback helper and now also guards backtest_page's walk-forward heatmap, which previously crashed the page on hosts without matplotlib. `matplotlib>=3.8.0` added to `fader/requirements.txt` (VPS venv lacked it; installed there so gradients render).
