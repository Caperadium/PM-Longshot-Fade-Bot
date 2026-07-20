"""dashboard/table_style.py

Shared styled-table rendering for dashboard pages.

Styler.background_gradient imports matplotlib lazily at render time
(inside st.dataframe's styler compute), so a deploy target without
matplotlib crashes the whole page mid-render instead of just losing
the color shading. Every gradient table in the dashboard goes through
styled_dataframe(), which probes for matplotlib and falls back to a
plain formatted table when it is absent.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd
import streamlit as st


def styled_dataframe(
    df: pd.DataFrame,
    fmt: Union[str, dict],
    subset: Optional[list] = None,
    axis: Optional[int] = 0,
) -> None:
    """Render df via st.dataframe with a RdYlGn background gradient when
    matplotlib is available, plain formatted table otherwise.

    subset/axis are passed through to Styler.background_gradient
    (subset=None + axis=None shades the whole frame, e.g. the backtest
    walk-forward heatmap; subset=["col"] shades one column).
    """
    try:
        import matplotlib  # noqa: F401  (availability probe only)

        styled = df.style.background_gradient(
            cmap="RdYlGn", subset=subset, axis=axis
        ).format(fmt)
    except ImportError:
        styled = df.style.format(fmt)
    st.dataframe(styled, width='stretch')
