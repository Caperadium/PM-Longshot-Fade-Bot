"""backtest/report.py

Shared report-formatting pieces for the five backtest CLIs (band_sweep,
allocation_analysis, grid_sweep, is_oos_backtest, crypto_sweep).

Phase 5 of temp/implementation-plan.md: "the five CLI formatters shrink to
section composition." Content-identical is the bar, not byte-identical --
each CLI's report layout (fixed-width text vs markdown tables, column
choices, header wording) is preserved as-is; this module only extracts the
mechanical parts that were duplicated: building a column-aligned table
from row dicts, rendering the "known backtest-vs-live filter gaps" caveats
section, and writing a report string to a file (creating parent dirs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# Default caveats every backtest report should carry -- historical
# Polymarket data cannot reconstruct these three live-engine filters.
# Kept as a fallback for reports that don't pass an explicit
# skipped_filters list (mirrors metrics.py's _DEFAULT_UNIVERSE_DISCREPANCY_KEYS).
DEFAULT_CAVEAT_LINES: Tuple[str, ...] = (
    "Backtest omits min_book_depth, min_24h_volume, min_total_volume filters.",
)


def metrics_table(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[Tuple[str, str, str]],
) -> str:
    """Render a fixed-width, column-aligned text table.

    Args:
        rows: sequence of dicts (or objects with matching keys via
            ``row[key]`` / ``row.get(key)``) -- one per output row.
        columns: sequence of ``(key, header, fmt)`` tuples, where ``fmt``
            is a str.format-style spec applied to ``row[key]`` (e.g.
            ``">8.3f"``, ``">6d"``, ``">5.1%"``). Header cells are
            right-aligned to the same width as the widest of (header,
            formatted value).

    Returns:
        Multi-line string: header row, separator, one line per row.
    """
    def _get(row: Any, key: str) -> Any:
        if isinstance(row, dict):
            return row.get(key)
        return getattr(row, key, None)

    formatted_rows: List[List[str]] = []
    for row in rows:
        cells = []
        for key, _header, fmt in columns:
            val = _get(row, key)
            try:
                cells.append(format(val, fmt) if val is not None else "N/A")
            except (ValueError, TypeError):
                cells.append(str(val))
        formatted_rows.append(cells)

    widths = []
    for i, (_key, header, _fmt) in enumerate(columns):
        col_vals = [header] + [r[i] for r in formatted_rows]
        widths.append(max(len(v) for v in col_vals))

    header_line = "  ".join(h.rjust(w) for (_, h, _), w in zip(columns, widths))
    sep_line = "-" * len(header_line)
    lines = [header_line, sep_line]
    for cells in formatted_rows:
        lines.append("  ".join(c.rjust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


def caveats_section(
    skipped_filters: Optional[Iterable[str]] = None,
    extra_lines: Optional[Sequence[str]] = None,
    title: str = "CAVEATS",
    rule: str = "=",
    width: int = 72,
) -> str:
    """Build a CAVEATS section.

    ``skipped_filters`` (e.g. a run's ``trades_df.attrs["skipped_filters"]``,
    Phase 4) is rendered as one line per known filter name via the same
    universe-discrepancy table metrics.py uses, falling back to the fixed
    historical 3-filter line when omitted -- reports that never plumbed
    skipped_filters through keep their original wording.
    ``extra_lines`` are appended verbatim after the filter-gap line(s),
    letting each CLI keep its own additional caveats (walk-forward
    caveats, data-source caveats, etc.).
    """
    from backtest.metrics import _UNIVERSE_DISCREPANCY_DEFS

    lines: List[str] = []
    lines.append(rule * width)
    lines.append(title)
    lines.append(rule * width)

    if skipped_filters is None:
        lines.append(DEFAULT_CAVEAT_LINES[0])
    else:
        known = [f for f in skipped_filters if f in _UNIVERSE_DISCREPANCY_DEFS]
        if known:
            names = ", ".join(known)
            lines.append(f"Backtest omits {names} filters (not reconstructable from historical data).")

    if extra_lines:
        lines.extend(extra_lines)

    return "\n".join(lines)


def write_report(path: Union[str, Path], title: str, sections: Sequence[str]) -> str:
    """Compose ``sections`` into one report string, write to ``path``, return it.

    Creates parent directories if needed (mirrors every CLI's
    ``out_path.parent.mkdir(parents=True, exist_ok=True)`` call before
    writing). ``title`` is written as the first line if non-empty.
    """
    path = Path(path)
    parts: List[str] = []
    if title:
        parts.append(title)
        parts.append("")
    parts.extend(sections)
    report = "\n".join(parts)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report written: {path}")
    return report
