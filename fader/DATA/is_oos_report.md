# IS / OOS Backtest -- Top Candidates

**Generated:** 2026-06-29 19:56  
**Data:** 2025-08-07 .. 2026-06-28 (24,408 rows)  
**IS period:** 2025-08-07 .. 2026-03-22 (16,679 rows, 70% of calendar time)  
**OOS period:** 2026-03-23 .. 2026-06-28 (7,729 rows, 30% of calendar time)  
**Split:** hard temporal cutoff (no lookahead)  
**Runtime:** 4s  

Candidates sourced from `fader/DATA/grid_sweep_findings.md` Pareto frontier.
All use alpha=0.0 (uniform sizing).

---

## 1. Max Calmar
*Highest risk-adjusted (IS Calmar 8.64)*

Band [0.50, 0.95] | DTE [2, 5] | alpha=0.0

| Metric | IS | OOS | IS->OOS |
|--------|-----|-----|---------|
| Trades (N) | 1126 | 403 | -- |
| Calmar | 12.95 | 24.15 | +87% |
| Sortino | 7.91 | 8.15 | +3% |
| Hit rate | 83.0% | 84.6% | +2% |
| Profit factor | 1.56 | 1.61 | +3% |
| Expectancy ($/trade) | 0.96 | 0.94 | -2% |
| Max drawdown | 43.4% | 32.8% | -24% |
| Total PnL ($) | 1077.66 | 378.69 | -65% |

> **PASS -- OOS Calmar retains >=70% of IS Calmar.** (OOS/IS Calmar = 187%)

---

## 2. Max Sortino
*Highest total PnL (IS Sortino 8.58)*

Band [0.50, 0.95] | DTE [3, 6] | alpha=0.0

| Metric | IS | OOS | IS->OOS |
|--------|-----|-----|---------|
| Trades (N) | 1191 | 401 | -- |
| Calmar | 12.73 | 24.58 | +93% |
| Sortino | 8.37 | 8.22 | -2% |
| Hit rate | 83.0% | 84.5% | +2% |
| Profit factor | 1.60 | 1.67 | +4% |
| Expectancy ($/trade) | 1.01 | 1.03 | +2% |
| Max drawdown | 51.2% | 39.3% | -23% |
| Total PnL ($) | 1204.77 | 413.85 | -66% |

> **PASS -- OOS Calmar retains >=70% of IS Calmar.** (OOS/IS Calmar = 193%)

---

## 3. Conservative
*Slight band tighten (IS Calmar 7.76)*

Band [0.55, 0.95] | DTE [0, 5] | alpha=0.0

| Metric | IS | OOS | IS->OOS |
|--------|-----|-----|---------|
| Trades (N) | 1506 | 680 | -- |
| Calmar | 11.05 | 14.18 | +28% |
| Sortino | 7.40 | 6.06 | -18% |
| Hit rate | 83.1% | 83.4% | +0% |
| Profit factor | 1.36 | 1.23 | -10% |
| Expectancy ($/trade) | 0.61 | 0.39 | -37% |
| Max drawdown | 41.6% | 28.0% | -33% |
| Total PnL ($) | 925.48 | 262.11 | -72% |

> **PASS -- OOS Calmar retains >=70% of IS Calmar.** (OOS/IS Calmar = 128%)

---

## 4. Balanced
*Lower drawdown (IS Calmar 6.58)*

Band [0.60, 0.95] | DTE [5, 7] | alpha=0.0

| Metric | IS | OOS | IS->OOS |
|--------|-----|-----|---------|
| Trades (N) | 862 | 240 | -- |
| Calmar | 8.98 | 16.11 | +79% |
| Sortino | 7.70 | 7.42 | -4% |
| Hit rate | 88.7% | 89.2% | +0% |
| Profit factor | 1.78 | 1.87 | +5% |
| Expectancy ($/trade) | 0.87 | 0.94 | +7% |
| Max drawdown | 39.3% | 20.9% | -47% |
| Total PnL ($) | 752.45 | 225.01 | -70% |

> **PASS -- OOS Calmar retains >=70% of IS Calmar.** (OOS/IS Calmar = 179%)

---

## 5. High PF
*Best PF without degenerate alpha*

Band [0.70, 0.95] | DTE [5, 7] | alpha=0.0

| Metric | IS | OOS | IS->OOS |
|--------|-----|-----|---------|
| Trades (N) | 735 | 205 | -- |
| Calmar | 8.54 | 11.33 | +33% |
| Sortino | 8.29 | 5.78 | -30% |
| Hit rate | 92.5% | 91.2% | -1% |
| Profit factor | 2.12 | 1.77 | -16% |
| Expectancy ($/trade) | 0.84 | 0.68 | -19% |
| Max drawdown | 32.0% | 14.7% | -54% |
| Total PnL ($) | 613.84 | 138.45 | -77% |

> **PASS -- OOS Calmar retains >=70% of IS Calmar.** (OOS/IS Calmar = 133%)

---

## Summary Comparison

| Candidate | IS Cal | IS Sort | IS N | OOS Cal | OOS Sort | OOS N | Calmar retain | Verdict |
|-----------|--------|---------|------|---------|----------|-------|---------------|---------|
| Max Calmar | 12.95 | 7.91 | 1126 | 24.15 | 8.15 | 403 | 187% | PASS |
| Max Sortino | 12.73 | 8.37 | 1191 | 24.58 | 8.22 | 401 | 193% | PASS |
| Conservative | 11.05 | 7.40 | 1506 | 14.18 | 6.06 | 680 | 128% | PASS |
| Balanced | 8.98 | 7.70 | 862 | 16.11 | 7.42 | 240 | 179% | PASS |
| High PF | 8.54 | 8.29 | 735 | 11.33 | 5.78 | 205 | 133% | PASS |

**Calmar retain**: OOS Calmar / IS Calmar. >=70% = PASS, 40-70% = MARGINAL, <40% = FAIL.

---

## Notes

1. **Hard temporal split (no shuffle).** IS = earlier data, OOS = later data. Matches real deployment -- bot never sees future data.
2. **OOS is a single hold-out window.** One 30% slice is not enough to distinguish luck from skill; use walkforward.py for per-window stability.
3. **OOS N may be lower** because DTE-filtered strategies require contracts with sufficient days remaining.
4. **No bootstrap CIs here** (speed). Run compute_all_metrics(n_bootstrap=10000) for CI bands.
5. **BTC only.** Results may not generalize to Seoul temperature or other slugs.