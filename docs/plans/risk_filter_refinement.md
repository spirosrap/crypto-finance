# Refined Risk Filter Plan

## Problem Statement

The current balance guard (minimum candidates per side) functions as a coarse risk filter. It blocks trades in highly skewed regimes, but it does not directly measure stability or volatility. This can prevent early trend entries and does not explicitly guard against volatility spikes or range-break regimes.

## Current Observed Filters (Short-Term)

These are already active in the short-term stack, so the refined risk filter should avoid duplicating them or over-tightening.

### Live Finder (always on)

- **Market-cap floor**: `min_market_cap` (default 100M; enforced in the data pipeline).
- **Liquidity gate**: `min_volume_24h` + `min_volume_market_cap_ratio`.
- **Spread gate**: `spread_bps * leverage / 100 <= max_spread_margin_pct` (default cap 20% at 50x leverage).
- **Risk-level gate**: `max_risk_level` (profile-dependent; `MEDIUM` in focused_no_llm_100).
- **ATR caps (sizing)**: `max_atr_usd` + `max_atr_bps` cap ATR used for stops/TP; not a hard gate but reduces sizing volatility.

### Gate-Scan Baseline (informational)

- **ATR ratio**: `atr_bps / cap_bps <= 1.5` (baseline flag in scan output, not a live gate).
- **Spread headroom**: uses a heuristic spread cap (volume-tiered) for baseline pass.
- **VMC ratio**: `min_volume_market_cap_ratio` check (with BTC/ETH exempt).

### Active Thresholds (from current env)

**Default profile (env shows `SHORT_FINDER_PROFILE=default`)**
- limit 30, max_results 20, top_per_side 10
- min_market_cap 100M
- min_volume_24h 15M, min_vmc_ratio 0.025
- max_atr_usd 3000, max_atr_bps 400
- max_spread_margin_pct 20, report_leverage 50
- max_risk_level None (no risk cap)
- intraday_lookback_days 14

**Focused profile (common live run: `focused_no_llm_100`)**
- limit 100, max_results 20, top_per_side 5
- min_market_cap 100M
- min_volume_24h 5M, min_vmc_ratio 0.03
- max_atr_usd 3000, max_atr_bps 400
- max_spread_margin_pct 20, report_leverage 50
- max_risk_level MEDIUM
- intraday_lookback_days 20, unique_by_symbol True

## Goals

- Add a **refined risk filter** that measures stability/volatility directly.
- Keep the existing **balance guard** as a secondary safety check (optional).
- Provide **transparent reasons** when entries are suppressed.
- Keep the implementation confined to gate-scan outputs and paper/live commands (no strategy engine changes).

## Non-Goals

- Do not change entry logic, TP/SL math, or finder scoring.
- Do not add new data sources beyond existing candle data.
- Do not remove the existing range-break circuit.

## Proposed Risk Signals

### 1) Volatility Guard (explicit)
Use ATR and realized volatility to block abnormal conditions.

Inputs (already available):
- ATR7 in USD (`atr_raw` and `atr_eff`)
- ATR cap ratio (ATR7 / cap)
- Candle returns (from OHLC data)

Signals:
- **ATR ratio**: `atr_raw / atr_cap` (block if too high, e.g., > 1.2)
- **ATR percentile**: ATR7 vs 30–60d window (block if > 90th percentile)
- **Vol spike**: last N candles’ realized vol vs 30d baseline (block if > 1.5×)

### 2) Stability Guard (explicit)
Measure whether price action is stable vs. break risk.

Inputs:
- Range-break status (`range_break_status.json`)
- Distance to range boundaries (if range-break active/inactive)
- Trend/chop measure (ADX or slope if available; else simple trend score)

Signals:
- **Range-break active**: suppress entries (already done)
- **Range proximity**: if price is within X% of range boundary, flag as higher break risk
- **Chop filter**: low trend strength + high vol = unstable (block)

### 3) Balance Guard (existing)
Keep `min_per_side` as a final safety rail (optional).

## Decision Policy (Example)

```
if range_break_active:
    suppress
elif vol_spike or atr_ratio_high:
    suppress
elif chop_high and vol_high:
    suppress
elif not enough candidates on both sides (min_per_side):
    suppress (optional)
else:
    allow
```

## Light-Touch Tightening Path (Not Too Strict)

Goal: nudge the risk filter up a notch without strangling trade count. Start with **warn-only** and low suppression targets.

**Phase 0 (observe only, 1–2 weeks)**
- Compute risk flags but do **not** suppress trades.
- Target: risk flags should fire on <20% of candidates on most days.

**Phase 1 (gentle suppress, single-signal)**
- `risk-atr-ratio-max`: **1.4** (from baseline 1.5)  
- `risk-atr-percentile-max`: **92**  
- `risk-vol-spike-mult`: **1.7**  
- Suppress **only** if *one* of the volatility signals fires (no chop/range proximity yet).

**Phase 2 (add stability, still loose)**
- Add chop condition but **only** when volatility is also high (two‑factor filter).
- Range proximity: **warn-only** unless within **0.5%** of range boundary.

**Stop conditions**
- If daily candidate count drops >30% for 3+ sessions, revert to Phase 0/1.
- Keep a “suppressed by reason” summary so you can see if a single gate is too aggressive.

This path gives a modest tightening (1.5 → 1.4 ATR ratio) while keeping the broader liquidity/spread/risk‑level gates as the primary constraints.

## CLI/Config Design

Add optional flags (default off to preserve current behavior):

- `--risk-filter refined|basic|off` (default: off)
- `--risk-atr-ratio-max 1.2`
- `--risk-atr-percentile-max 90`
- `--risk-vol-spike-mult 1.5`
- `--risk-range-prox-pct 1.0`
- `--risk-chop-threshold <value>`
- `--risk-require-balance` (keeps existing min-per-side behavior)

## Implementation Plan

1) **Risk helpers in `scripts/symbol_snapshot.py`**
   - Add helpers to compute ATR percentile, realized vol, and range proximity.
   - Read `range_break_status.json` (already used).
   - Tag each candidate with risk flags and a `risk_ok` boolean.

2) **Gate-scan filtering**
   - Apply `risk_ok` before selection and before printing commands.
   - Show a summary of suppressed reasons (counts by reason).

3) **Command output changes**
   - Print a short “Risk filter summary” block, e.g.:
     - `Risk filter: ON (refined)`
     - `Suppressed: vol_spike=4, atr_ratio=2, range_prox=1`

4) **Paper/live parity**
   - Gate-scan output affects both paper and live command lists.
   - No changes to paper simulator or live execution logic.

## Validation Plan

- **Backtest-style sanity** using recent gate-scan logs:
  - Compare output counts with/without refined risk filter.
  - Check that known high-volatility days are suppressed.
- **Manual spot checks** for a few symbols:
  - Ensure ATR percentile and vol spike are computed correctly.
  - Verify range proximity triggers only near boundary.

## Rollout Plan

1) Add flags + helpers (default off).
2) Turn on refined risk filter in `scripts/run_gate_scan_paper.sh` (paper only).
3) Observe trade counts + expectancy for 1–2 weeks.
4) If stable, enable for live.

## Success Criteria

- Reduced exposure during volatility spikes without collapsing trade count to zero.
- Fewer trades during range-break risk windows.
- No regression in baseline expectancy over a 30–60 trade sample.

## Risks / Trade-offs

- Over-filtering could starve the pipeline.
- ATR percentile thresholds may need tuning per regime.
- Range proximity could block good breakout entries (acceptable per “no breakouts” request).
