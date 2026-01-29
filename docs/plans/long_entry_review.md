# Long Entry Review + Symbol-Specific Adjustments

## Status
Proposal (future work)

## Context
Recent live stats show longs underperforming, especially on majors:
- BTC/ETH longs: low win rates (24% / 19%)
- APT/SEI: more consistent performance relative to other longs

Goal: tighten long entry criteria without choking overall trade flow.

## Plan 2: Review Long Entry Criteria

### Objectives
- Reduce low‑quality long entries while keeping overall trade count stable.
- Apply changes only to **longs** to avoid harming the short edge.

### Proposed Adjustments (Longs Only)
- Tighten ATR caps for longs (e.g., 1.5 → 1.4 or use percentile cap for longs only).
- Raise RR threshold for longs (e.g., 2.0 → 2.2) while leaving shorts unchanged.
- Add a small spread/volatility buffer for longs during high‑vol regimes.

### Rollout
1. Paper first: 30–50 trades, confirm no large drop in total trade count.
2. If expectancy and drawdown improve, apply to live with conservative settings.

### Metrics to Track
- Long win rate, expectancy, and max drawdown
- Total trade count impact (target: <20–25% drop)

## Plan 3: Symbol‑Specific Adjustments

### Objectives
- Avoid blanket long tightening where a few symbols are the main drag.
- Preserve sizing for consistent symbols (APT/SEI).

### Proposed Adjustments
- **BTC/ETH longs**: require higher RR (e.g., >2.5) or slightly wider stops to avoid noise stops.
- **APT/SEI**: keep current sizing/thresholds (no change).
- If needed, add symbol‑level RR overrides for known weak/strong long performers.

### Rollout
1. Simulate symbol‑specific RR overrides in paper logs for 2–4 weeks.
2. Promote to live only if long expectancy improves and shorts are unaffected.

## Decision Gates
- If long expectancy stays negative after adjustments → pause long changes and re‑evaluate regime tilt plan.
- If total trade count drops too much → revert to baseline long filters.

## Sequencing (Recommended)
Implement **one change at a time** to avoid masking the true impact:
1. **Regime tilt** (if stats are favorable at 150 trades). Reversible, doesn’t alter entry mechanics.
2. **Long‑only tightening** (ATR cap or RR bump, not both). Evaluate for 30–50 trades.
3. **Symbol‑specific overrides** (BTC/ETH long RR > 2.5; keep APT/SEI unchanged) only if longs remain weak.

Reassess after each step and stop if expectancy improves or trade count drops too much.
