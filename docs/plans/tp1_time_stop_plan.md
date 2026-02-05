# TP1 Time-Stop (No-TP1 Drag Control)

## Status
Proposal (future enhancement)

## Context
- Recent live behavior: winners tend to finish early, while many losers drag until expiry.
- Current guard: hard expiry at 24h via `watchdog_close_old_positions.py --max-age-hours 24`.
- Goal: reduce slow-bleed losses without tightening global SL/TP or reducing trade count.

## Goals
- Cut drawdown from trades that fail to reach TP1 within a reasonable window.
- Preserve fast winners and overall trade cadence.
- Keep change side-selective if one side (e.g., longs) underperforms.

## Non-Goals
- No global SL/TP changes.
- No new hard filters or gate changes.
- No reduction in trade count.

## Proposed Behavior
Keep the 24h hard expiry. Add an optional **TP1 time-stop**:

- If **TP1 has not hit** within **12–16 hours** from entry, apply a *soft* action:
  - **Preferred:** tighten SL to breakeven (or entry +/- small buffer).
  - **Alternative:** close 50% at market and leave remainder with original SL/TP.
- Apply this rule **once per position**, and only if TP1 was not hit.
- Optionally apply the rule only to the weaker side (e.g., longs) if side stats diverge.

## Configuration (proposed)
- `--tp1-time-stop-hours 12` (or 16)
- `--tp1-time-stop-action tighten_sl|partial_close`
- `--tp1-time-stop-close-pct 50`
- `--tp1-time-stop-side longs|shorts|both` (optional)

### Defaults (when implemented)
- Hours: `12`
- Action: `tighten_sl`
- Close pct: `50` (only applies if `partial_close`)
- Side: `both`
- Grace window: `60` minutes since last order/fill

## Implementation Notes
- Leverage existing partial-fill detection: TP1 hit can be inferred from `partial_tp` fills or bracket status.
- Add a new closure reason label if a partial close is triggered (e.g., `time_stop_no_tp1`).
- Ensure the action is idempotent (only runs once per position).
- Respect a recent-order grace window (do not act if last order was placed in the last 30–60 minutes).

## Rollout Plan
1. Paper-only for 2–4 weeks (or 30–50 trades).
2. Compare expectancy and drawdown vs. baseline.
3. If positive and trade count is stable, enable for live with a higher threshold (16h).

## Metrics to Track
- Expectancy (overall and last 20 trades)
- Max loss streak
- Avg loss vs avg win
- % of trades affected by time-stop
- Side-specific performance (long vs short)

## Risks
- Late winners may get cut before recovery.
- If set too strict, may reduce total edge; prefer 12–16h and soft actions.

## Decision Checkpoints
- Re-evaluate after 50 and 100 trades under the new rule.
