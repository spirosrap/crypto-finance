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

### Conditional Action (optional)
If we want the alternative to trigger only when the trade is already underwater:
- **LONG**: if `mark_price < entry_price` at the time-stop, use `partial_close`; otherwise `tighten_sl`.
- **SHORT**: if `mark_price > entry_price` at the time-stop, use `partial_close`; otherwise `tighten_sl`.
This keeps “soft protection” for slow winners but cuts drag when price has already slipped past entry.

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

### Grace Window Clarification
The grace window is a short cooldown applied **after any recent fill/order activity**
(most commonly late/partial entry fills), so the time-stop does not fire immediately
after a fresh fill. It is **not** tied to TP1 and does not replace the 12h timer; it
only delays the action when recent fills occur.

### Timing Anchor
The 12h timer is anchored to the **first entry fill** (not the last partial fill).

## Prerequisites / Wiring
- This is enforced by `watchdog_close_old_positions.py` (not the gate-scan script).
- TP1 detection needs fill logging to be running:
  - `watchdog_close_old_positions.py --log-fills ...` must run regularly so `partial_tp` rows exist in `trade_logs/watchdog_closed_positions.csv`.
- Cron wiring: add the flag to the existing watchdog cron line (not `scripts/run_gate_scan_paper.sh`).
  - Example: append `--tp1-time-stop-enable` to the `watchdog_close_update` job in `docs/pipelines/cron.md`.

## Implementation Notes
### TP1 Hit Detection (robust + simple)
- Primary: infer TP1 hit from logged fill events.
  - When `--log-fills` is enabled, partial take-profit fills are recorded as `closure_reason=partial_tp` rows in `trade_logs/watchdog_closed_positions.csv`.
  - The time-stop should treat “TP1 hit” as: a matching `partial_tp` row exists for the same `product_id` and the same `opened_at` (within a tolerance).
- Secondary (optional): if fill-logs are unavailable, a safer default is “do nothing”.
  - Avoid time-stopping based on missing data, otherwise you’ll incorrectly tighten/close winners.

### Idempotency
- Add a checkpoint so the time-stop runs once per position, even if cron runs every 3–5 minutes.
- Recommended checkpoint file: `trade_logs/watchdog_tp1_time_stop_checkpoint.json`.
- Key: `PRODUCT_ID|OPENED_AT_ENTRY_ISO` where `opened_at_entry` is the first entry fill timestamp.
- Keep only the last ~500 keys to prevent unbounded growth.

### Timing Model
- Define two timestamps:
  - `opened_at_entry`: first entry fill (anchor for the 12h time-stop).
  - `latest_fill`: most recent fill/order completion time (used only for the grace window).
- Apply time-stop when:
  - `now - opened_at_entry >= tp1_time_stop_hours`, AND
  - `TP1 not hit`, AND
  - not in grace window (`now - latest_fill > tp1_time_stop_grace_minutes`), AND
  - not already processed (checkpoint).

### Actions
#### `tighten_sl` (preferred)
Goal: keep the position open, but stop the drag by removing downside beyond breakeven.
- Fetch open orders for the product and filter to trigger bracket orders.
- For each bracket:
  - Read `base_size`, `take_profit` (TP), current `stop_trigger_price` (SL), and `end_time`.
  - Compute new SL at entry (`entry_price`), but clamp if it would be invalid vs current price:
    - LONG: if `entry >= mark`, clamp SL to `mark * (1 - buffer_bps)`.
    - SHORT: if `entry <= mark`, clamp SL to `mark * (1 + buffer_bps)`.
  - Replace bracket safely:
    - Place the replacement bracket first (same size, same TP, new SL, preserve `end_time`).
    - Only after replacement succeeds, cancel the old bracket order(s).
- No-op if SL is already at/inside entry.

#### `partial_close` (alternative)
Goal: reduce exposure when TP1 never arrived, without fully abandoning the trade.
- Submit a reduce-only market IOC to close `close_pct` of the current base size.
- **Important safety note**: do not leave brackets sized for the pre-reduction position.
  - Robust behavior is: after the partial close, re-place brackets sized to the *remaining* position and cancel the old ones.
  - If you can’t safely resync brackets, prefer `tighten_sl` (or paper-only rollout for `partial_close`).
- Logging:
  - Optionally append a log row with `closure_reason=time_stop_no_tp1_partial` so the dashboard can report “time-stop usage”.
  - If you already run `--log-fills`, you may rely on fills-derived logs instead of manual logging.

#### Conditional Underwater Alternative (optional)
- If enabled:
  - Determine underwater at trigger time using `mark_price` vs `entry_price`.
  - If underwater: `partial_close`, else `tighten_sl`.

### CLI / Config Surface (recommended)
- `--tp1-time-stop-enable`
- `--tp1-time-stop-hours 12`
- `--tp1-time-stop-action tighten_sl|partial_close`
- `--tp1-time-stop-close-pct 50`
- `--tp1-time-stop-side both|longs|shorts`
- `--tp1-time-stop-grace-minutes 60`
- `--tp1-time-stop-underwater-alt` (optional)
- Optional quality-of-life:
  - `--tp1-time-stop-dry-run` (log what would happen; safest first enable)

### Paper-First Rollout (recommended)
To follow the rollout plan strictly, implement the analogous rule in `paper_finder_simulator.py update`:
- Use `opened_at` as the anchor and `partial_tp_done` (or equivalent) to detect TP1.
- Tighten SL: set `stop_loss = entry_price` after `tp1_time_stop_hours`.
- Partial close: reduce `position_usd` by `close_pct` and keep the remainder.
- Persist idempotency in the open-positions CSV with a boolean like `tp1_time_stop_done` and a timestamp.

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
