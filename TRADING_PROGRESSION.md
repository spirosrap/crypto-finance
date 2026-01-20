# Trading Journal


## Quick Links
- [January 2026](#january-2026)
- [December 2025](#december-2025)
- [November 2025](#november-2025)
- [October 2025](#october-2025)
- [June 2025](#june-2025)

## January 2026
**State at a glance (January recap):** Baseline ATR exits (0.8× ATR stop, 1.5R target) with ATR ≤ 1.5× cap, spread/VMC gates, and cluster caps (10 total / 3 per bucket). Daily stop (−4%/−$40) + BTC range‑break circuit auto‑close live/paper and suppress entries until reset or confirmed re‑entry (latched; intraday doesn’t clear). Risk defaults live in `config/risk_thresholds.yaml`. Automation: gate‑scan every 4h; paper updates + fills polling + live snapshot every 5m. Live TP1 now reissues brackets with SL moved to entry (fill‑derived, clamped/rounded) and preserves TP; dust threshold bumped to $10 after a ~$9 BTC remainder. Dashboard now tracks fees + exit slippage; CCXT guardrails reduce v3 “index out of range” errors. Recent issue: Coinbase accepted leverage=50 on entry orders but positions reported 1×, so margin headroom is tighter than expected. Checkpoints: 100 live closes and 150 paper closes for scale/tweak decisions.

## January 20, 2026 - Next Paper Checkpoint Rules
- The Jan 15 paper checkpoint is now met, so size stays at full unless the next checkpoint fails.
- Next checkpoint: evaluate **60 new paper trades** (still anchored to the Jan 15 change date).
- Keep full size if last‑60 paper expectancy >= 0.25 (>= 0.30 buffer) **and** profit factor >= 1.0.
- **Half size** if expectancy is between 0.00 and 0.24 **or** profit factor is between 0.90 and 0.99.
- **Pause live** if expectancy < 0.00 **or** profit factor < 0.90.

## January 20, 2026 - Partial Exit Labeling Fixed
- Reclassified prior `partial_take` rows in `trade_logs/watchdog_closed_positions.csv` as `partial_tp` or `partial_sl`.
- Watchdog now labels partial fills as TP vs SL based on entry/exit direction so new rows are tagged correctly.

## January 20, 2026 - Margin Buffer for 1x Leverage
- Live positions are still reporting leverage=1 even when entries send leverage=50, which tightens margin headroom.
- Operationally keeping a larger INTX margin buffer to avoid “insufficient funds” on new entries.

## January 20, 2026 - Fix: Move-SL Brackets Send Leverage
- Fix applied so watchdog includes `leverage` + `margin_type=CROSS` when reissuing TP/SL brackets after TP1.
- Bracket leverage is pulled from the live position payload, mirroring the ccxt trade entry behavior (prevents defaulting to 1x).
- Documented the move‑SL leverage behavior in `README.md`.

## January 20, 2026 - BTC Dust Threshold Raised
- BTC position closed; ~$9 dust remained open because the dust threshold was $5.
- Raising dust threshold to $10 to auto-close small remnants like this.

## January 20, 2026 - Leverage Request vs Position Leverage
- Gate-scan entries sent `leverage=50` in Coinbase order payloads, but live positions reported leverage 1 on the INTX portfolio.
- Order fetch confirmed leverage=50 was accepted on the entry orders; positions still reflected 1x, so Coinbase appears to enforce 1x at the position/account level.

## January 19, 2026 - DOGE Excluded from Perps Pipeline
- Reviewed 168 closed trades from `watchdog_closed_positions.csv`; DOGE showed a poor win rate with 10+ stop losses vs ~3 take profits.
- Added `DOGE-PERP-INTX` to `config/excluded_perps.txt` to filter it from future gate-scan entries.
- Other symbols flagged for monitoring: SEI and ARB showed repeated stop-outs; CRV, RENDER, INJ showed stronger win rates.

## January 19, 2026 - Move-SL Bracket Safety + Price Formatting
- Watchdog now places the replacement TP/SL bracket *before* canceling existing stops so a failed replace never leaves positions unprotected.
- Replacement bracket prices are rounded/formatted to the product’s increment (cache/CCXT) to reduce INVALID_LIMIT_PRICE preview rejections.
- Move-SL now prefers the fill-derived entry price (pre‑partial) over Coinbase’s post‑partial VWAP for the new bracket SL.

## January 18, 2026 - Move-SL Bracket Clamped to Market
- When moving SL to entry after TP1, watchdog now clamps the SL to a valid side of market if the entry is on the wrong side (prevents Coinbase invalid stop errors).
- The replacement bracket still preserves TP and remaining size; cron `--move-sl-after-tp1` now succeeds even if entry is above/below market.

## January 18, 2026 - Live Move-SL Keeps TP Bracket
- Watchdog `--move-sl-after-tp1` now reissues a trigger bracket with TP preserved while moving SL to entry (no more stop-limit-only SL).
- SL is moved to the **post‑partial entry (Coinbase VWAP)** for the remaining size, not the original entry.

## January 18, 2026 - Dashboard Fees & Slippage (Live)
- Watchdog dashboard can now pull Coinbase fills to display fee totals and exit slippage vs order targets.
- Exit slippage now pulls from raw close rows (with order IDs) so results populate under live filters.
- Slippage lookup now handles Coinbase GetOrderResponse mappings so target prices resolve.
- Slippage target extraction now handles Coinbase order configuration objects (trigger bracket/stop-limit payloads).
- Slippage panel now sums exit fees alongside slippage to show total exit cost (fees + slip).
- Added a guard for older cached slippage tables so missing `fee_usd` doesn't crash; prompt to recompute.
- Added slippage breakdown by closure reason, per-cycle exit cost, and top-5 worst exits.
- Partial exits are now classified against TP vs SL targets (partial_tp / partial_sl) so slippage attribution is correct.
- Slippage panel now labels the combined metric as exit fees + net slippage (exit-order-only).

## January 18, 2026 - Paper TP1 Move-SL Logging
- Added explicit logging when paper trades move SL to entry after TP1 so the journal reflects those adjustments.

## January 18, 2026 - Watchdog Move-SL Flag
- Added `--move-sl-after-tp1` to the watchdog so live TP1 can move SL to entry when desired.

## January 18, 2026 - Claude Ops Docs Added
- Added a Claude skill and expanded `CLAUDE.md` with operational notes for the trading workflow.

## January 18, 2026 - Limit Bracket Placement Fix
- `ccxt_trade_perp.py` now waits for limit fills before sending brackets and offers `--place-brackets` to attach TP/SL to a filled position if brackets were skipped.
- Added a configurable bracket wait window (`--bracket-wait-seconds`) and corrected INTX position side detection so short brackets attach correctly.

## January 18, 2026 - Gate-Scan Live Entries Back to Market
- Reverted gate-scan live execution to market orders (removed `--baseline-limit-bps` from the runner).

## January 17, 2026 - Dust Cleanup Added to Watchdog
- Watchdog close runner can now optionally close tiny residual positions by notional (`--dust-notional-usd`) while preserving the usual age-based expiry flow.

## January 17, 2026 - PEM Secret Normalization for CCXT
- Normalized perps API secrets to convert escaped `\\n` into real newlines so CCXT signing works in cron/env runs.

## January 17, 2026 - CCXT Negative-Index Guard Applied
- Added a local guard to prevent CCXT from indexing empty lists with -1, which was throwing `index out of range` during v3 account/ticker/markets calls.
- Applied the same guard in the trade conda environment so non-repo CCXT usage inherits the fix until upstream resolves it.

## January 17, 2026 - Stability Improving, Profit Still Soft
- Recent changes appear to stabilize trade behavior, but profitability is still muted; wait for more closes before making new adjustments.

## January 16, 2026 - CCXT Local Guard + Issue Filed
- Applied a local CCXT site-packages guard for Coinbase v3 empty `accounts`/`trades` lists to avoid `index out of range` crashes.
- Filed upstream CCXT issue: https://github.com/ccxt/ccxt/issues/27694.

## January 16, 2026 - Dashboard Partial Exit Toggle
- Watchdog dashboard now supports showing partial exits on demand; default view still waits for full closes so trade counts/metrics stay per-trade.
- Paper view now preserves partial-only rows when the toggle is enabled (no longer dropped by batch filtering).

## January 16, 2026 - REST TP1 Fallback Restored
- When CCXT entry fails, REST fallback now places split brackets for TP1 + TP2 so live trades keep the partial-exit plan.

## January 16, 2026 - CCXT Entry Retry + Diagnostics
- CCXT order placement now retries once after an “index out of range” with a forced markets reload and logs the last CCXT request/response for debugging before falling back to REST.

## January 16, 2026 - Gate-Scan Cron Issues + Scan Limit Reduced
- Reduced the gate-scan universe to 200 symbols to speed the 4h cron runs.
- Live orders sometimes fail when placed via the cron runner (CCXT “index out of range” errors); manual runs still succeed, so this needs troubleshooting in the cron path.

## January 15, 2026 - Gate-Scan Filters + TP1 Added
- Gate-scan now filters underperforming products (paper log lookback, min trades, drop worst) and applies side-specific score gates.
- Added TP1 partials (default 0.8R / 50%) for both paper and live; paper can move SL to entry after TP1, live uses two brackets and does not auto-move SL.
- Live trading restarts today after these changes.
- New checkpoint: evaluate **30 new paper trades** from this change date (do not mix with prior sample).
  - Keep full size only if last‑30 paper expectancy >= 0.25 (>= 0.30 buffer) **and** profit factor >= 1.0.
  - **Half size** if expectancy is between 0.00 and 0.24 **or** profit factor is between 0.90 and 0.99.
  - **Pause live** if expectancy < 0.00 **or** profit factor < 0.90.

## January 15, 2026 - Paper 30+ Trade Checkpoint
- Paper trades reached 32; expectancy remains negative at -0.92, so live stays paused per the 30-trade rule.

## January 14, 2026 - Paper Progress Update
- Currently 26/30 paper trades with expectancy at -0.58.

## January 13, 2026 - Paper Progress Update
- Currently 19/30 paper trades with expectancy at -0.45, so it doesn’t look like I’ll be starting live soon.

## January 12, 2026 - Paper Drawdown Continues
- Yesterday’s paper trading saw more losses; currently at 13 trades.
- I don’t expect the picture to improve until I reach 30 trades.
- After 30 paper trades complete, only resume live if the dashboard **Expectancy** is >= 0.25 (USD per trade), or >= 0.30 for a buffer; otherwise pause.
- Note: `--balanced` only balances the gate-scan candidate list; baseline filters, open-position skips, and cluster caps can still reduce the final command count below 5.

## January 11, 2026 - Gate-Scan Runner Streams Output
- The gate-scan paper runner now streams output live (no buffering), so long scans no longer look stuck.
- It also runs with `--balanced` and scans 400 products via the 400-focused profile.

## January 11, 2026 - Considering a Live Restart
- Yesterday’s paper trading finished breakeven and the tape looks like it might reverse, so I’m considering restarting live.
- I will only resume after the paper last‑30 turns positive, and I’ll re-enter at half size.

## January 10, 2026 - Dashboard Filters Persist
- The dashboard now persists Start/End date and count filters per mode (live vs paper) across refreshes.

## January 10, 2026 - Live Pause Until Paper Improves
- Pausing live entries and letting paper continue.
- Resume live only after the paper last‑30 turns positive; start back at half size and reassess after another 30 trades.
- Re-entry plan: restart live at 25–50% size for 10–20 trades, then scale only if **both** paper last‑30 and live last‑20 are positive; otherwise pause again.

## January 10, 2026 - Breakeven Drift
- Trading continues.
- Yesterday’s trades closed around breakeven with a small loss.

## January 09, 2026 - Halt Lifted, New Trades Added
- The halt ended and 6 new live trades were added.
- Paper trading continues.
- Most trades are hovering around small loss or breakeven; I expect them to settle better by expiry.

## January 08, 2026 - Daily Stop Close Skip When Flat
- Daily stop guard now checks for open positions first; if there are none, it skips running close commands (live and paper). Reduces noise without changing the halt behavior.

## January 08, 2026 - Another Daily Halt
- Daily stop hit again today as crypto kept falling.
- The drawdown erased most gains from the prior upward move.

## January 08, 2026 - Finder Cron Disabled
- Commented out the `short_term_crypto_finder.py`-related cron outputs (archive + alert + finder/breakout jobs) since I’m no longer using that flow.

## January 07, 2026 - Daily Stop Hit, Short Reset Window
- Both live and paper halted after hitting the −4%/−$40 daily stop.
- The halt lifts on the next UTC day, which in this case is ~4 hours away.
- My take: the reset cadence is OK for now — last time the next day flipped to profitable conditions, so I’m keeping it unless I see repeated same‑day reentries.

## January 07, 2026 - Dashboard Heartbeats
- Added log heartbeat ages to the dashboard so stale cron pipelines are visible even when no trades close.


## January 07, 2026 - Swingy Day, Back Toward Steady
- Yesterday saw losses, then gains. Equity pulled back from the ~$260 peak to just under $200, then recovered after a TP.
- Things look steadier and pointed toward a good day, but crypto remains unpredictable.

## January 07, 2026 - 150-Trade Hold Plan
- Plan after the ~100 live checkpoint: keep rules unchanged and push to 150 closed live trades before any tweaks.
- Only consider scaling after the 150 checkpoint if expectancy/PF hold and drawdown stays contained; otherwise hold size and review.
- No parameter changes until the 150 trade review is complete.
- Concrete plan: finish the 100‑trade live checkpoint as‑is, then push to 150 with zero changes. If PF stays > 1.3 and max drawdown stays < 10%, keep the system; if weakness persists, first adjustment is to reduce shorts or strengthen the short filter (one knob only).

- Next tweak after the 150‑trade checkpoint (if needed): add a short filter — only short when BTC 3‑day return < 0 or when the trend filter is bearish.

## January 06, 2026 - Daily Stop Raised to $40
- Increased the daily stop to −$40 (4% of the $1k baseline) so it aligns with 6–8 concurrent $250 notionals.
- This keeps the guard meaningful without cutting trades too early when the book is fully deployed.

## January 06, 2026 - Long Bias After Strong Day
- Yesterday was another heavy green day for crypto.
- All suggested trades were LONGs and continue to skew LONG today.
- I expect the streak to end soon, but for now the long bias persists.

## January 05, 2026 - Edge vs Risk Control Framing
- Edge shows up on the good days; risk control keeps those gains from getting erased.
- Guardrails are meant to preserve the upside, not create it.

## January 05, 2026 - Range-Break Trigger Uses Confirmed Close
- Range-break now triggers using **confirmed daily closes** only, to avoid intraday flip‑flops.
- Latch still clears only after a confirmed daily close returns inside the range±buffer.

## January 05, 2026 - Range-Break Latch Clarified
- Range-break is now latched until a **confirmed daily close** re-enters the 7‑day range ± 0.5× ATR.
- Intraday reversals do not clear the pause; only the next daily close can unlock entries.

## January 05, 2026 - Breakeven Closes After Drawdown
- Most of yesterday’s trades recovered from a deep drawdown (near −$20 overall) and closed around breakeven.
- Paper trades showed a similar pattern.
- Trading continues today even though I expected the daily stop to trigger.

## January 05, 2026 - Dashboard Health Threshold Adjusted
- Raised the “pipeline attention” alert threshold from 6h to 12h so it matches the current cadence.

## January 05, 2026 - Daily Stop Streak Guard Added
- Added a live pause rule: 3 daily stops in 7 days triggers a 3‑day live pause.
- New warnings: 5 daily stops in 14 days → reduce size 50%; 7 in 21 days → tighten filters or go paper‑only.
- State is stored in `logs/daily_stop_history.json` and configured via `config/risk_thresholds.yaml`.

## January 04, 2026 - Positive Closes, Cooling Conditions
- A lot of trades closed positive yesterday and live trading is considerably up.
- Conditions seem to be cooling down now.
- Paper trading also closed some positive trades.

## January 04, 2026 - Guard Auto-Closures Added
- Gate-scan runner now auto-closes live positions on daily stop or BTC range break, and closes all paper positions with a matching reason.
- New entries are suppressed when a guard triggers, so the cron runner exits safely.

## January 04, 2026 - Live Snapshot Auto Refresh
- Added `scripts/update_live_snapshot.py` and a 5‑minute cron to refresh live positions + USDC balance.
- Dashboard now loads `logs/live_snapshot.json` on refresh, so Ctrl+R shows the latest snapshot without hitting Coinbase.

## January 03, 2026 - Equity Spike Then Reversal
- Yesterday showed a big uptick in earnings, but it later reversed and the top equity gave back a significant chunk.
- Paper trading resumed, but the timing was imperfect; equity is negative again even though nothing closed yet.

## January 03, 2026 - Guard Split Snapshot
- Split performance at the 2025-12-30 UTC guard change shows live improving while paper deteriorated.
- Live post-guard: 16 trades, PF 1.48, avg +1.75 (avg% +0.75%), maxDD -30.54; last 20 trades PF 1.58.
- Paper post-guard: 26 trades, PF 0.47, avg -1.68 (avg% -0.67%), maxDD -64.66; last 20 trades PF 0.25.

## January 02, 2026 - Risk Thresholds Centralized
- Moved shared risk defaults (daily stop, range-break, baseline sizing/gates) into `config/risk_thresholds.yaml`.
- Gate-scan and dashboard now read this file; CLI flags still override per run.

## January 02, 2026 - Checkpoint Thresholds Defined
- Paper (150 closes): keep if expectancy_pct ≥ +0.10%, PF ≥ 1.15, Max DD ≥ -10%; tweak one knob if expectancy_pct between -0.10% and +0.10% or PF 0.95–1.15; stop if expectancy_pct ≤ -0.10% and PF < 0.95 or Max DD < -12%.
- Live (100 closes): scale +10–20% if expectancy_pct ≥ +0.10%, PF ≥ 1.10, Max DD ≥ -6%; hold if expectancy_pct between -0.05% and +0.10%; reduce/pause if expectancy_pct ≤ -0.10% or Max DD < -8%.

## January 02, 2026 - Daily Stop Reset + Paper Halt
- Live halt resumed after the UTC day rolled over (daily stop resets on the next day, not a rolling 24h).
- Paper trading hit the −2R daily stop and is paused for the current UTC day.

## January 01, 2026 - Live Daily Stop Triggered
- Live trading entered the 24h pause after the daily stop hit the −2R drawdown threshold.

## January 01, 2026 - Live + Paper Steady
- Both live and paper trading continue without issues.
- 7 open live trades and 7 open paper trades, keeping exposure reasonable without overtrading.
- Meme coins are showing up less as opportunities; possibly their window has passed.

## December 2025
**State at a glance (December recap):** Live trading uses baseline ATR exits (0.8× ATR stop, 1.5R target) with ATR ≤ 1.5× cap plus spread/VMC gates; strict RR gate is archived. Risk guards in place: daily stop (−2%/−$20) and BTC range‑break pause. Gate‑scan runs every 4h with cluster caps (10 total, 3 per bucket) and skips open symbols; paper auto‑updates every 5m and fills poll every 5m. Paper experiment continues to 150 trades; live sizing is $250 notional on $1k assumed equity.

## December 31, 2025 - Live Resume Time Set
- Resume live trading after the 48h cooldown: 2025-12-31 12:41:48 UTC (14:41:48 Greece), based on last live close at 2025-12-29 12:41:48 UTC.

## December 31, 2025 14:40 - Live Notional Reset
- Switched live notional per trade back to $250 (from $125) because clustering caps mean fewer trades per day; staying with $1,000 assumed equity.
- Codex take: acceptable if max-open + daily stop gates remain active; watch drawdown and reduce size if the loss streaks return.

## December 31, 2025 08:20 - Paper Auto Flow + Live Restart Plan
- Fully automatic paper trading is running smoothly, with risk filters active.
- Planning to resume live trading later today (around 14:00).
- Hoping the new risk filters help avoid the larger drawdowns from earlier runs.

## December 31, 2025 - Paper Auto-Update Cron
- Added a dedicated cron job to run `paper_finder_simulator.py update` every 10 minutes.
- Tested it at 1-minute frequency first, confirmed it writes to `logs/paper_finder_update.log`, then restored to 10 minutes.
- Updated the cadence to every 5 minutes for faster P/L refresh.

## December 31, 2025 - Watchdog Fill Poll Cron
- Added a dedicated cron job to run `watchdog_close_old_positions.py --log-fills --skip-close --fills-limit 200 --verbose` every 5 minutes.
- Tested at 1-minute cadence, confirmed it writes to `logs/watchdog_close_update.log`, then restored to 5 minutes.
- Purpose: pull the latest fills from the Coinbase account so closed‑trade logs stay current.

## December 31, 2025 - Extend Paper Experiment
- Keep the fully automated paper experiment running to 150 trades (no manual tinkering) to get a cleaner sample after the abnormal manual day.
- The abnormal day was caused by repeatedly scanning and manually opening trades, leading to a much higher count (16 closes) than normal cadence days.

## December 30, 2025 - Live-Only Runner Toggle
- Added `RUN_PAPER=0|1` to `scripts/run_gate_scan_paper.sh` so I can switch to live-only execution after the paper experiment finishes.

## December 30, 2025 - Breakout Autotrade Paused
- Paused the hourly `run_breakout_autotrade.py` cron line while I archive the breakout flow.

## December 30, 2025 - Paper Trades Back Green
- Paper equity turned positive again.
- The process feels more streamlined now that new paper trades are added on a 4-hour cadence instead of random manual pulls.

## December 30, 2025 18:21 - Resisted Breaking Hiatus
- I felt the urge to break the 48h live-trade pause, but decided not to proceed after Codex’s suggestion.
- Paper experiment is at 66/100 closed trades.
- The auto gate-scan + paper runner is now taking new trades on cadence, which feels calmer and reduces the urge to check constantly or overtrade.

## December 30, 2025 - Daily Stop + Range Break Gate
- Added a daily stop gate (−2% or −$20 on $1k) to `symbol_snapshot.py`; gate-scan now suppresses new commands when triggered.
- Added a range-break circuit: if BTC closes outside the 7‑day range by >0.5× ATR, commands are suppressed until manual review.
- Dashboard now shows both “Daily stop” and “Range break” status (OK/ACTIVE).
- Range break status meaning: **inside** = OK to trade; **outside** = pause until manual review.

## December 29, 2025 - Drawdown Day
- Yesterday saw a major drawdown in both paper and live systems.
- Many stop losses hit; take-profits were scarce.
- Live trades likely stopped on spikes faster than the paper trades, which lagged the SL hits.
- Decision: pause new live entries for 48 hours (paper runs continue); cooldown active now.
- Set non-negotiables: 1–2% equity risk per trade (scaled down when many trades are open), daily loss stop at 2%, and no rule changes until 50–100 closed trades.
- Next live restart will assume $1,000 starting equity and $125 notional per trade (about 1% risk if avg stop ~8%).
- Gate-scan now supports `--baseline-live-position-usd` so live commands can size at $125 while paper sizing stays unchanged.
- Added `scripts/run_gate_scan_paper.sh` and a 4h cron runner (5 */4) to auto-run gate scan and open paper trades; tested with a 1-minute cron and restored to 4h. Log: `logs/gate_scan_paper.log`.
- Added `scripts/drawdown_breakdown.py` to compare live vs paper drawdowns (closure reasons, loss symbols, ATR buckets, and spread‑OK stop rates) over a rolling window.
- Added capacity guards for gate-scan commands: `--baseline-max-open` (default 10) and `--baseline-max-per-cluster` (default 3) to limit concurrent positions by majors/memecoins/alts.
- Cleaned duplicate open paper positions (kept oldest per symbol/side) and added a guard in `baseline_finder_from_snapshot.py` to skip opening duplicates unless `--include-open` is used.
- Note: the “16 closes in one day” spike was a manual over‑trading day (repeated gate scans + manual entries). The 4‑hour auto cadence + cluster caps should prevent this pattern going forward.

## December 28, 2025 - More Live Trades
- Placed more live trades today; many more opportunities are surfacing now, so I’m holding ~10–15 simultaneous trades. Not sure yet if I should cap the number of concurrent positions; will monitor.
- Gate-scan now supports `--baseline-paper-command` to emit a one-shot paper-trade command (skips open live/paper positions unless overridden).

## December 27, 2025 - Live Baseline Trades Started
- Started taking live trades using the looser baseline gates (same as paper trades).
- This is riskier than waiting for the 100-trade paper sample, but it saves time even if it may cost.
- Re-entry rule: after a trade expires/closes, it’s OK to take the same symbol/direction again if gates still pass (no mid-trade flips; optional cooldown if needed).
- Gate-scan now prints baseline command lines only for symbols that pass the baseline gates and are not already open; it also lists open positions being skipped.
- Decision: live trading now follows the same baseline system used for paper trades to save time, even though it increases risk. The stricter RR-gate system is archived for now because it produced too few trades and felt late.
- Live testing command:
  - `python scripts/symbol_snapshot.py --gate-scan --profile focused_no_llm_100 --top 15 --scan-limit 100 --baseline-commands --baseline-portfolio-usd 5000 --baseline-position-pct 5 --baseline-atr-mult 0.8 --baseline-rr 1.5 --baseline-atr-mode clipped --baseline-leverage 50 --baseline-expiry 30d`

## December 26, 2025 - Live Trades Went Sideways
- Live trades went positive briefly, then moved sideways; DOGE hit SL and the rest faded similarly.
- Paper trades continue to balance more often; the looser baseline gate seems to catch both sides of the spikes more evenly.
- Opened new paper trades (baseline RR): BTC LONG, ETH LONG, SOL SHORT.

## December 26, 2025 - Long-Heavy Paper Day
- Yesterday’s paper trades skewed long; the day was positive overall with longs in profit and shorts slightly down.
- LTC was in both paper trades and briefly cleared the live gate; the live trade stayed near breakeven after a small uptick.
- DOGE and SOL briefly cleared the strict RR gate, so I took them live as LONGs.
- Added three more paper trades today: LINK, DOGE, SEI.

## December 25, 2025 - BTC Paper Trade Added
- Took another paper trade on BTC (LONG) after it met the baseline conditions.
- Equity curve is starting to stabilize even with a recent stop loss and other expired losses.
- No live trades meet the stricter RR gate yet.

## December 25, 2025 - LTC Live Trade
- LTC briefly cleared the strict RR gate while ATR stayed within ≤1.5× cap, so I took it live.
- TP/SL were set using the baseline ATR RR bracket (same setup as the paper trade).

## December 24, 2025 - Paper Trades Green Again
- Paper open P/L is back in the green (+17.51 unrealized across 7 open trades), with several positions up ~1–2% as they head toward expiry.
- Gate scan confirms the opportunity window is narrowing again: top‑15 RR gaps are wider and no symbols are near the 2.0 gate.
- Replayed the old short-term entries with baseline exits (`atr_mult=0.8`, `rr=1.5`, `atr_clipped`) and saved the dashboard-ready output to `trade_logs/watchdog_closed_positions_baseline_short_term.csv`.

## December 23, 2025 - Paper Trades Tracking
- Paper trading is ongoing and currently green, with most trades on track to expire positive.
- The PAXG paper trade hit SL; noting it as a less predictable asset in this regime.
- Added MAE/MFE tracking for paper trades on close (uses Coinbase candles when perps keys are available).
- Added `scripts/backfill_paper_closed_positions.py` to backfill MAE/MFE and reclassify expiry outcomes as `expired_profit`, `expired_loss`, or `expired_breakeven` (±0.10% band).
- Placed three new paper trades: DOGE SHORT, SOL SHORT, ETH SHORT after yesterday’s expiries closed in profit. (Note: consider whether the 24h expiry should be longer.)

## December 22, 2025 - Baseline Paper Trades Begin
- Took two paper trades this morning using the baseline exits (ATR*RR) rather than the strict RR geometry.
- Early impression: opportunities are far more frequent; I’ll allow ATR <= 1.5x cap for paper testing and explicitly skip anything > 1.5x (safer cutoff).
- Feeling optimistic but keeping it in paper mode until results confirm the edge.
- Added a shareable paper equity report: `scripts/paper_equity_report.py` (HTML + PNG) and export in `watchdog_dashboard.py`. It uses daily aggregation, so a single trading day shows one point and drawdown reads 0%.
- Added DuckDB + PyArrow tooling: `scripts/convert_to_parquet.py` to store finder/trade logs/backtests as parquet and `scripts/duckdb_query.py` for ad-hoc SQL diagnostics.
- Benefit: parquet keeps historical analytics fast and lightweight as the logs grow.
- Added `scripts/paper_trade_progress.py` to report paper-trade count vs target, win%, avg%, expectancy, and TP/SL/expiry split in one line.

## December 21, 2025 - Baseline Option 1 (ATR*RR) Re-Test
- Ran baseline stats with `atr_mult=0.8`, `rr=1.5`, `atr_mode=clipped` on `trade_logs/watchdog_closed_positions_short_term.csv` (output: `trade_logs/watchdog_closed_positions_baseline_atr_clipped_rr1p5_mult0p8.csv`).
- Result: improved win% and avg% in the 1.0–1.5× ATR buckets versus the earlier baseline, suggesting a better fit for 24h expiry.
- Table (ATR cap ratio buckets):
```
ATR cap ratio buckets (ATR_bps / cap_bps):
bucket        n   win%    avg%    med%  avg_mae  avg_mfe  avg_ratio
-------------------------------------------------------------------
1.0-1.25     45   57.8    0.35    0.33    -2.08     2.19       1.15
1.25-1.5     56   62.5    0.60    1.41    -2.59     2.77       1.37
1.5-2.0      78   47.4    0.13   -0.16    -3.46     3.45       1.77
<=1.0        79   44.3   -0.27   -0.06    -1.06     0.56       0.44
>2.0         73   52.1    0.50    0.14    -5.73     7.55       2.39
```
- Decision: use `atr_mult=0.8`, `rr=1.5`, `atr_mode=clipped` for the next baseline paper trades.
- Confidence check: this is a strong hint but not proof; I’m treating it as ~60% likely better and will build confidence via parallel paper runs before changing live rules.

## December 21, 2025 - Baseline Snapshot Paper Helper Added
- Added `scripts/baseline_finder_from_snapshot.py` to turn snapshot picks into a baseline finder file (`finder_short_baseline.txt`) with fixed ATR*RR exits.
- Optional `--open-paper` hands the baseline file to `paper_finder_simulator.py`, so I can stage paper trades and review them in `watchdog_dashboard.py`.
- New paper rule: when ATR is within ~1.25x cap and spread/VMC gates pass, I can paper-trade baseline RR exits even if geometry would fail (no live execution yet).

## December 21, 2025 - ZEC Trade Resurfaced, Still Too Hot
- ZEC showed up again this morning, but ATR is still above my safe band (ATR ~938 bps vs safe threshold ≤500 bps at 1.25× cap).
- Decision: skip the trade and stay aligned with the low‑volatility preference.

## December 20, 2025 - Baseline vs Current Exit Comparison (Short-Term)
- Compared actual exits vs baseline exits on `trade_logs/watchdog_closed_positions_short_term.csv` (n=331).
- Baseline (ATR raw) outperformed: win% 55.0 vs 42.9, avg% 0.187 vs -0.006, baseline better on 57.7% of trades (65 loss→win flips).
- Baseline (ATR clipped) also outperformed: win% 53.8 vs 42.9, avg% 0.175 vs -0.006, baseline better on 55.3% of trades (62 loss→win flips).
- Decision: keep running the **new RR/ATR rules** until I log ~50 trades, then re‑evaluate before switching exit logic. Currently only ~5 trades into the new regime, so it’s too early to judge.

## December 20, 2025 - ATR Clip Ratio Confirms Conservative Stance
- Ran `scripts/watchdog_atr_clip_analysis.py` on the short-term closed-trade log.
- Results: heavy clipping (>2x cap) shows higher average upside but much worse MAE; lower ATR buckets are flatter but safer.
- Decision: keep my low-ATR preference (<= ~1.25–1.5x cap) even if compounding is slower, because it reduces drawdown pain and fast SL churn.
- Reference table (short-term closed trades):
```
ATR cap ratio buckets (ATR_bps / cap_bps):
bucket        n   win%    avg%    med%  avg_mae  avg_mfe  avg_ratio
-------------------------------------------------------------------
1.0-1.25     45   60.0   -0.00    0.12    -7.54     8.02       1.15
1.25-1.5     56   33.9   -0.47   -0.31    -7.80     7.06       1.37
1.5-2.0      78   41.0   -0.29   -0.21   -10.62     9.82       1.77
<=1.0        79   32.9   -0.19    0.00    -5.10     3.55       0.44
>2.0         73   52.1    0.85    0.28   -13.32    23.06       2.39
```
- Quick read:
  - `>2.0` shows the highest avg% and MFE but also the worst MAE (high upside, high pain).
  - `1.0–1.25` has solid win% but flat avg% (small wins/losses).
  - `<=1.0` has low MAE but also low MFE (safer, less payoff).
  - `1.25–2.0` buckets look weaker on average.

## December 20, 2025 - Snapshot Output Now Uses Rich Tables (ASCII)
- Installed `rich` and updated `scripts/symbol_snapshot.py` + `scripts/long_term_snapshot.py` to render ASCII tables when available.
- Gate-scan and per-symbol outputs are now more readable while keeping a plain-text fallback if Rich isn’t installed.

## December 20, 2025 - ATR Clip Ratio Outcome Test Added
- Added `scripts/watchdog_atr_clip_analysis.py` to bucket closed trades by `ATR_bps / cap_bps`.
- Goal: quantify which volatility regimes (within cap, mildly clipped, heavily clipped) perform best before changing gates.

## December 20, 2025 - ATR Stays Elevated = Disagreement/Two-Sided Flow
- BTC ATR7 is still in the ~3–4k band, which suggests ongoing **price disagreement** rather than a one-way trend.
- The persistence of high ATR likely reflects two-sided flow (dip-buying vs selling, hedging/liquidations, uncertainty), keeping daily ranges wide even when direction is unclear.
- Takeaway: high ATR doesn’t just mean “sellers”—it can signal sustained uncertainty and volatility clustering.

## December 19, 2025 - Snapshot Now Explains RR Drivers
- `scripts/symbol_snapshot.py` now prints an `RR drivers:` line for both long/short (e.g., `tp_clamp, risk=atr_floor`).
- This tells me *why* RR is low: reward capped by structure vs the RR target, and whether risk is dominated by raw stop distance, ATR floor, fee/slippage, or the 0.1% tick floor.
- This makes it easier to see if a low RR is due to market geometry (tight targets) or due to volatility/fees inflating the risk side.

## December 19, 2025 - Gate-Scan Shows RR Still Far (Conditions Worsening)
- Recent `scripts/symbol_snapshot.py --gate-scan` shows most of the top-10 names still stuck at RR≈0.80 (gap ~1.20), so conditions look worse again vs earlier optimism.
- Only PAXG showed ATR within cap, but RR is still far from 2.0; most other names were ATR-clipped, indicating volatility remains too high for clean RRs.
- Spreads were mixed: some acceptable, but at least one candidate (OP) showed a very wide spread, reinforcing the “wait for better conditions” stance.

## December 19, 2025 - Restlessness Check (No Action Taken)
- Feeling restless and impatient with the lack of short-term trades. Not taking action outside the system.
- Considering a small SPOT BTC buy as an outlet, but **no trade placed** yet.
- Decision: stay disciplined and keep the short-term finder as the primary system until conditions improve.

## December 19, 2025 - ZEC Short Skipped (ATR Overcap)
- ZEC surfaced with RR≈2.11 on the short side, but ATR was ~1123 bps versus a 400 bps cap (≈2.8× over cap). That’s heavy clipping, so the RR is not reliable under my current safety stance.
- Decision: skip the trade. Even for maximum risk appetite I’m capping at ~2× the ATR cap (≈800 bps); above that the volatility is too hot.

## December 19, 2025 - Snapshot Shows SL/TP Distance in ATR Multiples
- `scripts/symbol_snapshot.py` now prints how far the stop/target are in ATR units (e.g., `SL 0.6x ATR, TP 1.2x ATR`) so I can spot “noise‑prone” setups quickly.
- This helps explain fast stop‑outs: if SL is much less than 1× ATR, normal volatility can hit it even when the thesis is right.

## December 18, 2025 - Long-Term Snapshot Tool (Daily Metrics)
- Added `scripts/long_term_snapshot.py` as a long-horizon companion to `scripts/symbol_snapshot.py`.
- It uses `LongTermCryptoFinder` (daily candles + long-term indicators like ATR(14), Sharpe, max drawdown) and prints LONG/SHORT entry/SL/TP plus risk level, so I can sanity-check long-term candidates without relying on short-term intraday readouts.
- Added `--gate-scan` to rank the profile universe by closeness to an RR target (default 2.0) using long-term metrics, so I can see when the long-horizon side is improving without forcing trades.
- Gate-scan excludes stablecoins by default (USDT/USDC/USD1/etc) to avoid meaningless “best RR” rows; use `--include-stables` only if needed.
- Reminder: spread + liquidity still matter for long-term trades (execution cost + exit ability), but short-term intraday fields (range position / 6h vol) are mostly entry-timing noise for multi-day holds.

## December 18, 2025 - Long-Term `focused_llm_100` Profile (Finder + Gate-Scan Match)
- Added a long-term profile `focused_llm_100` so my finder run and long-term gate-scan use the same core knobs: `limit=100`, `max_results=10`, `min_volume_24h=5M`, `min_vmc_ratio=0.03`, and `use_openai_scoring=1`.
- This removes the “why don’t these match?” confusion: I can run `long_term_crypto_finder.py --profile focused_llm_100` and then sanity-check the same universe with `scripts/long_term_snapshot.py --gate-scan --profile focused_llm_100`.

## December 18, 2025 - BTC Whipsaw Candles; Relief Rally Talk
- Yesterday BTC printed a big green candle followed quickly by a big red candle (~+3% then ~-3%). The sudden whipsaw sparked discussion on Twitter.
- Some traders are framing it as potential “relief rally” setup, but I’m staying rules-based and waiting for my gates (RR/ATR/liquidity/spread) to align before acting.

## December 17, 2025 - Gate-Scan Now Shows Liquidity + Spread “Distance to Acceptable”
- `scripts/symbol_snapshot.py --gate-scan` now prints three quick “can I actually trade this?” lines alongside RR/ATR:
  - `liq vol=... (x... vs min=..., src=...)`: 24h USD volume, how many times above `min_volume_24h` it is, and the volume source (typically CoinGecko).
  - `vmc=...% (...pp vs 3.0%)`: volume/market-cap ratio and its gap vs `min_volume_market_cap_ratio` (3% on my focused profile). Majors can be marked `(exempt)` from the ratio rule.
  - `spr=... bps (...; cap=...)`: current spread in bps and the “headroom” vs a heuristic acceptable spread cap.
- Spread cap meaning: this “cap” is **not a hard gate**; it’s a quick safety heuristic based on liquidity tier:
  - ≥ `$1B` vol/day → cap `3 bps`
  - ≥ `$100M` vol/day → cap `5 bps`
  - otherwise → cap `10 bps`
- How I read it (safe-first): prefer symbols where volume and VMC both clear their mins **and** spread headroom is positive (spread below cap). Negative headroom means the spread is wider than the acceptable range, so costs/slippage risk are higher even if RR/ATR look good.
- VMC quick read: `vmc` is “volume / market cap”. If `vmc` is above the minimum (e.g., `7.49%` vs `3.0%`), that ratio is healthy and the VMC liquidity check is passing.

## December 16, 2025 - Added Baseline Backtest CSV Generator (Watchdog Trades)
- Added `scripts/watchdog_baseline_backtest.py` to replay entries from `trade_logs/watchdog_closed_positions.csv` with a simple baseline exit model: ATR(7)-based SL, 2R TP, and a 24h expiry.
- The script writes a dashboard-compatible CSV so I can compare equity curves in `watchdog_dashboard.py` by switching the sidebar “CSV path” to the baseline file.
- Baseline modes:
  - `--mode atr_raw`: uses raw ATR(7) (no cap).
  - `--mode atr_clipped`: uses ATR(7) clipped by the same USD + tiered bps caps as the short-term finder, so I can isolate the effect of clipping vs “geometry” (level-based TP/SL rules).
- Goal: establish a realistic benchmark before changing gates/rules again, so I’m not mistaking regime noise for strategy improvement.

## December 16, 2025 - Added Volatility Regime Readout to Snapshots
- Updated `scripts/symbol_snapshot.py` (and `--gate-scan`) to print a quick “volatility regime” line: `ATR21`, `ATR7/ATR21`, and `TR1/ATR7`.
- Purpose: help tell whether volatility is *temporarily spiking* vs *persistently elevated*:
  - `ATR7/ATR21 > 1`: short-term volatility is hotter than the last ~month (possible spike).
  - `TR1/ATR7 < 1`: the most recent candle is calmer than the recent average (possible cooling).
- This is a sanity-check signal only (not a gate): heavy ATR clipping can still hide real stop-out/slippage risk, but these ratios help judge whether that risk might be reverting.
- Decision rule (for safety): only consult these ratios when ATR is **near the cap** (or mildly clipped within my risk tolerance). If ATR is **far above cap** (heavy clipping), treat it as a no-trade regardless of ratios and stay flat until ATR cools back toward the cap.

## December 16, 2025 - Breakout Scanner Long Bias; BTC ATR Near 3K Despite Dip
- `scripts/breakout_scanner.py`: a LONG setup is close to triggering — `KTA (KTA-USDC) LONG RR=1.77 (gap 0.23) | ATR 1635 bps, cap 400 bps, ATR CLIPPED (over cap by 1235 bps)`. Notable because it’s a different directional bias than yesterday’s near-signals.
- BTC snapshot: price `86,357.91`, ATR7 `3,133.03` and still slightly above the tiered cap (`-38 bps` headroom vs `325 bps` cap). RR is still sub-2 (`LONG 0.31`, `SHORT 0.75`), but ATR continues drifting closer to the “tradable” zone.
- Takeaway: even with yesterday’s sharp BTC decline, ATR staying near ~3k suggests volatility may still be stabilising; if RR expands while ATR continues easing, windows should reopen.
- Decision: ZEC surfaced as a candidate (briefly showing RR near/above 2 on one side), but I skipped it because raw ATR was far above the clipping cap (heavy ATR clipping), which implies higher real stop-out risk than the RR number suggests.
- ATR clipping tolerance (rule-of-thumb): evaluate `ratio = ATR_bps / cap_bps` (or equivalently “over cap bps”).
  - Safe: `ratio ≤ 1.25` (≈ over-cap ≤ 25% of cap; e.g., cap 400 → over-cap ≤ 100 bps).
  - Medium: `ratio ≤ 1.5` (e.g., cap 400 → over-cap ≤ 200 bps).
  - Aggressive: `ratio ≤ 2.0` (e.g., cap 400 → over-cap ≤ 400 bps).
  - Beyond `2.0×` is heavy clipping (slippage/stop-out risk rises fast) — generally skip.
- Current stance: I’m staying **medium risk or safer** for now; avoid trades that only qualify because ATR is heavily clipped (e.g., ZEC at ~3×+ cap).

## December 15, 2025 - FET Trade Taken; Signals Still Sparse
- This morning I checked the finder logs and a FET trade surfaced; I took it.
- Outcome: the FET trade hit TP (win).
- Observation: only ~1–2 trades/day are surfacing now, which suggests conditions are still less favorable than they used to be (the stricter gates are doing their job).
- Insight: when only 1–2 trades pass the filters, it’s often a sign the overall market regime is still “unfriendly” to the strategy (most symbols are failing ATR/RR/liquidity gates). The few that sneak through can be borderline and more prone to stop-outs. It may be safer to treat “opportunities ≥ 5” as a regime/breadth confirmation before taking trades, but this should be validated with log stats before hard-coding any gate.
- Gate-scan now shows early signs of a regime shift: ATR headroom is getting closer to zero on majors and RR gaps are shrinking (e.g., BTC ATR is only slightly above its cap, and SOL is within ~0.2 RR of the 2.0 gate). Many alts still fail the ATR cap even when RR looks good, but overall the market appears to be moving toward more “tradable” conditions.
- Clarified ATR behavior: the “ATR gate” is not an explicit reject/allow filter. The correct term is **ATR clipping**: when raw ATR is above the cap, the finder uses the capped ATR value in stop/TP sizing and in the RR calculation. That means trades can still be suggested even when raw ATR is above the cap, because the RR gate is evaluated using the clipped ATR (and other geometry/fee inputs).

## December 14, 2025 - Cron Surfaced a SYRUP Long
- `short_term_crypto_finder.py` cron surfaced a new SYRUP long in `finder_short.txt`.
- Executed via: `python add_position_from_finder.py --file finder_short.txt --portfolio-usd 5000 --leverage 50 --order market`.
- Trade: `SYRUP-PERP-INTX` LONG — entry `0.2775`, TP `0.29371`, SL `0.26982`.
- Size: `5%` of `$5,000` (≈ `$250` position notional), leverage `50x`, expiry `30d`.
- Planned PnL vs position: TP `+$14.60` (`+5.84%`) | SL `-$6.92` (`-2.77%`).
- Outcome: SL hit (loss).
- Gate-scan (top 10 closest to RR 2.0): several alts are near/at RR, but most are still blocked by ATR being far above cap (headroom negative). This suggests the market has “almost tradable” RR geometry, but volatility is still too elevated for the strict ATR filter to allow most of these.
  - XCN (XCN-USDC) SHORT RR=2.25 (gap 0.00) | ATR 1210 bps, cap 400 bps, headroom -810 bps
  - BARD (BARD-USDC) LONG RR=1.79 (gap 0.21) | ATR 1210 bps, cap 400 bps, headroom -810 bps
  - ZORA (ZORA-USDC) LONG RR=1.25 (gap 0.75) | ATR 833 bps, cap 400 bps, headroom -433 bps
  - PAXG (PAXG-USDC) LONG RR=0.80 (gap 1.20) | ATR 125 bps, cap 350 bps, headroom +225 bps
  - LINK (LINK-USDC) SHORT RR=0.80 (gap 1.20) | ATR 614 bps, cap 400 bps, headroom -214 bps
  - SUI (SUI-USDC) LONG RR=0.80 (gap 1.20) | ATR 718 bps, cap 400 bps, headroom -318 bps
  - CRV (CRV-USDC) SHORT RR=0.80 (gap 1.20) | ATR 741 bps, cap 400 bps, headroom -341 bps
  - AAVE (AAVE-USDC) LONG RR=0.80 (gap 1.20) | ATR 681 bps, cap 400 bps, headroom -281 bps
  - SYRUP (SYRUP-USDC) SHORT RR=0.80 (gap 1.20) | ATR 961 bps, cap 400 bps, headroom -561 bps
  - FARTCOIN (FARTCOIN-USDC) SHORT RR=0.80 (gap 1.20) | ATR 1135 bps, cap 400 bps, headroom -735 bps

## December 13, 2025 - BTC RR Rebounds as ATR7 Drifts Toward 3k
- BTC: ATR7=3,190 (only ~-28 bps over the 325 bps cap) and RR has improved sharply (LONG 1.42 / SHORT 1.47), now within ~0.5–0.6 of the 2:1 gate. This is the closest BTC has been to qualifying in weeks, suggesting conditions are stabilising.
- ETH: ATR still well above its cap (-210 bps headroom) and RRs remain below 2 (LONG 0.98 / SHORT 0.57) despite momentum picking up; still not close.
- SOL: RR remains far (LONG 0.25 / SHORT 0.80) with ATR above cap (-193 bps headroom); not near.
- XRP: RR still far (LONG 0.31 / SHORT 0.80) with ATR slightly above cap (-55 bps headroom); not near.
- Takeaway: majors are mostly still blocked, but BTC is trending toward a tradable window if ATR continues easing and RR can push above 2.
- Execution: today’s `short_term_crypto_finder.py` cron surfaced a FARTCOIN short in `finder_short.txt`, and I took the trade per plan.
- Outcome: the FARTCOIN short hit SL (Dec 13).

## December 12, 2025 - Email Burst Faded, Volatility Still Sticky
- The repeated >5‑opportunity email batches from `short_term_crypto_finder.py` have tapered off again after yesterday’s burst, so the ATR bps tiering probably wasn’t the only driver of signals returning.
- BTC ATR7 continues to hover above 3k (cap still binding), suggesting volatility is still sticky and high‑quality windows may remain rare.
- Mindset: fewer but cleaner trade windows should be less stressful and more profitable than forcing daily trades, while accepting that some risk is unavoidable even with strict gates.
- Clarified tooling: finder profiles set filter/score presets and a default scan breadth, but don’t hard‑code products. In gate‑scan mode, `--scan-limit N` is just a speed knob to scan only the top N symbols from the profile‑filtered universe.

## December 11, 2025 - Signals Surge, RRs Still Sub-2
- short_term_crypto_finder: Email batch overnight with >5 candidates (finder_short.txt now lists 7 opps). RR remains <2 on majors (BTC long/short 0.31/0.80; ETH 0.78/0.29; XRP 0.25/0.60; SOL 0.25/0.80). ATR caps binding (e.g., BTC cap ~325 bps; ETH cap ~350 bps; alts ~400 bps), but RR geometry still the blocker.
- breakout_autotrade: multiple near SHORT breakdowns logged, but no trigger (price hasn’t closed through the swing with RR≥2).
- Snapshot (07:00 UTC): BTC 90,156; ATR7=3,458 (cap binding, -59 bps headroom); ETH 3,201; ATR7=187 (cap binding, -234 bps headroom); XRP ATR7 ~0.10 (-111 bps headroom); SOL ATR7 ~8.58 (-254 bps headroom). RRs still sub-2 across the board.
- Added observability: `scripts/symbol_snapshot.py --gate-scan` now surfaces the closest symbols to RR/ATR gates across the profile, so we can see when the universe is near tradable without touching filters.
- Gate-scan takeaways: the only symbols clearing both gates were stables (USDT/USD1) — treat as non-actionable. A couple of alts (KITE, FARTCOIN) had RR≥2 but ATR was ~800–1000 bps over cap, so volatility still blocks them. The rest of the “closest 15” were still 0.5–1.2 RR away and mostly above ATR caps, meaning most of the universe remains far from true tradable conditions.

## December 10, 2025 (Night) - Two Trades Closed
- FARTCOIN-PERP long: TP hit (opened 17:52Z, closed 19:00Z) at ~4.75R on notional (+11.8% of position); clean exit per plan.
- ZK-PERP short: SL hit (opened 17:49Z, closed 20:17Z) at -3.08% of position (~-7.7 bps on notional); taken per plan.
- Net: one TP, one SL from the first batch after a month of no signals. ATR tiering may have helped surface these; continue to monitor if more alts meet gates.


## December 10, 2025 (Evening) - Finder Finally Surfaced Trades
- First valid finder_short.txt in ~1 month (7 opps). Took two trades: ZK PERP short, FARTCOIN PERP long (both ~2.1R brackets). Unsure if the tiered ATR bps cap change tipped it, but signals reappeared.
- List highlights: mix of alts (ZK, FARTCOIN, JASMY, POPCAT, ZORA), a SOL short, and even USDT short. ATR caps now bind across assets; RR=2+ printed for all seven.
- Action: holding ZK/FARTCOIN per plan; monitor the rest if risk budget allows. Stay strict on stops/TPs.

## December 10, 2025 (Later) - Tiered ATR Caps Rolled Out
- Metrics: ATR7=3,408 (BTC); top RR long/short ~0.80/0.31 (BTC); trades taken=0.
- Change: Added tiered ATR bps caps on top of the 3k USD cap — BTC/mega caps ≈325 bps, large caps ≈350 bps, majors ≈400 bps, small alts ≈450 bps. Stops/TPs now bind sensibly across prices; ATR gate no longer “not binding” for alts. RR gate remains 2:1.
- Impact: BTC still above cap (~-44 bps headroom); ETH/DOT/SOL/XRP now show ATR headroom in bps, making the cap meaningful. RRs remain <2, so no trades. Safety intact, clearer gating info in `symbol_snapshot`.

## December 10, 2025 - ATR Stalling Just Above 3k, RR Still Sub-2
- Metrics: ATR7=3,285.60 (BTC); top RR long/short=0.80/0.31 (BTC); near-breakouts (today)=yes (logged via autotrader); trades taken=0.

- Snapshot (focused_no_llm_100): BTC 92,737.99; ATR7=3,285.60 (hovering just above the 3k cap), RR long/short 0.80/0.31; intraday_range_pos ~0.94; daily_vol_30d ~0.0237; spread ~0.001 bps.
- ETH: 3,325.21; ATR7=171.07; RR 0.77/0.19; RSI14 ~71 (overbought), trend still negative (~-0.65%/d); volatility elevated intraday (~0.0304 over 6h).
- XRP: 2.0851; ATR7=0.10; RR long/short 0.61/1.70 — closest to gate but still below 2:1; intraday_range_pos ~0.95; spread ~0.48 bps.
- SOL: 139.12; ATR7=8.17; RR 0.77/0.17; intraday_range_pos ~0.96; spread ~0.72 bps.
- Takeaway: BTC ATR7 has stalled above the 3k cap, keeping RR suppressed; none of the majors clear the RR ≥ 2 gate. XRP short is the nearest at 1.70 but still a skip. Stay flat until ATR eases further or RR expands.

## December 9, 2025 - Near-Breakouts Faded Overnight
- Metrics: ATR7 ~3.1–3.3k (BTC); top RR long/short ~0.8/0.3; near-breakouts=none after 00–08 UTC; trades taken=0.

- Latest hourly logs (00:00–08:00 UTC) show no near-breakouts; prior LTC/ATOM near-long flags (22:00–23:00 UTC on Dec 8) faded as price backed away from swing levels.
- Autotrader continues hourly on USDC majors, flat until a bar actually closes through a swing level with RR=2.

## December 8, 2025 - Breakout Signals Close, No Fills Yet
- Metrics: ATR7 ~3.2–3.4k (BTC); top RR long/short ~0.8/0.3; near-breakouts=multiple per hour within ±0.5%; trades taken=0.

- Hourly breakout autotrader logs show multiple near-long breakouts (BTC, ETH, SOL, AVAX, LINK, UNI, INJ, etc.) and one near-short breakdown (MKR) within ±0.5% of swing levels, but none have closed through to trigger trades.
- All runs (through 06:00 UTC) ended with “No breakouts found / No qualifying breakout found,” so no positions were opened yet. Staying strict with RR=2:1 and the 24h lock.
- Backtest check since autotrader start (Dec 6) shows brief breakouts on Dec 7–8 (e.g., BTC/USDC 2025-12-07 18:00 UTC, ETH/SOL/DOGE/OP/ARB/UNI/INJ 2025-12-08 09:00 UTC, AAVE 2025-12-08 14:00 UTC). Cron runs didn’t catch sustained closes through the swing levels—signals were fleeting and didn’t align with the hourly run/close. Market conditions: shallow pushes to swing levels without follow-through, reinforcing a wait stance.

## December 8, 2025 - Identity Check: Short-Term, Selective, Systematic

- My style is a niche, rules-based short-term approach (intraday-to-24h) with strict RR/ATR gates and a willingness to sit flat for days. Not classic day trading (no daily trade mandate) and not multi-day swing; more “short-term systematic” with rare entries.
- Breakout autotrader + short-term finder both enforce hard SL/TP and (optionally) a 24h timeout; downtime is a feature for risk control, not a bug.
- Expect low frequency; when conditions line up, execute mechanically and let the process, not impatience, drive action.

## December 7, 2025 - Breakout Autotrader Observability

- Breakout scanner/auth: normalized Coinbase PEM secrets; autotrader now authenticates reliably and falls back to Kraken if needed.
- Added near-breakout logging (±0.5% to swing highs/lows) to `breakout_scanner.py` so cron logs show when setups are close even if no trade triggers.
- Autotrader scans USDC majors hourly with RR=2:1, $500/50x, 24h lock; still no signals today, but alerts will show when price is kissing breakout levels.

## December 6, 2025 - Added Breakout Autotrader

- Built and scheduled an hourly breakout auto-runner scanning USDC majors (BTC/ETH/SOL/XRP/ADA/DOT/AVAX/LINK/LTC/DOGE/OP/ARB/ATOM/UNI/AAVE/MKR/INJ) with 1h lookback; it writes finder-style output and places $500/50x trades when RR≥2.
- Coinbase auth now normalizes PEM secrets; falls back to Kraken. Current runs show no signals today, but this strategy should fire more often than the strict short-term finder.
- Hourly cron is active; lock file prevents multiple concurrent trades (24h).

## December 4, 2025 - Spreads Spiking, ATR Still High, RR Weak

- Snapshots (focused_no_llm_100):
  - BTC 93,208.07; ATR7=3,488.42; daily_vol_30d=0.0255; intraday_range_pos=0.934; intraday_vol_6h=0.0191; spread_bps≈0.121.
  - ETH 3,203.39; ATR7=167.54; daily_vol_30d=0.0402; intraday_range_pos=0.930; intraday_vol_6h=0.0162; spread_bps≈0.031.
- Spreads flared versus usual ~0.001 bps, and ATRs remain elevated; RRs deteriorated (BTC long/short 0.53/0.27, ETH 0.57/0.19). The restart window is pushed back again.
- Liquidity strong (BTC Vol24h ~77.3B; MCAP ~1.857T; ETH Vol24h ~31.8B; MCAP ~385B). RSI14s ~59 (BTC) and ~65 (ETH) with mildly negative trends (~-0.55% to -0.75%/d); momentum split (~0 vs 100), reinforcing a stay-flat stance until volatility and spreads cool.
- Adjustment: implemented a 3k USD ATR cap in the short-term finder (configurable via SHORT_MAX_ATR_USD) so stops/TP sizing won’t be blown out by extreme ATR; goal is to let RR recover sooner while keeping the RR ≥ 2 gate.

## December 3, 2025 - ATR Still High, Restart Pushed Back

- Snapshot (focused_no_llm_100): BTC 92,848.59; ATR7=3,807.43 (elevated again); daily_vol_30d=0.0262; intraday_range_pos=0.528; intraday_vol_6h=0.0036; spread_bps≈0.001.
- RR remains weak (long 0.70 / short 0.24), so the “day to restart trading” is pushed back until ATR cools and RR clears 2:1.
- Liquidity strong (Vol24h ~91.3B; MCAP ~1.856T); RSI14 ~58.7 with a negative trend (-0.563%/d). Momentum still split (mom ~0.01 vs 99.99), reinforcing a wait-and-see stance.

## December 2, 2025 - RR Improving, Still Below 2:1

- Snapshot (focused_no_llm_100): BTC 87,118.43; ATR7=3,271.40; daily_vol_30d=0.0243; intraday_range_pos=0.761; intraday_vol_6h=0.0103; spread_bps≈0.001.
- RR has picked up: long 1.18 / short 1.05 (first time above 1 recently), but still below the 2:1 gate.
- Liquidity strong (Vol24h ~76.1B; MCAP ~1.746T); RSI14 ~35.7 with a negative trend (-0.58%/d); momentum split (mom ~0 vs 100). Need further ATR easing and/or a cleaner trend break to clear the RR threshold.

## December 2, 2025 (Late) - Big Green Candle, ATR7 Jumps Back Toward 4k

- Snapshot (focused_no_llm_100): BTC 92,023.64; ATR7=3,937.20; daily_vol_30d=0.0271; intraday_range_pos=0.539; intraday_vol_6h=0.0004; spread_bps≈0.001.
- Large daily candle lifted price and volatility; RR deteriorated again (long 0.70 / short 0.23), well below the 2:1 gate.
- Liquidity strong (Vol24h ~83.1B; MCAP ~1.836T); RSI14 ~55.8 with still-negative trend (-0.571%/d). Need ATR to cool back down to resume RR improvements seen earlier in the day.

## December 1, 2025 - ATR7 Tick Up on BTC, RR Still < 2:1

- Snapshot (focused_no_llm_100): BTC 86,146.33; ATR7=3,324.63 (picked up vs yesterday); daily_vol_30d=0.0243; intraday_range_pos=0.897; intraday_vol_6h=0.0030; spread_bps≈0.001.
- RR remains below 2:1 (long 0.80 / short 1.08); short-side geometry improved but still not clearing the gate.
- Liquidity strong (Vol24h ~63.8B; MCAP ~1.722T); RSI14 ~30 with a negative trend (-0.578%/d). Await lower ATR and/or cleaner trend to re-open qualifying setups.

## November 2025

## November 30, 2025 - ATR7 Near 3k, Conditions Calming

- Snapshot (focused_no_llm_100): BTC 91,457.78; ATR7=3,006.34 (still drifting lower, signalling stabilisation); daily_vol_30d=0.0233; intraday_range_pos=0.435; intraday_vol_6h=0.0031; spread_bps≈0.001.
- RR inching up but still sub-2:1 (long 0.80 / short 0.30); need further ATR cooling or cleaner trend to unlock valid entries.
- Liquidity healthy (Vol24h ~39.3B; MCAP ~1.823T); RSI14 ~53 with a mild negative trend (~-0.57%/d). Momentum remains split (mom ~0.02 vs 99.98), reinforcing the “wait for edge” stance.

## November 29, 2025 - ATR7 Continues Lower, RR Edging Up

- Snapshot (focused_no_llm_100): BTC 90,585.15; ATR7=3,286.35 (downtrend intact); daily_vol_30d=0.0234; intraday_range_pos=0.522; intraday_vol_6h=0.0061; spread_bps≈0.001.
- RR creeping closer to 2:1 but still short (long 0.80 / short 0.28); falling ATR is helping but TP/SL geometry remains too tight to clear the gate.
- Liquidity solid (Vol24h ~59.3B; MCAP ~1.806T); RSI14 ~48 with a mild negative trend (-0.566%/d). Need further ATR cooling or a cleaner directional move to unlock valid entries.

## November 28, 2025 - ATR7 Drifting Down, Still Sub-2 RRs

- Snapshot (focused_no_llm_100): BTC 91,566.18; ATR7=3,454.74 (continuing downward, early signs of stabilising); daily_vol_30d=0.0236; intraday_range_pos=0.28; intraday_vol_6h=0.0232; spread_bps≈0.001.
- RR remains below 2:1 (long 0.80 / short 0.27); tighter ATR helps but TP/SL geometry still fails the gate.
- Liquidity solid (Vol24h ~51.7B; MCAP ~1.826T). Momentum/RSI modest (RSI14 ~53), trend still negative (~-0.55%/d); need further ATR cooling or cleaner trend to open up entries.

## November 27, 2025 - Zero Setups Is Expected in High ATR Chop

- With the stricter RR ≥ 2:1 gate, it’s normal to see zero qualified setups for weeks across the full Coinbase perps universe (~200 assets) when ATR7 is elevated and ranges are tight.
- Elevated ATR7 widens stops while TPs stay near recent extremes, keeping RR < 2 even on majors; liquidity filters aren’t the blocker.
- Until volatility cools and trends allow TPs to sit farther from entry, the model will keep printing “no opportunities,” which is an intended safety feature, not a data failure.

## November 26, 2025 - BTC ATR7 Easing Further

- Snapshot (focused_no_llm_100): BTC price 87,373.77; ATR7=3,720.86; daily_vol_30d=0.0222; intraday_range_pos=0.71; intraday_vol_6h=0.0053; spread_bps≈0.001.
- RR still sub-2:1 (0.23 long / 0.80 short) despite ATR drifting lower; stops remain wide while TPs sit close to recent lows.
- Volume/liquidity solid (Vol24h ~63.8B; MCAP ~1.744T); the ATR-driven RR gate is the blocker. Need ATR7 to cool toward ~1–2k before expecting more setups.

## November 24, 2025 - ATR7 Easing, Still Elevated

- BTC ATR(7) is starting to edge down but remains far above the ~1–2k “quiet” band; likely weeks away from that zone at the current pace.
- RR remains sub-2:1 across majors; primary gate is still wide stops vs. close TPs. Continue watching ATR(7) drift lower before expecting a meaningful increase in setups.
- Insight: the strict RR gate is forcing “no trade” during high-vol chop—use this as discipline training to avoid chasing marginal setups until the edge returns.

## November 23, 2025 - ATR(7) Too High to Clear RR Gate

- Snapshots for majors (BTC/ETH/SOL/XRP) still show RR well below 2:1 (BTC ~0.17/0.80; ETH ~0.14/0.61) because daily ATR(7) is elevated.
- Daily ATR(7) on BTC is ~4.3k (≈5% of price), versus the ~1–2k range from quieter months; wide stops + TPs clamped near recent range keep reward/risk < 2.
- Liquidity/score filters aren’t the blocker; mapping is fixed and volume data is present. The primary gate is the 2:1 RR failing under high-vol chop near lows.
- Visual cue: on the 1D chart with ATR set to length 7, ATR needs to cool toward prior ~1–2k levels and/or price break cleanly so TP can sit farther from entry. Until then, expect 0–1 candidates.

## November 22, 2025 - Tight Filters, Few Setups

- After tightening `short_term_crypto_finder.py` (RR ≥ 2:1, liquidity/risk caps), the last few days show 0–1 viable setups.
- Market is high-volatility and choppy near range lows; stops widen and TPs stay close, so RR falls below 2:1 even on big movers.
- Mapping fixes for majors (BTC/ETH/SOL/XRP/USDT/USDC/etc.) are in; lack of signals is due to conditions, not data gaps.

## November 16, 2025 - Paper Trading Continues + Finder Archive

- Resumed the daily paper-trade cycle with the short-term finder, keeping the “5 balanced trades held for 24h” rule intact.
- Added `scripts/archive_finder_output.sh` so every finder run saves a dated text file under `finder_logs/`. That way, once I have ~30 days of logs I can replay the exact signals with `paper_finder_backtest.py` instead of relying solely on forward paper trading.
- Plan: keep logging the five trades per day via `paper_finder_simulator.py` while the archiver builds the data set for a more comprehensive backtest next month.

## November 14, 2025 - Bear Market Pressure

- Stayed flat for another session while Bitcoin legged lower and altcoins got demolished across the board.
- Feels like the bear market is reasserting itself with more force, so I’m staying sidelined until the tape calms down.

## November 13, 2025 - Red Day on the Sidelines

- Market bled across the board and Bitcoin briefly fell below $100K, but I honored the ongoing pause and didn’t put on any trades.
- Logging it anyway so the equity curve shows that the red day belonged to price action, not new execution mistakes.

## November 12, 2025 - Weird Tape

- Another day where the five finder positions floated green for most of the 24h cycle and then two of them tagged their stops near the finish, flipping the basket into a loss. Market feels choppy: signals behave for 18+ hours and then reverse right before expiry. Keeping the 24h discipline, but it’s tough to sit through.

### Trading Pause
- String of back-to-back drawdowns has fried my focus. I’m stepping away from execution for a bit—no live trades until I reset and review the playbook with fresh eyes.

## November 11, 2025 - Waiting the Full 24h Still Hurts

- All five finder trades spent most of the session green—at one point the basket was up roughly $30—but every exit was an expiry/stop and I locked in a $10 net loss.
- Staying disciplined with the 24h rule kept me out of worse chop, but it’s frustrating to watch solid intraday P/L vanish right before the deadline.

## November 10, 2025 - Finder Balancing Rules

- Two full days of running `short_term_crypto_finder.py` to completion (no early exits before the 24h cycle closes) has made the equity curve noticeably calmer and expectancy positive again.
- Execution rule going forward: **trade only when the finder surfaces at least five signals and they split 2L/3S or 2S/3L**. If the list is all longs, all shorts, or fewer than five total, I skip the entire session.
- That directional balance keeps me from leaning into a single regime and avoids forcing trades on thin days when the universe isn’t offering enough quality.

## November 8, 2025 - Returning to Short-Term Finder Discipline

### Reservoir Takeaways
- The multi-coin reservoir push fizzled: R/R = 2:1 on paper but live trades routinely hit the stop long before reaching target.
- Winners like PROMPT closed quickly, yet most positions bled out well ahead of the 24h expiry, leaving the strategy net negative despite the theoretical edge.
- Reordering risk checks and tinkering with ATR multipliers didn’t solve the early-stop problem, so the promised diversification never materialised.

### Pivot Back to `short_term_crypto_finder.py`
- Moving back to the finder workflow, which has historically been steadier and easier to reason about.
- Key tweak: **must** let every finder trade run its full 24h evaluation window (unless the stop/TP is hit). Cutting them early to “lock” gains clearly ruined expectancy.
- No LLM overlay for now; I’ll rely on the new `focused_no_llm_100` profile to keep scoring deterministic while I rebuild confidence.

### Next Steps
1. Run the finder daily with strict adherence to the 24h completion rule; review P/L only after the cycle ends.
2. Track R/R vs realised outcomes in the Watchdog CSV to verify whether holding full-cycle restores the historical hit rate.
3. Keep reservoir tooling in paper mode as a monitoring aid, but don’t let it override finder executions until its own metrics improve.

## November 7, 2025 - Watchdog P/L Snapshot & Guardrails

### What the CSV Shows
- Pulled `watchdog_closed_positions.csv`: only 8 trades logged since the reservoir go-live, expectancy at **–3.81 USDC/trade** with a **37.5% win rate**.
- Losses cluster in recent shorts (`ICP`, `PROMPT`, `TOWNS`); longs remain positive (+86 vs –116 on shorts), so directional bias slipped during the latest volatility spike.
- MINA and B3 winners can’t offset the outsized short losers—median trade is negative, confirming the edge disappeared quickly once regime changed.

### Immediate Adjustments
- Treat reservoir signals as “pilot” size until `_evaluation.csv` reports Sharpe > 0.2; finder trades stay half-size unless the reservoir agrees.
- Write a quick expectancy tripwire that scans the last N trades in the CSV and auto-flips execution to paper mode after three consecutive negative expectancy readings.
- Temporarily blacklist `ICP` and `PROMPT` perps (or widen ATR multipliers) until the books settle; both delivered heavier losses than the average win.

### Follow-up
- Weekend task: re-run the reservoir with a slightly higher threshold (0.0045) and compare hit rate before reinstating full size on Monday.
- Instrument the Watchdog dashboard with long vs short cumulative P/L so directional drift is visible at a glance.
- Keep logging finder vs reservoir divergence; give the consensus gate higher weight whenever volatility breaks the prior regime.

## November 6, 2025 - Regime Protection Playbook

### Finder Vulnerability
- Noted that `short_term_crypto_finder.py` bleeds quickly after regime shifts; rule-based scoring keeps firing legacy setups and fixed TP/SL brackets become too tight in fresh volatility.
- Outcome: when the environment turns, the finder alone can go from edge to drag in a handful of runs unless someone intervenes manually.

### Reservoir Guardrails
- Pairing the finder with `multi_coin_reservoir_daytrader.py` adds an adaptive view: the ridge readout retrains every run and ATR-based brackets expand or contract with current volatility.
- New workflow: require reservoir confirmation for full-size positions; if reservoir disagrees, either skip the trade or scale the size down sharply.

### Automation Safeties
- TODO: build a “tripwire” script that watches the finder’s rolling expectancy (last N trades) and auto-pauses execution after three consecutive negative readings.
- Log the reservoir `_evaluation.csv` daily; if 24h Sharpe dips below a set floor, force signals into paper-trade mode until metrics recover.
- Medium-term idea: intersection engine that only approves trades when finder and reservoir agree on side and comparable TP/SL distances.

### Action Items
1. Prototype the expectancy tripwire and hook it into the execution pipeline.
2. Surface finder vs reservoir P/L divergence on the Watchdog dashboard for quick visual checkpoints.
3. Test a 15m reservoir profile off-line; if it stabilises the evaluation metrics, add it as an additional confirmation layer.

## November 5, 2025 - Multi-Coin Reservoir Day Trader Testing

### Current Strategy Testing
- Actively testing `multi_coin_reservoir_daytrader.py` for automated trade signal generation
- Taking the top 5 best trades per day from the reservoir computing model
- Each trade has a 24-hour expiration period
- System generates signals across multiple crypto assets using shared echo state network readout

### Technical Implementation
- Script uses reservoir computing (echo state network) to predict short-term price movements
- Evaluates multiple crypto pairs for long/short opportunities
- Trade targets and stops are derived from each asset's current volatility regime
- Outputs ranked CSV signals and plain text reports for execution

### Risk Management
- 24-hour time horizon for each trade position
- ATR-based stop losses and take profit levels
- Position sizing based on volatility-adjusted risk parameters
- Focus on high-volume USDC pairs for better liquidity

### Monitoring & Evaluation
- Tracking signal quality and execution performance
- Evaluating hit rates across different market conditions
- Monitoring for overfitting and adapting parameters as needed
- Comparing reservoir predictions against actual market outcomes

### Next Steps
- Accumulate performance data over multiple trading days
- Fine-tune threshold parameters and reservoir hyperparameters
- Assess correlation between predicted returns and actual outcomes
- Consider integrating with existing automated execution framework

---

## October 2025
## October 11, 2025 - Coinbase EURCUSDC Execution Anomaly

### Market Context
- Significant sell-off unfolded on October 10, 2025.
- EURCUSDC pair on Coinbase dipped below the configured stop-loss level during the move.
- Coinbase did not execute the stop-loss order despite the breach.
- Position ultimately flattened at roughly breakeven; unable to fully verify the exact exit mechanics.
- Coinbase activity log shows a ticket for the stop but reports twice the configured stop-loss distance, suggesting the order was never honored at the intended price.
- Subsequent order details (ID `b1b517c9-f48b-4fec-a4a4-4d3f3863102e`) confirm the exit filled at 1.1314 USDC versus the 1.1487 USDC stop—classic slippage as liquidity vanished, leaving the bracket stop to execute as a market fill well below target.

### Current Execution Process
- Running five automated trades each day: top two short setups, top two long setups, plus whichever remaining candidate has the highest score regardless of side.
- Signal generation relies on `short_term_crypto_finder.py`, and `add_top5_from_finder.py` auto-selects the top-scoring entries for execution.
- Monitoring order-routing reliability closely after the missed stop-loss event.

### Follow-up Notes
- Revisit Coinbase order execution logs to confirm whether the stop was acknowledged.
- Assess contingency routing in case of future exchange-level execution gaps.
- Consider redundant exit automation (for example, failover to manual or secondary API call) when the stop threshold is crossed.
- Review sizing/stop-width on thin EURC-PERP books to reduce future slippage and consider wider offsets or manual intervention protocols during fast sell-offs.
- Watchdog performance snapshot (51 trades through Oct 12): expectancy +0.43R with average winners 1.72R vs losers 1.0R, confirming edge despite modest 49% hit rate. Sharpe 0.26 / Sortino 0.37 highlight that equity swings are driven primarily by crypto’s intrinsic volatility rather than poor trade selection.

## June 2025
## June 19, 2025 - Single Bitcoin Position Reflection

### Personal Trading Philosophy Evolution
Having a single Bitcoin position feels much better and safer. The ups and downs don't matter as much when you're focused on one position rather than constantly monitoring multiple trades or small price movements.

### Previous Approach Analysis
I think I previously focused way too much on such small price action, which really messed me up. That kind of intense focus on tiny fluctuations created unnecessary stress and pressure that wasn't sustainable.

### Current Mindset
I don't think I could accept that kind of constant ups and downs anymore. The single position approach provides peace of mind and allows for a more patient, long-term perspective that aligns better with my current trading psychology.

### Key Realizations
1. **Simplified Focus**: One position means one thing to monitor, reducing decision fatigue
2. **Emotional Stability**: Less stress from multiple position management
3. **Better Risk Control**: Single position allows for clearer risk management
4. **Long-term Perspective**: Enables focus on broader market trends rather than micro-movements

### Going Forward
This approach feels more sustainable and aligned with where I want to be mentally. The goal is to maintain this simplified strategy while continuing to develop the supporting systems and analysis tools.

---

## June 18, 2025 - Strategic Pivot: Moving to Long-Term Approach

### Trading Philosophy Shift
- Stopped relying on cursor coding ChatGPT suggestions for trade execution
- Recognized that short-term trade suggestions lack sustainable edge
- Pivoting to a more long-term, systematic approach to trading
- Focus shifting from immediate trade opportunities to building robust trading systems

### Key Changes
1. **Reduced Trade Frequency**: Moving away from high-frequency trading based on AI suggestions
2. **System Development Focus**: Emphasizing long-term strategy development over immediate execution
3. **Risk Management Priority**: Strengthening position sizing and risk management frameworks
4. **Patience and Discipline**: Accepting that sustainable trading requires patience and systematic approaches

### Rationale
- Realized that ChatGPT suggestions, while informative, don't provide a consistent edge
- Short-term trading based on external suggestions leads to inconsistent results
- Long-term success requires developing and following proven systems
- Better to build sustainable processes than chase immediate opportunities

### Forward Strategy
- Focus on developing comprehensive trading systems with clear entry/exit criteria
- Implement proper backtesting and forward testing before live execution
- Build portfolio management approaches that compound over time
- Document and refine processes based on actual market behavior

---

## June 17, 2025 - Bitcoin RSI Dip Trading

### Market Context
- Executed RSI dip strategy using btc_entry_conditions.py
- Initial trade exited at break-even
- Re-entered position and achieved profitable outcome
- Using 20x leverage with $300 margin

### Position Analysis
- Initial entry based on RSI dip conditions
- First position closed at break-even
- Second entry proved successful with profit
- Position size: $300 with 20x leverage ($6,000 exposure)

### Key Takeaways
1. Strategy demonstrated resilience with successful re-entry
2. Break-even exit on first attempt showed good risk management
3. Second entry capitalized on continued market opportunity

### Follow-up Actions
- Monitor RSI conditions for future entries
- Document any pattern improvements in the strategy
- Continue tracking performance of RSI dip strategy

---

## June 16, 2025 - TSLA and Bitcoin Market Update

### Market Context
- TSLA trading around $330, still below $350 target
- Bitcoin showing significant strength, reaching $107,700
- Market showing mixed signals with crypto strength but TSLA target not yet reached

### Position Analysis
- TSLA position still in play with $350 target
- Stop loss at $300 remains in place
- Monitoring for potential correlation between crypto strength and TSLA movement
- Position showing resilience despite target not being reached

### Key Takeaways
1. Patience required as TSLA target remains unhit
2. Notable divergence between crypto and TSLA performance
3. Importance of maintaining discipline with existing targets

### Follow-up Actions
- Continue monitoring TSLA price action
- Assess potential impact of Bitcoin's strength on overall market sentiment
- Maintain existing stop loss and target levels
- Document any significant market correlation patterns

---

## June 12, 2025 - TSLA Target Not Yet Reached

### Market Context
- TSLA has not yet reached the $350 target
- Market momentum has slowed
- Increased uncertainty ahead of Friday's session

### Position Analysis
- Still holding remaining 50% of original position
- No further profit-taking executed
- Stop loss at $300 remains in place
- Monitoring for signs of reversal or renewed strength

### Key Takeaways
1. Patience required as target remains unhit
2. Prepared for potential downside risk on Friday
3. Importance of sticking to the plan and risk management

### Follow-up Actions
- Closely monitor price action on Friday
- Be ready to act if downside accelerates
- Reassess position if market conditions change

---

## June 11, 2025 - TSLA Price Movement Analysis

### Market Context
- TSLA showing strong premarket movement, reaching $332
- Building on previous day's momentum
- Potential path to $350 target by Friday becoming more visible
- Following successful partial exit at $320 on June 10 (50% of position)

### Position Analysis
- Current position showing continued strength
- Previous partial profit-taking strategy at $320 proving effective
- Remaining position approaching new target of $350
- Stop loss at $300 providing adequate protection
- Managing remaining 50% of original position

### Key Takeaways
1. Market momentum aligning with trading thesis
2. Initial partial profit-taking strategy validated
3. Remaining position showing potential for extended gains
4. Successful execution of planned exit strategy

### Follow-up Actions
- Monitor price action throughout the day
- Consider adjusting stop loss if momentum continues
- Document final outcome of position
- Review trading strategy effectiveness

---

## June 10, 2025 - TSLA Position Management

### Market Context
- TSLA approaching target level of $320
- Position showing strong performance

### Position Management Plan
- Planning to sell 50% of position at $320 target
- Remaining 50% to be held with:
  - New target: $350
  - Stop loss: $300
- Strategy demonstrates proper risk management by:
  - Taking partial profits at initial target
  - Letting winners run with trailing stop
  - Maintaining clear exit levels

### Key Takeaways
1. Implementing partial profit-taking strategy
2. Using trailing stop to protect gains
3. Maintaining clear risk management parameters

### Follow-up Actions
- Execute partial sale at $320
- Monitor remaining position
- Adjust stop loss if market conditions change
- Document final outcome of both positions

---

## June 9, 2025 - TSLA Trading Observation

### Market Context
- TSLA opened approximately 3.5% down in premarket and at market open
- Notable social media activity: A Tesla influencer on Twitter reported that people were liquidating their positions

### Personal Analysis & Decision
- Considered buying more TSLA shares given the price drop
- Decided to maintain trading discipline and wait for a more obvious opportunity
- Rationale: Avoided emotional trading based on social media sentiment and short-term price movements

### Key Takeaways
1. Maintained discipline by not immediately reacting to price movements
2. Recognized the importance of waiting for clear opportunities rather than following social media sentiment
3. Demonstrated risk management by not increasing position size during uncertain market conditions

### Follow-up Actions
- Monitor TSLA price action and fundamentals
- Look for clear technical or fundamental signals before considering new positions
- Continue to evaluate market sentiment vs. actual company performance

---
