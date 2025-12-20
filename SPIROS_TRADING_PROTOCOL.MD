# SPIROS — Daily Trading Protocol

*Objective: Protect capital. Take only asymmetric trades. Review fast. Iterate slowly.*

## Core Guardrails (review before login)
| Guardrail | Limit | Triggered Action |
| --- | --- | --- |
| Risk per trade | <= 0.50% of equity | Position size via Section 3 formula; never round up. |
| Daily loss stop | 1.50% of equity or 3 losing trades | Flatten, halt live trading, shift to review/SIM. |
| Circuit breaker A | 2 consecutive losses with MAE > 0.7% | Stop for day, screenshot charts, write quick debrief. |
| Circuit breaker B | 5 consecutive losses (any size) | Stop for day, schedule deep review before next live session. |
| News blackout | +/- 15 min around top-tier news on instrument | Observe only; queue setup for post-release if still valid. |

## 0. Non-Negotiables
- Capital preservation outranks opportunity. No exceptions.
- Only live trade when rested, calm, and platform connectivity confirmed.
- Never move a stop further from entry. Never add size outside of a pre-planned add point.
- If any guardrail is breached, follow the triggered action immediately and document it.

## 1. Pre-Session (10 minutes)
- [ ] Sleep >= 6.5 h, caffeine level stable, stress <= 4/10. Else run SIM only.
- [ ] Platform ready: data live, order routing tested with a micro-lot, OCO templates intact.
- [ ] Charts synced: correct time zone, indicators refreshed, no stale drawings.
- [ ] Economic calendar checked; mark top-tier events in personal notes.
- [ ] Market scan: trade only instruments with clean structure and active range/volatility.
- [ ] Write intent for session (focus metric, e.g., "no chase", "respect stop").

## 2. Setup Qualification (must pass all)
- [ ] Direction: clear trend or well-defined range; no ambiguous bias.
- [ ] Location: entry at a level other traders respect (prior high/low, VWAP, daily level).
- [ ] Confluence: at least two aligned signals (e.g., level + volume shift, break + retest).
- [ ] Volatility: ATR% or recent range supports >= 2R target before session end.
- [ ] Liquidity: spreads tight; funding/fees acceptable; depth stable.
- [ ] Narrative intact: catalyst or order flow context still valid at execution.

## 3. Entry Rules
- Plan first, click second: write entry, invalidation, Target1 (>= 2R), Target2/Trail.
- Use only limit/stop orders at planned levels. If level does not trade, log as missed.
- Position size = (0.5% * Equity) / (Entry - Stop). Round down to nearest allowable lot.
- Do not stack multiple correlated positions that push total risk > 0.75% equity.

## 4. Trade Management
- After fill, immediately stage OCO (hard stop + TP). No manual widening.
- If price moves +1R, trail to breakeven only when structure supports it (swing break).
- Adds allowed only at pre-defined levels and only if total open risk <= 0.75% initial equity.
- Monitor spread and funding; if costs expand materially, consider discretionary exit.

## 5. Exit Logic
- Hard stop equals invalidation. Never cancel it.
- Take TP1 at planned level. For runners, trail below last confirmed swing or structure line.
- If thesis changes before stop, exit immediately and tag as "thesis break".
- Record slippage if exit deviates >0.1R from plan.

## 6. Post-Trade Note (under 60 seconds)
- Log in `trade_logs/` journal: date, ticker, setup tag, R taken, MAE, MFE, result, one lesson.
- Tag avoidable errors separately (chase, size creep, stop move, revenge, news fade).
- Screenshot before/after chart snippets for patterns worth archiving in `backtest_results/` reference.

## 7. Daily Review (<= 5 minutes)
- Count process errors. If >= 2, no next-day live until a SIM block with 0 errors is completed.
- Update equity curve and compare to its 5-day EMA (use helper in `trading/visualization.py`).
  - If equity closes below 5-day EMA, reduce risk per trade by 50% for tomorrow.
- Capture a single improvement lever for tomorrow; defer system redesign.

## 8. Weekly Maintenance (Friday close)
- Compile stats: win rate, average R, expectancy, error rate, news avoidance.
- Only adjust rules with >= 30-trade sample showing expectancy improvement.
- Run a baseline comparison on the same entries to sanity-check whether rule changes are improving exits (not just picking different trades):
  - `python scripts/watchdog_baseline_backtest.py --mode atr_raw`
  - `python scripts/watchdog_baseline_backtest.py --mode atr_clipped`
  - Load the output CSVs in `watchdog_dashboard.py` by changing the sidebar “CSV path” to `trade_logs/watchdog_closed_positions_baseline_*.csv`.
- Refresh watchlist levels and news schedule; archive outdated screenshots.
- Reset position sizing if weekly drawdown <= plan and mental capital restored.

## 9. Red Flags (immediate stop)
- Urge to "win it back", widening stops, doubling size, or skipping logs.
- Emotional spike > 7/10 or attention drift. Switch to SIM, walk 10 minutes, review Section 0.
- Technical issues (latency, data stalls). Flatten positions and pause until resolved.

## 10. Mindset Anchors
- Small losses are rent; big losses are self-inflicted fines.
- Edge = (clean setups x disciplined execution) - error tax.
- Cash is a position; flat is acceptable when clarity fades.
- Trade what is happening, not what should happen.

## 11. Current Style (Dec 2025)
- Identity: short-term systematic trader (intraday-to-24h), strict RR/ATR gates, willing to sit flat for days. Not classic day trading; not multi-day swing.
- Playbooks: short_term_crypto_finder (RR>=2 with ATR clipping cap) and breakout_autotrade (fixed 2R structure). Both use hard SL/TP; optional 24h timeout for closes.
- Short-term finder selection criteria (engine-level):
  - Universe: Coinbase “online” products filtered to preferred quotes; stablecoin bases are excluded (e.g., USDT/USDC/USD1).
  - Prefilters: market cap >= `min_market_cap` (real MC preferred; estimated MC optional) + liquidity checks (min 24h USD volume and volume/market-cap ratio, with major-symbol exceptions).
  - Hard gate: candidates are only kept if `risk_reward_ratio >= 2.0` (RR includes fee/slippage floor and uses ATR clipping caps).
  - Quality filters: `overall_score >= min_overall_score` and `risk_level <= max_risk_level` (I’m currently medium-or-safer).
  - Output shaping: `unique_by_symbol` keeps best side per symbol; `top_per_side` caps each side; final list sorted by `overall_score`.
- Frequency: low by design. No trades if conditions are poor; downtime is acceptable.
- Automation: breakout_autotrade runs hourly on USDC majors with $500 notional/50x, RR=2, 24h lock to avoid stacking. Near-breakouts are logged; no trade unless conditions trigger.
- Housekeeping: `watchdog_close_old_positions.py` runs (or can run) alongside to enforce 24h expiry on open positions; `watchdog_dashboard.py` provides status/metrics visibility on active trades and cron health.
- Observability: `scripts/symbol_snapshot.py --gate-scan` surfaces which symbols are closest to clearing RR/ATR gates (uses tiered ATR caps and RR target; defaults RR=2, top=15) and now also includes quick liquidity/spread “distance” readouts:
  - `liq vol=... (x... vs min=..., src=...)`: 24h USD volume, how many times above `min_volume_24h`, and the data source (typically CoinGecko).
  - `vmc=...% (...pp vs X%)`: volume/market-cap ratio and its gap vs `min_volume_market_cap_ratio` (majors may show `(exempt)`).
    - VMC quick read: if `vmc` is above the minimum (positive pp gap), the VMC liquidity check is passing.
  - `spr=... bps (...; cap=...)`: current spread in bps and headroom vs a heuristic “acceptable” spread cap (3/5/10 bps by volume tier). This is informational, not a hard gate.
  - Safe-first read: prefer `x>=1`, VMC gap positive, and spread headroom positive; negative spread headroom means the spread is wider than the acceptable range (higher cost/slippage risk).
- Diagnostics: `scripts/watchdog_atr_clip_analysis.py` buckets closed trades by `ATR_bps / cap_bps` so I can see which ATR regimes perform best before changing gates.
- Snapshot formatting: when `rich` is installed, both `scripts/symbol_snapshot.py` and `scripts/long_term_snapshot.py` render ASCII tables for easier scanning (plain-text fallback remains).
- Discipline: boredom is part of the edge. Waiting is intentional; do not loosen gates out of impatience. Flat is a valid position.
- ATR handling (important):
  - Timeframe: the strategy uses **daily ATR(7)** (ATR computed on daily candles with `atr_period=7`), not intraday ATR.
  - “Cap” vs “gate”: the short-term finder does **not** reject trades because ATR is high. Instead it **clips** ATR before using it in stop/TP sizing and RR math:
    - Conceptually: `atr_used = min(atr_raw, SHORT_MAX_ATR_USD, price * eff_bps / 10_000)`.
    - `eff_bps` is tiered by price (≈325 bps for BTC, 350 bps for ETH, 400 bps for mid/large alts, 450 bps for small alts) and bounded by `SHORT_MAX_ATR_BPS` if set.
  - Meaning: if ATR is above cap, the model uses the capped ATR value. This can prevent “ATR spikes” from making everything look untradable, but it also means stops/TPs can be tight relative to real volatility, increasing stop-out risk.
  - Practical read: in `scripts/symbol_snapshot.py --gate-scan`, **ATR within cap** means raw ATR <= cap; **ATR CLIPPED** means raw ATR > cap (the model is relying on the cap).
  - Clipping tolerance by risk mode (rule-of-thumb):
    - Compute `ratio = ATR_bps / cap_bps` (or use the “over cap by X bps” number from gate-scan).
    - Safe: `ratio ≤ 1.25` (≈ over-cap ≤ 25% of cap).
    - Medium: `ratio ≤ 1.5`.
    - Aggressive: `ratio ≤ 2.0`.
    - If `ratio > 2.0`, treat as heavy clipping and usually skip (stop-outs/slippage risk rises quickly).
    - Example (cap = `400 bps`):
      - Safe: raw ATR `< 500 bps`
      - Medium: raw ATR `< 600 bps`
      - Aggressive/high-risk: raw ATR `< 800 bps`
  - Outcome check (short-term log): higher clip ratios (>2x) show more upside but materially worse MAE. Lower ratios are flatter but safer, so staying ≤1.25–1.5x cap matches my current safety-first stance even if compounding is slower.
  - Volatility regime ratios (only useful near the cap):
    - I only look at `ATR7/ATR21` and `TR1/ATR7` when ATR is **within cap** or only **mildly clipped** (within my risk tolerance). If ATR is heavily clipped, I skip regardless of ratios.
    - When near-cap, use the ratios as context/timing:
      - `ATR7/ATR21 > 1` → volatility is expanding (spike/heating); be cautious.
      - `ATR7/ATR21 < 1` → volatility is easing vs the last ~month; safer timing.
      - `TR1/ATR7 > 1` → latest candle is larger than the recent average (still hot).
      - `TR1/ATR7 < 1` → latest candle is calmer than the recent average (cooling).
  - Current risk stance: **medium or safer** until the system proves itself across a larger sample; do not take heavily clipped trades just because RR prints ≥2.

## Historical Notes
- Prior iterations leaned on short_term_crypto_finder with 24h hold rule and balanced 5-trade baskets; tightened RR/ATR filters in late 2025.
- November 2025: introduced deterministic profiles (focused_no_llm_100) and stopped forcing trades when fewer than five signals appeared.
- December 2025: added breakout autotrader (USDC majors) with RR=2:1, fallback exchanges, near-breakout logging.

### Active vs Historical Rules
- Active: Sections 0–5 (risk, entry/exit discipline), 6–9 (logging/review), Section 11 (current style/automation).
- Historical/optional: Balanced 5-trade baskets and mandatory 24h holds from earlier finder practice are no longer required; apply 24h timeout only for breakout_autotrade or when explicitly configured.

### Retired/Not In Use
- Mandatory daily trading or “must take X trades”: no longer applicable; flat is acceptable.
- Balanced 5-trade baskets and forced 24h holds across all strategies: retired; only use 24h timeout where explicitly configured (e.g., breakout_autotrade lock).
- Reservoir/multi-basket experiments: not active; stick to current short-term finder and breakout playbooks.

## 11. Reset Protocol
- If daily or circuit breaker stop triggered: review trade log, annotate charts, schedule SIM session.
- After weekly maintenance changes: update playbooks in `docs/` and communicate adjustments to accountability partner.
- For extended drawdown (> 5% equity): halt live trading, run 3 successful SIM days before resuming.

## Glossary
- R: Risk unit defined by (Entry - Stop) sized to 0.5% equity.
- MAE: Maximum Adverse Excursion; largest unrealized loss before exit.
- MFE: Maximum Favorable Excursion; largest unrealized gain before exit.
- OCO: One-Cancels-Other order pairing (stop + target).

### Signature Check (say aloud before first trade)
"I will risk 0.5% per idea, stop at 1.5% daily loss or 3 losers, and never move a stop away."

## 12. Finder Cadence (Short vs Long Horizons) — Historical
- **Long-Term Finder**: built on 365-day lookback with 60–90 bar momentum and 3R+/ATH targets; plan to hold winners for multi-week swings (≈6–24 weeks) unless stop/target or thesis change triggers an exit.
- **Short-Term Finder**: uses daily candles plus 3/10/21-day pulses and 1.3× ATR stops; expect holds of 3–15 trading days. Runs rely on the current daily candle, so reruns intraday introduce noise until the UTC session closes.
- **Daily Rotation Guardrails**: it is acceptable to close a short-term idea inside a day and replace it with fresher momentum, provided (a) the updated scan shows a lower score/risk reclassification or better risk-reward elsewhere, (b) the instrument remains liquid (tight spread, stable depth), and (c) costs (spread + fees) stay within plan. Avoid churning purely because the list reshuffled.
- **Refresh Rhythm**: anchor one scan just after the Coinbase daily bar (≈00:00 UTC); optional midday check is fine for fast conditions. Document every forced exit reason in `trade_logs/` and respect existing stop/target discipline when holding between refreshes.
- **ATR “Gatekeeper” (nuance)**: The RR≥2:1 filter is influenced by daily ATR because ATR affects stop sizing and the RR risk-distance floor. However, the short-term finder uses an ATR **clip** (USD + bps caps) rather than a hard “reject if ATR too high” rule. True favorable conditions show up when many symbols have RR≥2 **and** ATR is **within cap** (not clipped) across the universe.
- **RR Confirmation / Sanity Checks**:
  - Short-term: use `scripts/symbol_snapshot.py` as a spot check; only re-engage sizing when multiple symbols show RR≥2.0 with current filters, not just one-off stables or odd tickers.
  - Long-term: use `scripts/long_term_snapshot.py` (daily candles + long-term indicators like ATR(14)/Sharpe/drawdown) to sanity-check spread/liquidity/volatility before taking multi-day holds; `--gate-scan` shows which symbols are closest to the RR target across the universe.
  - ATR multiples: in `scripts/symbol_snapshot.py`, review the `SL/TP x ATR` line. If SL is well below 1× ATR, the setup is likely noise-prone; prefer wider-than-ATR stops when conditions are hot.
    - Safety-first baseline: **prefer ≥1.0× ATR**.
    - Aggressive floor (max risk): **0.5×–0.75× ATR**.
    - **<0.5× ATR** = skip (noise range).
  - RR drivers: `scripts/symbol_snapshot.py` also prints a short `RR drivers` line (e.g., `tp_clamp, risk=atr_floor`) so I can see whether low RR is driven by target caps or by the risk floor (ATR/fees/tick).

## 13. Current Short-Term Execution (Nov 2025)
- Run `short_term_crypto_finder.py` (profile `focused_no_llm_100`) daily after the UTC candle; use the live-spread RR calc and RR≥2:1 gate.
- Only trade when the finder surfaces at least 5 setups that pass filters; take the top 5 and hold for 24h unless stop/TP hits. If fewer than 5 qualify, skip the day.
- Respect existing guardrails: no early exits, no widening stops, and keep directional balance when possible (avoid all-long or all-short baskets).
- Reassess sizing/risk only after a day of 5 setups prints RR≥2.0 across multiple symbols; the stricter regime stays until ATR cools and RR improves.
