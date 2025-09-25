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

## 12. Finder Cadence (Short vs Long Horizons)
- **Long-Term Finder**: built on 365-day lookback with 60–90 bar momentum and 3R+/ATH targets; plan to hold winners for multi-week swings (≈6–24 weeks) unless stop/target or thesis change triggers an exit.
- **Short-Term Finder**: uses daily candles plus 3/10/21-day pulses and 1.3× ATR stops; expect holds of 3–15 trading days. Runs rely on the current daily candle, so reruns intraday introduce noise until the UTC session closes.
- **Daily Rotation Guardrails**: it is acceptable to close a short-term idea inside a day and replace it with fresher momentum, provided (a) the updated scan shows a lower score/risk reclassification or better risk-reward elsewhere, (b) the instrument remains liquid (tight spread, stable depth), and (c) costs (spread + fees) stay within plan. Avoid churning purely because the list reshuffled.
- **Refresh Rhythm**: anchor one scan just after the Coinbase daily bar (≈00:00 UTC); optional midday check is fine for fast conditions. Document every forced exit reason in `trade_logs/` and respect existing stop/target discipline when holding between refreshes.
