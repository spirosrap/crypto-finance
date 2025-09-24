SPIROS — DAILY TRADING PROTOCOL
Objective: Protect capital. Take only asymmetric trades. Review fast. Iterate slowly.

0) NON-NEGOTIABLES
- Max risk per trade: 0.50% of equity.
- Max daily loss: 1.50% of equity OR 3 losing trades, whichever hits first → stop.
- Circuit breakers: 
  A) 2 consecutive losses with MAE > 0.7% → stop day.
  B) 5 consecutive losses any size → stop day and review.
- No trades 15 min before/after top-tier news on the instrument.

1) PRE-SESSION (10 min)
- Sleep ≥ 6.5h, caffeine level stable, no agitation → else SIM only.
- Platform checks: data live, orders route, TP/SL tested.
- Market scan: trade only instruments with clean structure and active range/volatility.

2) SETUP QUALIFICATION (must pass all)
- Direction: clear trend or clear range with defined edges. No “maybe.”
- Location: entry at a level others care about (prior high/low, VWAP, daily level).
- Confluence: need ≥2 signals (e.g., level + volume shift; break + retest).
- Volatility: ATR% or recent range supportive of 2R target before session end.
- Liquidity: spreads tight; no abnormal funding/fees that kill R.

3) ENTRY RULES
- Plan first, click second.
- Define exact: Entry, Invalid level (hard stop), Target1 (≥2R), Target2/Trail logic.
- No market order if level not touched. No chasing. If missed → log as “missed,” move on.
- Size = (0.5% * Equity) / (Entry − Stop). Round down to nearest lot.

4) MANAGEMENT
- After fill: place OCO (Stop + TP). No manual widen.
- If price moves +1R, move stop to entry only if structure supports it; otherwise untouched.
- Scale only on planned add levels that keep total risk ≤ 0.75% initial.

5) EXIT LOGIC
- Hard stop = invalidation. Never cancel it.
- Take TP1 at planned level. For runners, trail below last confirmed swing or structure line.
- If thesis changes before stop, exit and record “thesis break.”

6) POST-TRADE NOTE (under 60 seconds each)
- Record: date, ticker, setup tag, R taken, MAE, MFE, result, one lesson.
- Tag avoidable errors separately: chase, size creep, stop move, revenge.

7) DAILY REVIEW (5 min max)
- Count errors. If ≥2 process errors → no next-day live until a SIM block with 0 errors.
- Equity curve check: below 5-day EMA of curve → reduce size by 50% tomorrow.
- One improvement lever only for tomorrow. No system redesign.

8) WEEKLY MAINTENANCE (Friday)
- Stats: win rate, avg R, expectancy, error rate. 
- Keep/change rule only if ≥30 trades evidence and expectancy improves. No mid-week tinkering.

9) RED FLAGS → IMMEDIATE STOP
- Urge to “win it back,” widening stops, doubling size, skipping log.
- Emotional spike >7/10. Switch to SIM. Walk 10 minutes. Review rule 0.

10) MINDSET
- Small losses are rent. Big losses are self-inflicted. Pay rent, avoid fines.
- Edge = (clean setups × disciplined execution) − (error tax).
- Go flat when unclear. Cash is a position.

SIGNATURE CHECK (say out loud before first trade)
“I will risk 0.5% per idea, stop at 1.5% daily loss or 3 losers, and never move a stop away.”
