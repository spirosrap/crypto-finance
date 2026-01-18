---
name: crypto-finance-ops
description: This skill should be used when the user asks to "run finder", "run snapshot", "gate scan", "check ATR", "check RR", "update trading journal", "update TRADING_PROGRESSION", "interpret RR output", "interpret ATR output", or discusses the crypto trading pipeline, volatility diagnostics, or journal updates.
version: 1.0.0
---

# Crypto Finance Ops

Use this skill when working in `/home/spiros/crypto-finance` on the active trading pipeline, snapshots, diagnostics, and journal updates.

## Quick Commands

- Short-term snapshot:
  - `python scripts/symbol_snapshot.py --symbols BTC,ETH --profile focused_no_llm_100`
- Short-term finder (focused LLM profile):
  - `python short_term_crypto_finder.py --profile focused_llm_100 --plain-output finder_short.txt --force-refresh`
- Short-term gate-scan:
  - `python scripts/symbol_snapshot.py --gate-scan --profile focused_no_llm_100 --top 15 --scan-limit 100`
- Long-term snapshot:
  - `python scripts/long_term_snapshot.py --symbols BTC,ETH --profile default`
- Long-term finder (focused LLM profile):
  - `python long_term_crypto_finder.py --profile focused_llm_100 --plain-output finder_long.txt --suppress-console-logs`
- Long-term gate-scan:
  - `python scripts/long_term_snapshot.py --gate-scan --profile focused_llm_100 --top 10 --scan-limit 100`
- ATR clip ratio diagnostic (use conda env if ccxt missing):
  - `conda run -n trade python scripts/watchdog_atr_clip_analysis.py --input trade_logs/watchdog_closed_positions.csv`

## Interpretation Cues (Short-Term)

- `RR drivers`: explain whether RR is low due to `tp_clamp` (structure capped reward) or risk floor dominance (`atr_floor`, `raw`, `fee`, `tick`).
- `ATR multiples`: SL/TP in ATR units; <0.5x ATR is noise-prone.
- `ATR CLIPPED`: ATR above cap; RR can still pass but real volatility risk is higher.

## Documentation Updates

- When a new feature, command, diagnostic, or workflow is added, update **all relevant docs**, not just the code:
  - `README.md` (current pipeline + active tools)
  - `README_SHORT_TERM_CRYPTO_FINDER.md` / `README_LONG_TERM_CRYPTO_FINDER.md` when applicable
  - `SPIROS_TRADING_PROTOCOL.MD` for rules/interpretation changes
  - `TRADING_PROGRESSION.md` to log the change with date context

## Journal Updates

- Add new entries to `TRADING_PROGRESSION.md` with newest dates at the top of the month.
- Keep entries concise, action-oriented, and consistent with the day's outputs (RR/ATR, signals taken/ignored).
- Update `SPIROS_TRADING_PROTOCOL.MD` when rules/interpretations change (keep historical sections intact).

## Git Hygiene

- Commit only files touched by the task.
- Push and report if network fails; don't force or rewrite history.

## Safety Notes

- Avoid live trading commands unless explicitly requested.
- Never paste secrets; rely on `.env`/config.
