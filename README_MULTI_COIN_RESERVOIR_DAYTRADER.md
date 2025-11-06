# Multi-Coin Reservoir Day Trader

`multi_coin_reservoir_daytrader.py` generates 24-hour long/short signals for leading Coinbase spot pairs using a shared echo state network (reservoir computing) model. It mirrors the short-term finder workflow, including an execution-ready `finder_short.txt`, so you can feed the results straight into `add_position_from_finder.py`.

Reservoir computing is a neural-forecasting technique where a large, randomly connected recurrent network (the “reservoir”) transforms raw inputs into a rich, high-dimensional state. Because the reservoir weights stay fixed, only a lightweight linear readout must be trained, making it fast and stable for streaming data. This setup excels at capturing non-linear temporal patterns without expensive backpropagation through time.

---

## Highlights
- **Coinbase-native data** – pulls OHLCV candles via the existing `HistoricalData` cache, respecting force-refresh toggles.
- **Shared reservoir** – one 1k-node echo state network produces latent states reused across all coins.
- **Ridge readouts per coin** – fast, deterministic predictions of the next-period log return.
- **Volatility-aware trade levels** – ATR% drives take-profit/stop-loss distances (2× / 1×) with a 24h expiry.
- **Automatic discovery profiles** – scan Coinbase product lists by liquidity/quote via presets or custom filters.
- **LLM-aware profile (optional)** – `focused_llm_100` mirrors the short-term finder’s LLM blend, layering an OpenAI “opinion” on top of reservoir scores.
- **Walk-forward reservoir scoring** – per-coin scaling and stepwise readout refits keep predictions out-of-sample by construction.
- **Signal-first ordering** – longs and shorts sorted together by absolute predicted return so the strongest conviction leads.
- **Finder-compatible output** – ranked CSV diagnostics plus a `finder_short.txt` formatted for `add_position_from_finder.py`.
- **Performance checks** – optional hit-rate and Sharpe summaries in `_evaluation.csv`.

---

## Prerequisites
1. **Environment** – Python 3.11+, dependencies from `requirements.txt` (notably `numpy`, `pandas`, `scikit-learn`, `coinbase-rest` SDK).
2. **Credentials** – Coinbase API key/secret exposed via:
   - `credentials.get_primary_credentials()`, or
   - `API_KEY` / `API_SECRET` environment variables, or
   - `config.API_KEY` / `config.API_SECRET` fallback.
3. **Data cache** – `historicaldata.py` writes to `candle_data/`; ensure the directory is writable.

---

## Quick Start

```bash
# Activate your virtual environment first
python multi_coin_reservoir_daytrader.py \
  --timeframe ONE_HOUR \
  --lookback 720 \
  --profile default \
  --threshold 0.003 \
  --reservoir-size 1000 \
  --alpha 0.3 \
  --spectral-radius 0.9 \
  --washout 50
```

### 15-minute Walk-Forward Profile (no LLM)

```bash
python multi_coin_reservoir_daytrader.py \
  --timeframe 15m \
  --lookback 2000 \
  --profile oos_15m_focus \
  --reservoir-size 400 \
  --alpha 0.25 \
  --spectral-radius 0.9 \
  --input-scaling 0.08 \
  --ridge-alpha 0.05 \
  --threshold 0.003 \
  --short-threshold 0.0045 \
  --force-refresh
```

This command produces three main artifacts:

| File | Description |
|------|-------------|
| `signals/multi_coin_reservoir_daytrader_signals.csv` | Execution CSV consumed by automation (`timestamp, coin, signal, tp_pct, sl_pct, expiry_h`). |
| `signals/multi_coin_reservoir_daytrader_signals_ranked.csv` | Ranked diagnostics with `predicted_return` and indicator snapshots. |
| `finder_short.txt` | Finder-style narrative blocks ready for `add_position_from_finder.py`. |

If evaluation metrics are available, a fourth file is emitted:

| File | Description |
|------|-------------|
| `signals/multi_coin_reservoir_daytrader_signals_evaluation.csv` | Hit-rate and 24h Sharpe computed from the most recent horizon. |

---

## Command-Line Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--symbols` | auto (profile discovery) | Explicit Coinbase product IDs to score (bypasses discovery). |
| `--profile` | `default` | Discovery preset (see table below). |
| `--list-profiles` | off | Print discovery presets and exit. |
| `--quotes` | profile value | Override discovery quotes (comma-separated). |
| `--max-products` | profile value | Cap the number of products analysed. |
| `--min-volume` | profile value | Minimum 24h volume required (quote units). |
| `--timeframe` | `ONE_HOUR` | Coinbase granularity (`1h`, `15m`, `ONE_DAY`, etc.). |
| `--lookback` | `720` | Number of candles per product. |
| `--reservoir-size` | `400` | Echo state network hidden units. |
| `--alpha` | `0.25` | Leak rate of the reservoir (a.k.a. `leaking_rate`). |
| `--spectral-radius` | `0.9` | Spectral radius scaling after random initialisation. |
| `--input-scaling` | `0.1` | Weight scale applied to input connections. |
| `--threshold` | `0.003` | Return cutoff for ±1 signals. |
| `--short-threshold` | `1.5 × threshold` | Optional, higher hurdle for shorts to enter. |
| `--ridge-alpha` | `0.05` | L2 penalty for the readout regression. |
| `--washout` | auto (`≈ 3/leak_rate`) | Warm-up steps discarded before fitting (override if desired). |
| `--force-refresh` | *off* | Bypass cached Coinbase candles. |
| `--output-csv` | `signals/multi_coin_reservoir_daytrader_signals.csv` | Base path for CSV exports. |
| `--plain-output` | `finder_short.txt` | Finder text path for execution scripts. |
| `--log-level` | `INFO` | Console/file logging level. |

---

## Discovery Profiles

| Profile | Max Products | Quotes | Min Volume | LLM | Description |
|---------|--------------|--------|------------|-----|-------------|
| `default` | 40 | USDC, USD | 2,000,000 | ✗ | Balanced coverage of the most liquid majors. |
| `wide` | 150 | USDC, USD, USDT | 500,000 | ✗ | Broad scan across high-volume spot pairs. |
| `focused` | 20 | USDC | 5,000,000 | ✗ | Tight basket of USDC majors for faster execution. |
| `focused_reservoir_100` | 100 | USDC | 5,000,000 | ✗ | Same universe as the LLM profile but ranked purely by reservoir scores. |
| `focused_llm_100` | 100 | USDC | 5,000,000 | ✓ | Focused basket with OpenAI LLM blend (mirrors `focused_llm_100` from the short-term finder). |
| `oos_15m_focus` | 100 | USDC | 2,000,000 | ✗ | Walk-forward friendly 15m preset for fast regime checks (LLM disabled). |

Run `python multi_coin_reservoir_daytrader.py --list-profiles` to view these presets inside the CLI, or mix `--quotes`, `--max-products`, and `--min-volume` for custom discovery filters.

---

## Integration with the Execution Stack

1. **Generate signals**
   ```bash
   python multi_coin_reservoir_daytrader.py --timeframe ONE_HOUR --lookback 720 --profile default
   ```
2. **Review the narrative**
   ```bash
   less finder_short.txt
   ```
3. **Dry-run orders**
   ```bash
   python add_position_from_finder.py \
     --file finder_short.txt \
     --portfolio-usd 25000 \
     --leverage 5 \
     --order limit
   ```
4. **Go live (optional)**
   ```bash
   python add_position_from_finder.py \
     --file finder_short.txt \
     --position-usd 500 \
     --leverage 5 \
     --order market \
     --execute
   ```

---

## Troubleshooting

| Issue | Hint |
|-------|------|
| `No signals generated` | Lower `--threshold`, increase `--lookback`, or ensure Coinbase data exists for each product. |
| `Coinbase API credentials not found` | Export `API_KEY` / `API_SECRET` or add them to `credentials.py`. |
| `Insufficient history to compute evaluation metrics` | Extend `--lookback` or choose a longer timeframe. |
| `finder_short.txt` missing entries | Signals with `0` (flat) are intentionally filtered out. |

---

## Notes & Future Ideas
- Swap in alternative indicators (`pandas-ta` or TA-Lib) by extending `compute_indicators`.
- Experiment with multiple reservoirs (e.g., separate per sector) or hybrid ensembles.
- Add backtest hooks by looping the script across historical days and persisting scoreboards to `backtest_results/`.

Feel free to submit PRs or open issues if you extend the architecture! =)
