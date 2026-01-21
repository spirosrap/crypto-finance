# Regime Tilt Implementation Plan

## Problem Statement

The current gate-scan pipeline uses a balanced 50/50 allocation between LONG and SHORT positions. In trending markets, one side consistently outperforms while the other bleeds, offsetting gains.

**Observed Jan 15-21, 2026:**
- Shorts: ~76% win rate, ~+$95
- Longs: ~29% win rate, ~-$85
- Net: ~+$10 (nearly breakeven despite strong short performance)

## Solution: Regime-Aware Position Allocation

Detect BTC trend regime and tilt position allocation:
- **Bullish regime**: 70% long slots, 30% short slots
- **Bearish regime**: 30% long slots, 70% short slots
- **Neutral**: Keep current 50/50 balanced allocation

This maintains diversification (no 100/0 exposure) while leaning into the favorable direction.

## Regime Detection Logic

**Method: BTC Daily Close vs 20 EMA**

```
IF BTC closes ABOVE 20 EMA for 2 consecutive days → BULLISH
IF BTC closes BELOW 20 EMA for 2 consecutive days → BEARISH
ELSE → NEUTRAL (no tilt, use balanced 50/50)
```

**Expected Accuracy:**
- Clear trends: ~70-75% correct
- Choppy/ranging: ~45-50% (whipsaws)
- Overall: ~55-65%

Even modest accuracy helps because 70/30 allocation limits downside when wrong.

## Imbalance as Reversal Protection

**Critical insight:** When the market shows extreme imbalance (e.g., 60 longs, 6 shorts), this is often a contrarian signal that a reversal is imminent. The current `--balanced` suppression behavior protects against entering at tops/bottoms.

**Design principle:** Regime tilt should only adjust allocation AFTER confirming healthy setup availability on both sides. If there's an imbalance, suppress output regardless of regime.

```
STEP 1: Check minimum candidates exist on BOTH sides (reversal protection)
        - Need at least min_per_side on unfavored side
        - If not enough → SUPPRESS (imbalance = reversal warning)

STEP 2: If balance check passes, apply regime tilt (70/30 allocation)
```

This preserves the natural brake against over-exposure at market extremes.

---

## Implementation Details

All changes are in `scripts/symbol_snapshot.py`. No changes to `run_gate_scan_paper.sh`.

### 1. Add Regime Detection Function

**Location:** After `_vol_regime_ratios()` function (~line 710)

```python
import logging

_logger = logging.getLogger(__name__)

# BTC symbol fallback list - finder may expect different formats
_BTC_SYMBOLS = ["BTC", "BTC-USD", "BTC-USDT", "BTCUSD"]


def _detect_btc_regime(
    finder: "ShortTermCryptoFinder",
    ema_period: int = 20,
    confirm_days: int = 2,
    buffer_bps: float = 0.0,
) -> str:
    """
    Detect BTC trend regime based on daily close vs EMA.

    Args:
        finder: ShortTermCryptoFinder instance for candle data
        ema_period: EMA period (default 20)
        confirm_days: Consecutive closes required to confirm regime (default 2)
        buffer_bps: Buffer in basis points to reduce whipsaw (e.g., 20 = 0.2%)
                    Bullish requires close > EMA * (1 + buffer_bps/10000)
                    Bearish requires close < EMA * (1 - buffer_bps/10000)

    Returns:
        'bullish' - BTC above EMA (+buffer) for confirm_days consecutive closes
        'bearish' - BTC below EMA (-buffer) for confirm_days consecutive closes
        'neutral' - Mixed or insufficient data
    """
    # Try multiple BTC symbol formats
    df = None
    used_symbol = None
    for btc_symbol in _BTC_SYMBOLS:
        try:
            df = finder._get_candle_data(btc_symbol, interval="1d", limit=ema_period + confirm_days + 5)
            if df is not None and len(df) >= ema_period + confirm_days:
                used_symbol = btc_symbol
                break
        except Exception:
            continue

    if df is None or len(df) < ema_period + confirm_days:
        _logger.warning("Regime detection: could not fetch BTC daily candles; defaulting to neutral")
        return "neutral"

    try:
        # Sort and drop incomplete current candle if present
        df = df.sort_index()
        # Check if last candle is incomplete (timestamp is today UTC)
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        last_ts = df.index[-1]
        if hasattr(last_ts, 'date') and last_ts.date() == now_utc.date():
            # Last candle is today (incomplete) - drop it
            df = df.iloc[:-1]
            if len(df) < ema_period + confirm_days:
                _logger.warning("Regime detection: insufficient confirmed candles after dropping incomplete")
                return "neutral"

        # Calculate EMA
        df["ema"] = df["close"].ewm(span=ema_period, adjust=False).mean()

        # Apply buffer
        buffer_mult = buffer_bps / 10000.0
        df["ema_upper"] = df["ema"] * (1 + buffer_mult)
        df["ema_lower"] = df["ema"] * (1 - buffer_mult)

        # Check last N confirmed closes vs EMA (with buffer)
        recent = df.tail(confirm_days)
        above_count = (recent["close"] > recent["ema_upper"]).sum()
        below_count = (recent["close"] < recent["ema_lower"]).sum()

        if above_count == confirm_days:
            _logger.info(f"Regime detection ({used_symbol}): BULLISH - {confirm_days} closes above EMA{ema_period} (+{buffer_bps}bps)")
            return "bullish"
        elif below_count == confirm_days:
            _logger.info(f"Regime detection ({used_symbol}): BEARISH - {confirm_days} closes below EMA{ema_period} (-{buffer_bps}bps)")
            return "bearish"
        else:
            _logger.info(f"Regime detection ({used_symbol}): NEUTRAL - mixed signals in last {confirm_days} closes")
            return "neutral"
    except Exception as e:
        _logger.error(f"Regime detection error: {e}")
        return "neutral"
```

### 2. Add CLI Arguments

**Location:** After `--balanced` argument (~line 2015)

```python
parser.add_argument(
    "--regime-tilt",
    action="store_true",
    help="Tilt long/short allocation based on BTC regime (70/30 split favoring trend direction).",
)
parser.add_argument(
    "--regime-ema-period",
    type=int,
    default=20,
    help="EMA period for regime detection (default: 20).",
)
parser.add_argument(
    "--regime-confirm-days",
    type=int,
    default=2,
    help="Consecutive days required to confirm regime (default: 2).",
)
parser.add_argument(
    "--regime-tilt-pct",
    type=float,
    default=70.0,
    help="Percentage allocation for favored side when regime active (default: 70).",
)
parser.add_argument(
    "--regime-buffer-bps",
    type=float,
    default=0.0,
    help="Buffer in basis points to reduce whipsaw (e.g., 20 = 0.2%%). Default: 0 (no buffer).",
)
```

### 3. Modify `_select_balanced_rows()` Function

**Location:** Line 867

**Current signature:**
```python
def _select_balanced_rows(
    rows: List[Dict[str, object]],
    top: int,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
```

**New signature:**
```python
def _select_balanced_rows(
    rows: List[Dict[str, object]],
    top: int,
    regime: str = "neutral",
    tilt_pct: float = 70.0,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
```

**Replace the slot calculation logic (lines 877-890):**

```python
    # STEP 1: Calculate regime-aware slot allocation
    if regime == "bullish":
        long_slots = int(top * (tilt_pct / 100.0))
        short_slots = top - long_slots
    elif regime == "bearish":
        long_slots = top - int(top * (tilt_pct / 100.0))
        short_slots = int(top * (tilt_pct / 100.0))
    else:  # neutral
        long_slots = top // 2
        short_slots = top // 2

    # STEP 2: STRICT BALANCE CHECK (reversal protection)
    # Even with regime tilt, we require minimum candidates on the UNFAVORED side.
    # This prevents entering when market is extremely one-sided (reversal signal).
    min_required_per_side = min(long_slots, short_slots)  # The smaller allocation

    if len(longs) < min_required_per_side or len(shorts) < min_required_per_side:
        # Imbalance detected - suppress output (potential reversal)
        return [], {
            "longs": len(longs),
            "shorts": len(shorts),
            "long_slots": long_slots,
            "short_slots": short_slots,
            "min_required": min_required_per_side,
            "regime": regime,
            "suppressed": True,
            "suppression_reason": "insufficient_balance",
        }

    # STEP 3: Balance check passed - apply regime tilt allocation
    selected: List[Dict[str, object]] = []
    selected.extend(longs[:long_slots])
    selected.extend(shorts[:short_slots])

    # Fill any remaining slots from best of remainder pool
    remaining = top - len(selected)
    if remaining > 0:
        remainder_pool = longs[long_slots:] + shorts[short_slots:]
        remainder_pool.sort(key=_gate_scan_sort_key)
        selected.extend(remainder_pool[:remaining])

    selected.sort(key=_gate_scan_sort_key)
    final_longs = sum(1 for r in selected if str(r.get("best_side") or "").upper() == "LONG")
    final_shorts = len(selected) - final_longs

    return selected, {
        "longs": len(longs),
        "shorts": len(shorts),
        "long_slots": long_slots,
        "short_slots": short_slots,
        "min_required": min_required_per_side,
        "selected_longs": final_longs,
        "selected_shorts": final_shorts,
        "regime": regime,
        "suppressed": False,
    }
```

### 4. Add Regime Detection Call in `gate_scan()`

**Location:** Inside `gate_scan()` function, after finder initialization (~line 1236)

```python
    # Detect regime if tilt enabled
    regime = "neutral"
    if regime_tilt:
        regime = _detect_btc_regime(
            finder,
            ema_period=regime_ema_period,
            confirm_days=regime_confirm_days,
            buffer_bps=regime_buffer_bps,
        )
        tilt_pct_display = regime_tilt_pct if regime != "neutral" else 50
        buffer_note = f", {regime_buffer_bps:.0f}bps buffer" if regime_buffer_bps > 0 else ""
        print(f"Regime: {regime.upper()} (BTC vs {regime_ema_period} EMA, {regime_confirm_days}d confirm{buffer_note}) → {tilt_pct_display:.0f}/{100-tilt_pct_display:.0f} tilt")
```

### 5. Update `gate_scan()` Function Signature

**Location:** Line 1186

Add new parameters:
```python
def gate_scan(
    ...
    balanced: bool,
    regime_tilt: bool,           # ADD
    regime_ema_period: int,      # ADD
    regime_confirm_days: int,    # ADD
    regime_tilt_pct: float,      # ADD
    regime_buffer_bps: float,    # ADD
    perf_filter: bool,
    ...
```

### 6. Pass Regime to `_select_balanced_rows()`

**Location:** Line 1511 (where `_select_balanced_rows` is called)

**Change from:**
```python
top_rows, balance_meta = _select_balanced_rows(rows, top)
```

**Change to:**
```python
top_rows, balance_meta = _select_balanced_rows(
    rows,
    top,
    regime=regime if regime_tilt else "neutral",
    tilt_pct=regime_tilt_pct,
)
```

### 7. Update Balance Note Output

**Location:** ~Line 1514

**Change from:**
```python
balance_note = (
    "Balanced gate-scan: LONG={longs}, SHORT={shorts}, min_per_side={min_per_side}.".format(
        ...
    )
)
```

**Change to:**
```python
# Handle suppression case (imbalance = reversal protection)
if balance_meta.get("suppressed"):
    if regime_tilt and regime != "neutral":
        balance_note = (
            "Regime-tilted gate-scan ({regime}): LONG={longs}, SHORT={shorts}, "
            "min_required={min_required}. Insufficient balance; output suppressed (reversal protection).".format(**balance_meta)
        )
    else:
        balance_note = (
            "Balanced gate-scan: LONG={longs}, SHORT={shorts}, "
            "min_required={min_required}. Insufficient balance; output suppressed.".format(**balance_meta)
        )
    print(balance_note)
    return  # Exit early - don't output commands

# Balance check passed - show allocation
if regime_tilt and regime != "neutral":
    balance_note = (
        "Regime-tilted gate-scan ({regime}): selected LONG={selected_longs}, SHORT={selected_shorts} "
        "(target {long_slots}/{short_slots}, available {longs}/{shorts}).".format(**balance_meta)
    )
else:
    balance_note = (
        "Balanced gate-scan: LONG={longs}, SHORT={shorts}, selected={selected_longs}/{selected_shorts}.".format(
            **balance_meta
        )
    )
```

### 8. Wire Up Arguments in `main()`

**Location:** Where `gate_scan()` is called in main (~line 2250+)

Add the new arguments to the `gate_scan()` call:
```python
gate_scan(
    ...
    balanced=args.balanced,
    regime_tilt=args.regime_tilt,
    regime_ema_period=args.regime_ema_period,
    regime_confirm_days=args.regime_confirm_days,
    regime_tilt_pct=args.regime_tilt_pct,
    regime_buffer_bps=args.regime_buffer_bps,
    perf_filter=args.perf_filter,
    ...
)
```

---

## Usage

### Enable Regime Tilt in Shell Script

**File:** `scripts/run_gate_scan_paper.sh`

Add `--regime-tilt` to `SCAN_CMD` array (no other changes needed):

```bash
SCAN_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/symbol_snapshot.py"
  --gate-scan
  --profile focused_no_llm_200
  --top 15
  --balanced
  --regime-tilt                    # ADD THIS
  --min-score-long 55
  ...
)
```

### CLI Examples

```bash
# Default 70/30 tilt with 20 EMA, 2-day confirm
python scripts/symbol_snapshot.py --gate-scan --balanced --regime-tilt --top 15

# Custom tilt percentage (60/40)
python scripts/symbol_snapshot.py --gate-scan --balanced --regime-tilt --regime-tilt-pct 60 --top 15

# Faster regime detection (10 EMA, 1-day confirm)
python scripts/symbol_snapshot.py --gate-scan --balanced --regime-tilt --regime-ema-period 10 --regime-confirm-days 1 --top 15

# Add buffer to reduce whipsaw in choppy markets (20bps = 0.2%)
python scripts/symbol_snapshot.py --gate-scan --balanced --regime-tilt --regime-buffer-bps 20 --top 15

# Conservative: 3-day confirm + buffer for choppy conditions
python scripts/symbol_snapshot.py --gate-scan --balanced --regime-tilt --regime-confirm-days 3 --regime-buffer-bps 20 --top 15

# Check regime only (dry run)
python scripts/symbol_snapshot.py --gate-scan --balanced --regime-tilt --top 15 2>&1 | grep "Regime:"
```

---

## Expected Output

**Normal case (balance check passes):**
```
Regime: BEARISH (BTC vs 20 EMA, 2d confirm) → 70/30 tilt
Performance filter: excluded 3/25 products (min_trades=5, drop_worst=0.25)
Score gates: min LONG score=55.0, min SHORT score=60.0.
Regime-tilted gate-scan (bearish): selected LONG=4, SHORT=11 (target 4/11, available 12/18).
...
[baseline commands output]
```

**Suppressed case (imbalance = reversal protection):**
```
Regime: BULLISH (BTC vs 20 EMA, 2d confirm) → 70/30 tilt
Performance filter: excluded 3/25 products (min_trades=5, drop_worst=0.25)
Score gates: min LONG score=55.0, min SHORT score=60.0.
Regime-tilted gate-scan (bullish): LONG=60, SHORT=3, min_required=4. Insufficient balance; output suppressed (reversal protection).
```

In the suppressed case, even though regime is bullish (wanting 70% longs), we only have 3 shorts available but need at least 4 (the 30% allocation). This imbalance suggests the market is overextended bullish and may reverse - so we don't take any trades.

---

## Rollback

To disable regime tilt without code changes, simply remove `--regime-tilt` from the shell script. The system will revert to balanced 50/50 allocation.

---

## Observability & Monitoring

### Suppression Logging

Track suppression frequency to ensure the reversal guard isn't overly strict. Add a counter in `gate_scan()`:

```python
# At module level or in gate_scan scope
_suppression_counts = {"total": 0, "regime_bullish": 0, "regime_bearish": 0, "regime_neutral": 0}

# After _select_balanced_rows returns
if balance_meta.get("suppressed"):
    _suppression_counts["total"] += 1
    _suppression_counts[f"regime_{regime}"] += 1
    _logger.info(f"Gate-scan suppressed (total={_suppression_counts['total']}): {regime} regime, L={balance_meta['longs']}, S={balance_meta['shorts']}")
```

If suppression rate exceeds ~30-40% over a week, consider loosening the minimum requirement or adding a fallback mode.

### Dashboard Integration

Surface regime in `watchdog_dashboard.py` so you can track if tilt helps:

1. **Current regime indicator** - Show detected regime (bullish/bearish/neutral) in dashboard header
2. **Regime history** - Log regime changes to a file for post-hoc analysis
3. **P&L by regime** - Split closed trade stats by the regime active at entry time

---

## Future Enhancements (Optional)

1. **Multi-timeframe confirmation**: Check 4h + daily alignment
2. **ADX filter**: Only apply tilt when ADX > 25 (trending market)
3. **Volatility adjustment**: Reduce tilt during high VIX/volatility periods
4. **Per-symbol regime**: Check each alt vs its own EMA (more complex)
5. **Gradual tilt**: 60/40 for weak regime, 70/30 for strong regime
6. **Regime buffer auto-tune**: Increase buffer automatically in choppy markets (high whipsaw count)

---

## Testing Checklist

- [ ] Regime detection returns correct value for current BTC state
- [ ] BTC symbol fallback works (tries BTC, BTC-USD, etc.)
- [ ] Incomplete daily candle is dropped (no partial-day flips)
- [ ] Buffer option reduces false signals (`--regime-buffer-bps 20`)
- [ ] Neutral regime maintains 50/50 split
- [ ] Bullish regime produces more longs than shorts (when balance passes)
- [ ] Bearish regime produces more shorts than longs (when balance passes)
- [ ] **Imbalance suppression works:** If unfavored side has fewer candidates than its allocation requires, output is suppressed
- [ ] Suppression message clearly indicates "reversal protection"
- [ ] Suppression count is logged for monitoring
- [ ] Output messages clearly indicate regime and allocation
- [ ] Shell script works with new flag
- [ ] No regression in existing balanced behavior when flag not used
- [ ] Logging uses project logger (not print)

### Specific Test Cases

| Regime | Available L/S | Target L/S | Min Required | Result |
|--------|---------------|------------|--------------|--------|
| Bullish | 60/10 | 10/5 | 5 | PASS - allocate 10L, 5S |
| Bullish | 60/3 | 10/5 | 5 | SUPPRESS - only 3S, need 5 |
| Bearish | 10/60 | 5/10 | 5 | PASS - allocate 5L, 10S |
| Bearish | 3/60 | 5/10 | 5 | SUPPRESS - only 3L, need 5 |
| Neutral | 60/6 | 7/7 | 7 | SUPPRESS - only 6S, need 7 |
| Neutral | 10/10 | 7/7 | 7 | PASS - allocate 7L, 7S |

---

## Summary: Two-Tier Protection System

```
┌─────────────────────────────────────────────────────────────┐
│                    GATE SCAN FLOW                           │
├─────────────────────────────────────────────────────────────┤
│  1. Detect regime (BTC vs 20 EMA)                          │
│     → bullish / bearish / neutral                          │
│                                                             │
│  2. Calculate target allocation                             │
│     → bullish: 70% L, 30% S                                │
│     → bearish: 30% L, 70% S                                │
│     → neutral: 50% L, 50% S                                │
│                                                             │
│  3. BALANCE CHECK (reversal protection)                     │
│     → Need at least [unfavored side %] candidates          │
│     → If not enough → SUPPRESS (market overextended)       │
│                                                             │
│  4. If balance OK → Apply tilt allocation                   │
│     → Output baseline commands with tilted sizing          │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** The imbalance check isn't just about having enough candidates - it's a market sentiment indicator. Extreme imbalance = overextension = potential reversal. By requiring minimum candidates on the unfavored side, we naturally avoid entering at tops/bottoms.
