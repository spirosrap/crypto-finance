#!/usr/bin/env python3
"""
Replay historical finder outputs as a systematic backtest.

The script mirrors ``paper_finder_simulator.py`` selection logic (top-N,
optional balanced split) but instead of paper-logging live trades it
reconstructs one day of candidates at a time and simulates each trade
with historical Coinbase perp prices. Days with fewer than the required
number of trades are skipped entirely (per the desk workflow).

Example:
    python paper_finder_backtest.py \\
        --finder-path logs/finder_archive/*.txt \\
        --days 30 \\
        --balanced-top \\
        --min-trades 5 \\
        --output-csv backtest_results/finder_backtest.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import paper_finder_simulator as sim
from add_position_from_finder import ParsedFinder

try:
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover - surfaced immediately for CLI users
    raise SystemExit("ccxt is required for finder backtests. Install with `pip install -r requirements.txt`.") from exc


UTC = sim.UTC

GEN_TIMESTAMP_RE = re.compile(r"Generated on\s*\(UTC\)\s*:\s*([^\n\r]+)", re.I)
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}


@dataclass
class FinderRun:
    path: Path
    generated_at: datetime
    candidates: List[sim.FinderCandidate]


@dataclass
class TradeResult:
    day: date
    source: Path
    product_id: str
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_usd: float
    opened_at: datetime
    closed_at: datetime
    exit_price: float
    exit_reason: str
    profit_loss: float
    profit_loss_pct: float
    finder_score: float
    finder_rank: int
    mae_pct: Optional[float] = None
    mfe_pct: Optional[float] = None


@dataclass
class DayResult:
    date: date
    source: Path
    trades: List[TradeResult] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    @property
    def pnl(self) -> float:
        return sum(trade.profit_loss for trade in self.trades)


@dataclass
class BacktestReport:
    trades: List[TradeResult]
    days: List[DayResult]


def _parse_generated_at(text: str) -> Optional[datetime]:
    match = GEN_TIMESTAMP_RE.search(text)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt


def _discover_files(patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            files.extend(Path(p).resolve() for p in glob.glob(pattern))
            continue
        path = Path(pattern).expanduser()
        if path.is_dir():
            files.extend(p.resolve() for p in path.glob("*.txt"))
        elif path.is_file():
            files.append(path.resolve())
        else:
            logging.warning("Finder path %s not found (skipping).", pattern)
    unique = sorted({p for p in files if p.exists()})
    return unique


def _load_finder_run(path: Path) -> Optional[FinderRun]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logging.warning("Unable to read %s: %s", path, exc)
        return None
    generated_at = _parse_generated_at(text)
    if not generated_at:
        logging.warning("Finder file %s missing 'Generated on' timestamp; skipping.", path)
        return None
    candidates = sim.gather_candidates(text)
    if not candidates:
        logging.warning("No finder candidates detected in %s; skipping.", path)
        return None
    return FinderRun(path=path, generated_at=generated_at, candidates=candidates)


def _select_runs_by_day(runs: Iterable[FinderRun]) -> Dict[date, FinderRun]:
    per_day: Dict[date, FinderRun] = {}
    for run in runs:
        day = run.generated_at.date()
        current = per_day.get(day)
        if current is None or run.generated_at > current.generated_at:
            per_day[day] = run
    return per_day


class OhlcvFetcher:
    """Thin CCXT wrapper that caches a Coinbase Advanced handle."""

    def __init__(self, timeframe: str, limit: int = 450):
        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported timeframe {timeframe!r}. Pick one of {sorted(TIMEFRAME_SECONDS)}.")
        self.timeframe = timeframe
        self.limit = limit
        self.interval_seconds = TIMEFRAME_SECONDS[timeframe]
        self.interval_ms = self.interval_seconds * 1000
        self.exchange: Optional[ccxt.coinbaseadvanced] = None
        self.logger = logging.getLogger("finder_backtest.fetcher")

    def _ensure_exchange(self) -> None:
        if self.exchange is not None:
            return
        params: Dict[str, object] = {"enableRateLimit": True}
        api_key = os.getenv("COINBASE_PERP_API_KEY") or os.getenv("API_KEY_PERPS")
        api_secret = os.getenv("COINBASE_PERP_API_SECRET") or os.getenv("API_SECRET_PERPS")
        if api_key and api_secret:
            params.update({"apiKey": api_key, "secret": api_secret})
        self.exchange = ccxt.coinbaseadvanced(params)
        self.exchange.load_markets()
        self.logger.debug("Coinbase Advanced markets loaded for OHLCV backtest fetcher.")

    def fetch(self, product_id: str, start: datetime, end: datetime) -> List[Dict[str, object]]:
        """Return OHLCV bars covering [start, end]."""
        self._ensure_exchange()
        assert self.exchange  # for mypy
        symbol = sim._ccxt_symbol(product_id)
        start_ms = max(0, int((start - timedelta(seconds=self.interval_seconds)).timestamp() * 1000))
        end_ms = int((end + timedelta(seconds=self.interval_seconds)).timestamp() * 1000)
        bars: List[Dict[str, object]] = []
        since = start_ms
        safety = 0
        while since <= end_ms and safety < 1000:
            safety += 1
            try:
                batch = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, since=since, limit=self.limit)
            except ccxt.NetworkError as exc:
                self.logger.warning("Network issue fetching %s %s: %s (retrying shortly)", product_id, self.timeframe, exc)
                time.sleep(1.0)
                continue
            except ccxt.ExchangeError as exc:
                self.logger.error("Exchange error fetching %s %s: %s", product_id, self.timeframe, exc)
                break
            if not batch:
                break
            for entry in batch:
                ts = entry[0]
                if ts < start_ms:
                    continue
                if ts > end_ms:
                    break
                bars.append(
                    {
                        "timestamp": datetime.fromtimestamp(ts / 1000, tz=UTC),
                        "open": float(entry[1]),
                        "high": float(entry[2]),
                        "low": float(entry[3]),
                        "close": float(entry[4]),
                    }
                )
            last_ts = batch[-1][0]
            if last_ts >= end_ms or len(batch) < self.limit:
                break
            next_since = last_ts + self.interval_ms
            if next_since <= since:
                next_since = since + self.interval_ms
            since = next_since
        return bars


class FinderBacktester:
    def __init__(
        self,
        *,
        top: int,
        balanced_top: bool,
        min_trades: int,
        expiry_hours: float,
        portfolio_usd: float,
        default_pct: float,
        timeframe: str,
        fixed_position_usd: Optional[float] = None,
    ) -> None:
        self.top = top
        self.balanced_top = balanced_top
        self.min_trades = min_trades
        self.expiry_hours = expiry_hours
        self.portfolio_usd = portfolio_usd
        self.default_pct = default_pct
        self.fixed_position_usd = fixed_position_usd
        self.fetcher = OhlcvFetcher(timeframe=timeframe)
        self.logger = logging.getLogger("finder_backtest")

    def run(self, runs_by_day: Dict[date, FinderRun]) -> BacktestReport:
        day_results: List[DayResult] = []
        for day in sorted(runs_by_day):
            run = runs_by_day[day]
            day_results.append(self._run_day(run))
        trades = [trade for day in day_results for trade in day.trades]
        return BacktestReport(trades=trades, days=day_results)

    def _run_day(self, run: FinderRun) -> DayResult:
        candidates = sim._filter_supported_candidates(run.candidates)
        selected = sim._select_candidates(
            candidates=candidates,
            symbols=None,
            picks=None,
            top=self.top,
            balanced_top=self.balanced_top,
        )
        if len(selected) < self.min_trades:
            reason = f"{len(selected)} trades available (<{self.min_trades})"
            self.logger.info("Skipping %s because %s.", run.path, reason)
            return DayResult(date=run.generated_at.date(), source=run.path, trades=[], skipped_reason=reason)

        trades = []
        for cand in selected:
            try:
                trades.append(self._simulate_trade(run, cand))
            except Exception as exc:
                self.logger.error("Failed to simulate %s rank %s: %s", cand.parsed.symbol, cand.rank, exc)
        return DayResult(date=run.generated_at.date(), source=run.path, trades=trades)

    def _simulate_trade(self, run: FinderRun, cand: sim.FinderCandidate) -> TradeResult:
        parsed = cand.parsed
        product_id = cand.product_id or sim._product_id(parsed.symbol)
        if not product_id:
            raise ValueError(f"Unable to derive product id for {parsed.symbol}")
        opened_at = run.generated_at
        expiry_at = opened_at + timedelta(hours=self.expiry_hours)
        candles = self.fetcher.fetch(product_id, opened_at, expiry_at)
        exit_price, exit_reason, closed_at, mae_pct, mfe_pct = self._determine_exit(parsed, candles, opened_at, expiry_at)
        position_usd = sim._desired_position_usd(
            parsed=parsed,
            portfolio_usd=self.portfolio_usd,
            fixed_position_usd=self.fixed_position_usd,
            default_pct=self.default_pct,
        )
        profit_pct = sim._compute_unrealized_pct(parsed.side, parsed.entry, exit_price)
        profit_usd = position_usd * profit_pct / 100.0
        return TradeResult(
            day=opened_at.date(),
            source=run.path,
            product_id=product_id,
            symbol=parsed.symbol,
            side=parsed.side,
            entry_price=parsed.entry,
            stop_loss=parsed.stop,
            take_profit=parsed.take_profit,
            position_usd=position_usd,
            opened_at=opened_at,
            closed_at=closed_at,
            exit_price=exit_price,
            exit_reason=exit_reason,
            profit_loss=round(profit_usd, 2),
            profit_loss_pct=round(profit_pct, 4),
            finder_score=cand.score,
            finder_rank=cand.rank,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
        )

    def _determine_exit(
        self,
        parsed: ParsedFinder,
        candles: List[Dict[str, object]],
        opened_at: datetime,
        expiry_at: datetime,
    ) -> tuple[float, str, datetime, Optional[float], Optional[float]]:
        if not candles:
            return parsed.entry, "no_market_data", opened_at, None, None

        interval = timedelta(seconds=self.fetcher.interval_seconds)
        usable = [bar for bar in candles if (bar["timestamp"] or opened_at) >= opened_at - interval]
        if not usable:
            usable = candles
        mae_pct, mfe_pct = self._compute_excursions(parsed.side, parsed.entry, usable)
        stop = parsed.stop
        tp = parsed.take_profit
        side = parsed.side.upper()

        for bar in usable:
            ts = bar["timestamp"]
            if not isinstance(ts, datetime):
                continue
            if ts >= expiry_at:
                break
            high = float(bar["high"])
            low = float(bar["low"])
            if side == "LONG":
                hit_stop = stop > 0 and low <= stop
                hit_tp = tp > 0 and high >= tp
            else:
                hit_stop = stop > 0 and high >= stop
                hit_tp = tp > 0 and low <= tp
            if hit_stop:
                return stop, "stop_loss", ts, mae_pct, mfe_pct
            if hit_tp:
                return tp, "take_profit", ts, mae_pct, mfe_pct

        last_bar = usable[-1]
        ts = last_bar["timestamp"]
        last_close = float(last_bar["close"])
        closed_at = ts if isinstance(ts, datetime) else expiry_at
        if closed_at > expiry_at:
            closed_at = expiry_at
        return last_close, "expired", closed_at, mae_pct, mfe_pct

    @staticmethod
    def _compute_excursions(side: str, entry: float, candles: List[Dict[str, object]]) -> tuple[Optional[float], Optional[float]]:
        if entry <= 0 or not candles:
            return None, None
        side = side.upper()
        mae: Optional[float] = None
        mfe: Optional[float] = None
        for bar in candles:
            high = float(bar["high"])
            low = float(bar["low"])
            if side == "LONG":
                adverse = (low - entry) / entry * 100.0
                favorable = (high - entry) / entry * 100.0
            else:
                adverse = (entry - high) / entry * 100.0
                favorable = (entry - low) / entry * 100.0
            mae = adverse if mae is None else min(mae, adverse)
            mfe = favorable if mfe is None else max(mfe, favorable)
        return mae, mfe


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _summarise(report: BacktestReport, initial_capital: float) -> None:
    trades = report.trades
    days = report.days
    total_trades = len(trades)
    executed_days = sum(1 for day in days if day.trades)
    skipped_days = len(days) - executed_days
    pnl_total = sum(t.profit_loss for t in trades)
    wins = len([t for t in trades if t.profit_loss > 0])
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    avg_trade = (pnl_total / total_trades) if total_trades else 0.0

    equity = initial_capital
    peak = equity
    max_drawdown = 0.0
    for day in days:
        equity += day.pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    print("Finder Backtest Summary")
    print("=======================")
    print(f"Total trades       : {total_trades}")
    print(f"Active days        : {executed_days}")
    print(f"Skipped days       : {skipped_days}")
    print(f"Total P/L (USD)    : {pnl_total:+.2f}")
    print(f"Win rate           : {win_rate:.2f}%")
    print(f"Avg P/L per trade  : {avg_trade:+.2f}")
    print(f"Ending equity      : {initial_capital + pnl_total:.2f}")
    print(f"Max drawdown (abs) : {max_drawdown:.2f}")
    print()

    if skipped_days:
        print("Skipped days:")
        for day in days:
            if not day.trades and day.skipped_reason:
                print(f"  {day.date}: {day.skipped_reason}")
        print()

    print("Daily breakdown:")
    for day in days:
        if day.trades:
            print(f"  {day.date}: {len(day.trades)} trades, P/L {day.pnl:+.2f}")
        else:
            print(f"  {day.date}: 0 trades (skipped)")


def _write_csv(trades: List[TradeResult], path: Path) -> None:
    if not trades:
        logging.info("No trades to write to %s.", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "day",
        "source",
        "product_id",
        "symbol",
        "side",
        "entry_price",
        "stop_loss",
        "take_profit",
        "position_usd",
        "opened_at",
        "closed_at",
        "exit_price",
        "exit_reason",
        "profit_loss",
        "profit_loss_pct",
        "finder_score",
        "finder_rank",
        "mae_pct",
        "mfe_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow(
                {
                    "day": trade.day.isoformat(),
                    "source": str(trade.source),
                    "product_id": trade.product_id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "entry_price": trade.entry_price,
                    "stop_loss": trade.stop_loss,
                    "take_profit": trade.take_profit,
                    "position_usd": trade.position_usd,
                    "opened_at": trade.opened_at.isoformat(),
                    "closed_at": trade.closed_at.isoformat(),
                    "exit_price": trade.exit_price,
                    "exit_reason": trade.exit_reason,
                    "profit_loss": trade.profit_loss,
                    "profit_loss_pct": trade.profit_loss_pct,
                    "finder_score": trade.finder_score,
                    "finder_rank": trade.finder_rank,
                    "mae_pct": trade.mae_pct,
                    "mfe_pct": trade.mfe_pct,
                }
            )
    logging.info("Wrote %s trades to %s", len(trades), path)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest finder selections by replaying historical outputs.")
    parser.add_argument(
        "--finder-path",
        action="append",
        default=[],
        help="Finder text file, directory, or glob pattern. Repeat for multiple sources.",
    )
    parser.add_argument("--days", type=int, default=30, help="Number of trailing days to include when start-date omitted.")
    parser.add_argument("--start-date", type=str, help="Inclusive UTC start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="Inclusive UTC end date (YYYY-MM-DD). Defaults to latest finder file.")
    parser.add_argument("--top", type=int, default=5, help="Pick the top-N candidates per day.")
    parser.add_argument(
        "--balanced-top",
        dest="balanced_top",
        action="store_true",
        default=True,
        help="Pick 2 longs, 2 shorts, then best remaining (default ON).",
    )
    parser.add_argument(
        "--no-balanced-top",
        dest="balanced_top",
        action="store_false",
        help="Treat --top purely by score (disable balanced split).",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=5,
        help="Skip the day if fewer than this number of trades would be opened.",
    )
    parser.add_argument("--expiry-hours", type=float, default=24.0, help="Trade expiry horizon in hours.")
    parser.add_argument("--portfolio-usd", type=float, default=25000.0, help="Portfolio size used for sizing math.")
    parser.add_argument("--initial-capital", type=float, default=25000.0, help="Initial equity for reporting.")
    parser.add_argument("--default-position-pct", type=float, default=3.0, help="Fallback position %% when finder omits it.")
    parser.add_argument("--fixed-position-usd", type=float, help="Override absolute USD allocation per trade.")
    parser.add_argument(
        "--timeframe",
        choices=sorted(TIMEFRAME_SECONDS),
        default="1m",
        help="CCXT timeframe for price simulation (shorter -> finer fill ordering).",
    )
    parser.add_argument("--output-csv", type=Path, help="Optional CSV destination for per-trade results.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, etc.).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    finder_paths = args.finder_path or ["finder_short.txt"]
    files = _discover_files(finder_paths)
    if not files:
        raise SystemExit("No finder files discovered. Provide --finder-path pointing at your archive.")

    runs = [run for path in files if (run := _load_finder_run(path))]
    if not runs:
        raise SystemExit("Finder files contained no parsable candidates.")

    latest_date = max(run.generated_at.date() for run in runs)
    if args.end_date:
        end_date = _parse_date(args.end_date)
    else:
        end_date = latest_date
    if args.start_date:
        start_date = _parse_date(args.start_date)
    else:
        start_date = end_date - timedelta(days=max(args.days - 1, 0))
    if start_date > end_date:
        raise SystemExit("start-date must be before or equal to end-date.")

    filtered = [run for run in runs if start_date <= run.generated_at.date() <= end_date]
    if not filtered:
        raise SystemExit("No finder runs fall within the requested date range.")

    runs_by_day = _select_runs_by_day(filtered)
    backtester = FinderBacktester(
        top=args.top,
        balanced_top=args.balanced_top,
        min_trades=args.min_trades,
        expiry_hours=args.expiry_hours,
        portfolio_usd=args.portfolio_usd,
        default_pct=args.default_position_pct,
        timeframe=args.timeframe,
        fixed_position_usd=args.fixed_position_usd,
    )
    report = backtester.run(runs_by_day)
    _summarise(report, initial_capital=args.initial_capital)

    if args.output_csv:
        _write_csv(report.trades, args.output_csv)


if __name__ == "__main__":
    main()
