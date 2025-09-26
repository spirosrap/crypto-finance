import unittest
from datetime import datetime, timedelta, UTC
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from trading.intraday_zero_fee_trader import (
    CoinbaseExecutionEngine,
    IntradayTraderConfig,
    PaperExecutionEngine,
    SignalDecision,
    SignalEngine,
    ZeroFeePerpTrader,
    build_feature_frame,
    compute_atr,
    compute_rsi,
)


class DummyHistorical:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_historical_data(self, product_id, start_date, end_date, granularity, force_refresh):
        subset = self.frame.loc[start_date:end_date]
        records = []
        for ts, row in subset.iterrows():
            records.append({
                "start": int(ts.timestamp()),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            })
        return records


class DummyService:
    def __init__(self, frame: pd.DataFrame):
        self.historical_data = DummyHistorical(frame)


def make_df(rows: int = 200, final_drop: float = 0.003, volume_spike: float = 1.7) -> pd.DataFrame:
    now = datetime.now(UTC)
    index = [now - timedelta(minutes=i) for i in reversed(range(rows))]
    base = np.linspace(30000, 30350, rows)
    base[-5:] = base[-5:] * 1.002
    base[-1] = base[-1] * (1 - final_drop)
    prices = base
    highs = prices + 20
    lows = prices - 20
    volumes = np.full(rows, 900.0)
    volumes[-1] *= volume_spike
    df = pd.DataFrame({
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=pd.DatetimeIndex(index))
    return df


class TestFeatureEngineering(unittest.TestCase):
    def test_compute_rsi_bounds(self):
        series = pd.Series(np.linspace(1, 100, 120))
        rsi = compute_rsi(series, 14)
        self.assertTrue(((rsi >= 0) & (rsi <= 100)).all())

    def test_compute_atr_positive(self):
        df = make_df(120)
        atr = compute_atr(df, 14)
        self.assertTrue((atr.dropna() >= 0).all())

    def test_build_feature_frame_columns(self):
        df = make_df(150)
        cfg = IntradayTraderConfig()
        cfg.lookback_bars = 120
        cfg.ema_trend = 30
        cfg.volume_lookback = 20
        features = build_feature_frame(df, cfg)
        self.assertIn("ema_fast", features.columns)
        self.assertIn("rsi", features.columns)
        self.assertIn("volume_ratio", features.columns)
        self.assertGreater(len(features), 0)


class TestSignalEngine(unittest.TestCase):
    def setUp(self):
        self.cfg = IntradayTraderConfig()
        self.cfg.lookback_bars = 160
        self.cfg.ema_fast = 5
        self.cfg.ema_slow = 11
        self.cfg.ema_trend = 21
        self.cfg.min_volume_ratio = 1.0
        self.cfg.rsi_buy_threshold = 60.0

    def test_long_signal_triggers_in_uptrend(self):
        df = make_df(180)
        # Create a pullback on the last few bars
        df.iloc[-1, df.columns.get_loc("low")] = df.iloc[-1, df.columns.get_loc("close")] - 20
        df.iloc[-1, df.columns.get_loc("high")] = df.iloc[-1, df.columns.get_loc("close")] + 20
        features = build_feature_frame(df, self.cfg)
        engine = SignalEngine(self.cfg)
        decision = engine.evaluate("BTC-PERP-INTX", features)
        self.assertIsNotNone(decision)
        if decision:
            self.assertEqual(decision.side, "buy")
            self.assertGreater(decision.size, 0)
            self.assertLess(decision.stop_loss, decision.entry_price)
            self.assertGreater(decision.take_profit, decision.entry_price)

    def test_volume_filter_blocks_signal(self):
        df = make_df(180)
        df.iloc[-1, df.columns.get_loc("close")] *= 0.998
        df.iloc[-1, df.columns.get_loc("volume")] *= 0.2
        features = build_feature_frame(df, self.cfg)
        engine = SignalEngine(self.cfg)
        decision = engine.evaluate("BTC-PERP-INTX", features)
        self.assertIsNone(decision)


class TestTraderIntegration(unittest.TestCase):
    def test_run_once_places_paper_trade(self):
        df = make_df(200)
        df.iloc[-1, df.columns.get_loc("low")] = df.iloc[-1, df.columns.get_loc("close")] - 20
        df.iloc[-1, df.columns.get_loc("high")] = df.iloc[-1, df.columns.get_loc("close")] + 20
        cfg = IntradayTraderConfig()
        cfg.product_ids = ["BTC-PERP-INTX"]
        cfg.cooldown_seconds = 0
        cfg.dry_run = True
        cfg.min_volume_ratio = 1.0
        cfg.rsi_buy_threshold = 60.0
        cfg.lookback_bars = 150
        cfg.ema_fast = 5
        cfg.ema_slow = 11
        cfg.ema_trend = 21
        service = DummyService(df)
        trader = ZeroFeePerpTrader(config=cfg, service=service, execution=PaperExecutionEngine(cfg.log_dir))
        trader.run_once()
        self.assertGreaterEqual(len(trader.positions), 1)

    def test_live_execution_logs_to_csv(self):
        class StubService:
            def __init__(self):
                self.calls = []

            def place_order(self, **kwargs):
                self.calls.append(kwargs)
                return {"order_id": "test-order"}

        decision = SignalDecision(
            product_id="BTC-PERP-INTX",
            side="buy",
            size=0.01,
            entry_price=40000.0,
            stop_loss=39800.0,
            take_profit=40200.0,
            rationale="unit-test",
            leverage=25.0,
        )

        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            engine = CoinbaseExecutionEngine(StubService(), log_dir)
            position = engine.enter(decision)
            self.assertIsNotNone(position)

            log_file = log_dir / "zero_fee_live_trades.csv"
            self.assertTrue(log_file.exists())
            lines = log_file.read_text().strip().splitlines()
            self.assertGreaterEqual(len(lines), 2)
            header = lines[0]
            row = lines[-1]
            self.assertEqual(
                header,
                "timestamp,product_id,side,price,size,stop_loss,take_profit,rationale,leverage,order_id",
            )
            fields = row.split(",")
            self.assertEqual(fields[1], "BTC-PERP-INTX")
            self.assertEqual(fields[2], "buy")
            self.assertEqual(fields[-1], "test-order")


if __name__ == "__main__":
    unittest.main()
