from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from freqtrade.strategy.interface import IStrategy
    from freqtrade.persistence import Trade
except ImportError as exc:  # pragma: no cover - strategy still useful for lint/tests
    raise RuntimeError(
        "finder_bridge_strategy requires freqtrade to be installed in "
        "the environment where it is executed."
    ) from exc

DEFAULT_SIGNAL_PATH = Path("signals") / "freqtrade_signals.json"


@dataclass
class FinderSignal:
    pair: str
    side: str
    entry: float
    take_profit: float
    stop_loss: float
    expires_at: Optional[datetime]
    leverage: float
    confidence: Optional[float] = None

    @property
    def is_long(self) -> bool:
        return self.side.upper() == "LONG" or self.side.upper() == "BUY"

    @property
    def is_short(self) -> bool:
        return self.side.upper() == "SHORT" or self.side.upper() == "SELL"


class FinderBridgeStrategy(IStrategy):
    """
    Bridge strategy that consumes signals exported by finder scripts and pipes
    them into Freqtrade's execution engine. It assumes `signals/freqtrade_signals.json`
    exists (or the path overridden via `config['finder_signal_path']`) with a payload:

    {
        "generated_at": "... iso8601 ...",
        "timeframe": "ONE_HOUR",
        "expiry_hours": 24,
        "signals": [
            {
                "pair": "1000SHIB/USDC",
                "side": "SHORT",
                "entry": 0.0100,
                "take_profit": 0.0090,
                "stop_loss": 0.0110,
                "leverage": 20,
                "confidence": 0.73,
                "expires_at": "..."
            },
        ]
    }
    """

    timeframe = "1h"
    can_short = True
    startup_candle_count = 1
    use_custom_stoploss = True
    position_adjustment_enable = False
    minimal_roi = {"0": 1000}  # disable ROI-based exits – finder provides TP
    stoploss = -0.99  # fallback only; real SL comes from signal payload

    # Provide sane defaults; actual sizing handled in config / position sizing rules
    custom_trailing_stop = False
    trailing_stop = False

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.signal_path: Path = Path(
            config.get("finder_signal_path", DEFAULT_SIGNAL_PATH)
        )
        self._loaded_at: Optional[float] = None
        self._signals: Dict[str, FinderSignal] = {}
        self._active_orders: Dict[int, FinderSignal] = {}

    # --- Signal loading helpers -------------------------------------------------
    def _load_signals(self) -> None:
        if not self.signal_path.exists():
            self.dp.logger.warning(
                "Finder signal file %s not found; skipping entries.",
                self.signal_path,
            )
            self._signals = {}
            return

        mtime = self.signal_path.stat().st_mtime
        if self._loaded_at and mtime <= self._loaded_at:
            return

        try:
            payload = json.loads(self.signal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.dp.logger.error("Invalid finder signal JSON: %s", exc)
            self._signals = {}
            return

        signals: Dict[str, FinderSignal] = {}
        expiry_hours = payload.get("expiry_hours")
        horizon = timedelta(hours=float(expiry_hours)) if expiry_hours else None
        generated_at = payload.get("generated_at")
        generated_ts = None
        if generated_at:
            try:
                generated_ts = (
                    datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                    .astimezone(timezone.utc)
                )
            except ValueError:
                generated_ts = None

        for entry in payload.get("signals", []):
            try:
                pair = entry["pair"]
                side = entry["side"]
                signal = FinderSignal(
                    pair=pair,
                    side=side,
                    entry=float(entry["entry"]),
                    take_profit=float(entry["take_profit"]),
                    stop_loss=float(entry["stop_loss"]),
                    leverage=float(entry.get("leverage", 1)),
                    confidence=entry.get("confidence"),
                    expires_at=self._resolve_expiry(entry, horizon, generated_ts),
                )
            except (KeyError, TypeError, ValueError) as exc:
                self.dp.logger.warning("Skipping malformed signal %s: %s", entry, exc)
                continue
            signals[signal.pair.upper()] = signal

        self._signals = signals
        self._loaded_at = mtime
        self.dp.logger.info(
            "Loaded %s finder signals (file=%s)",
            len(signals),
            self.signal_path,
        )

    @staticmethod
    def _resolve_expiry(
        entry: dict, horizon: Optional[timedelta], generated: Optional[datetime]
    ) -> Optional[datetime]:
        if "expires_at" in entry:
            try:
                return datetime.fromisoformat(entry["expires_at"].replace("Z", "+00:00"))
            except ValueError:
                return None
        if horizon and generated:
            return generated + horizon
        return None

    def _get_signal(self, pair: str) -> Optional[FinderSignal]:
        self._load_signals()
        sig = self._signals.get(pair.upper())
        if not sig:
            return None
        if sig.expires_at and datetime.now(timezone.utc) > sig.expires_at:
            return None
        return sig

    # --- Freqtrade hooks --------------------------------------------------------
    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict
    ) -> pd.DataFrame:
        # Finder already provides price levels; no indicators required
        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict
    ) -> pd.DataFrame:
        dataframe.loc[:, ["enter_long", "enter_short"]] = 0
        pair = metadata["pair"]
        signal = self._get_signal(pair)
        if signal is None:
            return dataframe

        last_index = dataframe.index[-1]
        if signal.is_long:
            dataframe.at[last_index, "enter_long"] = 1
        elif signal.is_short:
            dataframe.at[last_index, "enter_short"] = 1

        dataframe.at[last_index, "enter_tag"] = "finder"
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict
    ) -> pd.DataFrame:
        dataframe.loc[:, ["exit_long", "exit_short"]] = 0
        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        signal = self._get_signal(pair)
        if signal is None:
            self.dp.logger.info("Finder signal missing for %s; blocking entry.", pair)
            return False
        self.dp.logger.info(
            "Confirming %s entry from finder signal: entry=%.6f tp=%.6f sl=%.6f",
            pair,
            signal.entry,
            signal.take_profit,
            signal.stop_loss,
        )
        return True

    def custom_entry_price(
        self, pair: str, current_time: datetime, proposed_rate: float
    ) -> Optional[float]:
        signal = self._get_signal(pair)
        if signal:
            return signal.entry
        return proposed_rate

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        signal = self._get_signal(pair)
        if signal is None:
            return self.stoploss

        entry = trade.open_rate
        if signal.is_long:
            stop_loss_pct = (signal.stop_loss - entry) / entry
            return max(stop_loss_pct, self.stoploss)
        if signal.is_short:
            stop_loss_pct = (entry - signal.stop_loss) / entry
            return max(-stop_loss_pct, self.stoploss)
        return self.stoploss

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        signal = self._get_signal(pair)
        if signal is None:
            return None

        if signal.is_long and current_rate >= signal.take_profit:
            return "finder-tp"
        if signal.is_short and current_rate <= signal.take_profit:
            return "finder-tp"
        if signal.expires_at and current_time > signal.expires_at:
            return "finder-expiry"
        return None
