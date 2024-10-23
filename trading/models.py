from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from datetime import datetime, timezone

@dataclass
class TradeRecord:
    date: int
    action: str
    price: float
    amount: float
    fee: float

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(portfolio_values: List[float], buy_and_hold_values: List[float]) -> Dict[str, float]:
        """Calculate various performance metrics."""
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        buy_and_hold_returns = np.diff(buy_and_hold_values) / buy_and_hold_values[:-1]
        
        metrics = {}
        if daily_returns.std() != 0:
            metrics["sharpe_ratio"] = np.sqrt(365) * daily_returns.mean() / daily_returns.std()
            metrics["sharpe_ratio_buy_and_hold"] = np.sqrt(365) * buy_and_hold_returns.mean() / buy_and_hold_returns.std()
            
            negative_returns = daily_returns[daily_returns < 0]
            negative_returns_buy_and_hold = buy_and_hold_returns[buy_and_hold_returns < 0]
            
            if len(negative_returns) > 0:
                downside_deviation = np.sqrt(np.mean(negative_returns**2))
                metrics["sortino_ratio"] = np.sqrt(365) * daily_returns.mean() / downside_deviation
                
                downside_deviation_buy_and_hold = np.sqrt(np.mean(negative_returns_buy_and_hold**2))
                metrics["sortino_ratio_buy_and_hold"] = np.sqrt(365) * buy_and_hold_returns.mean() / downside_deviation_buy_and_hold
            else:
                metrics["sortino_ratio"] = float('inf')
                metrics["sortino_ratio_buy_and_hold"] = float('inf')
        else:
            metrics["sharpe_ratio"] = 0
            metrics["sortino_ratio"] = 0
            metrics["sharpe_ratio_buy_and_hold"] = 0
            metrics["sortino_ratio_buy_and_hold"] = 0

        # Calculate maximum drawdown
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (cumulative_max - portfolio_values) / cumulative_max
        metrics["max_drawdown"] = drawdown.max() * 100
        
        return metrics
