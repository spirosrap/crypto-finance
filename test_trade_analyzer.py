import unittest
from datetime import datetime
from trade_outcome_analyzer import parse_single_trade, analyze_trade_outcome

class TestTradeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.sample_trade = """====== 🤖 AI Trading Recommendation (2025-03-04 18:57:15) ======
{"BUY AT":83702.62,"SELL BACK AT":87887.75,"STOP LOSS":81215.547,"PROBABILITY":79.2,"CONFIDENCE":"High","R/R_RATIO":1.68,"VOLUME_STRENGTH":"Weak","VOLATILITY":"Low","MARKET_REGIME":"Bullish","REGIME_CONFIDENCE":"High","IS_VALID":true}
Switching to market order: Limit price ($83702.62) is above or close to current price ($83711.86)

Executing trade with parameters:
Product: BTC-PERP-INTX
Side: BUY
Market Regime: Bullish
Regime Confidence: High
Initial Margin: $80.0
Leverage: 10x
Reducing position size by 30% - SL > 2% (2.97%)
Position Size: $560.00
Entry Price: $83702.62
Current Price: $83711.86
Price Deviation: 0.01%
Take Profit: $87887.75 (5.00% / $28.00)
Stop Loss: $81215.55 (2.97% / $16.64)
Probability: 79.2%
Signal Confidence: High
R/R Ratio: 1.680
Volume Strength: Weak
Order Type: Market
Trade executed successfully!

=== Order Summary ===
Product: BTC-PERP-INTX
Side: BUY
Position Size: $560.0 (≈0.0067 BTC)
Leverage: 10.0x
Required Margin: $80.0
Current Price: $83679.9
Take Profit Price: $87887.0
Stop Loss Price: $81215.0
Order Type: Market"""

    def test_parse_trade_with_reduced_leverage(self):
        trade_details = parse_single_trade(self.sample_trade)
        
        # Test basic trade details
        self.assertEqual(trade_details['side'], 'LONG')
        self.assertEqual(trade_details['entry_price'], 83702.62)
        self.assertEqual(trade_details['take_profit'], 87887.75)
        self.assertEqual(trade_details['stop_loss'], 81215.547)
        
        # Test leverage reduction
        self.assertEqual(trade_details['position_size'], 560.0)
        self.assertEqual(trade_details['initial_margin'], 80.0)
        self.assertEqual(trade_details['effective_leverage'], 7.0)  # 560/80 = 7

    def test_profit_calculation_with_reduced_leverage(self):
        trade_details = parse_single_trade(self.sample_trade)
        
        # Mock historical data class
        class MockHistoricalData:
            def get_historical_data(self, product_id, start_date, end_date, granularity):
                return [{'low': 81000.0, 'high': 88000.0, 'start': int(datetime.now().timestamp())}]
        
        # Test take profit scenario
        outcome = analyze_trade_outcome(trade_details, MockHistoricalData())
        
        # Calculate expected profit
        # Price change: (87887.75 - 83702.62) / 83702.62 = 0.05 (5%)
        # With 7x leverage: 5% * 7 = 35%
        self.assertEqual(outcome['outcome'], 'SUCCESS')
        self.assertAlmostEqual(outcome['outcome_pct'], 35.0, places=1)

    def test_loss_calculation_with_reduced_leverage(self):
        trade_details = parse_single_trade(self.sample_trade)
        
        # Mock historical data class with price hitting stop loss
        class MockHistoricalData:
            def get_historical_data(self, product_id, start_date, end_date, granularity):
                return [{'low': 81000.0, 'high': 81000.0, 'start': int(datetime.now().timestamp())}]
        
        # Test stop loss scenario
        outcome = analyze_trade_outcome(trade_details, MockHistoricalData())
        
        # Calculate expected loss
        # Price change: (83702.62 - 81215.547) / 83702.62 = 0.0297 (2.97%)
        # With 7x leverage: 2.97% * 7 = 20.79%
        self.assertEqual(outcome['outcome'], 'STOP LOSS')
        self.assertAlmostEqual(outcome['outcome_pct'], -20.79, places=1)

if __name__ == '__main__':
    unittest.main() 