Example of regime proxies:
	•	Jan 2023 → sideways, post-FTX chop
	•	Mar 2023 → news-driven (SVB crash)
	•	Jun 2023 → trending recovery
	•	Nov 2023 → altcoin spikes, short-lived breakouts
	•	Jan 2024 → macro-driven slow climb
	•	Mar 2024 → BTC ETFs, bullish breakout
	•	Mar 2025 → current
	•   Jan 2025
	•   Feb 2025


----


python backtest_trading_bot.py --start_date 2023-01-01 --end_date 2023-02-01

Trade history exported to: backtest_trades_BTC-USDC_20250403_203826.csv

=== ATR Regime Distribution ===
Sideways   :   4684 bars (52.75%)
Moderate   :   3849 bars (43.35%)
Trending   :    346 bars (3.90%)
2025-04-03 20:38:26,971 - INFO - 
Backtest Results:
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10588.99 |
+-----------------+-----------+
| Total Profit    | $588.99   |
+-----------------+-----------+
| Total Trades    | 22        |
+-----------------+-----------+
| Winning Trades  | 12        |
+-----------------+-----------+
| Losing Trades   | 10        |
+-----------------+-----------+
| Win Rate        | 54.55%    |
+-----------------+-----------+
| Profit Factor   | 2.60      |
+-----------------+-----------+
2025-04-03 20:38:26,971 - INFO - 
TP Mode Statistics:
+----------------------+---------+
| Fixed TP Trades      | 21      |
+----------------------+---------+
| Fixed TP Win Rate    | 52.38%  |
+----------------------+---------+
| Adaptive TP Trades   | 1       |
+----------------------+---------+
| Adaptive TP Win Rate | 100.00% |
+----------------------+---------+
2025-04-03 20:38:26,971 - INFO - 
Risk Metrics:
+---------------------------+------------+
| Maximum Drawdown          | 1.05%      |
+---------------------------+------------+
| Maximum Drawdown Duration | 9.83 hours |
+---------------------------+------------+

----

python backtest_trading_bot.py -start_date 2023-03-01 --end_date 2023-04-01

Trade history exported to: backtest_trades_BTC-USDC_20250403_203826.csv

=== ATR Regime Distribution ===
Sideways   :   4684 bars (52.75%)
Moderate   :   3849 bars (43.35%)
Trending   :    346 bars (3.90%)
2025-04-03 20:38:26,971 - INFO - 
Backtest Results:
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10588.99 |
+-----------------+-----------+
| Total Profit    | $588.99   |
+-----------------+-----------+
| Total Trades    | 22        |
+-----------------+-----------+
| Winning Trades  | 12        |
+-----------------+-----------+
| Losing Trades   | 10        |
+-----------------+-----------+
| Win Rate        | 54.55%    |
+-----------------+-----------+
| Profit Factor   | 2.60      |
+-----------------+-----------+
2025-04-03 20:38:26,971 - INFO - 
TP Mode Statistics:
+----------------------+---------+
| Fixed TP Trades      | 21      |
+----------------------+---------+
| Fixed TP Win Rate    | 52.38%  |
+----------------------+---------+
| Adaptive TP Trades   | 1       |
+----------------------+---------+
| Adaptive TP Win Rate | 100.00% |
+----------------------+---------+
2025-04-03 20:38:26,971 - INFO - 
Risk Metrics:
+---------------------------+------------+
| Maximum Drawdown          | 1.05%      |
+---------------------------+------------+
| Maximum Drawdown Duration | 9.83 hours |
+---------------------------+------------+


----


python backtest_trading_bot.py --start_date 2023-06-01 --end_date 2023-07-01

=== ATR Regime Distribution ===
Moderate   :   3600 bars (41.90%)
Sideways   :   4748 bars (55.27%)
Trending   :    243 bars (2.83%)
2025-04-03 20:43:39,858 - INFO - 
Backtest Results:
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10815.53 |
+-----------------+-----------+
| Total Profit    | $815.53   |
+-----------------+-----------+
| Total Trades    | 34        |
+-----------------+-----------+
| Winning Trades  | 18        |
+-----------------+-----------+
| Losing Trades   | 16        |
+-----------------+-----------+
| Win Rate        | 52.94%    |
+-----------------+-----------+
| Profit Factor   | 2.43      |
+-----------------+-----------+
2025-04-03 20:43:39,858 - INFO - 
TP Mode Statistics:
+----------------------+--------+
| Fixed TP Trades      | 34     |
+----------------------+--------+
| Fixed TP Win Rate    | 52.94% |
+----------------------+--------+
| Adaptive TP Trades   | 0      |
+----------------------+--------+
| Adaptive TP Win Rate | 0.00%  |
+----------------------+--------+
2025-04-03 20:43:39,858 - INFO - 
Risk Metrics:
+---------------------------+------------+
| Maximum Drawdown          | 1.39%      |
+---------------------------+------------+
| Maximum Drawdown Duration | 5.67 hours |
+---------------------------+------------+


----



python backtest_trading_bot.py --start_date 2023-11-01 --end_date 2023-12-01

=== ATR Regime Distribution ===
Sideways   :   4600 bars (53.54%)
Moderate   :   3808 bars (44.33%)
Trending   :    183 bars (2.13%)
2025-04-03 20:45:32,619 - INFO - 
Backtest Results:
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10237.86 |
+-----------------+-----------+
| Total Profit    | $237.86   |
+-----------------+-----------+
| Total Trades    | 34        |
+-----------------+-----------+
| Winning Trades  | 13        |
+-----------------+-----------+
| Losing Trades   | 21        |
+-----------------+-----------+
| Win Rate        | 38.24%    |
+-----------------+-----------+
| Profit Factor   | 1.32      |
+-----------------+-----------+
2025-04-03 20:45:32,619 - INFO - 
TP Mode Statistics:
+----------------------+--------+
| Fixed TP Trades      | 33     |
+----------------------+--------+
| Fixed TP Win Rate    | 39.39% |
+----------------------+--------+
| Adaptive TP Trades   | 1      |
+----------------------+--------+
| Adaptive TP Win Rate | 0.00%  |
+----------------------+--------+
2025-04-03 20:45:32,619 - INFO - 
Risk Metrics:
+---------------------------+--------------+
| Maximum Drawdown          | 2.38%        |
+---------------------------+--------------+
| Maximum Drawdown Duration | 117.17 hours |
+---------------------------+--------------+


----



python backtest_trading_bot.py --start_date 2024-01-01 --end_date 2024-02-01

=== ATR Regime Distribution ===
Sideways   :   3005 bars (33.84%)
Moderate   :   5307 bars (59.77%)
Trending   :    567 bars (6.39%)
2025-04-03 20:47:38,225 - INFO - 
Backtest Results:
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10162.87 |
+-----------------+-----------+
| Total Profit    | $162.87   |
+-----------------+-----------+
| Total Trades    | 49        |
+-----------------+-----------+
| Winning Trades  | 17        |
+-----------------+-----------+
| Losing Trades   | 32        |
+-----------------+-----------+
| Win Rate        | 34.69%    |
+-----------------+-----------+
| Profit Factor   | 1.14      |
+-----------------+-----------+
2025-04-03 20:47:38,225 - INFO - 
TP Mode Statistics:
+----------------------+--------+
| Fixed TP Trades      | 46     |
+----------------------+--------+
| Fixed TP Win Rate    | 34.78% |
+----------------------+--------+
| Adaptive TP Trades   | 3      |
+----------------------+--------+
| Adaptive TP Win Rate | 33.33% |
+----------------------+--------+
2025-04-03 20:47:38,226 - INFO - 
Risk Metrics:
+---------------------------+-------------+
| Maximum Drawdown          | 2.08%       |
+---------------------------+-------------+
| Maximum Drawdown Duration | 28.92 hours |
+---------------------------+-------------+


----



python backtest_trading_bot.py --start_date 2024-03-01 --end_date 2024-04-01

=== ATR Regime Distribution ===
Moderate   :   5674 bars (63.90%)
Trending   :   1800 bars (20.27%)
Sideways   :   1405 bars (15.82%)
2025-04-03 20:49:39,709 - INFO - 
Backtest Results:
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10475.50 |
+-----------------+-----------+
| Total Profit    | $475.50   |
+-----------------+-----------+
| Total Trades    | 65        |
+-----------------+-----------+
| Winning Trades  | 24        |
+-----------------+-----------+
| Losing Trades   | 41        |
+-----------------+-----------+
| Win Rate        | 36.92%    |
+-----------------+-----------+
| Profit Factor   | 1.32      |
+-----------------+-----------+
2025-04-03 20:49:39,709 - INFO - 
TP Mode Statistics:
+----------------------+--------+
| Fixed TP Trades      | 58     |
+----------------------+--------+
| Fixed TP Win Rate    | 36.21% |
+----------------------+--------+
| Adaptive TP Trades   | 7      |
+----------------------+--------+
| Adaptive TP Win Rate | 42.86% |
+----------------------+--------+
2025-04-03 20:49:39,709 - INFO - 
Risk Metrics:
+---------------------------+--------------+
| Maximum Drawdown          | 4.20%        |
+---------------------------+--------------+
| Maximum Drawdown Duration | 237.58 hours |
+---------------------------+--------------+


----




python backtest_trading_bot.py --start_date 2025-03-01 --end_date 2025-04-01

=== ATR Regime Distribution ===
Moderate   :   4951 bars (55.76%)
Sideways   :   2808 bars (31.63%)
Trending   :   1120 bars (12.61%)
2025-04-03 20:51:40,920 - INFO - 
Backtest Results:
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10830.48 |
+-----------------+-----------+
| Total Profit    | $830.48   |
+-----------------+-----------+
| Total Trades    | 63        |
+-----------------+-----------+
| Winning Trades  | 27        |
+-----------------+-----------+
| Losing Trades   | 36        |
+-----------------+-----------+
| Win Rate        | 42.86%    |
+-----------------+-----------+
| Profit Factor   | 1.63      |
+-----------------+-----------+
2025-04-03 20:51:40,920 - INFO - 
TP Mode Statistics:
+----------------------+--------+
| Fixed TP Trades      | 57     |
+----------------------+--------+
| Fixed TP Win Rate    | 43.86% |
+----------------------+--------+
| Adaptive TP Trades   | 6      |
+----------------------+--------+
| Adaptive TP Win Rate | 33.33% |
+----------------------+--------+
2025-04-03 20:51:40,920 - INFO - 
Risk Metrics:
+---------------------------+-------------+
| Maximum Drawdown          | 1.82%       |
+---------------------------+-------------+
| Maximum Drawdown Duration | 70.50 hours |
+---------------------------+-------------+


----


python backtest_trading_bot_v1_2.py --start_date 2025-01-01 --end_date 2025-02-01


=== ATR Regime Distribution ===
Sideways   :   3348 bars (37.71%)
Moderate   :   4818 bars (54.26%)
Trending   :    713 bars (8.03%)
2025-04-04 10:39:12,813 - INFO - 
Backtest Results:
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $9624.84  |
+-----------------+-----------+
| Total Profit    | $-375.16  |
+-----------------+-----------+
| Total Trades    | 49        |
+-----------------+-----------+
| Winning Trades  | 12        |
+-----------------+-----------+
| Losing Trades   | 37        |
+-----------------+-----------+
| Win Rate        | 24.49%    |
+-----------------+-----------+
| Profit Factor   | 0.71      |
+-----------------+-----------+
2025-04-04 10:39:12,813 - INFO - 
TP Mode Statistics:
+----------------------+--------+
| Fixed TP Trades      | 46     |
+----------------------+--------+
| Fixed TP Win Rate    | 23.91% |
+----------------------+--------+
| Adaptive TP Trades   | 3      |
+----------------------+--------+
| Adaptive TP Win Rate | 33.33% |
+----------------------+--------+
2025-04-04 10:39:12,813 - INFO - 
Risk Metrics:
+---------------------------+--------------+
| Maximum Drawdown          | 4.89%        |
+---------------------------+--------------+
| Maximum Drawdown Duration | 510.92 hours |
+---------------------------+--------------+


----


python backtest_trading_bot_v1_2.py --start_date 2025-02-01 --end_date 2025-03-01

=== ATR Regime Distribution ===
Sideways   :   3534 bars (44.09%)
Moderate   :   3674 bars (45.84%)
Trending   :    807 bars (10.07%)
2025-04-04 10:44:05,696 - INFO - 
Backtest Results:
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $9578.90  |
+-----------------+-----------+
| Total Profit    | $-421.10  |
+-----------------+-----------+
| Total Trades    | 51        |
+-----------------+-----------+
| Winning Trades  | 12        |
+-----------------+-----------+
| Losing Trades   | 39        |
+-----------------+-----------+
| Win Rate        | 23.53%    |
+-----------------+-----------+
| Profit Factor   | 0.69      |
+-----------------+-----------+
2025-04-04 10:44:05,696 - INFO - 
TP Mode Statistics:
+----------------------+--------+
| Fixed TP Trades      | 46     |
+----------------------+--------+
| Fixed TP Win Rate    | 21.74% |
+----------------------+--------+
| Adaptive TP Trades   | 5      |
+----------------------+--------+
| Adaptive TP Win Rate | 40.00% |
+----------------------+--------+
2025-04-04 10:44:05,696 - INFO - 
Risk Metrics:
+---------------------------+--------------+
| Maximum Drawdown          | 5.66%        |
+---------------------------+--------------+
| Maximum Drawdown Duration | 357.33 hours |
+---------------------------+--------------+

