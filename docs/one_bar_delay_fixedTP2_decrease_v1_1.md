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



python backtest_trading_bot.py --start_date 2023-03-01 --end_date 2023-04-01



python backtest_trading_bot.py --start_date 2023-06-01 --end_date 2023-07-01




python backtest_trading_bot.py --start_date 2023-11-01 --end_date 2023-12-01

=== BACKTEST RESULTS ===
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10178.10 |
+-----------------+-----------+
| Total Profit    | $178.10   |
+-----------------+-----------+
| Total Trades    | 32        |
+-----------------+-----------+
| Winning Trades  | 14        |
+-----------------+-----------+
| Losing Trades   | 18        |
+-----------------+-----------+
| Win Rate        | 43.75%    |
+-----------------+-----------+
| Profit Factor   | 1.28      |
+-----------------+-----------+

=== TP MODE STATISTICS ===
+----------------------+--------+
| Fixed TP Trades      | 31     |
+----------------------+--------+
| Fixed TP Win Rate    | 45.16% |
+----------------------+--------+
| Adaptive TP Trades   | 1      |
+----------------------+--------+
| Adaptive TP Win Rate | 0.00%  |
+----------------------+--------+

=== MARKET REGIME STATISTICS ===
+--------------------+--------+
| Trending Trades    | 21     |
+--------------------+--------+
| Trending Win Rate  | 38.10% |
+--------------------+--------+
| Chop Trades        | 0      |
+--------------------+--------+
| Chop Win Rate      | 0.00%  |
+--------------------+--------+
| Uncertain Trades   | 11     |
+--------------------+--------+
| Uncertain Win Rate | 54.55% |
+--------------------+--------+



python backtest_trading_bot.py --start_date 2024-01-01 --end_date 2024-02-01


=== BACKTEST RESULTS ===
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10432.09 |
+-----------------+-----------+
| Total Profit    | $432.09   |
+-----------------+-----------+
| Total Trades    | 45        |
+-----------------+-----------+
| Winning Trades  | 21        |
+-----------------+-----------+
| Losing Trades   | 24        |
+-----------------+-----------+
| Win Rate        | 46.67%    |
+-----------------+-----------+
| Profit Factor   | 1.50      |
+-----------------+-----------+

=== TP MODE STATISTICS ===
+----------------------+--------+
| Fixed TP Trades      | 42     |
+----------------------+--------+
| Fixed TP Win Rate    | 47.62% |
+----------------------+--------+
| Adaptive TP Trades   | 3      |
+----------------------+--------+
| Adaptive TP Win Rate | 33.33% |
+----------------------+--------+

=== MARKET REGIME STATISTICS ===
+--------------------+--------+
| Trending Trades    | 34     |
+--------------------+--------+
| Trending Win Rate  | 50.00% |
+--------------------+--------+
| Chop Trades        | 0      |
+--------------------+--------+
| Chop Win Rate      | 0.00%  |
+--------------------+--------+
| Uncertain Trades   | 11     |
+--------------------+--------+
| Uncertain Win Rate | 36.36% |
+--------------------+--------+



python backtest_trading_bot.py --start_date 2024-03-01 --end_date 2024-04-01

=== BACKTEST RESULTS ===
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10362.80 |
+-----------------+-----------+
| Total Profit    | $362.80   |
+-----------------+-----------+
| Total Trades    | 51        |
+-----------------+-----------+
| Winning Trades  | 21        |
+-----------------+-----------+
| Losing Trades   | 30        |
+-----------------+-----------+
| Win Rate        | 41.18%    |
+-----------------+-----------+
| Profit Factor   | 1.34      |
+-----------------+-----------+

=== TP MODE STATISTICS ===
+----------------------+--------+
| Fixed TP Trades      | 45     |
+----------------------+--------+
| Fixed TP Win Rate    | 42.22% |
+----------------------+--------+
| Adaptive TP Trades   | 6      |
+----------------------+--------+
| Adaptive TP Win Rate | 33.33% |
+----------------------+--------+

=== MARKET REGIME STATISTICS ===
+--------------------+--------+
| Trending Trades    | 46     |
+--------------------+--------+
| Trending Win Rate  | 43.48% |
+--------------------+--------+
| Chop Trades        | 0      |
+--------------------+--------+
| Chop Win Rate      | 0.00%  |
+--------------------+--------+
| Uncertain Trades   | 5      |
+--------------------+--------+
| Uncertain Win Rate | 20.00% |
+--------------------+--------+


python backtest_trading_bot.py --start_date 2025-03-01 --end_date 2025-04-01


=== BACKTEST RESULTS ===
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10693.42 |
+-----------------+-----------+
| Total Profit    | $693.42   |
+-----------------+-----------+
| Total Trades    | 55        |
+-----------------+-----------+
| Winning Trades  | 26        |
+-----------------+-----------+
| Losing Trades   | 29        |
+-----------------+-----------+
| Win Rate        | 47.27%    |
+-----------------+-----------+
| Profit Factor   | 1.66      |
+-----------------+-----------+

=== TP MODE STATISTICS ===
+----------------------+--------+
| Fixed TP Trades      | 50     |
+----------------------+--------+
| Fixed TP Win Rate    | 48.00% |
+----------------------+--------+
| Adaptive TP Trades   | 5      |
+----------------------+--------+
| Adaptive TP Win Rate | 40.00% |
+----------------------+--------+

=== MARKET REGIME STATISTICS ===
+--------------------+--------+
| Trending Trades    | 41     |
+--------------------+--------+
| Trending Win Rate  | 56.10% |
+--------------------+--------+
| Chop Trades        | 1      |
+--------------------+--------+
| Chop Win Rate      | 0.00%  |
+--------------------+--------+
| Uncertain Trades   | 13     |
+--------------------+--------+
| Uncertain Win Rate | 23.08% |
+--------------------+--------+


python backtest_trading_bot.py --start_date 2025-01-01 --end_date 2025-02-01


=== BACKTEST RESULTS ===
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10419.91 |
+-----------------+-----------+
| Total Profit    | $419.91   |
+-----------------+-----------+
| Total Trades    | 47        |
+-----------------+-----------+
| Winning Trades  | 20        |
+-----------------+-----------+
| Losing Trades   | 27        |
+-----------------+-----------+
| Win Rate        | 42.55%    |
+-----------------+-----------+
| Profit Factor   | 1.43      |
+-----------------+-----------+

=== TP MODE STATISTICS ===
+----------------------+---------+
| Fixed TP Trades      | 46      |
+----------------------+---------+
| Fixed TP Win Rate    | 41.30%  |
+----------------------+---------+
| Adaptive TP Trades   | 1       |
+----------------------+---------+
| Adaptive TP Win Rate | 100.00% |
+----------------------+---------+

=== MARKET REGIME STATISTICS ===
+--------------------+--------+
| Trending Trades    | 34     |
+--------------------+--------+
| Trending Win Rate  | 35.29% |
+--------------------+--------+
| Chop Trades        | 0      |
+--------------------+--------+
| Chop Win Rate      | 0.00%  |
+--------------------+--------+
| Uncertain Trades   | 13     |
+--------------------+--------+
| Uncertain Win Rate | 61.54% |
+--------------------+--------+


python backtest_trading_bot.py --start_date 2025-02-01 --end_date 2025-03-01

=== BACKTEST RESULTS ===
+-----------------+-----------+
| Initial Balance | $10000.00 |
+-----------------+-----------+
| Final Balance   | $10114.01 |
+-----------------+-----------+
| Total Profit    | $114.01   |
+-----------------+-----------+
| Total Trades    | 49        |
+-----------------+-----------+
| Winning Trades  | 19        |
+-----------------+-----------+
| Losing Trades   | 30        |
+-----------------+-----------+
| Win Rate        | 38.78%    |
+-----------------+-----------+
| Profit Factor   | 1.11      |
+-----------------+-----------+

=== TP MODE STATISTICS ===
+----------------------+--------+
| Fixed TP Trades      | 46     |
+----------------------+--------+
| Fixed TP Win Rate    | 39.13% |
+----------------------+--------+
| Adaptive TP Trades   | 3      |
+----------------------+--------+
| Adaptive TP Win Rate | 33.33% |
+----------------------+--------+

=== MARKET REGIME STATISTICS ===
+--------------------+--------+
| Trending Trades    | 34     |
+--------------------+--------+
| Trending Win Rate  | 47.06% |
+--------------------+--------+
| Chop Trades        | 0      |
+--------------------+--------+
| Chop Win Rate      | 0.00%  |
+--------------------+--------+
| Uncertain Trades   | 15     |
+--------------------+--------+
| Uncertain Win Rate | 20.00% |
+--------------------+--------+
