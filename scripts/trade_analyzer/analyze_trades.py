#!/usr/bin/env python3
"""
Trade Analytics Script
Analyzes closed trades from watchdog_closed_positions.csv
Provides performance breakdowns by symbol, side, day, hour, and closure reason.
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Configuration
TRADES_CSV = "/home/spiros/crypto-finance/trade_logs/watchdog_closed_positions.csv"
OUTPUT_DIR = Path(__file__).parent


def load_trades(filepath):
    """Load trades from CSV file."""
    trades = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            closed_at = row.get('closed_at', '')
            if not closed_at:
                continue
            try:
                closed_dt = datetime.fromisoformat(closed_at.replace('Z', '+00:00'))
                pl = float(row.get('profit_loss', 0) or 0)
                trades.append({
                    'closed_at': closed_dt,
                    'date': closed_dt.date(),
                    'hour': closed_dt.hour,
                    'day_name': closed_dt.strftime('%A'),
                    'product_id': row.get('product_id', ''),
                    'side': row.get('position_side', ''),
                    'pl': pl,
                    'pl_pct': float(row.get('profit_loss_pct', 0) or 0),
                    'closure_reason': row.get('closure_reason', ''),
                    'duration_hours': float(row.get('duration_seconds', 0) or 0) / 3600,
                })
            except (ValueError, TypeError):
                continue
    return trades


def analyze_by_key(trades, key):
    """Group trades by key and compute stats."""
    groups = defaultdict(list)
    for t in trades:
        groups[t[key]].append(t)
    
    results = []
    for k, group in sorted(groups.items(), key=lambda x: sum(t['pl'] for t in x[1]), reverse=True):
        total_pl = sum(t['pl'] for t in group)
        wins = sum(1 for t in group if t['pl'] > 0)
        losses = sum(1 for t in group if t['pl'] < 0)
        avg_pl = total_pl / len(group) if group else 0
        results.append({
            'key': k,
            'count': len(group),
            'total_pl': total_pl,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(group) * 100 if group else 0,
            'avg_pl': avg_pl,
        })
    return results


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    trades = load_trades(TRADES_CSV)
    
    if not trades:
        print("No trades found!")
        return
    
    print(f"\n{' '*20}TRADE ANALYTICS REPORT")
    print(f"{' '*15}Loaded {len(trades)} trades from {TRADES_CSV}")
    print(f"  Date range: {min(t['date'] for t in trades)} to {max(t['date'] for t in trades)}")
    
    # Overall stats
    total_pl = sum(t['pl'] for t in trades)
    wins = sum(1 for t in trades if t['pl'] > 0)
    losses = sum(1 for t in trades if t['pl'] < 0)
    win_rate = wins / len(trades) * 100
    
    print_section("OVERALL PERFORMANCE")
    print(f"  Total P/L:      ${total_pl:>10.2f}")
    print(f"  Total trades:   {len(trades):>10}")
    print(f"  Win rate:       {win_rate:>9.1f}%")
    print(f"  Avg P/L/trade:  ${total_pl/len(trades):>10.2f}")
    print(f"  Wins/Losses:    {wins:>4}W / {losses:>4}L")
    
    # By Side (Long vs Short)
    by_side = analyze_by_key(trades, 'side')
    print_section("PERFORMANCE BY SIDE")
    print(f"  {'Side':<8} {'Count':>6} {'Total P/L':>12} {'Win%':>8} {'Avg P/L':>10}")
    print(f"  {'-'*8} {'-'*6} {'-'*12} {'-'*8} {'-'*10}")
    for s in by_side:
        print(f"  {s['key']:<8} {s['count']:>6} ${s['total_pl']:>10.2f} {s['win_rate']:>7.1f}% ${s['avg_pl']:>9.2f}")
    
    # By Day of Week
    by_day = analyze_by_key(trades, 'day_name')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    by_day_sorted = sorted(by_day, key=lambda x: day_order.index(x['key']) if x['key'] in day_order else 99)
    
    print_section("PERFORMANCE BY DAY OF WEEK")
    print(f"  {'Day':<10} {'Count':>6} {'Total P/L':>12} {'Win%':>8} {'Avg P/L':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*12} {'-'*8} {'-'*10}")
    for d in by_day_sorted:
        print(f"  {d['key']:<10} {d['count']:>6} ${d['total_pl']:>10.2f} {d['win_rate']:>7.1f}% ${d['avg_pl']:>9.2f}")
    
    # By Closure Reason
    by_reason = analyze_by_key(trades, 'closure_reason')
    print_section("PERFORMANCE BY CLOSURE REASON")
    print(f"  {'Reason':<20} {'Count':>6} {'Total P/L':>12} {'Win%':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*12} {'-'*8}")
    for r in by_reason:
        print(f"  {r['key']:<20} {r['count']:>6} ${r['total_pl']:>10.2f} {r['win_rate']:>7.1f}%")
    
    # Top/Bottom Symbols (min 5 trades)
    symbol_counts = defaultdict(int)
    for t in trades:
        symbol_counts[t['product_id']] += 1
    
    symbol_trades = defaultdict(list)
    for t in trades:
        symbol_trades[t['product_id']].append(t)
    
    significant_symbols = {k: v for k, v in symbol_trades.items() if len(v) >= 5}
    by_symbol = analyze_by_key(trades, 'product_id')
    by_symbol_significant = [s for s in by_symbol if s['count'] >= 5]
    
    print_section(f"TOP 5 SYMBOLS (min 5 trades, {len(by_symbol_significant)} qualify)")
    print(f"  {'Symbol':<25} {'Count':>6} {'Total P/L':>12} {'Win%':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*12} {'-'*8}")
    for s in by_symbol_significant[:5]:
        print(f"  {s['key']:<25} {s['count']:>6} ${s['total_pl']:>10.2f} {s['win_rate']:>7.1f}%")
    
    print_section("BOTTOM 5 SYMBOLS (min 5 trades)")
    print(f"  {'Symbol':<25} {'Count':>6} {'Total P/L':>12} {'Win%':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*12} {'-'*8}")
    for s in by_symbol_significant[-5:]:
        print(f"  {s['key']:<25} {s['count']:>6} ${s['total_pl']:>10.2f} {s['win_rate']:>7.1f}%")
    
    # By Hour
    by_hour = analyze_by_key(trades, 'hour')
    print_section("PERFORMANCE BY HOUR (UTC)")
    print(f"  {'Hour':>5} {'Count':>6} {'Total P/L':>12} {'Win%':>8}")
    print(f"  {'-'*5} {'-'*6} {'-'*12} {'-'*8}")
    for h in sorted(by_hour, key=lambda x: x['key']):
        print(f"  {h['key']:>02d}:00 {h['count']:>6} ${h['total_pl']:>10.2f} {h['win_rate']:>7.1f}%")
    
    # Export summary to JSON
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_trades': len(trades),
        'date_range': {
            'start': str(min(t['date'] for t in trades)),
            'end': str(max(t['date'] for t in trades)),
        },
        'overall': {
            'total_pl': total_pl,
            'win_rate': win_rate,
            'avg_pl': total_pl / len(trades),
            'wins': wins,
            'losses': losses,
        },
        'by_side': by_side,
        'by_day': by_day_sorted,
        'by_reason': by_reason,
        'top_symbols': by_symbol_significant[:5],
        'bottom_symbols': by_symbol_significant[-5:],
    }
    
    json_path = OUTPUT_DIR / "analytics_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n📁 Summary exported to: {json_path}")


if __name__ == "__main__":
    main()
