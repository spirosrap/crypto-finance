#!/usr/bin/env python3
"""
Enhanced Trade Analytics Script
Analyzes closed trades from watchdog_closed_positions.csv
Provides comprehensive performance breakdowns and insights.
"""

import csv
import json
import math
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
                    'day_idx': closed_dt.weekday(),
                    'product_id': row.get('product_id', ''),
                    'side': row.get('position_side', ''),
                    'pl': pl,
                    'pl_pct': float(row.get('profit_loss_pct', 0) or 0),
                    'closure_reason': row.get('closure_reason', ''),
                    'duration_hours': float(row.get('duration_seconds', 0) or 0) / 3600,
                })
            except (ValueError, TypeError):
                continue
    return sorted(trades, key=lambda x: x['closed_at'])


def analyze_by_key(trades, key, compute_expectancy=False):
    """Group trades by key and compute comprehensive stats."""
    groups = defaultdict(list)
    for t in trades:
        groups[t[key]].append(t)
    
    results = []
    for k, group in sorted(groups.items(), key=lambda x: sum(t['pl'] for t in x[1]), reverse=True):
        pls = [t['pl'] for t in group]
        total_pl = sum(pls)
        gross_profit = sum(p for p in pls if p > 0)
        gross_loss = abs(sum(p for p in pls if p < 0))
        wins = sum(1 for p in pls if p > 0)
        losses = sum(1 for p in pls if p < 0)
        avg_winner = gross_profit / wins if wins else 0
        avg_loser = gross_loss / losses if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss else float('inf')
        win_rate = wins / len(group) * 100 if group else 0
        
        expectancy = (win_rate/100 * avg_winner) - ((1 - win_rate/100) * avg_loser) if (wins + losses) > 0 else 0
        
        results.append({
            'key': k,
            'count': len(group),
            'total_pl': total_pl,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(group) * 100 if group else 0,
            'avg_pl': total_pl / len(group) if group else 0,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_win': avg_winner,
            'avg_loss': avg_loser,
        })
    return results


def calculate_max_drawdown(trades):
    """Calculate maximum drawdown from cumulative P/L."""
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t['pl']
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calculate_streaks(trades):
    """Calculate longest win/loss streaks."""
    current_streak = 0
    current_type = None
    max_win_streak = 0
    max_loss_streak = 0
    
    for t in trades:
        if t['pl'] > 0:
            if current_type == 'win':
                current_streak += 1
            else:
                current_type = 'win'
                current_streak = 1
            max_win_streak = max(max_win_streak, current_streak)
        elif t['pl'] < 0:
            if current_type == 'loss':
                current_streak += 1
            else:
                current_type = 'loss'
                current_streak = 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            current_streak = 0
            current_type = None
    
    return max_win_streak, max_loss_streak


def calculate_kelly(win_rate, avg_win, avg_loss):
    """Calculate Kelly fraction (simplified)."""
    if avg_loss == 0:
        return 1.0
    kelly = (win_rate/100 * avg_win - (1 - win_rate/100) * avg_loss) / avg_loss
    return max(0, min(kelly, 1.0))  # Clamp between 0 and 1


def print_section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print('='*62)


def print_stats(label, value, fmt=".2f"):
    """Print a stat line with consistent formatting."""
    if isinstance(value, float):
        if "P/L" in label or "$" in label:
            print(f"  {label:<25} ${value:{fmt}}")
        elif "%" in label or "Win" in label or "Rate" in label:
            print(f"  {label:<25} {value:{fmt}}%")
        else:
            print(f"  {label:<25} {value:{fmt}}")
    else:
        print(f"  {label:<25} {value}")


def main():
    trades = load_trades(TRADES_CSV)
    
    if not trades:
        print("No trades found!")
        return
    
    print(f"\n{' '*18}ENHANCED TRADE ANALYTICS")
    print(f"{' '*13}Loaded {len(trades)} trades")
    print(f"  Date range: {trades[0]['date']} to {trades[-1]['date']}")
    
    # Overall stats
    pls = [t['pl'] for t in trades]
    total_pl = sum(pls)
    wins = sum(1 for p in pls if p > 0)
    losses = sum(1 for p in pls if p < 0)
    breakevens = sum(1 for p in pls if p == 0)
    win_rate = wins / len(trades) * 100
    gross_profit = sum(p for p in pls if p > 0)
    gross_loss = abs(sum(p for p in pls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss else float('inf')
    avg_winner = gross_profit / wins if wins else 0
    avg_loser = gross_loss / losses if losses else 0
    expectancy = (win_rate/100 * avg_winner) - ((1 - win_rate/100) * avg_loser)
    avg_trade = total_pl / len(trades) if trades else 0
    max_dd = calculate_max_drawdown(trades)
    max_win_streak, max_loss_streak = calculate_streaks(trades)
    avg_duration = sum(t['duration_hours'] for t in trades) / len(trades) if trades else 0
    kelly = calculate_kelly(win_rate, avg_winner, avg_loser)
    
    # Cumulative equity curve
    cumulative = 0
    equity_points = []
    for t in trades:
        cumulative += t['pl']
        equity_points.append(cumulative)
    
    print_section("OVERALL PERFORMANCE")
    print_stats("Total P/L", total_pl)
    print_stats("Total Trades", len(trades))
    print_stats("Win Rate", win_rate)
    print_stats("Profit Factor", profit_factor)
    print_stats("Expectancy (per trade)", expectancy)
    print_stats("Avg P/L per Trade", avg_trade)
    print_stats("Max Drawdown", max_dd)
    print_stats("Kelly Fraction", kelly * 100)
    print(f"\n  Trade Count:     {wins:>4}W / {losses:>4}L / {breakevens}B")
    print(f"  Avg Winner:      ${avg_winner:>9.2f}")
    print(f"  Avg Loser:       ${avg_loser:>9.2f}")
    print(f"  Avg Duration:    {avg_duration:.1f} hours")
    print(f"  Max Win Streak:  {max_win_streak}")
    print(f"  Max Loss Streak: {max_loss_streak}")
    
    # Equity curve text visualization
    print_section("EQUITY CURVE (text)")
    min_eq = min(equity_points)
    max_eq = max(equity_points)
    eq_range = max_eq - min_eq if max_eq > min_eq else 1
    width = 50
    
    print(f"  Start: ${equity_points[0]:.2f} → End: ${equity_points[-1]:.2f}")
    print(f"  Peak:  ${max(equity_points):.2f}  |  Valley: ${min(equity_points):.2f}")
    print("  " + "─" * (width + 6))
    
    # Show first 10, last 10, and sample middle if many trades
    if len(equity_points) <= 20:
        indices = list(range(len(equity_points)))
    else:
        indices = list(range(10)) + list(range(10, len(equity_points)-10, max(1, (len(equity_points)-20)//10))) + list(range(len(equity_points)-10, len(equity_points)))
        indices = sorted(set(indices))
    
    last_eq = None
    for i, eq in enumerate(equity_points):
        if i not in indices and i != len(equity_points) - 1:
            continue
        if i > 0 and last_eq is not None:
            direction = "↑" if eq > last_eq else "↓" if eq < last_eq else "→"
        else:
            direction = "●"
        pos = int((eq - min_eq) / eq_range * (width - 1))
        bar = " " * pos + direction
        print(f"  [{i+1:3d}] {bar} ${eq:>8.2f}")
        last_eq = eq
    
    # By Side
    by_side = analyze_by_key(trades, 'side')
    print_section("PERFORMANCE BY SIDE")
    print(f"  {'Side':<8} {'Count':>6} {'P/L':>10} {'Win%':>7} {'PF':>6} {'Exp':>8} {'Kelly':>7}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*7} {'-'*6} {'-'*8} {'-'*7}")
    for s in by_side:
        kelly_s = calculate_kelly(s['win_rate'], s['avg_win'], s['avg_loss'])
        print(f"  {s['key']:<8} {s['count']:>6} ${s['total_pl']:>8.2f} {s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f} ${s['expectancy']:>7.2f} {kelly_s*100:>5.1f}%")
    
    # By Day
    by_day = analyze_by_key(trades, 'day_name')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    by_day_sorted = sorted(by_day, key=lambda x: day_order.index(x['key']) if x['key'] in day_order else 99)
    
    print_section("PERFORMANCE BY DAY")
    print(f"  {'Day':<10} {'Count':>6} {'P/L':>10} {'Win%':>7} {'PF':>6} {'Exp':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*7} {'-'*6} {'-'*8}")
    for d in by_day_sorted:
        print(f"  {d['key']:<10} {d['count']:>6} ${d['total_pl']:>8.2f} {d['win_rate']:>6.1f}% {d['profit_factor']:>5.2f} ${d['expectancy']:>7.2f}")
    
    # By Hour
    by_hour = analyze_by_key(trades, 'hour')
    print_section("PERFORMANCE BY HOUR (UTC)")
    print(f"  {'Hour':>5} {'Count':>6} {'P/L':>10} {'Win%':>7} {'PF':>6}")
    print(f"  {'-'*5} {'-'*6} {'-'*10} {'-'*7} {'-'*6}")
    for h in sorted(by_hour, key=lambda x: x['key']):
        print(f"  {h['key']:>02d}:00 {h['count']:>6} ${h['total_pl']:>8.2f} {h['win_rate']:>6.1f}% {h['profit_factor']:>5.2f}")
    
    # By Closure Reason
    by_reason = analyze_by_key(trades, 'closure_reason')
    print_section("PERFORMANCE BY CLOSURE REASON")
    print(f"  {'Reason':<20} {'Count':>6} {'P/L':>10} {'Win%':>7}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*7}")
    for r in by_reason:
        print(f"  {r['key']:<20} {r['count']:>6} ${r['total_pl']:>8.2f} {r['win_rate']:>6.1f}%")
    
    # Side + Day combinations
    print_section("BEST SIDE + DAY COMBINATIONS")
    side_day = defaultdict(list)
    for t in trades:
        key = f"{t['day_name']} {t['side']}"
        side_day[key].append(t)
    
    combos = []
    for k, group in side_day.items():
        pls = [t['pl'] for t in group]
        total = sum(pls)
        wins = sum(1 for p in pls if p > 0)
        combos.append({
            'key': k,
            'count': len(group),
            'total_pl': total,
            'win_rate': wins / len(group) * 100,
        })
    combos.sort(key=lambda x: x['total_pl'], reverse=True)
    
    print(f"  {'Combination':<20} {'Count':>6} {'P/L':>10} {'Win%':>7}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*7}")
    for c in combos[:8]:
        print(f"  {c['key']:<20} {c['count']:>6} ${c['total_pl']:>8.2f} {c['win_rate']:>6.1f}%")
    
    # Majors vs Alts
    majors = {'BTC-PERP-INTX', 'ETH-PERP-INTX', 'SOL-PERP-INTX'}
    major_trades = [t for t in trades if t['product_id'] in majors]
    alt_trades = [t for t in trades if t['product_id'] not in majors]
    
    print_section("MAJORS vs ALTS")
    for label, subset in [("Majors (BTC, ETH, SOL)", major_trades), ("Alts", alt_trades)]:
        if not subset:
            continue
        pls = [t['pl'] for t in subset]
        total = sum(pls)
        wins = sum(1 for p in pls if p > 0)
        pf = sum(p for p in pls if p > 0) / abs(sum(p for p in pls if p < 0)) if sum(p for p in pls if p < 0) else float('inf')
        print(f"  {label}:")
        print(f"    Trades: {len(subset)} | P/L: ${total:.2f} | Win%: {wins/len(subset)*100:.1f}% | PF: {pf:.2f}")
    
    # Top/Bottom Symbols (min 5 trades)
    by_symbol = analyze_by_key(trades, 'product_id')
    significant = [s for s in by_symbol if s['count'] >= 5]
    
    print_section(f"TOP 5 SYMBOLS (min 5 trades, {len(significant)} qualify)")
    print(f"  {'Symbol':<25} {'Count':>6} {'P/L':>10} {'Win%':>7} {'PF':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*7} {'-'*6}")
    for s in significant[:5]:
        print(f"  {s['key']:<25} {s['count']:>6} ${s['total_pl']:>8.2f} {s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f}")
    
    print_section("BOTTOM 5 SYMBOLS (min 5 trades)")
    print(f"  {'Symbol':<25} {'Count':>6} {'P/L':>10} {'Win%':>7} {'PF':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*7} {'-'*6}")
    for s in significant[-5:]:
        print(f"  {s['key']:<25} {s['count']:>6} ${s['total_pl']:>8.2f} {s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f}")
    
    # Recent trend (last 20 trades)
    recent = trades[-20:]
    recent_pl = sum(t['pl'] for t in recent)
    recent_wins = sum(1 for t in recent if t['pl'] > 0)
    print_section("RECENT TREND (last 20 trades)")
    print(f"  P/L: ${recent_pl:.2f} | {recent_wins}/20 wins ({recent_wins/20*100:.0f}%)")
    
    # Export comprehensive summary to JSON
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_trades': len(trades),
        'date_range': {'start': str(trades[0]['date']), 'end': str(trades[-1]['date'])},
        'overall': {
            'total_pl': total_pl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_pl': avg_trade,
            'max_drawdown': max_dd,
            'kelly': kelly,
            'wins': wins,
            'losses': losses,
            'breakevens': breakevens,
            'avg_duration_hours': avg_duration,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
        },
        'by_side': by_side,
        'by_day': by_day_sorted,
        'by_hour': by_hour,
        'by_reason': by_reason,
        'top_symbols': significant[:5],
        'bottom_symbols': significant[-5:],
        'recent_20': {
            'pl': recent_pl,
            'wins': recent_wins,
            'win_rate': recent_wins / 20 * 100,
        },
    }
    
    json_path = OUTPUT_DIR / "analytics_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n📁 Summary exported to: {json_path}")


if __name__ == "__main__":
    main()
