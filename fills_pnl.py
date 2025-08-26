#!/usr/bin/env python3
import argparse
import csv
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS


UTC = timezone.utc


def to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_time(ts: Any) -> datetime:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=UTC)
    if isinstance(ts, str):
        # Coinbase timestamps usually ISO8601 with Z
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
        except Exception:
            try:
                return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            except Exception:
                return datetime.now(tz=UTC)
    return datetime.now(tz=UTC)


def fetch_fills(cb: CoinbaseService, limit: int = 500) -> List[Dict[str, Any]]:
    # Use the underlying client directly; normalize to dict list
    # Some SDKs expose get_fills instead of list_fills
    if hasattr(cb.client, 'list_fills'):
        resp = cb.client.list_fills(limit=limit)
    else:
        resp = cb.client.get_fills(limit=limit)
    fills: List[Dict[str, Any]] = []
    if isinstance(resp, dict) and 'fills' in resp:
        raw_fills = resp['fills']
    elif hasattr(resp, 'fills'):
        raw_fills = resp.fills
    else:
        raw_fills = []

    for f in raw_fills:
        d = f if isinstance(f, dict) else getattr(f, '__dict__', {})
        fills.append({
            'product_id': d.get('product_id') or d.get('symbol') or '',
            'side': (d.get('side') or '').upper(),
            'size': to_float(d.get('size') or d.get('base_size') or 0),
            'price': to_float(d.get('price') or d.get('average_price') or 0),
            'fee': to_float(d.get('fee') or d.get('commission') or 0),
            'liquidity': (d.get('liquidity_indicator') or d.get('liquidity') or '').upper(),
            'order_id': d.get('order_id') or d.get('trade_id') or '',
            'time': parse_time(d.get('trade_time') or d.get('time') or d.get('created_time') or d.get('ts')),
        })
    # Sort ascending time for FIFO
    fills.sort(key=lambda x: x['time'])
    return fills


def compute_fifo_realized_pnl(fills: List[Dict[str, Any]], symbol: str = None, start: datetime = None) -> Dict[str, Any]:
    # Group by product
    by_product: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in fills:
        if symbol and f['product_id'] != symbol:
            continue
        if start and f['time'] < start:
            continue
        by_product[f['product_id']].append(f)

    summary: Dict[str, Any] = {}
    for product_id, pfills in by_product.items():
        # Maintain FIFO inventory of buys for long, sells for short via signed quantity
        # We will treat BUY size as positive, SELL size as negative
        inventory: deque = deque()  # each item: {'qty': float, 'price': float, 'fee': float}
        realized_pnl = 0.0
        realized_fees = 0.0
        gross_pnl = 0.0

        for f in pfills:
            qty = f['size'] if f['side'] == 'BUY' else -f['size']
            price = f['price']
            fee = f['fee']
            realized_fees += fee

            if qty > 0:
                # Add to inventory
                inventory.append({'qty': qty, 'price': price})
            else:
                qty_to_match = -qty
                while qty_to_match > 0 and inventory:
                    lot = inventory[0]
                    match_qty = min(qty_to_match, lot['qty'])
                    entry_price = lot['price']
                    exit_price = price
                    gross_pnl += (exit_price - entry_price) * match_qty
                    realized_pnl += (exit_price - entry_price) * match_qty
                    lot['qty'] -= match_qty
                    qty_to_match -= match_qty
                    if lot['qty'] <= 1e-12:
                        inventory.popleft()
                # If inventory empty and still qty_to_match > 0, this indicates opening a short; push negative lot
                if qty_to_match > 0:
                    inventory.append({'qty': -qty_to_match, 'price': price})

        net_position = sum(lot['qty'] for lot in inventory) if inventory else 0.0
        avg_entry = 0.0
        if net_position != 0 and inventory:
            total_cost = sum(lot['qty'] * lot['price'] for lot in inventory)
            avg_entry = total_cost / net_position

        summary[product_id] = {
            'realized_pnl': round(realized_pnl - realized_fees, 2),
            'gross_pnl': round(gross_pnl, 2),
            'fees': round(realized_fees, 2),
            'net_position': round(net_position, 8),
            'avg_entry_price': round(avg_entry, 2) if net_position else 0.0,
            'fills_count': len(pfills),
        }

    return summary


def compute_round_trip_cycles(
    fills: List[Dict[str, Any]], symbol: Optional[str] = None, start: Optional[datetime] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Group fills into closed round-trip trade cycles per product.

    A cycle begins when net position leaves 0 and ends when it returns to 0.
    Fees are included in the cycle PnL.
    """
    # Filter and group
    filtered: List[Dict[str, Any]] = []
    for f in fills:
        if symbol and f['product_id'] != symbol:
            continue
        if start and f['time'] < start:
            continue
        filtered.append(f)
    filtered.sort(key=lambda x: x['time'])

    by_product: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in filtered:
        by_product[f['product_id']].append(f)

    result: Dict[str, List[Dict[str, Any]]] = {}
    for product_id, pfills in by_product.items():
        cycles: List[Dict[str, Any]] = []
        position = 0.0
        cycle_start_time: Optional[datetime] = None
        cycle_gross = 0.0
        cycle_fees = 0.0
        inventory: deque = deque()

        def close_cycle(end_time: datetime):
            nonlocal cycle_gross, cycle_fees, cycle_start_time
            cycles.append({
                'start_time': cycle_start_time,
                'end_time': end_time,
                'gross_pnl': round(cycle_gross, 2),
                'fees': round(cycle_fees, 2),
                'realized_pnl': round(cycle_gross - cycle_fees, 2),
            })
            cycle_gross = 0.0
            cycle_fees = 0.0
            cycle_start_time = None

        for f in pfills:
            qty = f['size'] if f['side'] == 'BUY' else -f['size']
            price = f['price']
            fee = f['fee']
            cycle_fees += fee

            # Start a new cycle when leaving flat
            if position == 0.0 and cycle_start_time is None:
                cycle_start_time = f['time']

            if qty > 0:  # buy
                inventory.append({'qty': qty, 'price': price})
                position += qty
            else:  # sell
                sell_qty = -qty
                position -= sell_qty
                # FIFO match
                remaining = sell_qty
                while remaining > 0 and inventory:
                    lot = inventory[0]
                    match_qty = min(remaining, lot['qty'])
                    cycle_gross += (price - lot['price']) * match_qty
                    lot['qty'] -= match_qty
                    remaining -= match_qty
                    if lot['qty'] <= 1e-12:
                        inventory.popleft()

            # If we returned to flat, close the cycle
            if abs(position) <= 1e-12 and cycle_start_time is not None:
                close_cycle(f['time'])
                inventory.clear()
                position = 0.0

        result[product_id] = cycles

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute realized PnL from Coinbase fills (fees included)")
    parser.add_argument('--symbol', type=str, help='Filter by product id, e.g., ETH-PERP-INTX')
    parser.add_argument('--days', type=int, default=30, help='How many days back to include fills')
    parser.add_argument('--limit', type=int, default=2000, help='Max fills to pull (default: 2000)')
    parser.add_argument('--save', type=str, help='Optional CSV to save the per-product summary')
    parser.add_argument('--cycles', type=int, default=8, help='Show last N closed trade cycles per symbol (default: 8)')
    args = parser.parse_args()

    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    fills = fetch_fills(cb, limit=args.limit)
    start = datetime.now(tz=UTC) - timedelta(days=args.days) if args.days else None
    if args.cycles:
        cycles = compute_round_trip_cycles(fills, symbol=args.symbol, start=start)
        for product_id, clist in cycles.items():
            if not clist:
                continue
            last = clist[-args.cycles:]
            total = round(sum(c['realized_pnl'] for c in last), 2)
            print(f"{product_id} last {len(last)} cycles: total={total} USD")
            for c in last:
                st = c['start_time'].strftime('%Y-%m-%d %H:%M') if c['start_time'] else 'N/A'
                et = c['end_time'].strftime('%Y-%m-%d %H:%M') if c['end_time'] else 'N/A'
                print(f"  {st} -> {et}: realized={c['realized_pnl']} (gross={c['gross_pnl']}, fees={c['fees']})")
    else:
        summary = compute_fifo_realized_pnl(fills, symbol=args.symbol, start=start)
        for product_id, s in summary.items():
            print(f"{product_id}: realized={s['realized_pnl']} USD, fees={s['fees']} USD, gross={s['gross_pnl']} USD, position={s['net_position']} @ {s['avg_entry_price']} (fills {s['fills_count']})")

    if args.save:
        with open(args.save, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['product_id', 'realized_pnl', 'gross_pnl', 'fees', 'net_position', 'avg_entry_price', 'fills_count'])
            for product_id, s in summary.items():
                writer.writerow([product_id, s['realized_pnl'], s['gross_pnl'], s['fees'], s['net_position'], s['avg_entry_price'], s['fills_count']])
        print(f"Saved summary to {args.save}")


if __name__ == '__main__':
    main()


