import os
import json
import logging
from typing import Optional, Dict, Tuple
import requests
from decimal import Decimal, ROUND_HALF_UP
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import argparse
import time

# Set up logging with more detailed format
logging.basicConfig(level=logging.WARNING,
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Add file handler for persistent logging
log_file = 'trade_btc_perp.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
logger.addHandler(file_handler)

_PRECISION_CACHE: Dict[str, float] = {}
_BASE_CONSTRAINTS_CACHE: Dict[str, Tuple[float, float]] = {}

def _coerce_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if v > 0 else None
    except Exception:
        return None

def _fetch_increment_public(product_id: str) -> Optional[float]:
    """Try public products list once and extract price increment for product_id."""
    try:
        url = "https://api.coinbase.com/api/v3/brokerage/products"
        resp = requests.get(url, params={"limit": 250}, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        prods = data.get("products", []) if isinstance(data, dict) else []
        for p in prods:
            if str(p.get("product_id", "")).upper() == product_id.upper():
                # Try several common keys
                for key in ("price_increment", "display_price_increment", "quote_increment"):
                    inc = p.get(key)
                    v = _coerce_float(inc)
                    if v:
                        return v
                # Some payloads encode increments as strings under nested fields
                try:
                    inc = p.get("trading_rules", {}).get("price_increment")
                    v = _coerce_float(inc)
                    if v:
                        return v
                except Exception:
                    pass
        return None
    except Exception:
        return None

def _fetch_increment_auth(product_id: str) -> Optional[float]:
    """Try authenticated products list to get price increment."""
    try:
        if not (API_KEY_PERPS and API_SECRET_PERPS):
            return None
        cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
        # Prefer direct get_product for perps that may not appear in list
        try:
            prod = cb.client.get_product(product_id=product_id)
            for key in ("price_increment", "display_price_increment", "quote_increment"):
                v = _coerce_float(prod.get(key)) if isinstance(prod, dict) else _coerce_float(getattr(prod, key, None))
                if v:
                    return v
        except Exception:
            pass

        # Fallback to products list
        try:
            from coinbase.rest import products as cb_products  # type: ignore
            auth = cb_products.get_products(cb.client)
            prods = auth.get("products", []) if isinstance(auth, dict) else getattr(auth, "products", []) or []
            for p in prods:
                try:
                    pid = p.get("product_id") if isinstance(p, dict) else getattr(p, "product_id", "")
                    if str(pid or "").upper() != product_id.upper():
                        continue
                    def _pget(obj, key: str):
                        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
                    for key in ("price_increment", "display_price_increment", "quote_increment"):
                        v = _coerce_float(_pget(p, key))
                        if v:
                            return v
                    tr = _pget(p, "trading_rules") or {}
                    v = _coerce_float(tr.get("price_increment") if isinstance(tr, dict) else None)
                    if v:
                        return v
                except Exception:
                    continue
        except Exception:
            pass
        return None
    except Exception:
        return None

def get_price_precision(product_id: str) -> float:
    """Return tick size (price precision) for a product.

    Order of resolution:
      1) In-memory cache
      2) Authenticated products listing (if API keys available)
      3) Public products listing
      4) Static fallback map (kept for known assets)
      5) Final default 0.01
    """
    if product_id in _PRECISION_CACHE:
        return _PRECISION_CACHE[product_id]

    # Try authenticated fetch
    inc = _fetch_increment_auth(product_id)
    if inc:
        _PRECISION_CACHE[product_id] = inc
        return inc

    # Try public fetch
    inc = _fetch_increment_public(product_id)
    if inc:
        _PRECISION_CACHE[product_id] = inc
        return inc

    # Fallback static map for common perps
    price_precision_map = {
        'BTC-PERP-INTX': 1.0,
        'ETH-PERP-INTX': 0.1,
        'DOGE-PERP-INTX': 0.0001,
        'SOL-PERP-INTX': 0.01,
        'XRP-PERP-INTX': 0.001,
        '1000SHIB-PERP-INTX': 0.000001,
        'NEAR-PERP-INTX': 0.001,
        'SUI-PERP-INTX': 0.0001,
        'ATOM-PERP-INTX': 0.001,
    }
    inc = price_precision_map.get(product_id)
    if inc:
        _PRECISION_CACHE[product_id] = inc
        return inc

    logger.warning(f"Using default tick size for {product_id}; could not resolve from API")
    _PRECISION_CACHE[product_id] = 0.01
    return 0.01


def _decimals_from_increment(inc: float) -> int:
    """Infer decimal places from a price increment, robust to scientific notation."""
    if inc >= 1:
        return 0
    # Format to 12 decimals, trim trailing zeros, then count
    s = f"{inc:.12f}".rstrip('0')
    if '.' in s:
        return max(0, len(s.split('.')[1]))
    return 0


def format_price_for_product(product_id: str, price: float) -> str:
    """Return a string price formatted to the product's tick size.

    Ensures we pass Coinbase exactly the allowed number of decimals
    to avoid PREVIEW_INVALID_PRICE_PRECISION.
    """
    inc = get_price_precision(product_id)
    decimals = _decimals_from_increment(inc)
    quant = Decimal(1).scaleb(-decimals)  # 1e-<decimals>
    qprice = Decimal(str(price)).quantize(quant, rounding=ROUND_HALF_UP)
    # Ensure fixed number of decimals
    return f"{qprice:.{decimals}f}"


def _fetch_base_constraints_public(product_id: str) -> Optional[Tuple[float, float]]:
    try:
        url = "https://api.coinbase.com/api/v3/brokerage/products"
        resp = requests.get(url, params={"limit": 250}, timeout=10)
        if resp.status_code != 200:
            return None
        prods = (resp.json() or {}).get("products", [])
        for p in prods:
            if str(p.get("product_id", "")).upper() == product_id.upper():
                # Common field names across variants
                for kmin in ("base_min_size", "min_size", "base_min_increment"):
                    vmin = _coerce_float(p.get(kmin))
                    if vmin:
                        break
                else:
                    vmin = None
                for kinc in ("base_increment", "base_min_increment"):
                    vinc = _coerce_float(p.get(kinc))
                    if vinc:
                        break
                else:
                    vinc = None
                if vmin and vinc:
                    return float(vmin), float(vinc)
        return None
    except Exception:
        return None


def _fetch_base_constraints_auth(product_id: str) -> Optional[Tuple[float, float]]:
    try:
        if not (API_KEY_PERPS and API_SECRET_PERPS):
            return None
        cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
        # Prefer direct get_product for perps
        try:
            prod = cb.client.get_product(product_id=product_id)
            # Direct keys present in get_product
            vmin = _coerce_float(prod.get("base_min_size") if isinstance(prod, dict) else getattr(prod, "base_min_size", None))
            vinc = _coerce_float(prod.get("base_increment") if isinstance(prod, dict) else getattr(prod, "base_increment", None))
            if vmin and vinc:
                return float(vmin), float(vinc)
        except Exception:
            pass

        # Fallback: list products
        try:
            from coinbase.rest import products as cb_products  # type: ignore
            auth = cb_products.get_products(cb.client)
            prods = auth.get("products", []) if isinstance(auth, dict) else getattr(auth, "products", []) or []
            for p in prods:
                pid = p.get("product_id") if isinstance(p, dict) else getattr(p, "product_id", "")
                if str(pid or "").upper() != product_id.upper():
                    continue
                def _pget(obj, key: str):
                    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
                vmin = None
                for kmin in ("base_min_size", "min_size", "base_min_increment"):
                    vmin = _coerce_float(_pget(p, kmin))
                    if vmin:
                        break
                vinc = None
                for kinc in ("base_increment", "base_min_increment"):
                    vinc = _coerce_float(_pget(p, kinc))
                    if vinc:
                        break
                if vmin and vinc:
                    return float(vmin), float(vinc)
        except Exception:
            pass
        return None
    except Exception:
        return None


def get_base_constraints(product_id: str) -> Tuple[float, float]:
    """Return (min_base_size, base_increment) for product.

    Order: cache -> auth -> public -> fallback map -> defaults (1.0, 0.001)
    """
    if product_id in _BASE_CONSTRAINTS_CACHE:
        return _BASE_CONSTRAINTS_CACHE[product_id]
    res = _fetch_base_constraints_auth(product_id) or _fetch_base_constraints_public(product_id)
    if res:
        _BASE_CONSTRAINTS_CACHE[product_id] = res
        return res
    # Fallbacks for common perps
    fallback = {
        'BTC-PERP-INTX': (0.0001, 0.0001),
        'ETH-PERP-INTX': (0.001, 0.001),
        'SOL-PERP-INTX': (0.01, 0.01),
        'DOGE-PERP-INTX': (1.0, 1.0),
        'XRP-PERP-INTX': (1.0, 1.0),
        'LINK-PERP-INTX': (0.1, 0.1),
        'NEAR-PERP-INTX': (1.0, 1.0),
        'SUI-PERP-INTX': (1.0, 1.0),
        'ATOM-PERP-INTX': (0.1, 0.1),
    }.get(product_id, (1.0, 1.0))
    _BASE_CONSTRAINTS_CACHE[product_id] = fallback
    return fallback

def round_to_precision(value: float, precision: float) -> float:
    """Round value to nearest exchange tick size.

    For integers (precision >= 1), use nearest integer. For fractional
    precisions, round to the nearest multiple to avoid preview rejections.
    """
    if precision is None or precision <= 0:
        return value
    # Avoid floating noise
    steps = round(value / precision)
    return steps * precision

def setup_coinbase():
    """Initialize CoinbaseService with API credentials."""
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    
    return CoinbaseService(api_key, api_secret)

def check_sufficient_funds(cb_service, size_usd: float, leverage: float) -> bool:
    """
    Check if there are sufficient funds for the trade.
    Required margin = Position Size / Leverage
    """
    try:
        # Get portfolio balance
        balance, _ = cb_service.get_portfolio_info(portfolio_type="INTX")
        required_margin = size_usd / leverage
        
        logger.info(f"Available balance: ${balance}")
        logger.info(f"Required margin: ${required_margin}")
        
        return balance >= required_margin
    except Exception as e:
        logger.error(f"Error checking funds: {e}")
        return False

def preview_order(cb_service, product_id: str, side: str, size: float, leverage: str, is_limit: bool = False, limit_price: float = None):
    """
    Preview the order before placing it to check for potential issues.
    """
    try:
        if is_limit:
            preview = cb_service.client.preview_limit_order_gtc(
                product_id=product_id,
                side=side.upper(),
                base_size=str(size),
                limit_price=str(limit_price),
                leverage=leverage,
                margin_type="CROSS"
            )
        else:
            preview = cb_service.client.preview_market_order(
                product_id=product_id,
                side=side.upper(),
                base_size=str(size),
                leverage=leverage,
                margin_type="CROSS"
            )
        
        logger.info(f"Order preview response: {preview}")
        
        # Check for preview errors
        if hasattr(preview, 'error_response'):
            error_msg = preview.error_response
            logger.error(f"Order preview failed: {error_msg}")
            return False, error_msg
            
        return True, None
        
    except Exception as e:
        logger.error(f"Error previewing order: {e}")
        return False, str(e)

def validate_params(product_id: str, side: str, size_usd: float, leverage: float, tp_price: float, sl_price: float, limit_price: float, cb_service):
    """Validate input parameters.

    Accept any BASE-PERP-INTX product id. If product is unknown to the
    exchange, later preview/order calls will fail with a clear message.
    """
    if '-PERP-' not in product_id.upper():
        raise ValueError("Product must be a perpetual, e.g., BASE-PERP-INTX")

    if side not in ['BUY', 'SELL']:
        raise ValueError("Side must be either 'BUY' or 'SELL'")
    
    if size_usd <= 0:
        raise ValueError("Position size must be positive")
    
    if not 1 <= leverage <= 100:
        raise ValueError("Leverage must be between 1 and 100")
    
    if tp_price <= 0:
        raise ValueError("Take profit price must be positive")
    
    if sl_price <= 0:
        raise ValueError("Stop loss price must be positive")
    
    if limit_price is not None and limit_price <= 0:
        raise ValueError("Limit price must be positive")
    
    # Get current price
    trades = cb_service.client.get_market_trades(product_id=product_id, limit=1)
    current_price = float(trades['trades'][0]['price'])
    logger.info(f"Current market price: ${current_price}")
    
    # Define maximum allowed price deviation (as percentage)
    max_deviation = 0.80  # 80% deviation
    
    # Calculate price bounds
    upper_bound = current_price * (1 + max_deviation)
    lower_bound = current_price * (1 - max_deviation)
    
    # Validate TP and SL based on side and price bounds
    if side == 'BUY':
        if tp_price <= sl_price:
            raise ValueError("For BUY orders, take profit price must be higher than stop loss price")
        if tp_price > upper_bound:
            raise ValueError(f"Take profit price (${tp_price}) is too high. Maximum allowed is ${upper_bound:.2f} (80% above current price ${current_price:.2f})")
        if sl_price < lower_bound:
            raise ValueError(f"Stop loss price (${sl_price}) is too low. Minimum allowed is ${lower_bound:.2f} (80% below current price ${current_price:.2f})")
        if limit_price is not None:
            if limit_price > current_price:
                raise ValueError(f"For BUY limit orders, limit price (${limit_price}) should be below current price (${current_price})")
            if limit_price <= sl_price:
                raise ValueError(f"For BUY limit orders, limit price (${limit_price}) should be above stop loss (${sl_price})")
            if limit_price >= tp_price:
                raise ValueError(f"For BUY limit orders, limit price (${limit_price}) should be below take profit (${tp_price})")
    else:  # SELL
        if tp_price >= sl_price:
            raise ValueError("For SELL orders, take profit price must be lower than stop loss price")
        if tp_price < lower_bound:
            raise ValueError(f"Take profit price (${tp_price}) is too low. Minimum allowed is ${lower_bound:.2f} (80% below current price ${current_price:.2f})")
        if sl_price > upper_bound:
            raise ValueError(f"Stop loss price (${sl_price}) is too high. Maximum allowed is ${upper_bound:.2f} (80% above current price ${current_price:.2f})")
        if limit_price is not None:
            if limit_price < current_price:
                raise ValueError(f"For SELL limit orders, limit price (${limit_price}) should be above current price (${current_price})")
            if limit_price >= sl_price:
                raise ValueError(f"For SELL limit orders, limit price (${limit_price}) should be below stop loss (${sl_price})")
            if limit_price <= tp_price:
                raise ValueError(f"For SELL limit orders, limit price (${limit_price}) should be above take profit (${tp_price})")

def get_min_base_size(product_id: str) -> float:
    """Get minimum base size for the given product via API if possible."""
    # Prefer cached constraints
    if product_id in _BASE_CONSTRAINTS_CACHE:
        return _BASE_CONSTRAINTS_CACHE[product_id][0]
    # Try auth fetch
    res = _fetch_base_constraints_auth(product_id) or _fetch_base_constraints_public(product_id)
    if res:
        _BASE_CONSTRAINTS_CACHE[product_id] = res
        return res[0]
    # Conservative default for INTX perps (most use integer contracts)
    return 1.0

def calculate_base_size(product_id: str, size_usd: float, current_price: float) -> float:
    """Calculate base size respecting min size and base increment.

    - Start from notional/price
    - Round to nearest base_increment
    - Enforce >= min_base_size
    """
    min_base_size, base_inc = get_base_constraints(product_id)
    est = max(size_usd / max(current_price, 1e-9), 0.0)
    # Quantize to increment (prefer rounding to nearest valid step)
    if base_inc > 0:
        steps = round(est / base_inc)
        base_size = steps * base_inc
    else:
        base_size = est
    # Ensure meets min size and aligned to increment
    if base_size < min_base_size:
        if base_inc > 0:
            steps = int((min_base_size + 1e-12) / base_inc)
            base_size = steps * base_inc
        else:
            base_size = min_base_size
    logger.info(f"Calculated base size: {base_size} (min: {min_base_size}, inc: {base_inc})")
    return base_size

def main():
    parser = argparse.ArgumentParser(description='Place a leveraged market or limit order for perpetual futures')
    parser.add_argument('--product', type=str, default='BTC-PERP-INTX',
                      help='Trading product (e.g., BTC-PERP-INTX, LINK-PERP-INTX, ETH-INTX-PERP)')
    parser.add_argument('--side', type=str, choices=['BUY', 'SELL'],
                      help='Trade direction (BUY/SELL)')
    parser.add_argument('--size', type=float,
                      help='Position size in USD')
    parser.add_argument('--leverage', type=float,
                      help='Leverage (1-20)')
    parser.add_argument('--tp', type=float,
                      help='Take profit price in USD')
    parser.add_argument('--sl', type=float,
                      help='Stop loss price in USD')
    parser.add_argument('--limit', type=float,
                      help='Limit price in USD (if not provided, a market order will be placed)')
    parser.add_argument('--no-confirm', action='store_true',
                      help='Skip order confirmation')
    # New arguments for placing bracket orders after fill
    parser.add_argument('--place-bracket', action='store_true',
                      help='Place bracket orders for an already filled limit order')
    parser.add_argument('--order-id', type=str,
                      help='Order ID of the filled limit order')

    args = parser.parse_args()

    # Normalize product id to BASE-PERP-INTX if user passed INTX-PERP or mixed
    def _normalize_perp(pid: str) -> str:
        s = (pid or '').upper().strip()
        parts = [p for p in s.split('-') if p]
        if not parts:
            return s
        base = parts[0]
        has_perp = any(p == 'PERP' for p in parts)
        has_intx = any(p == 'INTX' for p in parts)
        if has_perp and has_intx:
            return f"{base}-PERP-INTX"
        if has_perp and not has_intx:
            return f"{base}-PERP-INTX"
        if has_intx and not has_perp:
            return f"{base}-PERP-INTX"
        return f"{base}-PERP-INTX"
    args.product = _normalize_perp(args.product)

    try:
        # Initialize CoinbaseService
        cb_service = setup_coinbase()
        
        # Handle placing bracket orders after fill
        if args.place_bracket:
            if not all([args.order_id, args.product, args.size, args.tp, args.sl]):
                raise ValueError("For placing bracket orders, --order-id, --product, --size, --tp, and --sl are required")
            
            logger.info(f"Attempting to place bracket order for {args.product}")
            logger.info(f"Parameters: size={args.size}, tp={args.tp}, sl={args.sl}, leverage={args.leverage}")
            
            result = cb_service.place_bracket_after_fill(
                product_id=args.product,
                order_id=args.order_id,
                size=args.size,
                take_profit_price=args.tp,
                stop_loss_price=args.sl,
                leverage=str(args.leverage) if args.leverage else None
            )
            
            if "error" in result:
                if result.get("status") == "pending_fill":
                    logger.warning("Limit order not filled yet")
                    print("\nLimit order not filled yet. Please try again once the order is filled.")
                    return
                logger.error(f"Failed to place bracket orders. Full error: {result}")
                print(f"\nError placing bracket orders: {result['error']}")
                if 'bracket_error' in result:
                    logger.error(f"Bracket error details: {result['bracket_error']}")
                    print(f"Error details: {result['bracket_error']}")
                return
                
            logger.info("Bracket orders placed successfully")
            print("\nBracket orders placed successfully!")
            print(f"Take Profit Price: ${result['tp_price']}")
            print(f"Stop Loss Price: ${result['sl_price']}")
            return
        
        # Regular order placement flow
        if not all([args.side, args.size, args.leverage, args.tp, args.sl]):
            raise ValueError("For new orders, --side, --size, --leverage, --tp, and --sl are required")
        
        # Apply exchange tick-size rounding to TP/SL (and limit if provided)
        price_precision = get_price_precision(args.product)
        rounded_tp = round_to_precision(args.tp, price_precision)
        rounded_sl = round_to_precision(args.sl, price_precision)
        rounded_limit = round_to_precision(args.limit, price_precision) if args.limit else None
        if rounded_tp != args.tp or rounded_sl != args.sl or (args.limit and rounded_limit != args.limit):
            logger.info(
                f"Rounded prices to tick size {price_precision}: TP {args.tp}→{rounded_tp}, SL {args.sl}→{rounded_sl}"
                + (f", LIMIT {args.limit}→{rounded_limit}" if args.limit else "")
            )

        # Format prices to exactly allowed decimals to avoid precision errors
        tp_str = format_price_for_product(args.product, rounded_tp)
        sl_str = format_price_for_product(args.product, rounded_sl)
        limit_str = format_price_for_product(args.product, rounded_limit) if rounded_limit else None

        # Validate parameters (using rounded prices)
        validate_params(args.product, args.side, args.size, args.leverage, rounded_tp, rounded_sl, rounded_limit, cb_service)
        
        # Check for sufficient funds
        if not check_sufficient_funds(cb_service, args.size, args.leverage):
            raise ValueError("Insufficient funds for this trade")
        
        # Replace the current size calculation with the new one
        trades = cb_service.client.get_market_trades(product_id=args.product, limit=1)
        current_price = float(trades['trades'][0]['price'])
        size = calculate_base_size(args.product, args.size, current_price)
        
        # Preview the order
        is_valid, error_msg = preview_order(
            cb_service=cb_service,
            product_id=args.product,
            side=args.side,
            size=size,
            leverage=str(args.leverage),
            is_limit=rounded_limit is not None,
            limit_price=rounded_limit
        )
        
        if not is_valid:
            raise ValueError(f"Order preview failed: {error_msg}")
        
        # Show order summary
        print("\n=== Order Summary ===")
        print(f"Product: {args.product}")
        print(f"Side: {args.side}")
        print(f"Position Size: ${args.size} (≈{size} {args.product.split('-')[0]})")
        print(f"Leverage: {args.leverage}x")
        print(f"Required Margin: ${args.size / args.leverage}")
        print(f"Current Price: ${current_price}")
        print(f"Take Profit Price: ${tp_str}")
        print(f"Stop Loss Price: ${sl_str}")
        
        # Calculate and display potential profit/loss in dollars
        if args.side == 'BUY':
            potential_profit = (args.tp - current_price) / current_price * args.size
            potential_loss = (current_price - args.sl) / current_price * args.size
        else:  # SELL
            potential_profit = (current_price - args.tp) / current_price * args.size
            potential_loss = (args.sl - current_price) / current_price * args.size
            
        print(f"Potential Profit: ${potential_profit:.2f}")
        print(f"Potential Loss: ${potential_loss:.2f}")
        print(f"Risk/Reward Ratio: 1:{(potential_profit/potential_loss):.2f}")
        
        if rounded_limit:
            print(f"Limit Price: ${limit_str}")
            # Recalculate potential profit/loss based on limit price
            if args.side == 'BUY':
                limit_potential_profit = (rounded_tp - rounded_limit) / rounded_limit * args.size
                limit_potential_loss = (rounded_limit - rounded_sl) / rounded_limit * args.size
            else:  # SELL
                limit_potential_profit = (rounded_limit - rounded_tp) / rounded_limit * args.size
                limit_potential_loss = (rounded_sl - rounded_limit) / rounded_limit * args.size
                
            print(f"Potential Profit (from limit): ${limit_potential_profit:.2f}")
            print(f"Potential Loss (from limit): ${limit_potential_loss:.2f}")
            print(f"Risk/Reward Ratio (from limit): 1:{(limit_potential_profit/limit_potential_loss):.2f}")
        else:
            print("Order Type: Market")
        
        # Place the order based on order type
        if rounded_limit:
            result = cb_service.place_limit_order_with_targets(
                product_id=args.product,
                side=args.side,
                size=size,
                entry_price=limit_str,
                take_profit_price=tp_str,
                stop_loss_price=sl_str,
                leverage=str(args.leverage)
            )
            
            if "error" in result:
                print(f"\nError placing limit order: {result['error']}")
                return
                
            print("\nLimit order placed successfully!")
            print(f"Order ID: {result['order_id']}")
            print(f"Entry Price: ${result['entry_price']}")
            print(f"Status: {result['status']}")
            
            # Ask if user wants to monitor for fill
            if not args.no_confirm:
                monitor = input("\nWould you like to monitor the order until filled? (yes/no): ").lower()
                if monitor != 'yes':
                    print(f"\n{result['message']}")
                    print("\nTo place take profit and stop loss orders after fill, run:")
                    print(f"python trade_btc_perp.py --place-bracket --order-id {result['order_id']} --product {args.product} --size {size} --tp {rounded_tp} --sl {rounded_sl} --leverage {args.leverage}")
                    return
            
            print("\nMonitoring limit order for fill...")
            monitor_result = cb_service.monitor_limit_order_and_place_bracket(
                product_id=args.product,
                order_id=result['order_id'],
                size=size,
                take_profit_price=tp_str,
                stop_loss_price=sl_str,
                leverage=str(args.leverage)
            )
            
            if monitor_result['status'] == 'success':
                print(f"\n{monitor_result['message']}")
                print(f"Take Profit Price: ${monitor_result['tp_price']}")
                print(f"Stop Loss Price: ${monitor_result['sl_price']}")
            else:
                print(f"\n{monitor_result['message']}")
                if monitor_result['status'] == 'timeout':
                    print("\nTo place take profit and stop loss orders after fill, run:")
                    print(f"python trade_btc_perp.py --place-bracket --order-id {result['order_id']} --product {args.product} --size {size} --tp {rounded_tp} --sl {rounded_sl} --leverage {args.leverage}")
                elif monitor_result['status'] == 'error':
                    print(f"Error: {monitor_result.get('error', 'Unknown error')}")
            
        else:
            result = cb_service.place_market_order_with_targets(
                product_id=args.product,
                side=args.side,
                size=size,
                take_profit_price=rounded_tp,
                stop_loss_price=rounded_sl,
                leverage=str(args.leverage)
            )
            
            if "error" in result:
                print(f"\nError placing order: {result['error']}")
                # Surface any nested bracket preview/placement errors
                bracket_err = result.get('bracket_error')
                if isinstance(bracket_err, dict):
                    print(f"Bracket error details: {bracket_err}")
                elif isinstance(result['error'], dict):
                    print(f"Error details: {result['error'].get('message', 'No message')}")
                    print(f"Preview failure reason: {result['error'].get('preview_failure_reason', 'Unknown')}")
            else:
                print("\nOrder placed successfully!")
                print(f"Order ID: {result['order_id']}")
                print(f"Take Profit Price: ${result['tp_price']}")
                print(f"Stop Loss Price: ${result['sl_price']}")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main() 
