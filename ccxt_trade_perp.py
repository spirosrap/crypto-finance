#!/usr/bin/env python3
"""
Place Coinbase Advanced perpetual futures orders (market/limit) using CCXT, with optional bracket
take-profit and stop-loss legs.

Example usage (market long with bracket):
    python ccxt_trade_perp.py --product BTC-PERP-INTX --side BUY --size 200 \
        --leverage 5 --tp 120000 --sl 105000

Example usage (limit short, dry run):
    python ccxt_trade_perp.py --product BTC-PERP-INTX --side SELL --size 150 \
        --leverage 5 --tp 98000 --sl 110000 --limit 109500 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import ccxt  # type: ignore

try:
    from credentials import get_perps_credentials
except ImportError:  # pragma: no cover
    def get_perps_credentials() -> Tuple[str, str]:
        return os.getenv("API_KEY_PERPS", ""), os.getenv("API_SECRET_PERPS", "")


logger = logging.getLogger("ccxt_trade_perp")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _apply_ccxt_negative_index_guard() -> None:
    """Prevent CCXT safe_* helpers from indexing empty lists with -1."""
    try:
        from ccxt.base.exchange import Exchange  # type: ignore
    except Exception:
        return
    if getattr(Exchange, "_negative_index_guard", False):
        return

    def _guarded_get_object_value(dictionary_or_list, key_list):
        is_data_array = isinstance(dictionary_or_list, list)
        is_data_dict = isinstance(dictionary_or_list, dict)
        for key in key_list:
            if is_data_dict:
                if key in dictionary_or_list and dictionary_or_list[key] not in (None, ""):
                    return dictionary_or_list[key]
            elif is_data_array and not isinstance(key, str):
                if key < 0:
                    continue
                if (key < len(dictionary_or_list)) and (dictionary_or_list[key] is not None) and (dictionary_or_list[key] != ""):
                    return dictionary_or_list[key]
        return None

    Exchange.get_object_value_from_key_list = staticmethod(_guarded_get_object_value)
    Exchange._negative_index_guard = True


@dataclass
class MarketMeta:
    symbol: str
    ccxt_symbol: str
    market: Dict
    price_precision: float
    amount_precision: float
    min_base_size: float


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MARKETS_CACHE_PATH = os.path.join(_REPO_ROOT, "cache", "coinbaseadvanced_markets.json")
MARGIN_CACHE_PATH = os.path.join(_REPO_ROOT, "cache", "perps_margin_cache.json")


def _save_markets_cache(markets: Dict) -> None:
    try:
        os.makedirs(os.path.dirname(MARKETS_CACHE_PATH), exist_ok=True)
        with open(MARKETS_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(markets, handle)
    except Exception as exc:
        logger.warning("Failed to write markets cache: %s", exc)


def _load_markets_cache() -> Optional[Dict]:
    if not os.path.exists(MARKETS_CACHE_PATH):
        return None
    try:
        with open(MARKETS_CACHE_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read markets cache: %s", exc)
        return None


def _save_margin_cache(value: float) -> None:
    try:
        os.makedirs(os.path.dirname(MARGIN_CACHE_PATH), exist_ok=True)
        payload = {"value": value, "ts": time.time()}
        with open(MARGIN_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception as exc:
        logger.warning("Failed to write margin cache: %s", exc)


def _load_margin_cache(max_age_seconds: int = 900) -> Optional[float]:
    if not os.path.exists(MARGIN_CACHE_PATH):
        return None
    try:
        with open(MARGIN_CACHE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        value = float(payload.get("value"))
        ts = float(payload.get("ts"))
        if time.time() - ts <= max_age_seconds:
            return value
    except Exception as exc:
        logger.warning("Failed to read margin cache: %s", exc)
    return None
    try:
        with open(MARKETS_CACHE_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read markets cache: %s", exc)
        return None


def load_exchange() -> ccxt.Exchange:
    """Initialise a CCXT Coinbase Advanced client with credentials."""
    _apply_ccxt_negative_index_guard()
    key_env = os.getenv("COINBASE_PERP_API_KEY")
    secret_env = os.getenv("COINBASE_PERP_API_SECRET")
    key_cfg, secret_cfg = get_perps_credentials()
    api_key = key_env or key_cfg
    api_secret = secret_env or secret_cfg
    if not api_key or not api_secret:
        raise RuntimeError("Missing API credentials for Coinbase perps (set API_KEY_PERPS/API_SECRET_PERPS).")
    cached = _load_markets_cache()
    if cached:
        exchange = ccxt.coinbaseadvanced(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )
        exchange.timeout = 30000
        exchange.options.setdefault("fetchCurrencies", False)
        try:
            if hasattr(exchange, "set_markets"):
                exchange.set_markets(cached)
            else:
                exchange.markets = cached
                exchange.markets_by_id = {m.get("id"): m for m in cached.values() if isinstance(m, dict)}
            if os.getenv("SKIP_LOAD_MARKETS") == "1":
                logger.warning("Using cached markets (SKIP_LOAD_MARKETS=1).")
                return exchange
        except Exception as exc:
            logger.warning("Failed to apply markets cache: %s", exc)

    for attempt in range(1, 4):
        try:
            exchange = ccxt.coinbaseadvanced(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                }
            )
            exchange.timeout = 30000
            exchange.options.setdefault("fetchCurrencies", False)
            if os.getenv("CCXT_VERBOSE") == "1":
                exchange.verbose = True
            exchange.load_markets()
            _save_markets_cache(exchange.markets)
            return exchange
        except Exception as exc:
            logger.warning("load_markets attempt %d failed: %s", attempt, exc)
            time.sleep(2 ** (attempt - 1))
    if cached:
        try:
            logger.warning("Using cached markets after load_markets failures.")
            return exchange
        except Exception as exc:
            logger.warning("Failed to apply markets cache: %s", exc)
    raise RuntimeError("Failed to load markets after 3 attempts.")


def compute_end_time(expiry: str) -> str:
    """Return ISO8601Z string for GTD expiries."""
    expiry = (expiry or "30d").lower()
    now = time.time()
    if expiry == "12h":
        delta = 12 * 3600
    elif expiry == "24h":
        delta = 24 * 3600
    elif expiry in ("30d", "gtd", "gtc"):
        delta = 30 * 86400
    else:
        delta = 30 * 86400
    end_ts = now + delta
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_ts))


def normalize_symbol(product: str) -> str:
    """Convert repo-style product ids to CCXT symbols."""
    if "/" in product:
        return product
    product = product.strip().upper()
    if product.endswith("-PERP-INTX"):
        base = product.split("-")[0]
        return f"{base}/USDC:USDC"
    raise ValueError(f"Unrecognised product format: {product}")


def get_market_meta(exchange: ccxt.Exchange, product: str) -> MarketMeta:
    ccxt_symbol = normalize_symbol(product)
    market = exchange.market(ccxt_symbol)
    price_precision = market["precision"].get("price") or 0.01
    amount_precision = market["precision"].get("amount") or 1.0
    min_base_size = market["limits"]["amount"].get("min") or amount_precision
    return MarketMeta(
        symbol=product,
        ccxt_symbol=ccxt_symbol,
        market=market,
        price_precision=price_precision,
        amount_precision=amount_precision,
        min_base_size=min_base_size,
    )


def round_to_increment(value: float, increment: float) -> float:
    if increment is None or increment <= 0:
        return value
    steps = round(value / increment)
    return steps * increment


def clamp_precision(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.trunc(value * factor) / factor


def quantize_price(price: float, precision: float) -> float:
    if precision >= 1:
        return round(price / precision) * precision
    decimals = max(0, -int(round(math.log10(precision))))
    return clamp_precision(round(price / precision) * precision, decimals)


def calculate_base_size(size_usd: float, price: float, meta: MarketMeta) -> float:
    est = size_usd / max(price, 1e-9)
    est = max(est, meta.min_base_size)
    base = round_to_increment(est, meta.amount_precision)
    if base < meta.min_base_size:
        steps = math.ceil(meta.min_base_size / meta.amount_precision)
        base = steps * meta.amount_precision
    return base


def fetch_reference_price(exchange: ccxt.Exchange, ccxt_symbol: str) -> float:
    try:
        ticker = exchange.fetch_ticker(ccxt_symbol)
        last = ticker.get("last") or ticker.get("close")
        if last is not None:
            return float(last)
    except Exception as exc:
        logger.warning("fetch_ticker failed for %s: %s", ccxt_symbol, exc)

    try:
        ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe="1m", limit=2)
        if ohlcv:
            close = ohlcv[-1][4]
            if close is not None:
                return float(close)
    except Exception as exc:
        logger.warning("fetch_ohlcv failed for %s: %s", ccxt_symbol, exc)

    try:
        book = exchange.fetch_order_book(ccxt_symbol, limit=1)
        bid = book.get("bids") or []
        ask = book.get("asks") or []
        best_bid = bid[0][0] if bid else None
        best_ask = ask[0][0] if ask else None
        if best_bid is not None and best_ask is not None:
            return (float(best_bid) + float(best_ask)) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
    except Exception as exc:
        logger.warning("fetch_order_book failed for %s: %s", ccxt_symbol, exc)

    raise RuntimeError("Unable to fetch reference price from ticker/ohlcv/order book.")


def fetch_perp_available_usdc(exchange: ccxt.Exchange) -> float:
    """Return available USDC in the INTX (perpetuals) portfolio."""
    total = 0.0
    try:
        accounts = exchange.v3PrivateGetBrokerageAccounts({"limit": 250})
        for account in accounts.get("accounts", []):
            if account.get("currency") != "USDC":
                continue
            platform = account.get("platform") or ""
            if "INTX" not in platform.upper():
                continue
            balance = account.get("available_balance") or {}
            try:
                total += float(balance.get("value") or 0.0)
            except (TypeError, ValueError):
                continue
    except ccxt.AuthenticationError as exc:
        raise RuntimeError(f"Authentication failed when fetching accounts: {exc}") from exc
    except Exception as exc:
        logger.warning("Failed to fetch INTX accounts via v3 API: %s", exc)

    if total > 0:
        _save_margin_cache(total)
        return total

    # Fallback: query INTX portfolio via Coinbase REST client.
    try:
        from coinbase.rest import RESTClient, portfolios
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Coinbase REST client unavailable for INTX balance: %s", exc)
    else:
        try:
            api_key, api_secret = get_perps_credentials()
            if not api_key or not api_secret:
                raise RuntimeError("Missing perps API credentials for REST client.")
            client = RESTClient(api_key=api_key, api_secret=api_secret)
            port_resp = portfolios.get_portfolios(client)
            intx = next((p for p in port_resp.portfolios if getattr(p, "type", None) == "INTX"), None)
            if not intx:
                raise RuntimeError("INTX portfolio not found in REST response.")
            breakdown = portfolios.get_portfolio_breakdown(client, portfolio_uuid=getattr(intx, "uuid", None))
            bd = getattr(breakdown, "breakdown", None)
            balances = getattr(bd, "portfolio_balances", None)
            if isinstance(balances, dict):
                total_balance = balances.get("total_balance", {})
                value = total_balance.get("value")
                if value is not None:
                    total = float(value)
                    _save_margin_cache(total)
                    logger.info("Available USDC margin (INTX total balance): %.2f", total)
                    return total
        except Exception as exc:
            logger.warning("Fallback INTX portfolio balance failed: %s", exc)

    # Fallback: use generic balance endpoint (may exclude INTX portfolios)
    try:
        balance = exchange.fetch_balance({"type": "swap"})
        value = float(balance.get("free", {}).get("USDC") or 0.0)
        if value > 0:
            _save_margin_cache(value)
        return value
    except Exception as exc:
        logger.warning("Fallback fetch_balance failed: %s", exc)
        cached = _load_margin_cache()
        if cached is not None:
            logger.warning("Using cached margin value (%.2f USDC).", cached)
            return cached
        return 0.0


def ensure_margin_balance(exchange: ccxt.Exchange, required_margin: float) -> None:
    available = fetch_perp_available_usdc(exchange)
    logger.info("Available USDC margin: %.2f", available)
    if available < required_margin:
        raise RuntimeError(f"Insufficient margin. Need {required_margin:.2f} USDC, have {available:.2f} USDC.")


def place_entry_order(
    exchange: ccxt.Exchange,
    meta: MarketMeta,
    side: str,
    amount: float,
    price: Optional[float],
    leverage: float,
    expiry: str,
    dry_run: bool,
) -> Dict:
    if price is None:
        payload = {
            "client_order_id": f"entry-{int(time.time()*1000)}",
            "product_id": meta.market["id"],
            "side": side,
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": exchange.amount_to_precision(meta.ccxt_symbol, amount),
                }
            },
        }
        if leverage:
            payload["leverage"] = str(leverage)
            payload["margin_type"] = "CROSS"

        if dry_run:
            logger.info("[Dry run] submit entry payload: %s", payload)
            return {"status": "dry_run"}

        response = exchange.v3PrivatePostBrokerageOrders(payload)
        if not bool(response.get("success", True)):
            raise RuntimeError(f"Entry order rejected: {response.get('error_response', response)}")
        logger.info("Entry order placed: %s", response.get("order_id", response))
        return response

    payload = {
        "client_order_id": f"entry-{int(time.time()*1000)}",
        "product_id": meta.market["id"],
        "side": side,
        "order_configuration": {},
    }
    order_config = {
        "base_size": exchange.amount_to_precision(meta.ccxt_symbol, amount),
        "limit_price": exchange.price_to_precision(meta.ccxt_symbol, price),
        "post_only": False,
    }
    if expiry != "GTC":
        order_config["end_time"] = compute_end_time(expiry)
        payload["order_configuration"]["limit_limit_gtd"] = order_config
    else:
        payload["order_configuration"]["limit_limit_gtc"] = order_config

    if leverage:
        payload["leverage"] = str(leverage)
        payload["margin_type"] = "CROSS"

    if dry_run:
        logger.info("[Dry run] submit limit entry payload: %s", payload)
        return {"status": "dry_run"}

    response = exchange.v3PrivatePostBrokerageOrders(payload)
    if not bool(response.get("success", True)):
        raise RuntimeError(f"Entry order rejected: {response.get('error_response', response)}")
    logger.info("Entry order placed: %s", response.get("order_id", response))
    return response


def place_trigger_bracket_order(
    exchange: ccxt.Exchange,
    meta: MarketMeta,
    entry_side: str,
    base_size: float,
    tp_price: float,
    sl_price: float,
    leverage: float,
    expiry: str,
    dry_run: bool,
) -> Dict:
    bracket_side = "SELL" if entry_side == "BUY" else "BUY"
    payload = {
        "client_order_id": f"bracket-{int(time.time()*1000)}",
        "product_id": meta.market["id"],
        "side": bracket_side,
        "order_configuration": {
            "trigger_bracket_gtd": {
                "base_size": exchange.amount_to_precision(meta.ccxt_symbol, base_size),
                "limit_price": exchange.price_to_precision(meta.ccxt_symbol, tp_price),
                "stop_trigger_price": exchange.price_to_precision(meta.ccxt_symbol, sl_price),
                "end_time": compute_end_time(expiry if expiry != "GTC" else "30d"),
            }
        },
    }
    if leverage:
        payload["leverage"] = str(leverage)
        payload["margin_type"] = "CROSS"

    if dry_run:
        logger.info("[Dry run] submit bracket order payload: %s", payload)
        return {"status": "dry_run"}

    response = exchange.v3PrivatePostBrokerageOrders(payload)
    if not bool(response.get("success", True)):
        raise RuntimeError(f"Bracket order rejected: {response.get('error_response', response)}")
    logger.info("Bracket order response: %s", response)
    return response


def place_market_order_with_targets_rest(
    product_id: str,
    side: str,
    base_size: float,
    tp_price: float,
    sl_price: float,
    leverage: float,
    expiry: str,
    tp1_price: Optional[float] = None,
    tp1_pct: float = 0.0,
) -> Dict:
    try:
        from coinbaseservice import CoinbaseService
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Coinbase REST client unavailable: {exc}") from exc

    api_key, api_secret = get_perps_credentials()
    if not api_key or not api_secret:
        raise RuntimeError("Missing perps API credentials for Coinbase REST client.")

    service = CoinbaseService(api_key=api_key, api_secret=api_secret)
    result = service.place_market_order_with_targets(
        product_id=product_id,
        side=side,
        size=base_size,
        take_profit_price=tp_price,
        stop_loss_price=sl_price,
        leverage=str(leverage) if leverage else None,
        expiry=expiry,
        tp1_price=tp1_price,
        tp1_pct=tp1_pct,
    )
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(f"REST market+bracket order failed: {result['error']}")
    logger.info("REST market+bracket order placed for %s (%s).", product_id, side)
    return {"success": True, "via": "rest", "response": result}


def _summarize_ccxt_response(response: object, max_len: int = 600) -> str:
    parsed = None
    if isinstance(response, (bytes, bytearray)):
        response = response.decode("utf-8", errors="replace")
    if isinstance(response, str):
        text = response.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
        if parsed is None:
            summary = text.replace("\n", " ").strip()
            return summary if len(summary) <= max_len else f"{summary[:max_len]}…(truncated)"

    if parsed is None:
        parsed = response

    if isinstance(parsed, dict):
        if "products" in parsed:
            products = parsed.get("products") or []
            sample_ids = []
            for product in products[:5]:
                if isinstance(product, dict):
                    pid = product.get("product_id")
                    if pid:
                        sample_ids.append(pid)
            count = parsed.get("num_products") or len(products)
            summary = f"products={count} sample_ids={sample_ids}"
        elif "trades" in parsed:
            trades = parsed.get("trades") or []
            best_bid = parsed.get("best_bid")
            best_ask = parsed.get("best_ask")
            summary = f"trades={len(trades)} best_bid={best_bid} best_ask={best_ask}"
        else:
            summary = json.dumps(parsed, separators=(",", ":"), default=str)
    elif isinstance(parsed, list):
        sample = parsed[:3]
        summary = f"list[{len(parsed)}] sample={sample}"
    else:
        summary = str(parsed)

    summary = summary.replace("\n", " ").strip()
    if len(summary) > max_len:
        summary = f"{summary[:max_len]}…(truncated)"
    return summary


def _log_ccxt_failure(exchange: ccxt.Exchange, exc: Exception) -> None:
    logger.warning("CCXT exception: %s", exc)
    try:
        last_url = getattr(exchange, "last_request_url", None)
        last_body = getattr(exchange, "last_request_body", None)
        last_resp = getattr(exchange, "last_http_response", None)
        if last_url:
            logger.warning("CCXT last_request_url: %s", last_url)
        if last_body:
            logger.warning("CCXT last_request_body: %s", last_body)
        if last_resp:
            summary = _summarize_ccxt_response(last_resp)
            logger.warning("CCXT last_http_response (summary): %s", summary)
            if os.getenv("CCXT_LOG_FULL_RESPONSE") == "1":
                logger.warning("CCXT last_http_response (full): %s", last_resp)
    except Exception:
        logger.warning("Unable to read CCXT last_* diagnostics.")


def wait_for_fill(
    exchange: ccxt.Exchange,
    symbol: str,
    order_id: Optional[str],
    timeout: int = 15,
    poll_interval: float = 1.0,
) -> Dict:
    if not order_id:
        return {}
    deadline = time.time() + timeout
    last_order = {}
    while time.time() < deadline:
        try:
            last_order = exchange.fetch_order(order_id, symbol)
        except Exception as exc:
            logger.debug("fetch_order failed for %s: %s", order_id, exc)
            time.sleep(poll_interval)
            continue
        status = last_order.get("status")
        if status in ("closed", "canceled"):
            break
        time.sleep(poll_interval)
    return last_order


def main() -> None:
    parser = argparse.ArgumentParser(description="Trade Coinbase perps using CCXT with optional bracket orders.")
    parser.add_argument("--product", default="BTC-PERP-INTX", help="Perpetual product id (e.g., BTC-PERP-INTX or CCXT symbol).")
    parser.add_argument("--side", choices=["BUY", "SELL"], required=True, help="Entry side.")
    parser.add_argument("--size", type=float, required=True, help="Position size in USD notional.")
    parser.add_argument("--leverage", type=float, default=5.0, help="Leverage (default 5x).")
    parser.add_argument("--tp", type=float, required=True, help="Take-profit price.")
    parser.add_argument("--sl", type=float, required=True, help="Stop-loss trigger price.")
    parser.add_argument("--tp1", type=float, help="Partial take-profit price (optional).")
    parser.add_argument("--tp1-pct", type=float, default=0.0, help="Percent of size to close at TP1.")
    parser.add_argument("--tp1-move-sl", action="store_true", help="Move SL to entry after TP1 (not supported yet).")
    parser.add_argument("--no-rest-fallback", action="store_true", help="Disable REST fallback on CCXT entry failure.")
    parser.add_argument("--limit", type=float, help="Optional entry limit price (omit for market).")
    parser.add_argument("--expiry", choices=["GTC", "12h", "24h", "30d"], default="30d", help="GTD expiry horizon for limit entries.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without placing orders.")
    args = parser.parse_args()

    try:
        exchange = load_exchange()
        if os.getenv("SKIP_LOAD_MARKETS") == "1" and (args.limit is not None or args.tp1 is not None):
            try:
                exchange.load_markets()
                logger.warning("Loaded fresh markets to support limit/TP1 entries.")
            except Exception as exc:
                logger.warning("load_markets failed under SKIP_LOAD_MARKETS=1: %s", exc)
        meta = get_market_meta(exchange, args.product)
        current_price = fetch_reference_price(exchange, meta.ccxt_symbol)
        entry_price = args.limit or current_price

        base_size = calculate_base_size(args.size, entry_price, meta)
        tp_price = quantize_price(args.tp, meta.price_precision)
        sl_price = quantize_price(args.sl, meta.price_precision)
        tp1_pct = float(args.tp1_pct or 0.0)
        tp1_price = None
        if args.tp1 is not None or tp1_pct > 0:
            if args.tp1 is None or tp1_pct <= 0:
                raise RuntimeError("TP1 requires both --tp1 and --tp1-pct > 0.")
            if tp1_pct <= 0 or tp1_pct >= 100:
                raise RuntimeError("--tp1-pct must be between 0 and 100 (exclusive).")
            tp1_price = quantize_price(float(args.tp1), meta.price_precision)
            logger.info("TP1: %.2f (%s%% of size)", tp1_price, tp1_pct)
            if args.no_rest_fallback:
                logger.warning("REST fallback disabled; TP1 requires CCXT order flow.")
        if args.tp1_move_sl:
            logger.warning("TP1 move-SL is not supported yet for live orders; ignoring --tp1-move-sl.")

        logger.info("Reference price: %.2f", current_price)
        logger.info("Computed base size: %.8f %s", base_size, meta.market["base"])
        logger.info("Take profit: %.2f, Stop loss: %.2f", tp_price, sl_price)

        required_margin = args.size / max(args.leverage, 1e-9)
        ensure_margin_balance(exchange, required_margin)

        potential_profit = (tp_price - entry_price) / entry_price * args.size if args.side == "BUY" else (entry_price - tp_price) / entry_price * args.size
        potential_loss = (entry_price - sl_price) / entry_price * args.size if args.side == "BUY" else (sl_price - entry_price) / entry_price * args.size
        logger.info("Potential profit (USD): %.2f, potential loss (USD): %.2f", potential_profit, abs(potential_loss))
        if abs(potential_loss) < 1e-6:
            raise RuntimeError("Stop loss too close to entry; risk zero.")

        skip_bracket = False
        entry_order = {}
        try:
            entry_order = place_entry_order(
                exchange,
                meta,
                args.side,
                base_size,
                quantize_price(args.limit, meta.price_precision) if args.limit else None,
                args.leverage,
                args.expiry,
                args.dry_run,
            )
        except Exception as exc:
            if (not args.dry_run) and ("index out of range" in str(exc)):
                _log_ccxt_failure(exchange, exc)
                logger.warning("Retrying CCXT entry after index error with fresh markets.")
                try:
                    exchange.load_markets(reload=True)
                    meta = get_market_meta(exchange, args.product)
                    base_size = calculate_base_size(args.size, entry_price, meta)
                    tp_price = quantize_price(args.tp, meta.price_precision)
                    sl_price = quantize_price(args.sl, meta.price_precision)
                    if tp1_price is not None:
                        tp1_price = quantize_price(float(args.tp1), meta.price_precision)
                except Exception as refresh_exc:
                    logger.warning("CCXT refresh before retry failed: %s", refresh_exc)
                try:
                    entry_order = place_entry_order(
                        exchange,
                        meta,
                        args.side,
                        base_size,
                        quantize_price(args.limit, meta.price_precision) if args.limit else None,
                        args.leverage,
                        args.expiry,
                        args.dry_run,
                    )
                except Exception as retry_exc:
                    if (not args.no_rest_fallback) and ("index out of range" in str(retry_exc)):
                        _log_ccxt_failure(exchange, retry_exc)
                        logger.warning("CCXT entry still failing; falling back to Coinbase REST.")
                        if tp1_price is not None:
                            logger.warning("Using REST fallback with TP1 split brackets.")
                        entry_order = place_market_order_with_targets_rest(
                            product_id=meta.market["id"],
                            side=args.side,
                            base_size=base_size,
                            tp_price=tp_price,
                            sl_price=sl_price,
                            leverage=args.leverage,
                            expiry=args.expiry,
                            tp1_price=tp1_price,
                            tp1_pct=tp1_pct,
                        )
                        skip_bracket = True
                    else:
                        raise
            else:
                raise

        bracket_response = {}
        if not args.dry_run and not skip_bracket:
            entry_order_id = entry_order.get("id")
            entry_status = entry_order.get("status")
            if entry_order_id and entry_status != "closed":
                logger.info("Waiting for entry order %s to fill before submitting bracket…", entry_order_id)
                final_state = wait_for_fill(exchange, meta.ccxt_symbol, entry_order_id)
                if final_state and final_state.get("status") != "closed":
                    logger.warning("Entry order %s not filled (status=%s); bracket submission skipped.", entry_order_id, final_state.get("status"))
                    bracket_response = {"error": "entry_not_filled"}
                else:
                    bracket_response = {}
                    if tp1_price is not None and tp1_pct > 0:
                        size_tp1 = base_size * (tp1_pct / 100.0)
                        size_tp2 = base_size - size_tp1
                        if size_tp1 > 0:
                            bracket_response["tp1"] = place_trigger_bracket_order(
                                exchange,
                                meta,
                                args.side,
                                size_tp1,
                                tp1_price,
                                sl_price,
                                args.leverage,
                                args.expiry,
                                args.dry_run,
                            )
                        if size_tp2 > 0:
                            bracket_response["tp2"] = place_trigger_bracket_order(
                                exchange,
                                meta,
                                args.side,
                                size_tp2,
                                tp_price,
                                sl_price,
                                args.leverage,
                                args.expiry,
                                args.dry_run,
                            )
                    else:
                        bracket_response = place_trigger_bracket_order(
                            exchange,
                            meta,
                            args.side,
                            base_size,
                            tp_price,
                            sl_price,
                            args.leverage,
                            args.expiry,
                            args.dry_run,
                        )
            else:
                bracket_response = {}
                if tp1_price is not None and tp1_pct > 0:
                    size_tp1 = base_size * (tp1_pct / 100.0)
                    size_tp2 = base_size - size_tp1
                    if size_tp1 > 0:
                        bracket_response["tp1"] = place_trigger_bracket_order(
                            exchange,
                            meta,
                            args.side,
                            size_tp1,
                            tp1_price,
                            sl_price,
                            args.leverage,
                            args.expiry,
                            args.dry_run,
                        )
                    if size_tp2 > 0:
                        bracket_response["tp2"] = place_trigger_bracket_order(
                            exchange,
                            meta,
                            args.side,
                            size_tp2,
                            tp_price,
                            sl_price,
                            args.leverage,
                            args.expiry,
                            args.dry_run,
                        )
                else:
                    bracket_response = place_trigger_bracket_order(
                        exchange,
                        meta,
                        args.side,
                        base_size,
                        tp_price,
                        sl_price,
                        args.leverage,
                        args.expiry,
                        args.dry_run,
                    )
    except Exception as exc:
        context = (
            f"product={args.product} side={args.side} size={args.size} "
            f"leverage={args.leverage} tp={args.tp} sl={args.sl} "
            f"limit={args.limit} expiry={args.expiry}"
        )
        raise RuntimeError(f"Trade failed ({context}): {exc}") from exc

    print("\n=== Order Summary ===")
    print(f"Entry order: {entry_order}")
    if args.dry_run:
        print("Bracket response: dry run (no orders sent).")
        print("\nDry run complete. No orders were sent.")
    elif entry_order.get("via") == "rest":
        print("Bracket response: placed via REST (see entry_order['response']).")
    elif bracket_response:
        print(f"Bracket response: {bracket_response}")
    else:
        print("Bracket response: skipped (entry not filled or not submitted).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Error: %s", exc)
        sys.exit(1)
