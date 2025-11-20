#!/usr/bin/env python3
"""
Quick sanity checker for CCXT credential loading.

This utility verifies that API keys are visible to the process, that the
Coinbase Advanced secret can be parsed as a PEM-encoded EC private key, and
optionally exercises `load_markets` to mirror the finder startup flow.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv() -> bool:
    """Best-effort .env load; returns True if dotenv was invoked."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return False
    try:
        load_dotenv()
        return True
    except Exception:
        return False


def _get_credentials() -> Tuple[str, str]:
    """Retrieve API credentials via the shared helper to mirror app behaviour."""
    try:
        from credentials import get_primary_credentials
    except Exception:
        return "", ""
    return get_primary_credentials()


def _secret_info(secret: str) -> str:
    """Return a short, non-sensitive summary of the secret string."""
    if not secret:
        return "empty"
    parts = [f"length={len(secret)}"]
    if "\\n" in secret:
        parts.append("contains literal \\n")
    if "\n" in secret:
        parts.append("contains newline characters")
    return ", ".join(parts)


def _can_parse_secret(secret: str) -> Tuple[bool, str]:
    """Attempt to parse the secret as a PEM EC private key using ccxt deps."""
    if not secret:
        return False, "API_SECRET is empty or not set"
    try:
        from ccxt.static_dependencies import ecdsa  # type: ignore
    except Exception:
        try:
            import ecdsa  # type: ignore
        except Exception:
            return False, "ecdsa dependency not available; install ccxt or ecdsa"

    def _parse(candidate: str) -> Tuple[bool, str]:
        try:
            ecdsa.SigningKey.from_pem(candidate.encode())
            return True, "Secret parsed as PEM EC private key"
        except Exception as exc:
            return False, f"Failed to parse: {exc}"

    ok, msg = _parse(secret)
    if ok:
        return True, msg

    # Retry with escaped newlines converted to real newlines for clearer hints.
    if "\\n" in secret:
        alt = secret.replace("\\n", "\n")
        alt_ok, alt_msg = _parse(alt)
        if alt_ok:
            return False, "Secret failed as-is but parsed after \\n -> newline conversion; update .env quoting"
        return False, f"{msg} (also failed after newline conversion: {alt_msg})"

    return False, msg


def _test_load_markets(exchange_id: str, api_key: str, api_secret: str) -> Tuple[bool, str]:
    """Optionally attempt ccxt.load_markets; requires network access."""
    try:
        import ccxt  # type: ignore
    except Exception as exc:
        return False, f"ccxt not available: {exc}"

    try:
        exchange_cls = getattr(ccxt, exchange_id)
    except AttributeError:
        return False, f"Unsupported exchange id '{exchange_id}'"

    params = {"enableRateLimit": True}
    if api_key and api_secret:
        params["apiKey"] = api_key
        params["secret"] = api_secret

    try:
        ex = exchange_cls(params)
        ex.load_markets()
        return True, f"load_markets() succeeded for {exchange_id}"
    except Exception as exc:
        return False, f"load_markets() failed for {exchange_id}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that CCXT credentials are correctly formatted and optionally reachable."
    )
    parser.add_argument(
        "--exchange",
        default=os.getenv("CRYPTO_CCXT_EXCHANGE_ID", "coinbaseadvanced"),
        help="CCXT exchange id to test (default: coinbaseadvanced or CRYPTO_CCXT_EXCHANGE_ID)",
    )
    parser.add_argument(
        "--load-markets",
        action="store_true",
        help="Also call load_markets() to mirror finder init (requires network access)",
    )
    parser.add_argument(
        "--skip-dotenv",
        action="store_true",
        help="Do not auto-load .env (default: load if python-dotenv present)",
    )
    args = parser.parse_args()

    dotenv_loaded = False
    if not args.skip_dotenv:
        dotenv_loaded = _load_dotenv()

    api_key, api_secret = _get_credentials()

    print(f"[info] dotenv loaded: {dotenv_loaded}")
    print(f"[info] API_KEY present: {bool(api_key)}")
    print(f"[info] API_SECRET summary: {_secret_info(api_secret)}")

    secret_ok, secret_msg = _can_parse_secret(api_secret)
    print(f"[check] secret parse: {secret_msg}")

    exit_code = 0
    if not secret_ok:
        exit_code = 1

    if args.load_markets:
        lm_ok, lm_msg = _test_load_markets(args.exchange.lower(), api_key, api_secret)
        print(f"[check] load_markets: {lm_msg}")
        if not lm_ok:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
