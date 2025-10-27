#!/usr/bin/env python3
"""
Shared helpers for resolving API credentials.

These functions centralise the environment/config fallback logic so every
module uses a consistent strategy.
"""

from __future__ import annotations

import os
from typing import Tuple

try:  # pragma: no cover - optional local config
    import config as _cfg  # type: ignore
except Exception:  # pragma: no cover
    _cfg = None


def _from_env_then_config(name: str) -> str:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    if _cfg is not None:
        return getattr(_cfg, name, "") or ""
    return ""


def get_perps_credentials() -> Tuple[str, str]:
    """Return (API_KEY_PERPS, API_SECRET_PERPS) using env first, then config."""

    key = _from_env_then_config("API_KEY_PERPS")
    secret = _from_env_then_config("API_SECRET_PERPS")
    return key, secret


def get_primary_credentials() -> Tuple[str, str]:
    """Return (API_KEY, API_SECRET) for spot endpoints."""

    key = _from_env_then_config("API_KEY")
    secret = _from_env_then_config("API_SECRET")
    return key, secret


def get_openai_api_key() -> str:
    """Return the OpenAI API key using env first, then config."""

    return _from_env_then_config("OPENAI_KEY")


__all__ = ["get_perps_credentials", "get_primary_credentials", "get_openai_api_key"]
