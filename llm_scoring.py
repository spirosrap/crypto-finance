"""Utilities for refining finder scores with OpenAI models."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

try:  # Lazy optional import to keep module usable without openai installed
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - handled gracefully via attribute guards
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_dict(candidate: Any) -> Dict[str, Any]:
    if candidate is None:
        return {}
    if isinstance(candidate, dict):
        return dict(candidate)
    if is_dataclass(candidate):
        return asdict(candidate)
    # Fallback: look for attributes
    data: Dict[str, Any] = {}
    for attr in dir(candidate):
        if attr.startswith("_"):
            continue
        try:
            data[attr] = getattr(candidate, attr)
        except Exception:
            continue
    return data


class LLMScoringDisabled(RuntimeError):
    """Raised when LLM scoring is requested but unable to initialize."""


class LLMScorer:
    """Thin wrapper around the OpenAI API to refine candidate scores."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini",
        weight: float = 0.25,
        max_candidates: int = 12,
        temperature: Optional[float] = None,
        max_retries: int = 2,
        request_timeout: Optional[float] = 45.0,
        client: Optional[Any] = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self.model = model
        self.weight = float(max(0.0, min(1.0, weight)))
        self.max_candidates = int(max(1, max_candidates))
        temp_value: Optional[float]
        if temperature is None:
            temp_value = None
        else:
            try:
                temp_value = float(temperature)
            except (TypeError, ValueError):
                temp_value = None
        if temp_value is not None and temp_value <= 0.0:
            temp_value = None
        self.temperature = temp_value
        self.max_retries = max(0, max_retries)
        self.request_timeout = request_timeout
        self.sleep_seconds = max(0.0, sleep_seconds)

        resolved_key = api_key or os.getenv("OPENAI_KEY")
        if not resolved_key:
            try:
                from config import OPENAI_KEY as CONFIG_KEY  # type: ignore
            except Exception:
                CONFIG_KEY = None
            if CONFIG_KEY:
                resolved_key = CONFIG_KEY

        self.api_key = resolved_key

        if client is not None:
            self.client = client
        elif OpenAI and resolved_key:
            self.client = OpenAI(api_key=resolved_key)  # type: ignore[call-arg]
        else:
            self.client = None

        self.enabled = bool(self.client)
        if not self.enabled:
            logger.warning(
                "LLM scoring disabled: missing OpenAI client or API key."
            )

    def combine_scores(self, base_score: float, llm_score: float) -> float:
        base = _coerce_float(base_score, 0.0)
        llm = _coerce_float(llm_score, base)
        combined = (1.0 - self.weight) * base + self.weight * llm
        return float(max(0.0, min(100.0, combined)))

    def score_candidates(self, candidates: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not self.enabled or not candidates:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        limited = list(candidates)[: self.max_candidates]

        for candidate in limited:
            cand = _as_dict(candidate)
            candidate_id = str(cand.get("candidate_id") or cand.get("symbol"))
            if not candidate_id:
                logger.debug("Skipping candidate without identifier: %s", cand)
                continue

            payload = self._build_messages(cand)
            response_content: Optional[str] = None

            for attempt in range(self.max_retries + 1):
                try:
                    request_kwargs = {
                        'model': self.model,
                        'messages': payload,
                        'timeout': self.request_timeout,
                    }
                    if self.temperature is not None:
                        request_kwargs['temperature'] = self.temperature

                    response = self.client.chat.completions.create(  # type: ignore[union-attr]
                        **request_kwargs
                    )
                    choice = response.choices[0]
                    message = getattr(choice, "message", None)
                    if message and getattr(message, "content", None):
                        response_content = message.content
                    elif getattr(choice, "message", None) and isinstance(choice.message, dict):
                        response_content = choice.message.get("content")  # type: ignore[index]
                    if response_content:
                        break
                except Exception as exc:  # pragma: no cover - network failure path
                    if attempt >= self.max_retries:
                        logger.warning("OpenAI scoring failed for %s: %s", candidate_id, exc)
                    else:
                        logger.debug(
                            "OpenAI scoring retry %s/%s for %s: %s",
                            attempt + 1,
                            self.max_retries,
                            candidate_id,
                            exc,
                        )
                        time.sleep(min(2 ** attempt, 5))
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)

            if not response_content:
                continue

            parsed = self._parse_response(response_content)
            if not parsed:
                continue

            score = _coerce_float(parsed.get("llm_score"), float(cand.get("base_score", 0.0)))
            score = float(max(0.0, min(100.0, score)))
            parsed["llm_score"] = score
            results[candidate_id] = parsed

        return results

    def _build_messages(self, cand: Dict[str, Any]) -> Sequence[Dict[str, str]]:
        description = self._format_candidate_summary(cand)
        base_score = _coerce_float(cand.get("base_score"), 0.0)
        side = cand.get("position_side", "LONG")
        symbol = cand.get("symbol", "UNKNOWN")

        system_prompt = (
            "You are a quantitative crypto analyst. "
            "Evaluate the setup quality using the provided metrics, "
            "balancing momentum, risk, fundamentals, and risk/reward. "
            "Return strict JSON with fields: llm_score (0-100), confidence (LOW, MEDIUM, HIGH), reason (<=220 chars)."
        )
        user_prompt = (
            f"Candidate: {symbol} | Side: {side}\n"
            f"Baseline overall score: {base_score:.2f}\n"
            "Metrics:\n"
            f"{description}\n"
            "Score closer to 100 if the setup is strong and the risk is acceptable, "
            "otherwise push it closer to 0." 
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_candidate_summary(self, cand: Dict[str, Any]) -> str:
        parts = []
        ordered_keys = [
            "technical_score",
            "fundamental_score",
            "momentum_score",
            "risk_score",
            "risk_level",
            "risk_reward_ratio",
            "trend_strength",
            "volatility_30d",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "price_change_7d",
            "price_change_30d",
        ]
        for key in ordered_keys:
            if key not in cand:
                continue
            value = cand[key]
            if isinstance(value, float):
                parts.append(f"{key}: {value:.3f}")
            else:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        if not text:
            return None

        match = re.search(r"\{.*\}", text, re.DOTALL)
        json_payload = match.group(0) if match else text
        try:
            data = json.loads(json_payload)
            if isinstance(data, dict):
                data.setdefault("confidence", "MEDIUM")
                data.setdefault("reason", "")
                return data
        except json.JSONDecodeError:
            logger.debug("Failed to parse LLM JSON: %s", text)
        return None


def build_llm_payload(candidate: Any) -> Dict[str, Any]:
    """Helper to assemble the dictionary payload expected by LLMScorer."""
    cand = _as_dict(candidate)
    cand.setdefault(
        "candidate_id",
        f"{cand.get('symbol', 'UNKNOWN')}:{cand.get('position_side', 'LONG')}",
    )
    cand.setdefault("base_score", cand.get("overall_score", 0.0))
    return cand
