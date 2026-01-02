"""Load risk threshold defaults from config/risk_thresholds.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO_ROOT / "config" / "risk_thresholds.yaml"


def _coerce_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            if cast is int and "." in value:
                continue
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        key = key.strip()
        if not key:
            continue
        data[key] = _coerce_value(raw)
    return data


def load_risk_thresholds(path: Optional[Path] = None) -> Dict[str, Any]:
    """Return risk threshold overrides from YAML (empty if missing)."""
    target = path or DEFAULT_PATH
    if not target.exists():
        return {}
    try:
        import yaml  # type: ignore

        with target.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        try:
            return _parse_simple_yaml(target.read_text(encoding="utf-8"))
        except Exception:
            return {}

