"""
Configuration utilities for Chain Gambler.

Centralizes config loading patterns to reduce code duplication and ensure consistency.
"""
import os
from typing import Any, Optional
from pathlib import Path

import yaml

DEFAULT_TRADING_SYMBOLS = ["BTCUSDm", "XAUUSDm"]

_PLACEHOLDERS = {
    "YOUR_BOT_TOKEN_HERE",
    "YOUR_CHAT_ID_HERE",
}

# Centralized project root path
# Use Path for cross-platform compatibility
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """Get the project root directory as a Path object."""
    return PROJECT_ROOT


def get_symbol_config(symbol: str, config_dir: str = "configs") -> Optional[dict]:
    """
    Load per-symbol configuration from YAML file.

    Args:
        symbol: Trading symbol (e.g., "EURUSDm")
        config_dir: Directory containing config files (default: "configs")

    Returns:
        Dict with symbol configuration, or None if file not found/invalid

    Usage:
        cfg = get_symbol_config("EURUSDm")
        if cfg:
            spread = cfg.get("max_spread_pips", 50)
    """
    config_path = PROJECT_ROOT / config_dir / f"{symbol}.yaml"

    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        # Log error but don't crash - return empty config
        import logging
        logging.getLogger(__name__).warning(
            f"Failed to load config for {symbol} from {config_path}: {e}"
        )
        return None


def get_main_config_path() -> Path:
    """Get the path to the main config.yaml file."""
    return PROJECT_ROOT / "config.yaml"


def load_yaml_config(path: Path | str, default: Any = None) -> Any:
    """
    Safely load a YAML configuration file.

    Args:
        path: Path to YAML file
        default: Default value to return if file not found or invalid

    Returns:
        Parsed YAML content, or default value on error
    """
    path = Path(path) if isinstance(path, str) else path

    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or default
    except (yaml.YAMLError, OSError) as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load YAML from {path}: {e}")
        return default


def parse_symbol_list(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    txt = str(raw or "").strip()
    if not txt:
        return []
    return [part.strip() for part in txt.split(",") if part.strip()]


def resolve_trading_symbols(
    cfg: dict[str, Any] | None,
    *,
    env_keys: tuple[str, ...] = (),
    fallback: list[str] | None = None,
) -> list[str]:
    fallback_symbols = list(fallback or DEFAULT_TRADING_SYMBOLS)
    for key in env_keys:
        raw = os.environ.get(key)
        if raw:
            symbols = parse_symbol_list(raw)
            if symbols:
                return symbols

    trading_cfg = (cfg or {}).get("trading", {}) if isinstance(cfg, dict) else {}
    symbols = parse_symbol_list(trading_cfg.get("symbols", []))
    return symbols or fallback_symbols


def load_project_config(project_root: str, live_mode: bool = False) -> dict[str, Any]:
    """
    Load config.yaml and enforce that live runs are not using placeholders.
    """
    cfg_path = os.path.join(project_root, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if live_mode:
        tel = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
        token = str(tel.get("token", "") or "").strip()
        chat_id = str(tel.get("chat_id", "") or "").strip()
        token_env = os.environ.get("TELEGRAM_TOKEN", "").strip()
        chat_env = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

        token_is_env_ref = token.upper() == "ENV:TELEGRAM_TOKEN"
        chat_is_env_ref = chat_id.upper() == "ENV:TELEGRAM_CHAT_ID"

        if token in _PLACEHOLDERS or chat_id in _PLACEHOLDERS:
            raise RuntimeError(
                "Live mode blocked: config.yaml contains Telegram placeholders. "
                "Use real secrets via environment variables."
            )
        if token_is_env_ref and not token_env:
            raise RuntimeError(
                "Live mode blocked: TELEGRAM_TOKEN env var is not set while config.yaml uses ENV:TELEGRAM_TOKEN."
            )
        if chat_is_env_ref and not chat_env:
            raise RuntimeError(
                "Live mode blocked: TELEGRAM_CHAT_ID env var is not set while config.yaml uses ENV:TELEGRAM_CHAT_ID."
            )

    return cfg
