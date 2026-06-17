import argparse
import json
import os
import sys

import numpy as np
import yaml
from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Python.config_utils import DEFAULT_TRADING_SYMBOLS
from Python.data_feed import fetch_training_data
from Python.feature_pipeline import ULTIMATE_150, build_env_feature_matrix, normalize_feature_version
from alerts.telegram_alerts import TelegramAlerter

torch = None
_TORCH_IMPORT_ERROR = None

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "dreamer_training.log"), rotation="10 MB", level="INFO")


def _require_dreamer_stack() -> None:
    global torch, _TORCH_IMPORT_ERROR
    if torch is not None:
        return
    try:
        import torch as _torch
    except Exception as exc:
        _TORCH_IMPORT_ERROR = exc
        raise RuntimeError(
            "Dreamer training requires torch to be importable in the current environment."
        ) from exc
    torch = _torch


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return str(default)
    return str(raw).strip()


class DreamerTradingEnvironment:
    def __init__(self, features: np.ndarray, returns: np.ndarray, window: int = 64, cost_per_trade: float = 0.0001):
        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)
        self.window = int(window)
        self.cost = float(cost_per_trade)
        self.T = len(self.r)
        self.reset()

    def reset(self):
        self.t = self.window
        self.pos = 0
        self.equity = 1.0
        return self._get_obs()

    def _get_obs(self):
        w = self.X[self.t - self.window : self.t]
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def step(self, action_onehot):
        action_idx = int(np.argmax(action_onehot))
        new_pos = 0 if action_idx == 0 else (1 if action_idx == 1 else -1)
        delta = abs(new_pos - self.pos)
        trade_cost = self.cost * delta
        pnl = self.pos * self.r[self.t]
        reward = pnl - trade_cost
        self.equity *= 1.0 + reward
        self.pos = new_pos
        self.t += 1
        done = self.t >= self.T
        obs = np.zeros_like(self._get_obs()) if done else self._get_obs()
        return obs, float(reward), done, {"equity": float(self.equity), "pos": int(self.pos)}


def _load_cfg() -> dict:
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_cfg_value(value):
    if isinstance(value, str) and value.startswith("ENV:"):
        return os.environ.get(value.split(":", 1)[1])
    return value


def _build_alerter(cfg: dict):
    tel = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
    token = os.environ.get("TELEGRAM_TOKEN") or _resolve_cfg_value(tel.get("token"))
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _resolve_cfg_value(tel.get("chat_id"))
    if not token or not chat_id:
        return TelegramAlerter(None, None)
    return TelegramAlerter(token, str(chat_id))


def _parse_symbol_list(raw) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    txt = str(raw or "").strip()
    if not txt:
        return []
    return [part.strip() for part in txt.split(",") if part.strip()]


def _configured_symbols(cfg: dict) -> list[str]:
    trading_cfg = cfg.get("trading", {}) if isinstance(cfg, dict) else {}
    symbols = _parse_symbol_list(trading_cfg.get("symbols", []))
    return symbols or list(DEFAULT_TRADING_SYMBOLS)


def _dreamer_cfg(cfg: dict) -> dict:
    drl_cfg = cfg.get("drl", {}) if isinstance(cfg, dict) else {}
    raw = drl_cfg.get("dreamer", {})
    return raw if isinstance(raw, dict) else {}


def _resolve_symbols(args, cfg: dict) -> list[str]:
    if getattr(args, "symbols", None):
        return _parse_symbol_list(args.symbols)
    if getattr(args, "symbol", None):
        return _parse_symbol_list([args.symbol])

    env_symbols = os.environ.get("AGI_DREAMER_SYMBOLS")
    if env_symbols:
        return _parse_symbol_list(env_symbols)

    env_symbol = os.environ.get("AGI_DREAMER_SYMBOL")
    if env_symbol:
        return _parse_symbol_list([env_symbol])

    dreamer_cfg = _dreamer_cfg(cfg)
    cfg_symbols = _parse_symbol_list(dreamer_cfg.get("symbols", []))
    return cfg_symbols or _configured_symbols(cfg)


def _prepare_training_arrays(symbol: str, period: str, interval: str, candles: int, feature_version: str):
    df = fetch_training_data(symbol, period=period, interval=interval, strict=True, bars=candles, min_bars=min(5000, candles))
    features = build_env_feature_matrix(df.reset_index(drop=False), feature_version=feature_version)
    close = df["close"].to_numpy(dtype=np.float32)
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / (np.abs(close[:-1]) + 1e-8)
    return features, returns


def _train_symbol(symbol: str, args, period: str, interval: str, candles: int, feature_version: str, device: str):
    _require_dreamer_stack()
    from drl.dreamer_agent import DreamerV3Agent

    features, returns = _prepare_training_arrays(symbol, period, interval, candles, feature_version)
    env = DreamerTradingEnvironment(features, returns, window=args.window)
    obs_dim = env.reset().shape[0]
    agent = DreamerV3Agent(obs_dim=obs_dim, action_dim=3, device=device)

    logger.info(
        f"Dreamer training start | symbol={symbol} | steps={args.steps} | window={args.window} | obs_dim={obs_dim} | device={device} | features={feature_version}"
    )

    obs = env.reset()
    warmup = max(1000, args.window * 50)
    for _ in range(warmup):
        action_onehot = np.zeros(3, dtype=np.float32)
        action_onehot[np.random.randint(0, 3)] = 1.0
        next_obs, reward, done, _ = env.step(action_onehot)
        agent.replay_buffer.add(obs, action_onehot, reward, done)
        obs = env.reset() if done else next_obs

    obs = env.reset()
    h, z = None, None
    for step in range(args.steps):
        action_onehot, (h, z) = agent.act(obs, h, z, deterministic=False)
        next_obs, reward, done, _ = env.step(action_onehot)
        agent.replay_buffer.add(obs, action_onehot, reward, done)
        if step % 4 == 0:
            agent.train_step(batch_size=args.batch_size)
        obs = env.reset() if done else next_obs
        if done:
            h, z = None, None

    out_dir = os.path.join(PROJECT_ROOT, "models", "dreamer")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"dreamer_{symbol}.pt")
    meta_path = os.path.join(out_dir, f"dreamer_{symbol}.json")
    agent.save(model_path)
    meta = {
        "symbol": symbol,
        "feature_version": feature_version,
        "window_size": args.window,
        "obs_dim": obs_dim,
        "steps": args.steps,
        "period": period,
        "interval": interval,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    logger.success(f"Dreamer artifact saved: {model_path}")
    return {
        "symbol": symbol,
        "model_path": model_path,
        "meta_path": meta_path,
        "obs_dim": obs_dim,
        "feature_version": feature_version,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train DreamerV3 policy on current trading features.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--feature-version", default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()

    cfg = _load_cfg()
    drl_cfg = cfg.get("drl", {}) if isinstance(cfg, dict) else {}
    dreamer_cfg = _dreamer_cfg(cfg)
    period = _env_str("AGI_DREAMER_PERIOD", str(drl_cfg.get("period", "90d")))
    interval = _env_str("AGI_DREAMER_INTERVAL", str(drl_cfg.get("interval", "M5")))
    candles = _env_int("AGI_DREAMER_CANDLES", int(drl_cfg.get("candles_per_symbol", 100000)))
    args.steps = int(args.steps or _env_int("AGI_DREAMER_STEPS", int(dreamer_cfg.get("steps", 5000))))
    args.batch_size = int(args.batch_size or _env_int("AGI_DREAMER_BATCH_SIZE", int(dreamer_cfg.get("batch_size", 16))))
    args.window = int(args.window or _env_int("AGI_DREAMER_WINDOW", int(dreamer_cfg.get("window", 64))))
    feature_seed = args.feature_version or os.environ.get("AGI_FEATURE_VERSION") or os.environ.get("AGI_DREAMER_FEATURE_VERSION") or dreamer_cfg.get("feature_version", ULTIMATE_150)
    feature_version = normalize_feature_version(feature_seed, default=ULTIMATE_150)
    symbols = _resolve_symbols(args, cfg)

    _require_dreamer_stack()
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    alerter = _build_alerter(cfg)
    alerter.training(
        "Dreamer",
        f"Batch start | symbols={symbols} | steps={args.steps} | window={args.window} | features={feature_version}",
    )

    results = []
    for symbol in symbols:
        result = _train_symbol(symbol, args, period, interval, candles, feature_version, device)
        results.append(result)
        alerter.training(
            "Dreamer",
            f"Complete {symbol} | steps={args.steps} | obs_dim={result['obs_dim']} | features={feature_version}",
        )

    manifest_path = os.path.join(PROJECT_ROOT, "models", "dreamer", "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "symbols": [row["symbol"] for row in results],
                "feature_version": feature_version,
                "steps": args.steps,
                "window": args.window,
                "period": period,
                "interval": interval,
            },
            handle,
            indent=2,
        )
    logger.success(f"Dreamer manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
