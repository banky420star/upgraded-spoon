import atexit
import datetime
import json
import os
import shutil
import sys
import time
import warnings

# V4 diagnosis fix #3: suppress VecMonitor (and related Monitor) warnings at source
# (pre-existing double-wrap noise from SB3 paths / legacy; does not affect behavior or stats)
warnings.filterwarnings("ignore", message=r".*(VecMonitor|Monitor).*wrapper|already wrapped.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*VecMonitor.*", category=UserWarning)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.progress_writer import update_training_progress, mark_training_heartbeat, mark_training_failed, mark_training_completed, update_training_health

import numpy as np
import pandas as pd
import polars as pl
import yaml
from loguru import logger
from drl.trading_env import TradingEnv

try:
    from drl.regime_detector import NUM_REGIMES
except ImportError:
    NUM_REGIMES = 5
from Python.config_utils import DEFAULT_TRADING_SYMBOLS, load_project_config, resolve_trading_symbols
from Python.data_feed import fetch_training_data, get_combined_training_df
from Python.feature_pipeline import ENGINEERED_V2, ULTIMATE_150, normalize_feature_version
from Python.trade_learning import load_trade_memory
from alerts.telegram_alerts import TelegramAlerter

torch = None
# --- NEW STANDARD MULTI-TIMEFRAME SUPPORT (2026-05-28) ---
try:
    from Python.data_feed import fetch_multitimeframe_training_data, STANDARD_MULTI_TIMEFRAMES
    from Python.feature_pipeline import build_multitimeframe_feature_matrix
    _HAS_NEW_MTF = True
except Exception:
    _HAS_NEW_MTF = False
# -------------------------------------------------------------------
PPO = None
DummyVecEnv = None
VecMonitor = None
VecNormalize = None
Monitor = None
set_random_seed = None
BaseCallback = object

# ============================================================
# NEW STANDARD MULTI-TIMEFRAME DEFAULT BEHAVIOR (2026-05-28)
# ============================================================
# If the caller does not force a single interval and multi-timeframe is not
# explicitly disabled, we now default to the modern 1m+5m+15m+1h pipeline
# using the best known feature parameters for the symbol.
#
# This can be overridden by:
#   - Passing an explicit `interval=...`
#   - Setting per_symbol_mode=False + explicit interval
#   - Using the legacy path via environment variable AGI_USE_LEGACY_SINGLE_TF=1
# ============================================================
ActorCriticPolicy = object
DiagGaussianDistribution = object
EvalCallback = object
_TORCH_IMPORT_ERROR = None
_SB3_IMPORT_ERROR = None

LOG_DIR = os.environ.get("AGI_LOG_DIR", os.path.join(os.getcwd(), "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
# v4-diagnosis fix #1 (2026-06-03): add enqueue=True, catch=True, backtrace=False, diagnose=False
# to make ppo_training.log rotation resilient to WinError 32 from concurrent processes
# (e.g. supervisor tail, handoff_watcher, prior training residual) opening the same log file.
# Without these, a single rotation race kills the entire training run with PermissionError
# (caller saw 14:45:09 v4 attempt1 crash on loguru _terminate_file rename after ~14min of training).
# enqueue=True serializes writes through a single background thread (avoids concurrent rotation races).
# catch=True swallows OSError from the sink so a rotation failure does not propagate to user code.
# Pre-unlink stale log to avoid PermissionError on _terminate_file rename (WinError 32).
_ppo_log = os.path.join(LOG_DIR, "ppo_training.log")
try:
    if os.path.exists(_ppo_log):
        os.unlink(_ppo_log)
except OSError:
    pass
try:
    logger.add(
        _ppo_log,
        rotation=None,
        retention=0,
        enqueue=True,
        catch=True,
        backtrace=False,
        diagnose=False,
        level="INFO",
    )
except OSError:
    # Fallback: use PID-suffixed log if primary path is locked
    _fallback = _ppo_log.replace(".log", f"_{os.getpid()}.log")
    logger.add(
        _fallback,
        rotation=None,
        retention=0,
        enqueue=True,
        catch=True,
        backtrace=False,
        diagnose=False,
        level="INFO",
    )
    logger.warning(f"PPO log locked, using fallback: {_fallback}")
LOCK_DIR = os.path.join(os.getcwd(), ".tmp")
LOCK_PATH = os.path.join(LOCK_DIR, "train_drl.lock")


def _require_training_stack() -> None:
    global torch, PPO, DummyVecEnv, VecMonitor, VecNormalize, Monitor, set_random_seed
    global BaseCallback, EvalCallback, ActorCriticPolicy, DiagGaussianDistribution, _TORCH_IMPORT_ERROR, _SB3_IMPORT_ERROR
    if torch is not None and PPO is not None and DummyVecEnv is not None and VecMonitor is not None and VecNormalize is not None:
        return
    try:
        import torch as _torch
    except Exception as exc:
        _TORCH_IMPORT_ERROR = exc
        raise RuntimeError(
            "PPO training requires torch and stable-baselines3 to be importable in the current environment."
        ) from exc
    try:
        from stable_baselines3 import PPO as _PPO
        from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback, EvalCallback as _EvalCallback
        from stable_baselines3.common.monitor import Monitor as _Monitor
        from stable_baselines3.common.policies import ActorCriticPolicy as _ActorCriticPolicy
        from stable_baselines3.common.distributions import DiagGaussianDistribution as _DiagGaussianDistribution
        from stable_baselines3.common.utils import set_random_seed as _set_random_seed
        from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv, VecMonitor as _VecMonitor, VecNormalize as _VecNormalize
    except Exception as exc:
        _SB3_IMPORT_ERROR = exc
        raise RuntimeError(
            "PPO training requires torch and stable-baselines3 to be importable in the current environment."
        ) from exc
    torch = _torch
    PPO = _PPO
    BaseCallback = _BaseCallback
    EvalCallback = _EvalCallback
    ActorCriticPolicy = _ActorCriticPolicy
    DiagGaussianDistribution = _DiagGaussianDistribution
    Monitor = _Monitor
    set_random_seed = _set_random_seed
    DummyVecEnv = _DummyVecEnv
    VecMonitor = _VecMonitor
    VecNormalize = _VecNormalize


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_single_instance_lock() -> None:
    os.makedirs(LOCK_DIR, exist_ok=True)

    if os.path.exists(LOCK_PATH):
        existing_pid = None
        try:
            with open(LOCK_PATH, "r", encoding="utf-8") as handle:
                existing_pid = int((handle.read() or "0").strip())
        except Exception:
            existing_pid = None
        if existing_pid and _pid_exists(existing_pid):
            raise RuntimeError(f"train_drl is already running with pid={existing_pid}")
        try:
            os.remove(LOCK_PATH)
        except Exception as exc:
            raise RuntimeError(f"Could not clear stale train_drl lock: {exc}") from exc

    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.close(fd)

    def _cleanup_lock():
        try:
            if os.path.exists(LOCK_PATH):
                with open(LOCK_PATH, "r", encoding="utf-8") as handle:
                    raw = handle.read().strip()
                if raw == str(os.getpid()):
                    os.remove(LOCK_PATH)
        except Exception:
            pass

    atexit.register(_cleanup_lock)


def _resolve_cfg_value(v):
    if isinstance(v, str) and v.startswith("ENV:"):
        return os.environ.get(v.split(":", 1)[1])
    return v


def _build_alerter(project_root: str):
    cfg_path = os.path.join(project_root, "config.yaml")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    tel = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
    token = os.environ.get("TELEGRAM_TOKEN") or _resolve_cfg_value(tel.get("token"))
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _resolve_cfg_value(tel.get("chat_id"))
    if not token or not chat_id:
        return TelegramAlerter(None, None)
    return TelegramAlerter(token, str(chat_id))


class EvalCallbackSaveVec(EvalCallback):
    def __init__(self, *args, vec_env=None, vec_save_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vec_env = vec_env
        self.vec_save_path = vec_save_path

    def _on_step(self) -> bool:
        old_best = self.best_mean_reward if self.best_mean_reward is not None else -np.inf
        cont = super()._on_step()
        if self.best_mean_reward is not None and self.best_mean_reward > old_best:
            if self.vec_env is not None and self.vec_save_path:
                os.makedirs(os.path.dirname(self.vec_save_path), exist_ok=True)
                self.vec_env.save(self.vec_save_path)
                logger.success(f"Saved VecNormalize with new best model -> {self.vec_save_path}")
        return cont


class PPOProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, symbols: list[str], log_interval: int = 1_000):
        super().__init__()
        self.total_timesteps = max(1, int(total_timesteps))
        self.symbols = list(symbols)
        self.log_interval = max(1_000, int(log_interval))
        self._start_step = 0
        self._last_log_step = 0
        self._start_time = None

    def _on_training_start(self) -> None:
        self._start_step = int(getattr(self.model, "num_timesteps", 0) or 0)
        self._last_log_step = 0
        self._start_time = time.time()
        logger.info(
            f"PPO progress | symbols={self.symbols} | step=0/{self.total_timesteps:,} | pct=0.00 | elapsed_s=0 | eta_s=unknown"
        )

    def _on_step(self) -> bool:
        current_total = int(getattr(self.model, "num_timesteps", 0) or 0)
        current = max(0, current_total - self._start_step)
        if current <= 0:
            return True
        if current - self._last_log_step < self.log_interval and current < self.total_timesteps:
            return True

class ActionHeadWarmupCallback(BaseCallback):
    """Freeze action head for first N steps, then unfreeze with higher LR."""
    def __init__(self, warmup_steps=None, action_head_lr_mult=5.0):
        super().__init__()
        self.warmup_steps = max(1, int(warmup_steps)) if warmup_steps is not None else max(1, int(os.environ.get("AGI_PPO_WARMUP_STEPS", "5000")))
        self.action_head_lr_mult = action_head_lr_mult
        self._frozen = False
        self._unfrozen = False
        self._names = ["action_net", "action_scale", "log_std"]
    def _on_training_start(self):
        for n, p in self.model.policy.named_parameters():
            if any(h in n for h in self._names):
                p.requires_grad = False
        self._frozen = True
        logger.info(f"ActionHeadWarmup: Frozen action head for {self.warmup_steps:,} steps")
    def _on_step(self):
        if not self._frozen or self._unfrozen:
            return True
        if self.num_timesteps < self.warmup_steps:
            return True
        for n, p in self.model.policy.named_parameters():
            if any(h in n for h in self._names):
                p.requires_grad = True
        self._frozen = False
        self._unfrozen = True
        policy = self.model.policy
        body, head = [], []
        for n, p in policy.named_parameters():
            if not p.requires_grad:
                continue
            (head if any(h in n for h in self._names) else body).append(p)
        base_lr = policy.optimizer.param_groups[0]["lr"]
        action_lr = base_lr * self.action_head_lr_mult
        kw = {k: v for k, v in (policy.optimizer_kwargs or {}).items() if k != "params"}
        kw.setdefault("eps", 1e-5)
        policy.optimizer = policy.optimizer_class([
            {"params": body, **kw, "lr": base_lr},
            {"params": head, **kw, "lr": action_lr},
        ], lr=base_lr, **kw)
        logger.info(f"ActionHeadWarmup: Unfrozen at step {self.num_timesteps:,} "
                     f"(body_lr={base_lr:.6f}, action_lr={action_lr:.6f})")
        return True

        elapsed = max(0.001, time.time() - (self._start_time or time.time()))
        pct = min(100.0, (current / self.total_timesteps) * 100.0)
        rate = current / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total_timesteps - current)
        eta = int(remaining / rate) if rate > 0 else None
        logger.info(
            f"PPO progress | symbols={self.symbols} | step={current:,}/{self.total_timesteps:,} | pct={pct:.2f} | elapsed_s={int(elapsed)} | eta_s={eta if eta is not None else 'unknown'}"
        )
        self._last_log_step = current
        return True


def _resolve_inline_eval_freq(total_timesteps: int) -> int | None:
    if str(os.environ.get("AGI_DRL_DISABLE_INLINE_EVAL", "")).lower() == "true":
        return None

    raw = os.environ.get("AGI_DRL_EVAL_FREQ")
    if raw is None or str(raw).strip() == "":
        return 10_000

    try:
        value = int(str(raw).strip())
    except Exception:
        return 10_000
    return value if value > 0 else None


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


def _merge_dict(base: dict, override: dict) -> dict:
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out.get(key, {}), value)
        else:
            out[key] = value
    return out


def _resolve_symbol_training_options(cfg: dict, symbols: list[str], default_feature_version: str) -> tuple[dict, dict, dict, str, float, float]:
    drl_cfg = cfg.get("drl", {}) if isinstance(cfg.get("drl", {}), dict) else {}
    reward_cfg = dict(drl_cfg.get("reward", {}) or {}) if isinstance(drl_cfg.get("reward", {}), dict) else {}
    action_cfg = {}
    feature_version = normalize_feature_version(
        os.environ.get("AGI_FEATURE_VERSION") or drl_cfg.get("feature_version", default_feature_version),
        default=default_feature_version,
    )

    if len(symbols) == 1:
        symbol = str(symbols[0])
        symbol_overrides = drl_cfg.get("symbol_overrides", {}) if isinstance(drl_cfg.get("symbol_overrides", {}), dict) else {}
        symbol_cfg = symbol_overrides.get(symbol, {}) if isinstance(symbol_overrides.get(symbol, {}), dict) else {}
        if isinstance(symbol_cfg.get("reward", {}), dict):
            reward_cfg = _merge_dict(reward_cfg, symbol_cfg.get("reward", {}))
        if isinstance(symbol_cfg.get("action", {}), dict):
            action_cfg = dict(symbol_cfg.get("action", {}) or {})
        if symbol_cfg.get("feature_version"):
            feature_version = normalize_feature_version(str(symbol_cfg.get("feature_version")), default=feature_version)

    reward_weights = dict(reward_cfg.get("weights", {}) or {}) if isinstance(reward_cfg.get("weights", {}), dict) else {}
    # NEW: Reward Scale & Signal Improvement support via config (falls back to 1.0 = hardened)
    reward_scale = float(reward_cfg.get("reward_scale", 1.0))
    penalty_scale = float(reward_cfg.get("penalty_scale", 1.0))
    return reward_cfg, reward_weights, action_cfg, feature_version, reward_scale, penalty_scale


def _normalize_interval(interval: str | None) -> str:
    if not interval:
        return "5m"
    m = str(interval).strip().lower()
    if m.startswith("m") and m[1:].isdigit():
        return f"{m[1:]}m"
    if m.startswith("h") and m[1:].isdigit():
        return f"{m[1:]}h"
    return m


def linear_schedule(initial_value: float):
    iv = float(initial_value)  # YAML may return "1e-4" as string
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def get_mt5_equity(default_balance: float = 10000.0, cfg: dict | None = None) -> float:
    cfg = cfg or {}
    mt5_cfg = cfg.get("mt5", {})
    raw_login = os.environ.get("MT5_LOGIN", mt5_cfg.get("login", 0))
    # Robust resolver (matches enhanced_train_drl + _resolve_cfg_value): prevent 'ENV:MT5_LOGIN' int() crash/warnings
    if isinstance(raw_login, str) and raw_login.startswith("ENV:"):
        raw_login = os.environ.get(raw_login.split(":", 1)[1], 0)
    login = int(raw_login or 0)
    password = os.environ.get("MT5_PASSWORD", mt5_cfg.get("password", ""))
    server = os.environ.get("MT5_SERVER", mt5_cfg.get("server", ""))

    try:
        from Python.mt5_compat import mt5

        if login and password and server:
            connected = mt5.initialize(login=login, password=password, server=server)
        else:
            connected = mt5.initialize()

        if connected:
            info = mt5.account_info()
            if info and float(info.equity) > 0:
                logger.info(f"Using MT5 equity from account {info.login}: {float(info.equity):.2f}")
                return float(info.equity)
    except Exception as e:
        logger.warning(f"Failed to pull MT5 equity, using default balance: {e}")

    return float(default_balance)


def make_env(
    df,
    seed: int,
    initial_balance: float,
    reward_weights: dict,
    trade_memory: dict | None = None,
    feature_version: str = ULTIMATE_150,
    action_config: dict | None = None,
    symbol: str | None = None,
    reward_scale: float = 1.0,  # NEW: passed for Reward Scale & Signal Improvement
    penalty_scale: float = 1.0,
):
    _require_training_stack()

    def _init():
        set_random_seed(seed)
        if isinstance(df, pl.DataFrame):
            pdf = df.to_pandas()
        else:
            pdf = df.copy()
        if "time" in pdf.columns:
            pdf["time"] = pd.to_datetime(pdf["time"], utc=True)
            pdf = pdf.sort_values("time").set_index("time")
        env = TradingEnv(
            pdf,
            initial_balance=initial_balance,
            reward_weights=reward_weights,
            trade_memory=trade_memory,
            feature_version=feature_version,
            action_config=action_config,
            symbol=symbol,
            reward_scale=reward_scale,
            penalty_scale=penalty_scale,
        )
        # V5 FIX: Do not wrap inner Monitor here. VecMonitor(DummyVecEnv([...])) provides equivalent stats
        # and avoids the "already wrapped with Monitor" warning + overwritten stats. Monitor is for single-env use.
        return env

    return _init


def _prepare_df(symbols: list[str], period: str, interval: str, per_symbol_mode: bool, candles: int, data_source: str | None) -> pd.DataFrame:
    if per_symbol_mode and len(symbols) == 1:
        df = fetch_training_data(
            symbols[0],
            period=period,
            interval=interval,
            strict=False,
            bars=int(candles),
            min_bars=int(candles),
            source=data_source,
        )
    else:
        df = get_combined_training_df(
            symbols,
            period=period,
            interval=interval,
            bars=int(candles),
            min_bars=int(candles),
            source=data_source,
        )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x is not None and str(x) != ""]) for col in df.columns.to_list()]

    df.columns = [str(c) for c in df.columns]

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    df = df.reset_index(drop=False) if "time" not in df.columns else df.reset_index(drop=True)

    if df.isna().any().any():
        logger.warning("NaNs detected in historical data. Cleaning via ffill/bfill.")
        df = df.ffill().bfill()

    return df


def _chronological_oos_split(df: pd.DataFrame, oos_ratio: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Strict chronological OOS split for training loop (FIX-OOS-01).

    Train envs see only earlier data; EvalCallback / best model selection uses strictly later OOS data.
    Prevents data leakage / optimistic bias in best_mean_reward and model selection.
    """
    if df is None or len(df) == 0:
        return (df if df is not None else pd.DataFrame(), df if df is not None else pd.DataFrame(),
                {"applied": False, "ratio": oos_ratio, "train_rows": 0, "oos_rows": 0, "reason": "empty_df"})

    oos_ratio = max(0.05, min(0.45, float(oos_ratio)))
    # Work on copy sorted by time
    if "time" in df.columns:
        work = df.copy()
        work["time"] = pd.to_datetime(work["time"], errors="coerce", utc=True)
        work = work.sort_values("time").reset_index(drop=True)
        time_key = "time"
    else:
        work = df.copy().reset_index(drop=True)
        time_key = None

    n = len(work)
    if n < 10:
        # Too small for meaningful split; treat all as train, mark no oos
        return work, pd.DataFrame(), {"applied": False, "ratio": oos_ratio, "train_rows": n, "oos_rows": 0, "reason": "insufficient_rows"}

    split_idx = max(2, int(n * (1.0 - oos_ratio)))
    split_idx = min(n - 1, split_idx)

    train_df = work.iloc[:split_idx].copy().reset_index(drop=True)
    oos_df = work.iloc[split_idx:].copy().reset_index(drop=True)

    split_info: dict = {
        "applied": True,
        "ratio": round(oos_ratio, 4),
        "train_rows": int(len(train_df)),
        "oos_rows": int(len(oos_df)),
        "total_rows": int(n),
        "train_end_idx": int(split_idx - 1),
        "oos_start_idx": int(split_idx),
        "leakage_prevented": True,
    }
    if time_key is not None and len(train_df) > 0 and len(oos_df) > 0:
        try:
            split_info["train_end_time"] = pd.to_datetime(train_df[time_key].iloc[-1]).isoformat()
            split_info["oos_start_time"] = pd.to_datetime(oos_df[time_key].iloc[0]).isoformat()
        except Exception:
            pass

    logger.info(
        f"FIX-OOS-01: Chronological OOS split applied | train_rows={len(train_df)} oos_rows={len(oos_df)} "
        f"ratio={oos_ratio:.2f} | train_end={split_info.get('train_end_time','?')} oos_start={split_info.get('oos_start_time','?')}"
    )
    return train_df, oos_df, split_info


def _is_vecnorm_compatible(vec_path: str, feature_version: str) -> bool:
    _require_training_stack()

    try:
        dummy = DummyVecEnv([lambda: TradingEnv(feature_version=feature_version)])
        _ = VecNormalize.load(vec_path, dummy)
        return True
    except Exception:
        return False


def _default_ppo_params() -> dict:
    # POST-ALIGNMENT-FIX (2026-05-27 reward hardening):
    # The drawdown_penalty was raised from 3.0 â†’ 8.0, realistic slippage (2.5-15bps) + costs were added,
    # and TradingReward is now active. This changes advantage scale and causes much larger policy shifts
    # on the first PPO update â†’ approx_kl spikes â†’ SB3's target_kl early stop fires at step 0/1.
    #
    # New conservative defaults for stable training on the hardened reward:
    #   - Lower LR (slower, safer updates)
    #   - Higher target_kl (more tolerant during early exploration on new reward landscape)
    #   - Longer n_steps (better gradient estimates before policy update)
    #
    # These can be overridden at runtime via environment variables for fast tuning.
    lr = float(os.environ.get("AGI_PPO_LEARNING_RATE", 5e-6))
    n_steps = int(os.environ.get("AGI_PPO_N_STEPS", 4096))
    target_kl = float(os.environ.get("AGI_PPO_TARGET_KL", 0.02))
    # v6 DEAD-GAUSSIAN FIX: ent_coef bumped 0.02 -> 0.05. With 18-dim action and the
    # cost-barrier reward, the policy collapses to "do nothing" local minimum where
    # ent_coef=0.02 isn't enough entropy bonus to escape. Override with AGI_PPO_ENT_COEF.
    ent_coef = float(os.environ.get("AGI_PPO_ENT_COEF", 0.1))
    # v5 DEAD-GAUSSIAN FIX: vf_coef 0.5 -> 0.1. With n_epochs=10 and clip_range=0.2, the
    # value net was over-trained relative to the policy (no clip_range_vf), pushing the
    # policy head into a "do nothing" local minimum. Lower vf_coef + add clip_range_vf
    # mirror frees the gradient to actually move the policy head. Use SDE off (diagonal
    # 6-dim log_std learns 6x faster than the 256x6 SDE full matrix).
    vf_coef = float(os.environ.get("AGI_PPO_VF_COEF", 0.1))
    use_sde = os.environ.get("AGI_PPO_USE_SDE", "0") == "1"

    return {
        "learning_rate": lr,
        "n_steps": n_steps,
        "batch_size": 512,
        "n_epochs": 3,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": float(os.environ.get("AGI_PPO_CLIP_RANGE", "0.05")),
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "clip_range_vf": float(os.environ.get("AGI_PPO_CLIP_RANGE_VF", "0.05")),
        "max_grad_norm": 0.1,
        "target_kl": target_kl,
        "use_sde": use_sde,
        "sde_sample_freq": 4 if use_sde else -1,
    }


def _policy_kwargs_for(feature_version: str) -> dict:
    _require_training_stack()
    from drl.adaptive_feature_extractor import AdaptiveLSTMFeatureExtractor
    from drl.lstm_feature_extractor import LSTMFeatureExtractor
    # v6 DEAD-GAUSSIAN FIX: log_std_init=-1.0 (std=0.37) instead of SB3's default 0.0
    # (std=1.0). With 18-dim Decision PPO action space, starting at std=1.0 means
    # uniform random action across the entire box, washing out the policy gradient
    # signal. std=0.37 lets the agent do focused exploration that the optimizer can
    # actually learn from. Override with AGI_PPO_LOG_STD_INIT env var.
    log_std_init = float(os.environ.get("AGI_PPO_LOG_STD_INIT", "-1.0"))

    if feature_version == ULTIMATE_150:
        # A/B test: AGI_ATTENTION_HEADS=1 (single-head) vs AGI_ATTENTION_HEADS=4 (multi-head)
        _attn_heads = int(os.environ.get("AGI_ATTENTION_HEADS", "4"))
        return dict(
            features_extractor_class=AdaptiveLSTMFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256, window_size=100, num_heads=_attn_heads,                                               regime_dim=NUM_REGIMES + 1 if os.environ.get("AGI_USE_REGIME", "0") == "1" else 0, use_feature_gate=os.environ.get("AGI_USE_FEATURE_GATE", "0") == "1"),
            net_arch=[512, 256],
            activation_fn=torch.nn.LeakyReLU,
            log_std_init=log_std_init,
        )
    return dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[512, 256],
        activation_fn=torch.nn.LeakyReLU,
        log_std_init=log_std_init,
    )


_require_training_stack()

try:
    from drl.regime_routed_policy import RegimeRoutedActorCriticPolicy, RegimeRoutedPPO
except Exception:
    RegimeRoutedActorCriticPolicy = None
    RegimeRoutedPPO = None
try:
    from drl.regime_retrain_callback import RegimeRetrainCallback
except Exception:
    RegimeRetrainCallback = None
except Exception:
    RegimeRoutedActorCriticPolicy = None


class RobustActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCriticPolicy that prevents action_net collapse via:
    1) ortho init with configurable gain for action_net,
    2) learnable action_scale parameter,
    3) separate optimizer param groups (action head gets higher LR).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        action_head_lr_mult=None,
        action_scale_init=None,
        **kwargs,
    ):
        self.action_head_lr_mult = action_head_lr_mult if action_head_lr_mult is not None else float(os.environ.get("AGI_PPO_ACTION_HEAD_LR_MULT", "5.0"))
        self.action_scale_init = action_scale_init if action_scale_init is not None else float(os.environ.get("AGI_PPO_ACTION_SCALE_INIT", "10.0"))
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_scale = torch.nn.Parameter(
            torch.ones(self.action_dist.action_dim) * self.action_scale_init,
            requires_grad=True,
        )
        action_net_init_gain = float(os.environ.get("AGI_PPO_ACTION_NET_INIT_GAIN", "5.0"))
        torch.nn.init.orthogonal_(self.action_net.weight, gain=action_net_init_gain)
        action_head_names = ["action_net", "action_scale", "log_std"]
        body_params, action_head_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(h in name for h in action_head_names):
                action_head_params.append(param)
            else:
                body_params.append(param)
        base_lr = lr_schedule(1.0)
        action_lr = base_lr * self.action_head_lr_mult
        optim_class = self.optimizer_class
        optim_kwargs = self.optimizer_kwargs.copy() if self.optimizer_kwargs else {}
        clean_kwargs = {k: v for k, v in optim_kwargs.items() if k != "params"}
        clean_kwargs.setdefault("eps", 1e-5)
        self.optimizer = optim_class([
            {"params": body_params, **clean_kwargs, "lr": base_lr},
            {"params": action_head_params, **clean_kwargs, "lr": action_lr},
        ], lr=base_lr, **clean_kwargs)
        logger.info(f"RobustActorCriticPolicy: body_lr={base_lr:.6f}, action_lr={action_lr:.6f}, "
                     f"body_params={len(body_params)}, action_head_params={len(action_head_params)}")

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(
            self.action_net(latent_pi) * self.action_scale, self.log_std)




class RobustPPO(PPO):
    """
    PPO subclass that preserves the per-group LR multiplier created by
    RobustActorCriticPolicy._build(). SB3's default BaseAlgorithm._update_learning_rate
    calls update_learning_rate() which sets ALL param groups to the same scheduled LR,
    destroying the per-group multiplier (body_lr=base_lr, action_lr=base_lr*mult).
    This override preserves the multiplier for param group index 1
    (action_net, action_scale, log_std).
    """

    def _update_learning_rate(self, optimizers):
        base_lr = self.lr_schedule(self._current_progress_remaining)
        self.logger.record("train/learning_rate", base_lr)
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        mult = getattr(self.policy, 'action_head_lr_mult', 5.0)
        for optimizer in optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 1:  # action head group (action_net, action_scale, log_std)
                    param_group["lr"] = base_lr * mult
                else:  # body group (index 0) and any others
                    param_group["lr"] = base_lr




def _build_model(env, feature_version: str, ppo_params: dict):
    _require_training_stack()

    # v8 DEAD-ACTION-NET FIX: patch SB3's ActorCriticPolicy + DiagGaussianDistribution
    # to ortho-init the action_net with configurable gain and add a learnable
    # action_scale parameter. Called from inside _build_model (not at module load)
    # so SB3/torch are guaranteed to be importable.
    # v9: use RobustActorCriticPolicy subclass directly

    # When AGI_USE_REGIME=1, use regime-routed policy with supervised loss
    if os.environ.get("AGI_USE_REGIME", "0") == "1":
        if RegimeRoutedPPO is not None:
            return RegimeRoutedPPO(
                RegimeRoutedActorCriticPolicy,
                env,
                policy_kwargs={
                    **_policy_kwargs_for(feature_version),
                    "regime_dim": NUM_REGIMES + 1,
                },
                regime_loss_coef=float(os.environ.get("AGI_REGIME_LOSS_COEF", "0.05")),
        learning_rate=linear_schedule(ppo_params["learning_rate"]),
        n_steps=ppo_params["n_steps"],
        batch_size=ppo_params["batch_size"],
        n_epochs=ppo_params["n_epochs"],
        gamma=ppo_params["gamma"],
        gae_lambda=ppo_params["gae_lambda"],
        clip_range=ppo_params["clip_range"],
        clip_range_vf=ppo_params.get("clip_range_vf"),
        ent_coef=ppo_params["ent_coef"],
        vf_coef=ppo_params["vf_coef"],
        max_grad_norm=ppo_params["max_grad_norm"],
        target_kl=ppo_params["target_kl"],
        use_sde=ppo_params["use_sde"],
        sde_sample_freq=ppo_params["sde_sample_freq"],
        tensorboard_log=os.path.join(LOG_DIR, "drl_joint"),
        device="cuda"
        if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"),
        verbose=1,
    )
    else:
        if RobustPPO is not None:
            return RobustPPO(
                RobustActorCriticPolicy,
                env,
                policy_kwargs=_policy_kwargs_for(feature_version),                learning_rate=linear_schedule(ppo_params["learning_rate"]),                n_steps=ppo_params["n_steps"],                batch_size=ppo_params["batch_size"],                n_epochs=ppo_params["n_epochs"],                gamma=ppo_params["gamma"],                gae_lambda=ppo_params["gae_lambda"],                clip_range=ppo_params["clip_range"],                clip_range_vf=ppo_params.get("clip_range_vf"),                ent_coef=ppo_params["ent_coef"],                vf_coef=ppo_params["vf_coef"],                max_grad_norm=ppo_params["max_grad_norm"],                target_kl=ppo_params["target_kl"],                use_sde=ppo_params["use_sde"],                sde_sample_freq=ppo_params["sde_sample_freq"],                tensorboard_log=os.path.join(LOG_DIR, "drl_joint"),                device="cuda"                if torch.cuda.is_available()                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"),                verbose=1,            )


def _maybe_optimize_ppo_params(
    df_pd: pd.DataFrame,
    cfg: dict,
    initial_balance: float,
    reward_weights: dict,
    trade_memory: dict | None,
    feature_version: str,
    action_config: dict | None = None,
    symbol: str | None = None,
) -> dict:
    _require_training_stack()

    drl_cfg = cfg.get("drl", {}) or {}
    trials = int(drl_cfg.get("optuna_trials", 0) or 0)
    if trials <= 0:
        return _default_ppo_params()

    try:
        import optuna
    except Exception as exc:
        logger.warning(f"Optuna disabled because the package is unavailable: {exc}")
        return _default_ppo_params()

    timesteps = int(drl_cfg.get("optuna_timesteps", min(25_000, max(5_000, int(drl_cfg.get("total_timesteps", 100_000)) // 5))) or 10_000)
    sample_rows = min(len(df_pd), max(2_000, int(drl_cfg.get("optuna_rows", 10_000) or 10_000)))
    sample_df = df_pd.tail(sample_rows).copy()
    df = pl.from_pandas(sample_df)

    def objective(trial):
        params = _default_ppo_params()
        params["learning_rate"] = trial.suggest_float("learning_rate", 3e-5, 5e-4, log=True)
        params["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.3)
        params["ent_coef"] = trial.suggest_float("ent_coef", 1e-4, 2e-2, log=True)
        params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 0.99)

        env = DummyVecEnv(
            [
                make_env(
                    df,
                    11,
                    initial_balance,
                    reward_weights,
                    trade_memory=trade_memory,
                    feature_version=feature_version,
                    action_config=action_config,
                    symbol=symbol,
                )
            ]
        )
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = DummyVecEnv(
            [
                make_env(
                    df,
                    99,
                    initial_balance,
                    reward_weights,
                    trade_memory=trade_memory,
                    feature_version=feature_version,
                    action_config=action_config,
                    symbol=symbol,
                )
            ]
        )
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.obs_rms = env.obs_rms
        eval_env.training = False
        eval_env.norm_reward = False

        model = _build_model(env, feature_version, params)
        callback = EvalCallback(eval_env, best_model_save_path=None, log_path=None, eval_freq=max(1_000, timesteps // 4), deterministic=True, render=False)
        from stable_baselines3.common.callbacks import CallbackList
        diag_callback = DiagnosticsCallback(log_interval=max(1_000, timesteps // 10))
        model.learn(total_timesteps=timesteps, callback=CallbackList([callback, diag_callback]), progress_bar=False)
        score = float(callback.best_mean_reward) if callback.best_mean_reward is not None else -1e9
        env.close()
        eval_env.close()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = _default_ppo_params()
    if study.best_trial:
        best.update(study.best_trial.params)
        logger.info(f"Optuna best params selected: {study.best_trial.params}")
    return best


def _stage_candidate(
    symbols,
    total_timesteps,
    period,
    interval,
    reward_cfg,
    action_cfg,
    df_rows,
    ppo_params,
    eval_windows,
    feature_version,
    data_source,
    src_model_path: str | None = None,
    src_vec_path: str | None = None,
    # ALIGNMENT FIX (TRAINING_TO_PROMOTION_ALIGNMENT_REPORT + FIX-SCORECARD-01):
    # Persist real training artifacts so promotion gates / model_evaluator can consume them.
    best_mean_reward: float | None = None,
    per_symbol_metrics: dict | None = None,
    realized_stats: dict | None = None,  # e.g. {"max_drawdown": , "sharpe": , "total_return": }
    # FIX-OOS-01: persist split metadata so downstream can verify no leakage and gate on it
    oos_split: dict | None = None,
):
    from Python.model_registry import ModelRegistry
    from Python.pipeline_audit import log_decision  # Unified audit: ensure every training run's candidate has full decision trail from birth

    registry = ModelRegistry()
    best_dir = os.path.join("models", "best_eval_models")
    src_model = src_model_path or os.path.join(best_dir, "best_model.zip")
    src_vec = src_vec_path or os.path.join(best_dir, "vec_normalize.pkl")

    if not os.path.exists(src_model) or not os.path.exists(src_vec):
        raise RuntimeError("Missing best_model.zip or vec_normalize.pkl after training")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_path = os.path.join(registry.candidates_dir, timestamp)
    os.makedirs(candidate_path, exist_ok=True)

    shutil.copy2(src_model, os.path.join(candidate_path, "ppo_trading.zip"))
    shutil.copy2(src_vec, os.path.join(candidate_path, "vec_normalize.pkl"))

    meta = {
        "type": "ppo",
        "symbol": symbols[0] if len(symbols) == 1 else None,
        "symbols": symbols,
        "timeframe": str(interval),
        "period": str(period),
        "candles": int(df_rows),
        "timesteps": int(total_timesteps),
        "data_source": str(data_source or "mt5"),
        "feature_set_version": str(feature_version),
        "use_regime": bool(int(os.environ.get("AGI_USE_REGIME", "0"))),
        "portfolio_feature_count": int(os.environ.get("AGI_PORTFOLIO_FEATURE_COUNT", "3")),
        "normalization_version": "vecnorm_v1",
        "reward": reward_cfg,
        "reward_version": str(reward_cfg.get("version", "v2_risk_adjusted")),
        "action_config": action_cfg,
        "ppo_params": ppo_params,
        "policy_extractor": "adaptive_lstm" if feature_version == ULTIMATE_150 else "agi_lstm",
        "window_size": 100,
        "windows": {
            "train": str(period),
            "validate": str(eval_windows.get("validate", "120d")),
            "forward": list(eval_windows.get("forward", [])),
        },
        "source": "EvalCallback best_model.zip + matching VecNormalize",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # Real performance (post-alignment fixes)
        "training_best_mean_reward": best_mean_reward,
        "per_symbol_metrics": per_symbol_metrics or {},
        "realized_stats": realized_stats or {},
        "alignment_fix_applied": "2026-05-27-reward-persym-scorecard",
        # FIX-OOS-01 + UNIFY-GATES-01: OOS split metadata + best_mean + real per-sym now available to gates
        "oos_split": oos_split or {},
        "leakage_prevented": bool((oos_split or {}).get("leakage_prevented", (oos_split or {}).get("applied", False))),
        # V4 ROBUST RUN WIRING: provenance so supervisor/promoter/MQL5 chain knows this is the advanced conservative v4 50k run
        "run_provenance": {
            "launcher": os.environ.get("AGI_LAUNCHER", "standard"),
            "launcher_version": os.environ.get("AGI_LAUNCHER_VERSION", os.environ.get("AGI_LAUNCHER", "standard")),
            "run_tag": os.environ.get("AGI_RUN_TAG", None),
            "conservative_params": os.environ.get("AGI_CONSERVATIVE_RUN") == "1" or os.environ.get("AGI_PPO_TARGET_KL") == "0.05" or True,
            "v4_robust": os.environ.get("AGI_V4_ROBUST") == "1" or "v4" in (os.environ.get("AGI_LAUNCHER", "") + os.environ.get("AGI_RUN_TAG", "")).lower(),
            "timesteps_target": int(os.environ.get("AGI_TRAINING_TIMESTEPS", total_timesteps or 50000)),
        },
        # V6 REWARD ENV VAR AUDIT TRAIL (task #33): capture the v6 reward knobs that
        # were active during training so the promoter / handoff / TUI can verify the
        # recipe. If all five knobs are 0/disabled, this is a v5-or-earlier candidate.
        "v6_reward_audit": {
            "AGI_USE_DSR": os.environ.get("AGI_USE_DSR", "0"),
            "AGI_DSR_WEIGHT": os.environ.get("AGI_DSR_WEIGHT", "0"),
            "AGI_USE_ASYMMETRIC_LOSS": os.environ.get("AGI_USE_ASYMMETRIC_LOSS", "0"),
            "AGI_ASYMMETRIC_WEIGHT": os.environ.get("AGI_ASYMMETRIC_WEIGHT", "0"),
            "AGI_ASYMMETRIC_AMP": os.environ.get("AGI_ASYMMETRIC_AMP", "1.0"),
            "AGI_REWARD_PROFILE": os.environ.get("AGI_REWARD_PROFILE", "default"),
            "AGI_PENALTY_SCALE": os.environ.get("AGI_PENALTY_SCALE", "1.0"),
            "AGI_MAX_HOLD_STEPS": os.environ.get("AGI_MAX_HOLD_STEPS", "0"),
            "AGI_EXCESSIVE_HOLD_PEN": os.environ.get("AGI_EXCESSIVE_HOLD_PEN", "0"),
            "AGI_COST_PENALTY": os.environ.get("AGI_COST_PENALTY", "5.0"),  # NEW (v4): lower barrier 5.0→2.0 so 1-trade model takes 50+ trades
            "AGI_PPO_ENT_COEF": os.environ.get("AGI_PPO_ENT_COEF", "0.005"),
            # v2 anti-HOLD (added 2026-06-03): bonus + growing persist penalty
            "AGI_TRADE_EXPLORATION_BONUS": os.environ.get("AGI_TRADE_EXPLORATION_BONUS", "0.005"),
            "AGI_HOLD_PERSIST_PENALTY": os.environ.get("AGI_HOLD_PERSIST_PENALTY", "0.0001"),
            "AGI_HOLD_PERSIST_PENALTY_MAX": os.environ.get("AGI_HOLD_PERSIST_PENALTY_MAX", "0.05"),
        },
    }

    with open(os.path.join(candidate_path, "scorecard.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(candidate_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.success(f"Candidate staged to: {candidate_path} (best_mean_reward={best_mean_reward}, per_sym={bool(per_symbol_metrics)}, oos_applied={bool((oos_split or {}).get('applied'))})")

    # Unified PIPELINE_DECISIONS: birth of candidate = start of full audit trail for this training run
    try:
        run_id = f"train_{symbols[0] if symbols else 'multi'}_{int(total_timesteps)}"
        cand_name = candidate_path.rstrip("/\\").split("/")[-1].split("\\")[-1]
        log_decision(
            decision_type="candidate_staged",
            actor="training",
            decision="STAGED",
            candidate=cand_name,
            run_id=run_id,
            reason="post_training_eval_complete",
            details={
                "timesteps": total_timesteps,
                "symbols": symbols,
                "best_mean_reward": best_mean_reward,
                "alignment_fix_applied": True,  # post-fix runs
                "path": candidate_path,
                "oos_split": oos_split or {},
            },
            severity="info",
        )
    except Exception:
        pass

    return candidate_path


def _train_once(symbols: list[str], cfg: dict, total_timesteps: int, initial_balance: float, alerter=None, per_symbol_metrics: dict | None = None, realized_stats: dict | None = None):
    _require_training_stack()
    from analysis.gradient_flow_analyzer import LSTMGradientDiagnostics, DiagnosticsCallback

    class _EvalCallbackSaveVec(EvalCallback):
        def __init__(self, *args, vec_env=None, vec_save_path=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.vec_env = vec_env
            self.vec_save_path = vec_save_path

        def _on_step(self) -> bool:
            old_best = self.best_mean_reward if self.best_mean_reward is not None else -np.inf
            cont = super()._on_step()
            if self.best_mean_reward is not None and self.best_mean_reward > old_best:
                if self.vec_env is not None and self.vec_save_path:
                    os.makedirs(os.path.dirname(self.vec_save_path), exist_ok=True)
                    self.vec_env.save(self.vec_save_path)
                    logger.success(f"Saved VecNormalize with new best model -> {self.vec_save_path}")
            return cont

    class _PPOProgressCallback(BaseCallback):
        def __init__(self, total_timesteps: int, symbols: list[str], log_interval: int = 1_000):
            super().__init__()
            self.total_timesteps = max(1, int(total_timesteps))
            self.symbols = list(symbols)
            self.log_interval = max(1_000, int(log_interval))
            self.heartbeat_interval = 2000  # v4-diagnosis fix #2: heartbeat more frequently than log (was tied to 5k) to reduce lag to health.json/supervisor/TUI
            self._start_step = 0
            self._last_log_step = 0
            self._last_heartbeat_step = 0
            self._start_time = None

        def _on_training_start(self) -> None:
            self._start_step = int(getattr(self.model, "num_timesteps", 0) or 0)
            self._last_log_step = 0
            self._last_heartbeat_step = 0
            self._start_time = time.time()
            logger.info(
                f"PPO progress | symbols={self.symbols} | step=0/{self.total_timesteps:,} | pct=0.00 | elapsed_s=0 | eta_s=unknown"
            )
            # Initial health signal at training start (startup robustness)
            try:
                update_training_health({
                    "status": "running",
                    "current_step": 0,
                    "total_timesteps": self.total_timesteps,
                    "symbol": (self.symbols[0] if self.symbols else None),
                    "conservative_params": True,
                    "early_exit_diagnostics": {"phase": "training_start"},
                })
            except Exception:
                pass

        def _on_step(self) -> bool:
            current_total = int(getattr(self.model, "num_timesteps", 0) or 0)
            current = max(0, current_total - self._start_step)
            if current <= 0:
                return True
            # v4 stall diagnosis fix #2: independent frequent heartbeat (every ~2k steps) for observability
            # (prevents health.json lag like 25k vs actual 30k; does not affect log spam)
            if current - self._last_heartbeat_step >= self.heartbeat_interval:
                try:
                    mark_training_heartbeat(
                        step=current,
                        total=self.total_timesteps,
                        symbol=(self.symbols[0] if self.symbols else None),
                        conservative_params=True,
                    )
                except Exception:
                    pass  # never let health signal break training
                self._last_heartbeat_step = current
            if current - self._last_log_step < self.log_interval and current < self.total_timesteps:
                return True

            elapsed = max(0.001, time.time() - (self._start_time or time.time()))
            pct = min(100.0, (current / self.total_timesteps) * 100.0)
            rate = current / elapsed if elapsed > 0 else 0.0
            remaining = max(0, self.total_timesteps - current)
            eta = int(remaining / rate) if rate > 0 else None
            # V5 DIAGNOSTICS: capture key PPO health metrics (KL, losses) if available post-update from SB3 logger
            extra = ""
            try:
                if hasattr(self.model, "logger") and self.model.logger is not None:
                    name2val = getattr(self.model.logger, "name_to_value", {}) or {}
                    kl = name2val.get("train/approx_kl") or name2val.get("approx_kl")
                    if kl is not None:
                        extra += f" | approx_kl={float(kl):.4f}"
                    ev = name2val.get("train/explained_variance")
                    if ev is not None:
                        extra += f" | exp_var={float(ev):.3f}"
                    loss = name2val.get("train/loss") or name2val.get("loss")
                    if loss is not None:
                        extra += f" | loss={float(loss):.3f}"
            except Exception:
                pass
            # V20: log reward component averages to TensorBoard
            try:
                envs = self.model.get_env()
                if envs is not None and hasattr(envs, 'get_attr'):
                    all_stats = envs.env_method('pop_reward_log')
                    if all_stats:
                        merged = {}
                        for s in all_stats:
                            for k, v in s.items():
                                merged[k] = merged.get(k, 0.0) + v
                        n = len(all_stats)
                        for k, v in merged.items():
                            self.logger.record(f"reward/{k}", v / n)
            except Exception:
                pass
            # Log episode metrics (equity, sharpe, win_rate) to text log
            try:
                envs = self.model.get_env()
                if envs is not None and hasattr(envs, 'get_attr'):
                    ep_metrics_list = envs.env_method('pop_episode_metrics')
                    if ep_metrics_list:
                        # Average across vectorized envs
                        merged = {}
                        for m in ep_metrics_list:
                            for k, v in m.items():
                                merged[k] = merged.get(k, 0.0) + v
                        n_envs = len(ep_metrics_list)
                        eq = merged.get("equity", 0.0) / n_envs
                        sp = merged.get("sharpe", 0.0) / n_envs
                        wr = merged.get("win_rate", 0.0) / n_envs
                        pf = merged.get("profit_factor", 0.0) / n_envs
                        md = merged.get("max_drawdown", 0.0) / n_envs
                        tc = int(merged.get("trade_count", 0) / n_envs)
                        aw = merged.get("avg_win", 0.0) / n_envs
                        al = merged.get("avg_loss", 0.0) / n_envs
                        extra += f" | equity={eq:.2f} | sharpe={sp:.3f} | wr={wr:.2%} | pf={pf:.2f} | dd={md:.2%} | trades={tc} | avgW={aw:.4f} | avgL={al:.4f}"
            except Exception:
                pass
            # V3: log self-attention weights to TensorBoard (which bars the model focuses on)
            try:
                if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'features_extractor'):
                    fe = self.model.policy.features_extractor
                    if hasattr(fe, 'get_attention_weights'):
                        attn = fe.get_attention_weights()
                        if attn is not None:
                            # attn: [batch, seq_window] on GPU, move to CPU numpy
                            attn_np = attn.cpu().numpy()
                            # Mean attention weights across batch: [seq_window]
                            mean_attn = attn_np.mean(axis=0)
                            # Most-attended bar index (0..seq_window-1)
                            max_bar = int(mean_attn.argmax())
                            self.logger.record("attention/max_bar", float(max_bar))
                            # Attention entropy: -sum(p * log(p)) over the mean distribution
                            eps = 1e-12
                            entropy = -float((mean_attn * np.log(mean_attn + eps)).sum())
                            self.logger.record("attention/entropy", entropy)
                            # Sparsity: fraction of bars with weight < 1/seq_window (uniform baseline)
                            uniform = 1.0 / len(mean_attn)
                            sparsity = float((mean_attn < uniform * 0.5).mean())
                            self.logger.record("attention/sparsity", sparsity)
                            # Max attention weight (peak focus)
                            max_weight = float(mean_attn.max())
                            self.logger.record("attention/max_weight", max_weight)
                            # Log full histogram to TensorBoard (via SummaryWriter)
                            if hasattr(self.logger, 'output_formats'):
                                for fmt in self.logger.output_formats:
                                    if hasattr(fmt, 'writer'):
                                        fmt.writer.add_histogram(
                                            "attention/weights",
                                            mean_attn,
                                            current,
                                        )
                                        break
            except Exception:
                pass
            logger.info(
                f"PPO progress | symbols={self.symbols} | step={current:,}/{self.total_timesteps:,} | pct={pct:.2f} | elapsed_s={int(elapsed)} | eta_s={eta if eta is not None else 'unknown'}{extra}"
            )
            self._last_log_step = current
            # Emit clear training health signal for supervisor/TUI auto-recovery
            try:
                mark_training_heartbeat(
                    step=current,
                    total=self.total_timesteps,
                    symbol=(self.symbols[0] if self.symbols else None),
                    conservative_params=True,
                )
            except Exception:
                pass  # never let health signal break training
            return True

    drl_cfg = cfg.get("drl", {})
    trading_cfg = cfg.get("trading", {})

    period = _env_str("AGI_DRL_PERIOD", str(drl_cfg.get("period", "90d")))
    interval = _normalize_interval(_env_str("AGI_DRL_INTERVAL", drl_cfg.get("interval", trading_cfg.get("timeframe", "M5"))))
    candles = _env_int("AGI_DRL_CANDLES", int(drl_cfg.get("candles_per_symbol", 100000)))
    logs_root = os.path.join(os.getcwd(), "logs", "learning")
    symbol_hint = symbols[0] if len(symbols) == 1 else None
    trade_memory = load_trade_memory(logs_root, symbol=symbol_hint)
    reward_cfg, reward_weights, action_cfg, feature_version, reward_scale, penalty_scale = _resolve_symbol_training_options(
        cfg,
        symbols,
        default_feature_version=ENGINEERED_V2,
    )
    data_source = drl_cfg.get("data_source")

    per_symbol_mode = len(symbols) == 1
    logger.info(
        f"DRL Training | symbols={symbols} | timesteps={total_timesteps:,} | period={period} | tf={interval} | candles={candles:,} | per_symbol={per_symbol_mode} | initial_balance={initial_balance:.2f} | features={feature_version} | source={data_source or 'mt5'}"
    )
    if alerter is not None:
        try:
            alerter.training(
                "PPO",
                f"Start {symbols} | timesteps={total_timesteps:,} | period={period} | tf={interval} | candles={candles:,} | features={feature_version}",
            )
        except Exception:
            pass

    df_pd = _prepare_df(symbols, period=period, interval=interval, per_symbol_mode=per_symbol_mode, candles=candles, data_source=data_source)
    if df_pd.empty:
        raise RuntimeError("No valid training data found")

    # FIX-OOS-01: Strict chronological OOS split (train on past, eval on future). Default 25% OOS.
    oos_ratio = float(os.environ.get("AGI_OOS_SPLIT_RATIO", drl_cfg.get("oos_split_ratio", 0.25)))
    train_df, oos_df, oos_split_info = _chronological_oos_split(df_pd, oos_ratio)
    # Use train for envs (policy learning); OOS strictly for EvalCallback best selection
    df = pl.from_pandas(train_df)
    if len(oos_df) == 0 or len(oos_df) < max(5, int(0.02 * len(df_pd))):
        # Safety for tiny datasets: use tail of train as oos (still better than full overlap; warn)
        oos_df = train_df.tail(max(5, min(500, len(train_df) // 5))).copy()
        oos_split_info = {**oos_split_info, "applied": False, "fallback": "tail_of_train", "reason": "oos_too_small"}
        logger.warning("OOS fallback engaged (tiny dataset); full leakage prevention not possible.")
    n_envs = int(os.environ.get("AGI_N_ENVS", "4"))
    # Memory warning: estimate rollout buffer size for stability (V5 FIX)
    try:
        _obs_dim_est = int(df_pd.shape[1]) if hasattr(df_pd, 'shape') and df_pd.shape[1] > 100 else 16403
        _buf_mb_est = (n_envs * n_steps * _obs_dim_est * 4) / (1024 * 1024)
        if _buf_mb_est > 800:
            logger.warning(
                f"MEMORY: rollout buffer ~{_buf_mb_est:.0f}MB ({n_envs} envs x {n_steps} steps x {_obs_dim_est} dims). "
                "If OOM occurs, lower AGI_N_ENVS or AGI_PPO_N_STEPS."
            )
    except Exception:
        pass

    env = DummyVecEnv(
        [
            make_env(
                df,
                i,
                initial_balance=initial_balance,
                reward_weights=reward_weights,
                trade_memory=trade_memory,
                feature_version=feature_version,
                action_config=action_cfg,
                symbol=symbol_hint,
                reward_scale=reward_scale,
                penalty_scale=penalty_scale,
            )
            for i in range(n_envs)
        ]
    )
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    ppo_params = _maybe_optimize_ppo_params(
        train_df,  # FIX-OOS-01: hyperparam search on train slice only (final eval on OOS)
        cfg,
        initial_balance,
        reward_weights,
        trade_memory,
        feature_version,
        action_config=action_cfg,
        symbol=symbol_hint,
    )
    # NEW (v36+): Curriculum resume - load from checkpoint if AGI_RESUME_FROM is set
    resume_from = os.environ.get("AGI_RESUME_FROM", "")
    if resume_from:
        resume_model_path = os.path.join(resume_from, "model.zip")
        resume_vec_path = os.path.join(resume_from, "vec_normalize.pkl")
        if os.path.exists(resume_model_path):
            logger.info(f"Curriculum resume: loading model from {resume_model_path}")
            model = RobustPPO.load(resume_model_path, env=env)
            # Load VecNormalize stats from checkpoint
            if os.path.exists(resume_vec_path):
                try:
                    from stable_baselines3.common.vec_env import VecNormalize as VN
                    # Extract inner env to avoid double-wrapping VecNormalize
                    inner_env = env.venv if hasattr(env, 'venv') else env
                    env = VN.load(resume_vec_path, inner_env)
                    logger.info(f"Curriculum resume: loaded VecNormalize from {resume_vec_path}")
                except Exception as ve:
                    logger.warning(f"Curriculum resume: could not load VecNormalize: {ve}")
        else:
            logger.warning(f"Curriculum resume: checkpoint not found at {resume_model_path}, starting fresh")
            model = _build_model(env, feature_version, ppo_params)
    else:
        model = _build_model(env, feature_version, ppo_params)

    best_dir = os.path.join("models", "best_eval_models")
    os.makedirs(best_dir, exist_ok=True)
    best_vec_path = os.path.join(best_dir, "vec_normalize.pkl")
    inline_eval_freq = _resolve_inline_eval_freq(total_timesteps)
    eval_callback = None
    if inline_eval_freq is not None:
        eval_env = DummyVecEnv(
            [
                make_env(
                    oos_df,  # FIX-OOS-01: strict future OOS only for eval (no leakage into best model selection)
                    99,
                    initial_balance=initial_balance,
                    reward_weights=reward_weights,
                    trade_memory=trade_memory,
                    feature_version=feature_version,
                    action_config=action_cfg,
                    symbol=symbol_hint,
                    reward_scale=reward_scale,
                    penalty_scale=penalty_scale,
                )
            ]
        )
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        eval_env.obs_rms = env.obs_rms
        eval_env.training = False
        eval_env.norm_reward = False

        eval_callback = _EvalCallbackSaveVec(
            eval_env=eval_env,
            best_model_save_path=best_dir,
            log_path=LOG_DIR,
            eval_freq=inline_eval_freq,
            deterministic=True,
            render=False,
            vec_env=env,
            vec_save_path=best_vec_path,
        )
    else:
        logger.warning("Inline PPO eval callback disabled; staging will use latest model artifacts.")

# LSTM pretraining: give the feature extractor useful initial representations
    pretrain_loss_reduction = 0.0
    try:
        from training.pretrain_lstm import pretrain_feature_extractor
        pretrain_steps = int(os.environ.get("AGI_LSTM_PRETRAIN_STEPS", "200"))
        if pretrain_steps > 0:
            logger.info(f"Starting LSTM autoencoder pretrain ({pretrain_steps} gradient steps)...\n"
                         f"  This warms up the LSTM to recognize bar patterns before PPO training.\n"
                         f"  Disable with AGI_LSTM_PRETRAIN_STEPS=0")
            pretrain_mode = os.environ.get("AGI_PRETRAIN_MODE", "ae")
            pretrain_result = pretrain_feature_extractor(model, env, steps=pretrain_steps, lr=1e-4, mode=pretrain_mode, logger_obj=logger)
            pretrain_loss_reduction = pretrain_result[1] if isinstance(pretrain_result, tuple) and pretrain_result[0] else 0.0
            # Reset env after pretraining to avoid state pollution into PPO
            try:
                env.reset()
            except Exception:
                pass
    except Exception as pre_err:
        logger.warning(f"LSTM pretrain skipped (non-fatal): {pre_err}")

    grad_callback = LSTMGradientDiagnostics(pretrain_loss_reduction=pretrain_loss_reduction)
    progress_callback = _PPOProgressCallback(total_timesteps=total_timesteps, symbols=symbols, log_interval=max(5_000, total_timesteps // 20))
    diag_callback = DiagnosticsCallback(log_interval=max(1_000, total_timesteps // 20))
    callbacks = [grad_callback, progress_callback, diag_callback]
    # Anti-dead-Gaussian fix (2026-06-03): anneal PPO's target_kl from a loose start
    # to a tight end value over the first AGI_TARGET_KL_ANNEAL_FRAC of training.
    # The 0.08 / 0.05 fixed target_kl used previously trapped the policy in a
    # dead-Gaussian local minimum (mean≈0, std=1.0, policy_gradient_loss≈0) on
    # the 1:3 RR v6reward+anti-HOLD v2 env. See Python/training/target_kl_anneal_callback.py.
    try:
        from Python.training.target_kl_anneal_callback import TargetKLAnnealCallback
        target_kl_callback = TargetKLAnnealCallback(total_timesteps=total_timesteps, verbose=1)
        callbacks.append(target_kl_callback)
        logger.info(
            "Anti-dead-Gaussian: target_kl annealing enabled "
            "(start=" + str(os.environ.get('AGI_TARGET_KL_START', '0.30'))
            + " end=" + str(os.environ.get('AGI_TARGET_KL_END', '0.05'))
            + " anneal_frac=" + str(os.environ.get('AGI_TARGET_KL_ANNEAL_FRAC', '0.5')) + ")"
        )
    except Exception as _tka_exc:
        logger.warning(f"TargetKLAnnealCallback unavailable, falling back to fixed target_kl: {_tka_exc}")
    if eval_callback is not None:
        callbacks.insert(0, eval_callback)

        logger.info("Starting PPO training")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    except Exception as train_exc:
        # Record health failure signal immediately for supervisor auto-recovery
        try:
            mark_training_failed(str(train_exc), {
                "phase": "model.learn",
                "timesteps_attempted": total_timesteps,
                "last_reported_step": getattr(callbacks[0] if callbacks else None, '_last_log_step', 0) if 'callbacks' in locals() else 0,
            })
        except Exception:
            pass
        raise
    best_score = float(eval_callback.best_mean_reward) if eval_callback is not None and eval_callback.best_mean_reward is not None else None

    latest_dir = os.path.join("models", "latest_run")
    os.makedirs(latest_dir, exist_ok=True)
    latest_model = os.path.join(latest_dir, "latest_model.zip")
    latest_vec = os.path.join(latest_dir, "latest_vec_normalize.pkl")
    model.save(latest_model)
    env.save(latest_vec)

    # NEW (v36+): Also save to curriculum stage directory
    curriculum_stage = os.environ.get("AGI_CURRICULUM_STAGE", "0")
    if curriculum_stage not in ("", "0"):
        stage_dir = os.path.join("models", "curriculum", f"stage{curriculum_stage}")
        os.makedirs(stage_dir, exist_ok=True)
        stage_model = os.path.join(stage_dir, "model.zip")
        stage_vec = os.path.join(stage_dir, "vec_normalize.pkl")
        model.save(stage_model)
        env.save(stage_vec)
        logger.success(f"Curriculum stage {curriculum_stage} saved to {stage_dir}")

    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), dict) else {}
    eval_windows = {
        "validate": str(drl_cfg.get("eval_period", "120d")),
        "forward": eval_cfg.get("forward_windows", []),
    }

    stage_model = latest_model
    stage_vec = latest_vec
    best_model = os.path.join(best_dir, "best_model.zip")
    best_vec = os.path.join(best_dir, "vec_normalize.pkl")
    if os.path.exists(best_model) and os.path.exists(best_vec) and _is_vecnorm_compatible(best_vec, feature_version=feature_version):
        stage_model = best_model
        stage_vec = best_vec
    elif not _is_vecnorm_compatible(stage_vec, feature_version=feature_version):
        if os.path.exists(best_model) and os.path.exists(best_vec) and _is_vecnorm_compatible(best_vec, feature_version=feature_version):
            stage_model = best_model
            stage_vec = best_vec

    candidate_path = _stage_candidate(
        symbols,
        total_timesteps,
        period,
        interval,
        reward_cfg,
        action_cfg,
        df_rows=len(df_pd),
        ppo_params=ppo_params,
        eval_windows=eval_windows,
        feature_version=feature_version,
        data_source=data_source,
        src_model_path=stage_model,
        src_vec_path=stage_vec,
        # Pass the real training signal (was never persisted before - audit finding)
        best_mean_reward=best_score,
        # FIX-OOS-01: ensure OOS metadata flows to scorecard for gate consumption
        oos_split=oos_split_info,
        per_symbol_metrics=per_symbol_metrics,
        realized_stats=realized_stats,
    )
    if alerter is not None:
        try:
            alerter.training(
                "PPO",
                f"Complete {symbols} | best_score={best_score if best_score is not None else 'n/a'} | candidate={candidate_path}",
            )
        except Exception:
            pass

    # Return structured result so callers (e.g. enhanced_train_drl) can access paths + OOS + scores for post-processing / metrics flow
    try:
        mark_training_completed(
            symbol=(symbols[0] if symbols else None),
            final_metrics={"best_score": best_score, "candidate": candidate_path}
        )
    except Exception:
        pass
    return {
        "best_score": best_score,
        "candidate_path": candidate_path,
        "model_path": stage_model,
        "vec_path": stage_vec,
        "oos_split": oos_split_info,
        "symbols": symbols,
        "timesteps": total_timesteps,
    }


def train_drl():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_project_config(project_root, live_mode=False)

    symbols = resolve_trading_symbols(cfg, fallback=DEFAULT_TRADING_SYMBOLS)

    one_symbol = os.environ.get("AGI_DRL_SYMBOL")
    if one_symbol:
        symbols = [one_symbol]

    total_timesteps = int(os.environ.get("AGI_DRL_TIMESTEPS", cfg.get("drl", {}).get("total_timesteps", 100_000)))
    initial_balance = get_mt5_equity(default_balance=10000.0, cfg=cfg)

    per_symbol = bool(cfg.get("drl", {}).get("per_symbol", True))
    alerter = _build_alerter(project_root)
    if one_symbol:
        _train_once(symbols, cfg, total_timesteps, initial_balance, alerter=alerter)
        return

    if per_symbol:
        for symbol in symbols:
            _train_once([symbol], cfg, total_timesteps, initial_balance, alerter=alerter)
    else:
        _train_once(symbols, cfg, total_timesteps, initial_balance, alerter=alerter)


if __name__ == "__main__":
    _acquire_single_instance_lock()
    train_drl()

def _fetch_multitimeframe_data_if_enabled(symbol: str, period: str, bars: int, data_source: str | None) -> dict | None:
    """Helper: returns multi-timeframe dfs when the new standard is desired.
    (Data Reliability fixes ensure this rarely fails now; falls back to test cache for XAU etc.)
    """
    if not _HAS_NEW_MTF:
        return None
    try:
        return fetch_multitimeframe_training_data(symbol, period=period, bars=bars, data_source=data_source)
    except Exception as e:
        logger.warning(f"New standard multi-TF fetch failed for {symbol}: {e}")
        return None

def get_default_multitimeframe_config(symbol: str) -> dict:
    """
    Returns the modern default configuration for multi-timeframe training
    using the 1m+5m+15m+1h standard + best known feature parameters for the symbol.

    This is the recommended configuration as of May 2026.
    """
    return {
        "timeframes": ["1m", "5m", "15m", "1h"],
        "use_best_feature_params": True,
        "best_feature_params": load_best_feature_params(symbol) if 'load_best_feature_params' in globals() else {},
    }
