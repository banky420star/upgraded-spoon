"""
Backtester using the same TradingEnv profile as training.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from loguru import logger

class _PPOProxy:
    @staticmethod
    def load(*args, **kwargs):
        _require_rl_stack()
        return globals()["PPO"].load(*args, **kwargs)


PPO = _PPOProxy
DummyVecEnv = None


class _VecNormalizeProxy:
    @staticmethod
    def load(*args, **kwargs):
        _require_rl_stack()
        return globals()["VecNormalize"].load(*args, **kwargs)


VecNormalize = _VecNormalizeProxy
_RL_IMPORT_ERROR = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Python.config_utils import DEFAULT_TRADING_SYMBOLS
from Python.data_feed import fetch_training_data
from Python.feature_pipeline import ENGINEERED_V2, ULTIMATE_150, feature_count_for_version, expected_obs_dim
from drl.trading_env import TradingEnv, DEFAULT_PORTFOLIO_FEATURE_COUNT, PORTFOLIO_FEATURE_COUNT

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
try:
    # Avoid Windows rotation file-lock contention when multiple backtests run in parallel.
    logger.add(
        os.path.join(LOG_DIR, "backtester.log"),
        level="INFO",
        enqueue=True,
        mode="a",
        backtrace=False,
        diagnose=False,
    )
except Exception:
    pass


def _require_rl_stack() -> None:
    global PPO, DummyVecEnv, VecNormalize, _RL_IMPORT_ERROR
    if PPO is not _PPOProxy and DummyVecEnv is not None and VecNormalize is not _VecNormalizeProxy:
        return
    try:
        from stable_baselines3 import PPO as _PPO
        from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv, VecNormalize as _VecNormalize
    except Exception as exc:
        _RL_IMPORT_ERROR = exc
        raise RuntimeError(
            "stable-baselines3 with a working torch runtime is required for PPO backtests."
        ) from exc
    PPO = _PPO
    DummyVecEnv = _DummyVecEnv
    VecNormalize = _VecNormalize


def _normalize_interval(interval: str | None) -> str:
    if not interval:
        return "5m"
    m = str(interval).strip().lower()
    if m.startswith("m") and m[1:].isdigit():
        return f"{m[1:]}m"
    if m.startswith("h") and m[1:].isdigit():
        return f"{m[1:]}h"
    return m


def _make_env(
    df_pd: pd.DataFrame,
    initial_balance: float = 10000.0,
    reward_weights: dict | None = None,
    portfolio_feature_count: int | None = None,
    feature_version: str = ENGINEERED_V2,
    use_regime: bool = False,
):
    _require_rl_stack()

    def _init():
        old_regime = os.environ.get("AGI_USE_REGIME", None)
        os.environ["AGI_USE_REGIME"] = "1" if use_regime else "0"
        try:
            return TradingEnv(
                df_pd,
                initial_balance=initial_balance,
                reward_weights=reward_weights,
                portfolio_feature_count=portfolio_feature_count,
                feature_version=feature_version,
            )
        finally:
            if old_regime is not None:
                os.environ["AGI_USE_REGIME"] = old_regime
            else:
                os.environ.pop("AGI_USE_REGIME", None)

    return DummyVecEnv([_init])



def run_ppo_backtest(
    symbol: str,
    model_path: str,
    vecnorm_path: str,
    period: str = "120d",
    interval: str = "5m",
    initial_balance: float = 10000.0,
    max_steps: int | None = None,
    reward_weights: dict | None = None,
    feature_version: str | None = None,
) -> dict | None:
    metadata_path = os.path.join(model_dir := os.path.dirname(model_path), "metadata.json")
    meta: dict = {}
    _inferred_use_regime = False
    if feature_version is None:
        feature_version = ENGINEERED_V2
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
                feature_version = str(meta.get("feature_set_version", ENGINEERED_V2) or ENGINEERED_V2)
            except Exception:
                feature_version = ENGINEERED_V2
    # VecNormalize shape fix: if metadata is missing OR feature_set_version
    # is the default ENGINEERED_V2, check the on-disk vecnorm dims and recover
    # the actual feature version that produced them. Without this, an OOS
    # eval that reads an ultimate_150 model but defaults to engineered_v2
    # crashes with "spaces must have the same shape: (16403,) != (4003,)".
    if (not os.path.exists(metadata_path)) or feature_version == ENGINEERED_V2:
        if os.path.exists(vecnorm_path):
            try:
                import pickle
                with open(vecnorm_path, "rb") as _vf:
                    _vn = pickle.load(_vf)
                _stored_dim = int(np.asarray(_vn.obs_rms.mean).shape[0])
                # Try known feature versions; window is 100 (the training default).
                _window = 100
                for _fv in (ULTIMATE_150, ENGINEERED_V2):
                    _expected = _window * expected_obs_dim(_fv, _window)
                    if _expected == _stored_dim:
                        if feature_version != _fv:
                            logger.info(
                                f"VecNormalize shape recovery: dim={_stored_dim} -> "
                                f"feature_version={_fv} (was {feature_version})"
                            )
                        feature_version = _fv
                        break
            except Exception as _e:
                logger.debug(f"vecnorm shape recovery failed: {_e}")
    df = fetch_training_data(symbol, period=period, interval=interval, strict=True, require_fresh=True)
    if df is None or df.empty or len(df) < 400:
        raise RuntimeError(f"Insufficient data for {symbol} (len={0 if df is None else len(df)})")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Missing model file: {model_path}")
    model = PPO.load(model_path, device="cpu")
    expected_obs_dim = None
    _model_obs_dim = None
    obs_space = getattr(model, "observation_space", None)
    if obs_space is not None and getattr(obs_space, "shape", None):
        _model_obs_dim = int(np.prod(obs_space.shape))
    # Read portfolio_feature_count from metadata (new models) or infer with regime awareness
    _stored_portfolio = meta.get("portfolio_feature_count", None)
    if _stored_portfolio is not None:
        portfolio_feature_count = int(_stored_portfolio)
    else:
        # Old model without metadata: infer by detecting regime from residual
        if _model_obs_dim is not None:
            _n_feat = feature_count_for_version(feature_version)
            _residual = int(_model_obs_dim) - 100 * _n_feat
            # _build_portfolio_state has 6 base features; default portfolio is 3.
            # If residual > 6, the excess is likely regime/chronos/sentiment dims.
            if _residual > 6:
                portfolio_feature_count = DEFAULT_PORTFOLIO_FEATURE_COUNT  # 3
                _inferred_use_regime = False
                # Enable regime if not already set by metadata
                _use_regime_from_meta = meta.get("use_regime", None)
                if _use_regime_from_meta is None:
                    # Override use_regime for _make_env - inferred from residual
                    _inferred_use_regime = True
            else:
                portfolio_feature_count = _residual
        else:
            portfolio_feature_count = None
    env = _make_env(
        df,
        initial_balance=initial_balance,
        reward_weights=reward_weights,
        portfolio_feature_count=portfolio_feature_count,
        feature_version=feature_version,
        use_regime=meta.get("use_regime", _inferred_use_regime) if isinstance(meta, dict) else bool(_inferred_use_regime),
    )

    if not os.path.exists(vecnorm_path):
        raise RuntimeError(f"Missing vecnorm file: {vecnorm_path}")

    # Load VecNormalize stats from pickle
    with open(vecnorm_path, "rb") as _vnf:
        _saved_vn = pickle.load(_vnf)
    _saved_dim = int(_saved_vn.obs_rms.mean.shape[0])

    # Check if alignment is needed
    _env_dim = int(env.observation_space.shape[0])
    if _env_dim != _saved_dim:
        _ul = env.envs[0]
        _ul_nf = int(_ul.feature_data.shape[1])
        _saved_obs_dim = int(_saved_vn.observation_space.shape[0])
        _target_feats = _saved_obs_dim // 100

        if _ul_nf != _target_feats and _target_feats > 0:
            logger.warning(f"Feature drift: {_ul_nf}->{_target_feats} (obs {_env_dim}->{_saved_dim})")
            if _ul_nf > _target_feats:
                _ul.feature_data = _ul.feature_data[:, :_target_feats].copy()
            else:
                _zp = np.zeros((_ul.feature_data.shape[0], _target_feats - _ul_nf), dtype=_ul.feature_data.dtype)
                _ul.feature_data = np.concatenate([_ul.feature_data, _zp], axis=1)
            _ul.n_features = _target_feats

        # Restore correct pf and regime from metadata (may have been corrupted)
        _ul.portfolio_feature_count = int(meta.get("portfolio_feature_count", DEFAULT_PORTFOLIO_FEATURE_COUNT))
        _ul._use_regime = bool(meta.get("use_regime", True))
        _ul.observation_space = _saved_vn.observation_space

        # Monkey-patch _get_obs to pad/trim output to exactly saved_dim
        # This handles cases where constants like NUM_REGIMES changed since training
        _orig_get_obs = _ul._get_obs
        def _padded_get_obs():
            raw = _orig_get_obs()
            if len(raw) == _saved_dim:
                return raw
            if len(raw) < _saved_dim:
                return np.pad(raw, (0, _saved_dim - len(raw)))
            return raw[:_saved_dim]
        _ul._get_obs = _padded_get_obs

        # Create new DummyVecEnv wrapping aligned TradingEnv (properly initializes buf_obs)
        from stable_baselines3.common.vec_env import DummyVecEnv as _DVE
        env = _DVE([lambda: _ul])

    # Create fresh VecNormalize wrapping DummyVecEnv, copy saved stats with alignment
    from stable_baselines3.common.vec_env import VecNormalize as _VN
    env = _VN(env, training=False, norm_obs=True, norm_reward=False)
    _new_dim = int(env.observation_space.shape[0])
    if _saved_dim == _new_dim:
        env.obs_rms.mean = _saved_vn.obs_rms.mean.copy()
        env.obs_rms.var = _saved_vn.obs_rms.var.copy()
    elif _saved_dim > _new_dim:
        env.obs_rms.mean = _saved_vn.obs_rms.mean[:_new_dim].copy()
        env.obs_rms.var = _saved_vn.obs_rms.var[:_new_dim].copy()
    else:
        _zp = _new_dim - _saved_dim
        env.obs_rms.mean = np.pad(_saved_vn.obs_rms.mean, (0, _zp))
        env.obs_rms.var = np.pad(_saved_vn.obs_rms.var, (0, _zp), constant_values=1.0)
    env.obs_rms.count = _saved_vn.obs_rms.count
    del _saved_vn

    obs = env.reset()
    equities, costs, positions, rewards, step_rets = [], [], [], [], []
    reward_component_sums = {
        "growth": 0.0,
        "payoff": 0.0,
        "sharpe_bonus": 0.0,
        "drawdown_penalty": 0.0,
        "cost_penalty": 0.0,
        "churn_penalty": 0.0,
    }

    steps = 0
    prev_eq = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        info0 = info[0] if isinstance(info, (list, tuple)) else info
        eq = float(info0.get("equity", np.nan))
        cost = float(info0.get("cost", 0.0))
        pos = float(info0.get("position", 0.0))

        rc = info0.get("reward_components", {}) if isinstance(info0, dict) else {}
        for k in reward_component_sums:
            reward_component_sums[k] += float(rc.get(k, 0.0))

        equities.append(eq)
        costs.append(cost)
        positions.append(pos)
        rewards.append(float(reward[0]) if hasattr(reward, '__len__') else float(reward))

        if prev_eq is not None and prev_eq > 0:
            step_rets.append((eq - prev_eq) / prev_eq)
        prev_eq = eq

        steps += 1
        if max_steps and steps >= max_steps:
            break
        if bool(done[0] if hasattr(done, '__len__') else done):
            break

    # Extract trade-level metrics from the environment
    try:
        _trade_metrics = env.env_method("pop_episode_metrics")[0]
    except Exception:
        _trade_metrics = {}

    equity = np.array(equities, dtype=np.float64)
    if len(equity) < 3:
        raise RuntimeError(f"Insufficient backtest steps for {symbol}: {len(equity)}")

    rets = np.array(step_rets, dtype=np.float64) if step_rets else np.zeros(1)
    vol = float(np.std(rets) + 1e-12)
    sharpe = float(np.mean(rets) / vol) if vol > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    total_return = float((equity[-1] / (equity[0] + 1e-12)) - 1.0)
    avg_cost = float(np.mean(costs)) if costs else 0.0
    pos_arr = np.array(positions, dtype=np.float64)
    turnover = float(np.mean(np.abs(np.diff(pos_arr)))) if len(pos_arr) > 2 else 0.0

    score = (total_return * 100.0) - (max_dd * 100.0 * 1.8) + (sharpe * 6.0) - (turnover * 2.0)

    n = max(1, steps)
    result = {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "candles": int(len(df)),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "avg_cost": float(avg_cost),
        "turnover": float(turnover),
        "steps": int(steps),
        "final_equity": float(equity[-1]),
        "score": float(score),
        "reward_component_avg": {k: float(v / n) for k, v in reward_component_sums.items()},
        "total_trades": int(_trade_metrics.get("trade_count", 0)),
        "win_rate": float(_trade_metrics.get("win_rate", 0.0)),
        "profit_factor": float(_trade_metrics.get("profit_factor", 0.0)),
        "avg_win": float(_trade_metrics.get("avg_win", 0.0)),
        "avg_loss": float(_trade_metrics.get("avg_loss", 0.0)),
    }

    logger.info(
        f"BACKTEST {symbol} | tf={interval} ret={total_return:.2%} sharpe={sharpe:.2f} "
        f"maxDD={max_dd:.2%} score={score:.2f} steps={steps}"
    )
    return result


def run_multi(
    symbols: list[str],
    model_dir: str,
    period: str = "120d",
    interval: str = "5m",
    reward_weights: dict | None = None,
) -> dict:
    model_path = os.path.join(model_dir, "ppo_trading.zip")
    vec_path = os.path.join(model_dir, "vec_normalize.pkl")

    per_symbol = []
    errors = []
    tf = _normalize_interval(interval)
    if not symbols:
        symbols = ["BTCUSDm"]
    for sym in symbols:
        try:
            r = run_ppo_backtest(
                sym,
                model_path,
                vec_path,
                period=period,
                interval=tf,
                reward_weights=reward_weights,
            )
            if r:
                per_symbol.append(r)
        except Exception as exc:
            msg = f"{sym}: {exc}"
            errors.append(msg)
            logger.error(f"BACKTEST_FAIL {msg}")

    if not per_symbol:
        return {"error": "No valid backtests", "errors": errors}

    scores = [x["score"] for x in per_symbol]
    rets = [x["total_return"] for x in per_symbol]
    dds = [x["max_drawdown"] for x in per_symbol]
    sharpes = [x["sharpe"] for x in per_symbol]

    agg = {
        "symbols": [x["symbol"] for x in per_symbol],
        "period": period,
        "interval": tf,
        "avg_score": float(np.mean(scores)),
        "avg_return": float(np.mean(rets)),
        "worst_drawdown": float(np.max(dds)),
        "avg_sharpe": float(np.mean(sharpes)),
        "per_symbol": per_symbol,
        "errors": errors,
    }
    return agg


if __name__ == "__main__":
    symbols = list(DEFAULT_TRADING_SYMBOLS)
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join("models", "registry", "champion")
    report = run_multi(symbols, md, period="120d", interval="5m")
    print(json.dumps(report, indent=2))
