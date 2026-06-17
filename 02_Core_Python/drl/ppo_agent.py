"""
PPO Agent — Trains and loads a Stable-Baselines3 PPO model
on the TradingEnv (with real or synthetic market data) using
VecNormalize for distribution normalization.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from drl.trading_env import TradingEnv
from loguru import logger

# Decision PPO support
try:
    from drl.decision_head import DecisionHead, DECISION_ACTION_DIM
    _HAS_DECISION_HEAD = True
except Exception:
    _HAS_DECISION_HEAD = False
    DECISION_ACTION_DIM = 18

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_trading.zip")
VEC_NORM_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")
LOG_DIR = os.path.join(ROOT, "logs", "drl")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
except Exception:
    DEVICE = "cpu"

def make_env(df=None):
    def _init():
        return TradingEnv(df=df)
    return _init

def load_model():
    """Load the trained PPO model & VecNormalize for inference."""
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH, device=DEVICE)
        
        vec_env = None
        if os.path.exists(VEC_NORM_PATH):
            obs_dim = int(model.observation_space.shape[0])
            portfolio_feature_count = TradingEnv.infer_portfolio_feature_count(obs_dim)
            dummy = DummyVecEnv([make_env(None)])
            dummy.env_method("set_portfolio_feature_count", portfolio_feature_count)
            vec_env = VecNormalize.load(VEC_NORM_PATH, dummy)
            vec_env.training = False
            vec_env.norm_reward = False
            logger.success("PPO VecNormalize parameters loaded.")
            
        logger.success(f"PPO Base Model loaded from {MODEL_PATH}")
        return model, vec_env
    else:
        logger.warning("No trained PPO model found — run training first!")
        return None, None

def predict(obs, model=None, vec_env=None):
    """Get a continuous action from the PPO model + VecNormalizer."""
    if model is None:
        model, vec_env = load_model()
    if model is None:
        return 0.0  # default HOLD (0 leverage)
        
    if vec_env is not None:
        obs = vec_env.normalize_obs(obs)
        
    action, _ = model.predict(obs, deterministic=True)
    return action[0]


# ============================================================
# DECISION PPO HELPERS (high-level brain integration)
# ============================================================

def get_decision_action_dim() -> int:
    """Return the action dimensionality used by Decision PPO rich head."""
    return DECISION_ACTION_DIM


def make_decision_head(obs_dim: int, **kwargs):
    """Return a DecisionHead instance ready for custom training / distillation."""
    if not _HAS_DECISION_HEAD:
        raise RuntimeError("DecisionHead not available (torch import issue in decision_head.py)")
    return DecisionHead(input_dim=obs_dim, **kwargs)


def decode_decision_action(action_vec, **decode_kwargs) -> dict:
    """Convenience wrapper to decode rich Decision PPO action into DecisionSpec."""
    from drl.trading_env import TradingEnv
    return TradingEnv.decode_action(action_vec, decision_ppo=True, **decode_kwargs)


def build_decision_ppo_env_kwargs(action_config: dict | None = None) -> dict:
    """Return kwargs to pass to TradingEnv for Decision PPO mode."""
    cfg = dict(action_config or {})
    cfg.setdefault("decision_ppo", True)
    cfg.setdefault("decision_action_dim", DECISION_ACTION_DIM)
    return {"action_config": cfg, "action_version": "decision_ppo_v1"}
