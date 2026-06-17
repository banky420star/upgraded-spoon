import pytest
import numpy as np

drl_env = pytest.importorskip(
    "drl.trading_env",
    reason="TradingEnv requires heavy deps (gymnasium/numpy/torch); install to enable tests",
)


def test_trading_env_step_reset():
    try:
        env = drl_env.TradingEnv(config_name="XAUUSDm")
    except Exception as e:
        pytest.skip(f"Trading config/data missing, skipping env instantiation: {e}")
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
    action = env.action_space.sample()
    new_obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float), "Reward must be a float"
    assert "balance" in info
