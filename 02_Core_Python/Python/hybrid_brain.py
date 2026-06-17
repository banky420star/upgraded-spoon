import asyncio
import json
import os
import threading
from typing import Optional

import numpy as np
import yaml
from loguru import logger

from Python.action_translator import translate_trade_action
from Python.config_utils import DEFAULT_TRADING_SYMBOLS, resolve_trading_symbols
from Python.feature_pipeline import ENGINEERED_V2, feature_count_for_version


class HybridBrain:
    """
    RL Executor — PPO-first policy with LSTM volatility gating,
    deadzones, and Canary risk scaling.
    """

    def __init__(self, risk, executor, confidence_threshold: float = None):
        self.risk = risk
        self.risk_engine = risk
        self.executor = executor
        self._autonomy_thread = None

        self.ppo_model = None
        self.vec_norm = None
        self.ppo_bundles = []
        self.ppo_bundles_by_symbol = {}
        self.ppo_metadata = {}
        self.ppo_metadata_by_symbol = {}
        self.dreamer_enabled = False
        self.dreamer_blend = 0.15
        self.active_symbols = []
        self.dreamer_symbols = []
        self.dreamer_policies_by_symbol = {}
        self.dreamer_metadata_by_symbol = {}
        self.dreamer_policies = {}
        self._last_action_meta = None
        self._last_action_meta_by_symbol = {}

        cfg = self._load_cfg()
        drl_cfg = (cfg.get("drl", {}) or {}) if isinstance(cfg, dict) else {}
        ensemble_cfg = (drl_cfg.get("ensemble", {}) or {}) if isinstance(drl_cfg.get("ensemble", {}), dict) else {}
        dreamer_cfg = (drl_cfg.get("dreamer", {}) or {}) if isinstance(drl_cfg.get("dreamer", {}), dict) else {}
        trading_cfg = (cfg.get("trading", {}) or {}) if isinstance(cfg, dict) else {}
        self.drl_symbol_overrides = (drl_cfg.get("symbol_overrides", {}) or {}) if isinstance(drl_cfg.get("symbol_overrides", {}), dict) else {}

        def _cfg_bool(env_key: str, cfg_value, default: bool = False) -> bool:
            raw = os.environ.get(env_key)
            if raw is None:
                raw = cfg_value
            if raw is None:
                return bool(default)
            return str(raw).strip().lower() == "true"

        def _cfg_float(env_key: str, cfg_value, default: float) -> float:
            raw = os.environ.get(env_key)
            if raw is None:
                raw = cfg_value
            try:
                return float(raw)
            except Exception:
                return float(default)

        self.dreamer_enabled = _cfg_bool("AGI_DREAMER_ENABLED", dreamer_cfg.get("enabled"), default=False)
        self.dreamer_blend = _cfg_float("AGI_DREAMER_BLEND", dreamer_cfg.get("blend"), default=0.15)
        self.active_symbols = resolve_trading_symbols(cfg, env_keys=("AGI_RUNTIME_SYMBOLS",), fallback=DEFAULT_TRADING_SYMBOLS)
        configured_dreamer_symbols = dreamer_cfg.get("symbols", []) if isinstance(dreamer_cfg.get("symbols", []), (list, tuple)) else []
        self.dreamer_symbols = [str(sym) for sym in (configured_dreamer_symbols or self.active_symbols) if str(sym).strip()]

        cfg_blend = float(drl_cfg.get("ppo_blend", 0.55))
        self.ppo_enabled = os.environ.get("AGI_PPO_ENABLED", "true").lower() == "true"
        self.ppo_blend = float(os.environ.get("AGI_PPO_BLEND", str(cfg_blend)))
        self.ppo_min_abs = float(os.environ.get("AGI_PPO_MIN_ABS", "0.03"))
        self.window_size = int(os.environ.get("AGI_PPO_WINDOW", str(drl_cfg.get("window_size", 100) or 100)))
        self.ppo_ensemble_enabled = os.environ.get("AGI_PPO_ENSEMBLE", str(bool(ensemble_cfg.get("enabled", False)))).lower() == "true"
        self.ppo_ensemble_min_votes = int(os.environ.get("AGI_PPO_ENSEMBLE_MIN_VOTES", str(ensemble_cfg.get("min_votes", 2) or 2)))
        self.ppo_ensemble_threshold = float(os.environ.get("AGI_PPO_ENSEMBLE_THRESHOLD", str(ensemble_cfg.get("agreement_threshold", 0.5) or 0.5)))

        self._ppo_error_count = 0
        self._ppo_error_count_by_symbol = {}
        self._vecnorm_disabled = False

        self._seed_symbol_registry_entries()
        self._load_ppo_from_registry()
        self._load_dreamer_policies()
        self._start_autonomy_if_enabled()

    def _symbol_action_config(self, symbol: str | None) -> dict:
        if not symbol:
            return {}
        cfg = self.drl_symbol_overrides.get(str(symbol), {})
        action_cfg = cfg.get("action", {}) if isinstance(cfg, dict) else {}
        return dict(action_cfg or {}) if isinstance(action_cfg, dict) else {}

    def _symbol_ppo_min_abs(self, symbol: str | None) -> float:
        action_cfg = self._symbol_action_config(symbol)
        try:
            return float(action_cfg.get("min_target_abs", self.ppo_min_abs))
        except Exception:
            return float(self.ppo_min_abs)

    def _load_cfg(self) -> dict:
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _infer_portfolio_feature_count(obs_dim: Optional[int], feature_version: str = ENGINEERED_V2) -> int:
        from drl.trading_env import TradingEnv

        n_features = feature_count_for_version(feature_version)
        return TradingEnv.infer_portfolio_feature_count(obs_dim, n_features=n_features)

    def _start_autonomy_if_enabled(self):
        if os.environ.get("AGI_AUTONOMY_ENABLED", "true").lower() != "true":
            return

        if self._autonomy_thread and self._autonomy_thread.is_alive():
            return

        from Python.autonomy_loop import AutonomyLoop

        loop = AutonomyLoop(self)

        def _runner():
            try:
                asyncio.run(loop.start())
            except Exception as exc:
                logger.warning(f"AutonomyLoop thread stopped: {exc}")

        self._autonomy_thread = threading.Thread(target=_runner, name="autonomy-loop", daemon=True)
        self._autonomy_thread.start()
        logger.info("AutonomyLoop background thread started")

    def _seed_symbol_registry_entries(self):
        try:
            from Python.model_registry import ModelRegistry

            reg = ModelRegistry()
            active = reg._read_active()
            global_champion = active.get("champion")
            symbols = active.setdefault("symbols", {})
            touched = False
            for symbol in self.active_symbols:
                cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
                cur.setdefault("champion", None)
                cur.setdefault("canary", None)
                cur.setdefault("canary_policy", {})
                cur.setdefault("canary_state", {})
                if cur.get("champion") and not self._candidate_targets_symbol(str(cur.get("champion")), str(symbol)):
                    cur["champion"] = None
                if cur.get("canary") and not self._candidate_targets_symbol(str(cur.get("canary")), str(symbol)):
                    cur["canary"] = None
                if not cur.get("champion") and global_champion and self._candidate_targets_symbol(str(global_champion), str(symbol)):
                    cur["champion"] = global_champion
                symbols[str(symbol)] = cur
                touched = True
            if touched:
                reg._write_active(active)
        except Exception as exc:
            logger.warning(f"symbol registry seed failed: {exc}")

    def _candidate_model_paths(self, symbol: str | None = None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = []
        seen = set()

        try:
            from Python.model_registry import ModelRegistry

            reg = ModelRegistry()
            for prefer_canary, role in ((True, "canary"), (False, "champion")):
                candidate_dir = reg.load_active_model(prefer_canary=prefer_canary, symbol=symbol)
                if candidate_dir and symbol and not self._candidate_targets_symbol(candidate_dir, symbol):
                    continue
                if candidate_dir and candidate_dir not in seen:
                    seen.add(candidate_dir)
                    prefix = f"registry:{symbol}" if symbol else "registry"
                    out.append((candidate_dir, f"{prefix}:{role}"))
            if self.ppo_ensemble_enabled:
                for item in reg.get_recent_champions(symbol=symbol):
                    candidate_dir = str((item or {}).get("path") or "")
                    if candidate_dir and candidate_dir not in seen:
                        seen.add(candidate_dir)
                        prefix = f"registry:{symbol}" if symbol else "registry"
                        out.append((candidate_dir, f"{prefix}:history"))
        except Exception:
            pass

        fallbacks = [
            (os.path.join(base, "models", "best_eval_models"), "best_eval"),
            (os.path.join(base, "models"), "models_root"),
        ]
        for candidate_dir, source in fallbacks:
            if candidate_dir not in seen:
                out.append((candidate_dir, source))
        return out

    def _load_candidate_metadata(self, candidate_dir: str) -> dict:
        meta_path = os.path.join(candidate_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _candidate_targets_symbol(self, candidate_dir: str, symbol: str | None) -> bool:
        if not symbol:
            return True
        meta = self._load_candidate_metadata(candidate_dir)
        if self._bundle_targets_symbol(meta, symbol):
            return True
        scorecard = os.path.join(candidate_dir, "scorecard.json")
        if os.path.exists(scorecard):
            try:
                with open(scorecard, "r", encoding="utf-8") as f:
                    payload = json.load(f) or {}
                return self._bundle_targets_symbol(payload if isinstance(payload, dict) else {}, symbol)
            except Exception:
                return False
        return False

    def _build_dummy_env(self, feature_version: str, portfolio_feature_count: int):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from drl.trading_env import TradingEnv

        return DummyVecEnv(
            [lambda fv=feature_version, pfc=portfolio_feature_count: TradingEnv(feature_version=fv, portfolio_feature_count=pfc)]
        )

    @staticmethod
    def _bundle_targets_symbol(meta: dict, symbol: str | None) -> bool:
        if not symbol:
            return True
        symbol_str = str(symbol)
        single = str(meta.get("symbol", "") or "").strip()
        scoped = {str(item).strip() for item in (meta.get("symbols", []) or []) if str(item).strip()}
        if single:
            return single == symbol_str
        if scoped:
            return symbol_str in scoped
        return True

    def _load_ppo_bundles_for_symbol(self, symbol: str | None = None):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        bundles = []
        for candidate_dir, source in self._candidate_model_paths(symbol=symbol):
            model_name = "ppo_trading.zip" if source.startswith("registry") else ("best_model.zip" if source == "best_eval" else "ppo_trading.zip")
            vec_name = "vec_normalize.pkl"
            model_path = os.path.join(candidate_dir, model_name)
            vec_path = os.path.join(candidate_dir, vec_name)
            if not os.path.exists(model_path):
                continue

            meta = self._load_candidate_metadata(candidate_dir)
            if not self._bundle_targets_symbol(meta, symbol):
                continue
            feature_version = str(meta.get("feature_set_version", ENGINEERED_V2) or ENGINEERED_V2)
            try:
                model = PPO.load(model_path)
            except Exception as exc:
                logger.warning(f"Skipping PPO artifact from {source} due to model load error: {exc}")
                continue
            vec_norm = None
            if os.path.exists(vec_path):
                try:
                    obs_dim = int(np.prod(model.observation_space.shape))
                    portfolio_feature_count = self._infer_portfolio_feature_count(obs_dim, feature_version=feature_version)
                    dummy = self._build_dummy_env(feature_version, portfolio_feature_count)
                    vec_norm = VecNormalize.load(vec_path, dummy)
                    vec_norm.training = False
                    vec_norm.norm_reward = False
                except Exception as vec_exc:
                    # Non-security improvement: make VecNormalize mismatch much more visible
                    # for champion models (the ones actually used for decisions).
                    is_champion = "champion" in str(source).lower() or "registry" in str(source).lower()
                    log = logger.error if is_champion else logger.warning
                    log(
                        f"VecNormalize mismatch for {model_path} (continuing without normalization): {vec_exc}"
                    )
                    vec_norm = None

            bundle = {
                "model": model,
                "vec_norm": vec_norm,
                "meta": meta,
                "source": source,
                "candidate_dir": candidate_dir,
                "symbol": symbol,
            }
            if not self._validate_loaded_ppo(bundle):
                logger.warning(f"Skipping incompatible PPO artifact from {source}: {model_path}")
                continue
            bundles.append(bundle)
            logger.success(f"Loaded PPO model from {source}: {model_path}")
            if not self.ppo_ensemble_enabled:
                break
        return bundles

    def _load_ppo_from_registry(self):
        if not self.ppo_enabled:
            return

        try:
            bundles = self._load_ppo_bundles_for_symbol()
            bundles_by_symbol = {}
            metadata_by_symbol = {}
            for symbol in self.active_symbols:
                symbol_bundles = self._load_ppo_bundles_for_symbol(symbol)
                if symbol_bundles:
                    bundles_by_symbol[str(symbol)] = symbol_bundles
                    metadata_by_symbol[str(symbol)] = symbol_bundles[0]["meta"]

            self.ppo_bundles = bundles
            self.ppo_bundles_by_symbol = bundles_by_symbol
            if bundles:
                self.ppo_model = bundles[0]["model"]
                self.vec_norm = bundles[0]["vec_norm"]
                self.ppo_metadata = bundles[0]["meta"]
            else:
                self.ppo_model = None
                self.vec_norm = None
                self.ppo_metadata = {}
                logger.warning("No PPO model found for live inference; using SmartAGI-only exposure")
            self.ppo_metadata_by_symbol = metadata_by_symbol
        except Exception as exc:
            self.ppo_model = None
            self.vec_norm = None
            self.ppo_bundles = []
            self.ppo_bundles_by_symbol = {}
            self.ppo_metadata = {}
            self.ppo_metadata_by_symbol = {}
            logger.warning(f"PPO load failed: {exc}")

    def _load_dreamer_for_symbol(self, symbol: str) -> dict | None:
        from pathlib import Path

        base = Path(__file__).resolve().parent.parent
        model_path = base / "models" / "dreamer" / f"dreamer_{symbol}.pt"
        meta_path = base / "models" / "dreamer" / f"dreamer_{symbol}.json"

        if not model_path.exists():
            return None

        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        return {
            "model_path": str(model_path),
            "meta": meta,
            "symbol": str(symbol),
        }

    def _load_dreamer_policies(self):
        if not self.dreamer_enabled:
            return
        try:
            from Python.dreamer_policy import DreamerPolicy

            self.dreamer_policies_by_symbol = {}
            self.dreamer_metadata_by_symbol = {}
            for symbol in self.active_symbols:
                info = self._load_dreamer_for_symbol(symbol)
                if not info:
                    continue
                policy = DreamerPolicy.load_symbol(symbol)
                if policy is not None:
                    self.dreamer_policies_by_symbol[str(symbol)] = policy
                    self.dreamer_metadata_by_symbol[str(symbol)] = dict(info.get("meta", {}) or {})
                    logger.success(f"Loaded Dreamer policy for {symbol}")
            self.dreamer_policies = dict(self.dreamer_policies_by_symbol)
        except Exception as exc:
            logger.warning(f"Dreamer load failed: {exc}")

    def _validate_loaded_ppo(self, bundle: dict) -> bool:
        model = bundle.get("model")
        vec_norm = bundle.get("vec_norm")
        if model is None:
            return False

        try:
            obs_dim = int(np.prod(model.observation_space.shape))
            obs = np.zeros(obs_dim, dtype=np.float32)
            if vec_norm is not None:
                obs = vec_norm.normalize_obs(obs.reshape(1, -1)).reshape(-1)
            model.predict(obs, deterministic=True)
            return True
        except Exception as exc:
            logger.warning(f"PPO compatibility check failed: {exc}")
            return False

    def _expected_obs_dim(self, bundle: dict | None = None) -> Optional[int]:
        target = bundle or (self.ppo_bundles[0] if self.ppo_bundles else None)
        if not target:
            return None
        try:
            return int(np.prod(target["model"].observation_space.shape))
        except Exception:
            return None

    def _build_ppo_observation(self, df, bundle: dict) -> Optional[np.ndarray]:
        req = ["open", "high", "low", "close", "volume"]
        if df is None or any(c not in df.columns for c in req):
            return None

        from drl.trading_env import TradingEnv

        meta = bundle.get("meta", {}) or {}
        feature_version = str(meta.get("feature_set_version", ENGINEERED_V2) or ENGINEERED_V2)
        obs_dim = self._expected_obs_dim(bundle)
        inferred_window = int(meta.get("window_size", self.window_size) or self.window_size)
        n_features = feature_count_for_version(feature_version)
        portfolio_feature_count = self._infer_portfolio_feature_count(obs_dim, feature_version=feature_version)
        if obs_dim is not None and obs_dim > portfolio_feature_count:
            if (obs_dim - portfolio_feature_count) % n_features == 0:
                inferred_window = max(10, int((obs_dim - portfolio_feature_count) / n_features))

        if len(df) < inferred_window:
            return None

        env = TradingEnv(
            df=df.copy(),
            window_size=inferred_window,
            portfolio_feature_count=portfolio_feature_count,
            feature_version=feature_version,
        )
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs_dim is not None and obs.shape[0] != obs_dim:
            return None
        return obs

    def _normalize_obs_safe(self, bundle: dict, obs: np.ndarray) -> Optional[np.ndarray]:
        if bundle.get("_vecnorm_failed"):
            return None
        vec_norm = bundle.get("vec_norm")
        if vec_norm is None:
            return obs

        try:
            return vec_norm.normalize_obs(obs.reshape(1, -1)).reshape(-1)
        except Exception as exc:
            bundle["_vecnorm_failed"] = True
            if not self._vecnorm_disabled:
                logger.warning(f"VecNormalize failed (bundle will be skipped): {exc}")
                self._vecnorm_disabled = True
            return None

    def _predict_bundle_action(self, symbol: str, df, bundle: dict) -> Optional[dict]:
        obs = self._build_ppo_observation(df, bundle)
        if obs is None:
            return None
        obs = self._normalize_obs_safe(bundle, obs)
        if obs is None:
            return None
        action, _ = bundle["model"].predict(obs, deterministic=True)
        from drl.trading_env import TradingEnv

        action_cfg = self._symbol_action_config(symbol)
        min_abs = self._symbol_ppo_min_abs(symbol)
        action_meta = TradingEnv.decode_action(
            action,
            max_leverage=1.0,
            min_direction_abs=float(action_cfg.get("min_direction_abs", min_abs)),
            min_size_abs=float(action_cfg.get("min_size_abs", min_abs)),
            min_target_abs=float(action_cfg.get("min_target_abs", min_abs)),
        )
        action_val = float(np.clip(action_meta["target"], -1.0, 1.0))
        if abs(action_val) < min_abs:
            return None
        source = str(bundle.get("source", "") or "")
        candidate_dir = str(bundle.get("candidate_dir", "") or "")
        model_version = os.path.basename(candidate_dir) if candidate_dir else None
        lane = "unknown"
        if ":canary" in source:
            lane = "canary"
        elif ":champion" in source:
            lane = "champion"
        elif ":history" in source:
            lane = "history"
        meta = bundle.get("meta", {}) or {}
        action_meta["lane"] = lane
        action_meta["model_source"] = source
        action_meta["model_candidate_dir"] = candidate_dir
        action_meta["model_version"] = model_version
        action_meta["model_family"] = str(meta.get("policy_extractor") or meta.get("type") or "ppo")
        action_meta["model_symbol"] = str(meta.get("symbol") or symbol)
        return action_meta

    def predict_ppo_action(self, symbol: str, df) -> Optional[dict]:
        # Legacy helper retained for backward compatibility. Live runtime uses
        # Server_AGI._blend_symbol_decision with symbol-scoped action metadata.
        if str(symbol) in self.active_symbols:
            bundles = self.ppo_bundles_by_symbol.get(str(symbol)) or []
        else:
            bundles = self.ppo_bundles
        if not bundles:
            self._last_action_meta = None
            self._last_action_meta_by_symbol[str(symbol)] = None
            return None

        try:
            metas = []
            for bundle in bundles:
                action_meta = self._predict_bundle_action(symbol, df, bundle)
                if action_meta is not None:
                    metas.append(action_meta)

            if not metas:
                self._ppo_error_count = 0
                self._ppo_error_count_by_symbol[str(symbol)] = 0
                self._last_action_meta = None
                self._last_action_meta_by_symbol[str(symbol)] = None
                return None

            symbol_action = None
            if self.ppo_ensemble_enabled and len(metas) > 1:
                min_abs = self._symbol_ppo_min_abs(symbol)
                votes = [float(np.sign(meta.get("target", 0.0))) for meta in metas if abs(float(meta.get("target", 0.0))) >= min_abs]
                if votes:
                    non_zero = [v for v in votes if v != 0.0]
                    agreement = abs(sum(non_zero)) / max(1, len(non_zero)) if non_zero else 0.0
                    same_side_votes = int(abs(sum(non_zero))) if non_zero else 0
                    if same_side_votes < self.ppo_ensemble_min_votes or agreement < self.ppo_ensemble_threshold:
                        self._last_action_meta_by_symbol[str(symbol)] = None
                        self._last_action_meta = None
                        return None
                blended = dict(metas[0])
                blended["target"] = float(np.mean([float(meta.get("target", 0.0)) for meta in metas]))
                blended["size"] = float(np.mean([float(meta.get("size", 0.0)) for meta in metas]))
                symbol_action = blended
            else:
                symbol_action = metas[0]

            self._ppo_error_count = 0
            self._ppo_error_count_by_symbol[str(symbol)] = 0
            self._last_action_meta = symbol_action
            self._last_action_meta_by_symbol[str(symbol)] = symbol_action
            return symbol_action
        except Exception as exc:
            count = int(self._ppo_error_count_by_symbol.get(str(symbol), 0)) + 1
            self._ppo_error_count_by_symbol[str(symbol)] = count
            self._ppo_error_count = count
            if count <= 3:
                logger.warning(f"PPO inference failed for {symbol}: {exc}")
            elif count == 4:
                logger.warning("PPO inference continues failing; suppressing further per-tick warnings")
            elif count >= 10:
                logger.warning(f"Disabling PPO for {symbol} in this runtime due to repeated inference failures")
                self.ppo_bundles_by_symbol[str(symbol)] = []
            self._last_action_meta = None
            self._last_action_meta_by_symbol[str(symbol)] = None
            return None

    def predict_ppo_exposure(self, symbol: str, df) -> Optional[float]:
        # Legacy wrapper retained outside the live decision path.
        action_meta = self.predict_ppo_action(symbol, df)
        if action_meta is None:
            return None
        return float(np.clip(action_meta["target"], -1.0, 1.0))

    def predict_dreamer_action(self, symbol: str, df) -> Optional[dict]:
        if not self.dreamer_enabled:
            return None
        policy = self.dreamer_policies_by_symbol.get(str(symbol))
        if policy is None:
            return None
        try:
            predicted_target = policy.predict_exposure(df)
            if predicted_target is None:
                return None
            return {
                "target": float(np.clip(predicted_target, -1.0, 1.0)),
                "symbol": str(symbol),
                "source": "dreamer",
                "meta": self.dreamer_metadata_by_symbol.get(str(symbol), {}),
            }
        except Exception as exc:
            logger.warning(f"Dreamer inference failed for {symbol}: {exc}")
            return None

    def predict_dreamer_exposure(self, symbol: str, df) -> Optional[float]:
        # Legacy wrapper retained outside the live decision path.
        action = self.predict_dreamer_action(symbol, df)
        if action is None:
            return None
        exposure = float(action.get("target", 0.0) or 0.0)
        if abs(exposure) < self._symbol_ppo_min_abs(symbol):
            return None
        return exposure

    def get_last_action_meta(self, symbol: str | None = None) -> Optional[dict]:
        if symbol is not None:
            return self._last_action_meta_by_symbol.get(str(symbol))
        return self._last_action_meta

    def blend_exposure(
        self,
        agi_exposure: float,
        ppo_exposure: Optional[float],
        confidence: float = 1.0,
        dreamer_exposure: Optional[float] = None,
    ) -> float:
        # Legacy blender retained only for non-live callers. Live runtime uses
        # Server_AGI._blend_symbol_decision exclusively.
        if ppo_exposure is None and dreamer_exposure is None:
            return float(agi_exposure)

        conf = float(np.clip(confidence, 0.0, 1.0))
        ppo_w = float(np.clip(self.ppo_blend + (1.0 - conf) * 0.25, 0.0, 0.9))
        dreamer_w = float(np.clip(self.dreamer_blend if dreamer_exposure is not None else 0.0, 0.0, 0.4))
        agi_w = max(0.0, 1.0 - ppo_w - dreamer_w)
        mixed = agi_w * float(agi_exposure)
        if ppo_exposure is not None:
            mixed += ppo_w * float(ppo_exposure)
        if dreamer_exposure is not None:
            mixed += dreamer_w * float(dreamer_exposure)
        return float(np.clip(mixed, -1.0, 1.0))

    def live_trade(self, symbol, exposure, max_lots, action_meta=None, execution_context=None):
        if not self.risk.can_trade(symbol):
            logger.warning(f"HybridBrain.live_trade blocked — risk.can_trade({symbol})=False | exposure={exposure}")
            return
        logger.info(f"HybridBrain.live_trade executing — {symbol} exposure={exposure} max_lots={max_lots}")
        meta = action_meta or self.get_last_action_meta()
        tick = self.executor.get_tick(symbol)
        order_meta = translate_trade_action(symbol, meta, exposure, max_lots, tick) if meta else None
        if order_meta is None:
            order_meta = {
                "symbol": str(symbol),
                "exposure": float(exposure),
                "entry_mode": "market",
                "volume_lots": round(abs(float(exposure) * float(max_lots)), 2),
            }
        if isinstance(meta, dict):
            for key in (
                "lane",
                "model_source",
                "model_candidate_dir",
                "model_version",
                "model_family",
                "model_symbol",
            ):
                if key in meta and meta.get(key) is not None:
                    order_meta[key] = meta.get(key)
        if isinstance(execution_context, dict):
            order_meta.update({k: v for k, v in execution_context.items() if v is not None})
        execution_meta = self.executor.reconcile_exposure(
            symbol,
            exposure,
            max_lots,
            order_meta=order_meta,
            execution_context=execution_context,
        )
        if isinstance(execution_meta, dict):
            order_meta.update(execution_meta)
        return order_meta
