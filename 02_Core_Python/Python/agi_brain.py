import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from Python.feature_pipeline import ENGINEERED_V2, ULTIMATE_150, ENGINEERED_LSTM_COLUMNS, build_lstm_feature_frame

# ── Rainforest pattern detector (optional) ──────────────────────────────────
try:
    from Python.rainforest_detector import RainforestDetector as _RainforestDetector
    _RAINFOREST_AVAILABLE = True
except Exception as _rf_err:
    _RainforestDetector = None  # type: ignore
    _RAINFOREST_AVAILABLE = False
    logger.debug(f"RainforestDetector not available in agi_brain: {_rf_err}")

# Default to ULTIMATE_150 — same feature set used during LSTM training.
# ENGINEERED_V2 (17 cols) is kept as a legacy fallback only.
FEATURE_COLUMNS = list(ENGINEERED_LSTM_COLUMNS)  # 17-col legacy size; overridden by bundle metadata

# Direction labels produced by create_sequences() in train_lstm.py:
#   0 = HOLD/NEUTRAL  (small move or ambiguous RSI)
#   1 = BUY           (future_ret > up_thr AND rsi > 52)
#   2 = SELL          (future_ret < dn_thr AND rsi < 48)
DIRECTION_LABELS = ["HOLD", "BUY", "SELL"]


def _direction_to_risk_scalar(direction: str) -> float:
    """Higher risk tolerance when the model has a clear directional conviction."""
    d = str(direction or "").upper()
    if d in ("BUY", "SELL"):
        return 0.90
    return 0.80  # HOLD / uncertain — reduce position sizing slightly


def _direction_to_trend_bias(direction: str) -> float:
    """Map direction class to a signed [-1, +1] bias used in HybridBrain blend.

    The bias is intentionally ±1.0 so that when no PPO champion is present
    (ppo_weight redistributed to agi_weight) the AGI signal drives real decisions.
    """
    d = str(direction or "").upper()
    if d == "BUY":
        return 1.0
    if d == "SELL":
        return -1.0
    return 0.0  # HOLD


def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0].astype(float)
    return obj.astype(float)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 150-feature frame used for both training and inference."""
    features, _ = build_lstm_feature_frame(df, feature_version=ULTIMATE_150)
    return features


class AGIModel(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_COLUMNS)):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, 3, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)



class SmartAGI:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.prediction_count = 0
        self.symbol_models = {}
        self._warned_missing_symbol = set()
        self._warned_incompatible_symbol = set()

        from Python.model_registry import ModelRegistry

        self.registry = ModelRegistry()
        self.active_dir = self._resolve_registry_default_dir()

        if self.active_dir:
            model_path = os.path.join(self.active_dir, "lstm_model.pth")
            scaler_path = os.path.join(self.active_dir, "lstm_scaler.pkl")
            logger.info(f"registry active model dir: {self.active_dir}")
        else:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            model_path = os.path.join(model_dir, "lstm_agi_trained.pt")
            scaler_path = os.path.join(model_dir, "lstm_scaler.pkl")

        self.default_bundle = self._load_bundle(model_path, scaler_path, "default")

        # Backward-compatible aliases used by PPO feature extractor.
        self.model = self.default_bundle["model"]
        self.scaler = self.default_bundle["scaler"]
        self.scaler_loaded = self.default_bundle["scaler_loaded"]

        # ── Rainforest detectors — one per symbol ───────────────────────────
        self.rf_detectors: dict[str, "_RainforestDetector"] = {}  # type: ignore
        self._rf_cfg: dict = self._load_rainforest_config()

    # ------------------------------------------------------------------
    # Rainforest helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rainforest_config() -> dict:
        """Read rainforest config from config.yaml, returning safe defaults."""
        try:
            import yaml
            cfg_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
            )
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    full = yaml.safe_load(f) or {}
                return full.get("rainforest", {}) or {}
        except Exception:
            pass
        return {}

    def _get_rf_detector(self, symbol: str) -> "_RainforestDetector | None":  # type: ignore
        """Return (and lazily initialise) the per-symbol RainforestDetector."""
        if not _RAINFOREST_AVAILABLE:
            return None
        if symbol not in self.rf_detectors:
            try:
                detector = _RainforestDetector(
                    n_estimators=int(self._rf_cfg.get("n_estimators", 200)),
                    max_depth=int(self._rf_cfg.get("max_depth", 12)),
                )
                n_bars = int(self._rf_cfg.get("n_bars", 5000))
                detector.train_from_mt5_data(symbol, n_bars=n_bars)
                self.rf_detectors[symbol] = detector
            except Exception as exc:
                logger.debug(f"RainforestDetector init failed for {symbol}: {exc}")
                return None
        return self.rf_detectors.get(symbol)

    def _get_rainforest_prediction(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Get rainforest regime prediction for a symbol DataFrame.
        Returns a safe dict with regime/confidence even on failure.
        """
        try:
            det = self._get_rf_detector(symbol)
            if det is not None and det.is_trained():
                pred = det.predict_regime(df)
                # Push prediction into api_server cache if available
                try:
                    from Python.api_server import set_rainforest_predictions
                    set_rainforest_predictions(symbol, pred)
                except Exception:
                    pass
                return pred
        except Exception as exc:
            logger.debug(f"Rainforest prediction failed for {symbol}: {exc}")
        return {
            "regime": "ranging",
            "confidence": 0.0,
            "probabilities": {},
            "feature_importances": {},
            "top_patterns": [],
        }

    def _resolve_registry_default_dir(self) -> str | None:
        preferred = self.registry.load_active_model(prefer_canary=True)
        if self._has_default_bundle(preferred):
            return preferred

        champion = self.registry.load_active_model(prefer_canary=False)
        if self._has_default_bundle(champion):
            return champion
        return None

    @staticmethod
    def _has_default_bundle(candidate_dir: str | None) -> bool:
        if not candidate_dir:
            return False
        model_path = os.path.join(candidate_dir, "lstm_model.pth")
        scaler_path = os.path.join(candidate_dir, "lstm_scaler.pkl")
        return os.path.exists(model_path) and os.path.exists(scaler_path)

    def _load_bundle(self, model_path: str, scaler_path: str, label: str):
        feature_columns = list(FEATURE_COLUMNS)
        feature_version = ULTIMATE_150  # default: same as training
        metadata_path = os.path.splitext(model_path)[0] + ".meta.json"
        if os.path.exists(metadata_path):
            try:
                import json

                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f) or {}
                cols = metadata.get("feature_columns")
                if isinstance(cols, list) and cols:
                    feature_columns = [str(col) for col in cols]
                feature_version = str(metadata.get("feature_version", ULTIMATE_150) or ULTIMATE_150)
            except Exception as exc:
                logger.warning(f"{label} metadata load failed: {exc}")

        model = AGIModel(input_dim=len(feature_columns)).to(self.device)
        scaler = MinMaxScaler()
        scaler_loaded = False

        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                model.eval()
                logger.success(f"AGI Brain loaded {label} model on {self.device.upper()}")
            except Exception as exc:
                logger.warning(f"{label} model load failed ({exc}); using fresh weights")
        else:
            logger.warning(f"no trained {label} model found at {model_path}; using fresh weights")

        if os.path.exists(scaler_path):
            import joblib

            try:
                scaler = joblib.load(scaler_path)
                scaler_loaded = True
                logger.success(f"loaded {label} feature scaler")
            except Exception as exc:
                logger.warning(f"{label} scaler load failed: {exc}")

        return {
            "model": model,
            "scaler": scaler,
            "scaler_loaded": scaler_loaded,
            "feature_columns": feature_columns,
            "feature_version": feature_version,
        }

    def _symbol_artifact_paths(self, symbol: str):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        safe = symbol.replace("/", "_")

        symbol_active = self.registry.load_active_model(prefer_canary=True, symbol=symbol)
        if symbol_active:
            reg_model = os.path.join(symbol_active, "per_symbol", f"lstm_{safe}.pt")
            reg_scaler = os.path.join(symbol_active, "per_symbol", f"lstm_scaler_{safe}.pkl")
            if os.path.exists(reg_model) and os.path.exists(reg_scaler):
                return reg_model, reg_scaler

        if self.active_dir:
            reg_model = os.path.join(self.active_dir, "per_symbol", f"lstm_{safe}.pt")
            reg_scaler = os.path.join(self.active_dir, "per_symbol", f"lstm_scaler_{safe}.pkl")
            if os.path.exists(reg_model) and os.path.exists(reg_scaler):
                return reg_model, reg_scaler

        model = os.path.join(root, "models", "per_symbol", f"lstm_{safe}.pt")
        scaler = os.path.join(root, "models", "per_symbol", f"lstm_scaler_{safe}.pkl")
        return model, scaler

    def _is_compatible_lstm_model(self, model_path: str) -> bool:
        if not os.path.exists(model_path):
            return False
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            w = state.get("lstm.weight_ih_l0")
            if w is None or len(w.shape) != 2:
                return False
            metadata_path = os.path.splitext(model_path)[0] + ".meta.json"
            expected = len(FEATURE_COLUMNS)
            if os.path.exists(metadata_path):
                try:
                    import json

                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f) or {}
                    cols = metadata.get("feature_columns")
                    if isinstance(cols, list) and cols:
                        expected = len(cols)
                except Exception:
                    pass
            got = int(w.shape[1])
            return got == expected
        except Exception:
            return False

    def _bundle_for_symbol(self, symbol: str):
        if symbol in self.symbol_models:
            return self.symbol_models[symbol]

        model_path, scaler_path = self._symbol_artifact_paths(symbol)
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            if not self._is_compatible_lstm_model(model_path):
                if symbol not in self._warned_incompatible_symbol:
                    try:
                        bad_model = model_path + ".incompatible"
                        if os.path.exists(model_path) and not os.path.exists(bad_model):
                            os.replace(model_path, bad_model)
                        bad_scaler = scaler_path + ".incompatible"
                        if os.path.exists(scaler_path) and not os.path.exists(bad_scaler):
                            os.replace(scaler_path, bad_scaler)
                    except Exception:
                        pass
                    logger.info(
                        f"incompatible per-symbol model for {symbol} (feature shape mismatch); using default model"
                    )
                    self._warned_incompatible_symbol.add(symbol)
                return self.default_bundle
            bundle = self._load_bundle(model_path, scaler_path, f"symbol[{symbol}]")
            self.symbol_models[symbol] = bundle
            return bundle

        return self.default_bundle

    def predict(self, df: pd.DataFrame, production: bool = False) -> dict:
        self.prediction_count += 1

        symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) else "UNKNOWN"
        bundle = self._bundle_for_symbol(symbol)

        feature_version = str(bundle.get("feature_version", ULTIMATE_150) or ULTIMATE_150)
        feat_df, available_columns = build_lstm_feature_frame(df, feature_version=feature_version)
        if len(feat_df) < 60:
            # Insufficient history — return neutral/hold with no bias
            return {
                "signal": "HOLD",
                "regime": "HOLD",
                "direction": "HOLD",
                "confidence": 0.0,
                "risk_scalar": _direction_to_risk_scalar("HOLD"),
                "trend_bias": 0.0,
                "trade_blocked": False,
                "symbol": symbol,
                "feature_version": feature_version,
            }

        bundle_columns = [str(col) for col in bundle.get("feature_columns", FEATURE_COLUMNS)]
        use_columns = bundle_columns if set(bundle_columns).issubset(set(available_columns)) else available_columns
        features = feat_df[use_columns].astype(float).values

        scaler = bundle["scaler"]
        if bundle["scaler_loaded"] and hasattr(scaler, "n_features_in_") and int(scaler.n_features_in_) == features.shape[1]:
            data = scaler.transform(features)
        else:
            data = scaler.fit_transform(features)

        seq = torch.tensor(data[-60:].reshape(1, 60, features.shape[1]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = bundle["model"](seq)
            probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
            pred = int(np.argmax(probs)) if production else int(np.random.choice(3, p=probs))

        # DIRECTION_LABELS = ["HOLD", "BUY", "SELL"]  — must match train_lstm.py create_sequences()
        # y=0 neutral/hold, y=1 buy (up + rsi>52), y=2 sell (down + rsi<48)
        direction = DIRECTION_LABELS[pred]
        confidence = round(float(probs[pred]), 4)

        # Scale bias by confidence so a low-confidence BUY doesn't over-commit
        raw_bias = _direction_to_trend_bias(direction)
        scaled_bias = round(raw_bias * confidence, 4)

        # ── Rainforest regime gate ──────────────────────────────────────────
        # Apply size reduction when rainforest confidence is below threshold.
        rf_regime = "ranging"
        rf_confidence = 0.0
        risk_scalar = _direction_to_risk_scalar(direction)
        try:
            rf_pred = self._get_rainforest_prediction(df, symbol)
            rf_regime = str(rf_pred.get("regime", "ranging"))
            rf_confidence = float(rf_pred.get("confidence", 0.0))
            rf_cfg = self._rf_cfg
            regime_gate = bool(rf_cfg.get("regime_gate", True))
            uncertainty_threshold = float(rf_cfg.get("uncertainty_threshold", 0.6))
            size_reduction_factor = float(rf_cfg.get("size_reduction_factor", 0.3))
            if regime_gate and rf_confidence < uncertainty_threshold:
                # Reduce risk scalar by size_reduction_factor (e.g. 30%)
                risk_scalar = round(risk_scalar * (1.0 - size_reduction_factor), 4)
        except Exception as _rf_gate_err:
            logger.debug(f"Rainforest gate error for {symbol}: {_rf_gate_err}")

        return {
            "signal": direction,
            "regime": direction,         # backward-compat key still used by logging
            "direction": direction,
            "confidence": confidence,
            "risk_scalar": risk_scalar,
            "trend_bias": scaled_bias,   # ±confidence (0..1) rather than capped ±0.10
            "trade_blocked": False,
            "symbol": symbol,
            "feature_version": feature_version,
            "probs": {
                "HOLD": round(float(probs[0]), 4),
                "BUY":  round(float(probs[1]), 4),
                "SELL": round(float(probs[2]), 4),
            },
            "rainforest_regime": rf_regime,
            "rainforest_confidence": round(rf_confidence, 4),
        }

    def extract_features(self, seq: torch.Tensor) -> torch.Tensor:
        seq = seq.to(self.device).float()
        self.model.train()

        expected = int(self.model.lstm.input_size)
        got = int(seq.shape[-1])
        if got < expected:
            pad = torch.zeros(seq.shape[0], seq.shape[1], expected - got, device=seq.device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=-1)
        elif got > expected:
            seq = seq[:, :, :expected]

        x, _ = self.model.lstm(seq)
        return x[:, -1, :]
