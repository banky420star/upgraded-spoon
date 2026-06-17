import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Python.agi_brain import AGIModel
from Python.config_utils import DEFAULT_TRADING_SYMBOLS, parse_symbol_list
from Python.data_feed import fetch_training_data
from Python.feature_pipeline import ENGINEERED_V2, ULTIMATE_150, build_lstm_feature_frame, normalize_feature_version
from alerts.telegram_alerts import TelegramAlerter

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "lstm_training.log"), rotation="10 MB", level="INFO")


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


def _resolve_cfg_value(v):
    if isinstance(v, str) and v.startswith("ENV:"):
        return os.environ.get(v.split(":", 1)[1])
    return v


def _build_alerter():
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
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


def create_sequences(data: np.ndarray, close_col: int, atr_col: int, rsi_col: int, seq_len: int = 60):
    X, y = [], []
    for i in range(seq_len, len(data) - 1):
        X.append(data[i - seq_len : i])

        prev_close = data[i - 1, close_col]
        next_close = data[i, close_col]
        future_ret = (next_close - prev_close) / (abs(prev_close) + 1e-8)

        atr_norm = abs(data[i, atr_col] / (abs(next_close) + 1e-8))
        rsi = data[i, rsi_col]

        up_thr = max(0.0007, atr_norm * 0.35)
        dn_thr = -up_thr

        if future_ret > up_thr and rsi > 52:
            y.append(1)
        elif future_ret < dn_thr and rsi < 48:
            y.append(2)
        else:
            y.append(0)

    return np.array(X), np.array(y)


def _train_one_symbol(
    symbol: str,
    epochs: int,
    seq_len: int,
    device: str,
    out_dir: str,
    period: str = "60d",
    interval: str = "5m",
    candles: int = 100_000,
    feature_version: str = ULTIMATE_150,
    data_source: str | None = None,
    alerter=None,
):
    if alerter is not None:
        try:
            alerter.training(
                "LSTM",
                f"Start {symbol} | epochs={epochs} | seq_len={seq_len} | period={period} | tf={interval} | features={feature_version}",
            )
        except Exception:
            pass

    df = fetch_training_data(
        symbol,
        period=period,
        interval=interval,
        strict=False,
        bars=int(candles),
        min_bars=int(candles),
        source=data_source,
    )
    if df.empty or len(df) < seq_len + 50:
        logger.warning(f"insufficient data for {symbol}, skipping")
        if alerter is not None:
            try:
                alerter.alert(f"LSTM skipped {symbol}: insufficient raw data")
            except Exception:
                pass
        return None

    fdf, feature_columns = build_lstm_feature_frame(df, feature_version=feature_version)
    if len(fdf) < seq_len + 50:
        logger.warning(f"insufficient engineered rows for {symbol}, skipping")
        if alerter is not None:
            try:
                alerter.alert(f"LSTM skipped {symbol}: insufficient engineered rows")
            except Exception:
                pass
        return None

    feat = fdf[feature_columns].values.astype(np.float32)
    scaler = MinMaxScaler()
    feat_scaled = scaler.fit_transform(feat)

    close_col = feature_columns.index("close") if "close" in feature_columns else 0
    atr_col = feature_columns.index("atr_14") if "atr_14" in feature_columns else close_col
    rsi_col = feature_columns.index("rsi_14") if "rsi_14" in feature_columns else close_col

    X, y = create_sequences(feat_scaled, close_col, atr_col, rsi_col, seq_len=seq_len)
    if len(X) == 0:
        logger.warning(f"no sequences for {symbol}")
        if alerter is not None:
            try:
                alerter.alert(f"LSTM skipped {symbol}: no sequences built")
            except Exception:
                pass
        return None

    model = AGIModel(input_dim=len(feature_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    batch_size = 64
    n_batches = max(1, len(X_tensor) // batch_size)

    model.train()
    last_loss = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(len(X_tensor))
        X_epoch = X_tensor[perm]
        y_epoch = y_tensor[perm]

        correct = 0
        total = 0
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, len(X_epoch))
            xb = X_epoch[start:end]
            yb = y_epoch[start:end]
            if len(xb) == 0:
                continue

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

        last_loss = epoch_loss / max(1, n_batches)
        acc = (correct / max(1, total)) * 100.0
        logger.info(f"{symbol} | epoch {epoch + 1}/{epochs} | loss {last_loss:.4f} | acc {acc:.2f}%")
        if alerter is not None and ((epoch + 1) == 1 or (epoch + 1) % 5 == 0 or (epoch + 1) == epochs):
            try:
                alerter.training(
                    "LSTM",
                    f"{symbol} epoch {epoch + 1}/{epochs} | loss={last_loss:.4f} | acc={acc:.2f}%",
                )
            except Exception:
                pass

    os.makedirs(out_dir, exist_ok=True)
    safe = symbol.replace("/", "_")
    model_path = os.path.join(out_dir, f"lstm_{safe}.pt")
    scaler_path = os.path.join(out_dir, f"lstm_scaler_{safe}.pkl")
    meta_path = os.path.join(out_dir, f"lstm_{safe}.meta.json")

    torch.save(model.state_dict(), model_path)

    import joblib

    joblib.dump(scaler, scaler_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": symbol,
                "feature_version": feature_version,
                "feature_columns": feature_columns,
                "seq_len": int(seq_len),
                "epochs": int(epochs),
                "samples": int(len(X)),
            },
            f,
            indent=2,
        )
    if alerter is not None:
        try:
            alerter.training("LSTM", f"Complete {symbol} | loss={last_loss:.4f} | samples={int(len(X))}")
        except Exception:
            pass

    return {
        "symbol": symbol,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "meta_path": meta_path,
        "loss": last_loss,
        "samples": int(len(X)),
    }


def train_lstm(
    symbols=None,
    epochs=20,
    seq_len=60,
    period="60d",
    interval="5m",
    candles=100_000,
    feature_version: str = ULTIMATE_150,
    data_source: str | None = None,
):
    if symbols is None:
        symbols = list(DEFAULT_TRADING_SYMBOLS)

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(
        f"LSTM per-symbol training on {device.upper()} | symbols={symbols} | epochs={epochs} | period={period} | tf={interval} | candles={candles:,} | features={feature_version}"
    )
    alerter = _build_alerter()
    try:
        alerter.training(
            "LSTM",
            f"Batch start | symbols={len(symbols)} | device={device.upper()} | epochs={epochs} | period={period} | tf={interval} | features={feature_version}",
        )
    except Exception:
        pass

    model_dir = os.path.join(PROJECT_ROOT, "models")
    per_symbol_dir = os.path.join(model_dir, "per_symbol")

    results = []
    for symbol in symbols:
        res = _train_one_symbol(
            symbol,
            epochs=epochs,
            seq_len=seq_len,
            device=device,
            out_dir=per_symbol_dir,
            period=period,
            interval=interval,
            candles=candles,
            feature_version=feature_version,
            data_source=data_source,
            alerter=alerter,
        )
        if res:
            results.append(res)

    if not results:
        logger.error("no symbol models trained")
        try:
            alerter.alert("LSTM batch failed: no symbol models trained")
        except Exception:
            pass
        return

    best = sorted(results, key=lambda x: x["loss"])[0]
    import shutil

    shutil.copy2(best["model_path"], os.path.join(model_dir, "lstm_agi_trained.pt"))
    shutil.copy2(best["scaler_path"], os.path.join(model_dir, "lstm_scaler.pkl"))
    shutil.copy2(best["meta_path"], os.path.join(model_dir, "lstm_agi_trained.meta.json"))
    logger.success(f"default lstm artifacts now point to best symbol model: {best['symbol']}")
    try:
        alerter.model(f"LSTM best model selected: {best['symbol']} | loss={best['loss']:.4f} | samples={best['samples']}")
    except Exception:
        pass


if __name__ == "__main__":
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    symbols = None
    epochs = 20
    period = "60d"
    interval = "5m"
    candles = 100_000
    feature_version = ULTIMATE_150
    data_source = None

    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        symbols = cfg.get("trading", {}).get("symbols")
        env_symbols = os.environ.get("AGI_LSTM_SYMBOLS")
        if env_symbols:
            symbols = parse_symbol_list(env_symbols)
        tcfg = cfg.get("training", {}) or {}
        epochs = _env_int("AGI_LSTM_EPOCHS", int(tcfg.get("lstm_epochs", 20)))
        period = _env_str("AGI_LSTM_PERIOD", str(tcfg.get("lstm_period", "90d")))
        interval = _env_str("AGI_LSTM_INTERVAL", str(tcfg.get("lstm_interval", cfg.get("trading", {}).get("timeframe", "M5"))))
        candles = _env_int("AGI_LSTM_CANDLES", int(tcfg.get("lstm_candles", 100000)))
        feature_version = normalize_feature_version(
            os.environ.get("AGI_FEATURE_VERSION") or tcfg.get("feature_version", ULTIMATE_150),
            default=ULTIMATE_150,
        )
        data_source = tcfg.get("data_source")
    else:
        feature_version = normalize_feature_version(os.environ.get("AGI_FEATURE_VERSION"), default=ULTIMATE_150)
        symbols = list(DEFAULT_TRADING_SYMBOLS)

    train_lstm(
        symbols=symbols,
        epochs=epochs,
        period=period,
        interval=interval,
        candles=candles,
        feature_version=feature_version,
        data_source=data_source,
    )
