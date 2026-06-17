# supreme-chainsaw

**Regime-routed PPO trading agent with real feature ablation testing for XAUUSDm (Gold).**

---

## Overview

This project trains reinforcement learning trading agents (PPO-based) on real market data and systematically ablates feature groups to identify which engineered features actually help — and which are noise.

The core question: *Which of our 59 engineered features drive performance, and which should be removed?*

---

## Prerequisites

**Python 3.14.5** — pinned in [`.python-version`](.python-version) for reproducible builds.

If you use `pyenv`, the correct version is auto-selected when entering the project directory.

### Setup

```bash
python -m venv venv
venv/Scripts/activate   # Windows: venv\Scripts\activate; Unix: source venv/bin/activate
pip install -r requirements.txt
```

Verify:

```bash
python --version   # should print Python 3.14.5
```

---

## Architecture

```
Market Data (XAUUSDm M5)
       ↓
ENGINEERED_V2 Feature Pipeline (59 columns)
       ↓
Windowed Observations (100 bars × 59 features + regime)
       ↓
AdaptiveLSTMFeatureExtractor
  ├── Bidirectional LSTM (2-layer, hidden=160)
  ├── Multi-Head Attention Pooling (4 heads)
  ├── Projection (→ 256-dim)
  ├── [opt] TrendMomentumBiasLayer (6 bias features)
  └── [opt] FeatureGroupGate (regime-conditional)
       ↓
RegimeRoutedPPO Policy
  ├── Regime classifier (5 regimes)
  ├── Actor heads (per-regime)
  └── Critic head
       ↓
Action: position size, direction, exits
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| **Real Feature Ablation Harness** | `training/run_real_feature_ablation.py` | Trains PPO with feature groups ablated, measures impact |
| **AdaptiveLSTMFeatureExtractor** | `drl/adaptive_feature_extractor.py` | Bidirectional LSTM + attention + projection |
| **TrendMomentumBiasLayer** | `drl/trend_momentum_bias.py` | Soft directional prior (6 bias features) — currently parked |
| **FeatureGroupGate** | `drl/adaptive_feature_extractor.py` | Regime-conditional feature weighting |
| **RegimeRoutedPPO** | `drl/regime_routed_policy.py` | Per-regime actor heads + shared critic |
| **ENGINEERED_V2 Pipeline** | `Python/feature_pipeline.py` | 59-column feature matrix builder |

---

## Feature Ablation Groups

The harness tests which feature groups help vs hurt by zeroing out specific columns:

| Group | Indices | What it removes | Columns |
|---|---|---|---|
| `ALL` | — | Nothing (baseline) | All 59 |
| `NO_TREND` | 19-20 | Trend indicator | `htf_trend`, `vol_bucket` |
| `NO_MOMENTUM` | 5-7 | Price momentum | `log_ret1`, `log_ret5`, `log_ret20` |
| `NO_VOLATILITY` | 12 | Realized volatility | `rv_20` |
| `NO_VOLUME` | 13-14 | Volume + spread | `rel_volume`, `spread_est_bps` |
| `NO_CROSS_ASSET` | 40-57 | Cross-asset correlations | 18 columns |
| `NO_ML_SIGNAL` | 58 | XGBoost direction | `ml_signal_prob` |
| `NO_PATTERN` | 29-39 | Candlestick patterns | 11 columns |
| `NO_REGIME` | — | Regime routing | Disables regime detector |
| `TREND_MOMENTUM_FIRST` | — | Bias layer ON | All features + bias layer |
| `NO_BIAS_SATURATION` | — | Fixed temp=0.1 | Bias layer with clamped temperature |

### Fingerprint Verification

Every group's feature matrix is fingerprinted (MD5 hash) before training and compared against the `ALL` baseline:

```python
def matrix_fingerprint(x: np.ndarray) -> str:
    arr = np.nan_to_num(x).astype(np.float32)
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]
```

If a group that *should* differ produces an identical fingerprint, the harness raises `AssertionError` — the ablation mask is broken.

---

## Key Findings

### Full 30K-step Ablation (with LSTM extractor)

| Rank | Group | Sharpe | WinRate | PF | MaxDD | Verdict |
|---|---|---|---|---|---|---|
| 1 | NO_TREND* | +10.94 | 51.2% | 1.12 | -2.0% | Trend features harmful |
| 2 | NO_MOMENTUM | +9.54 | 51.2% | 1.12 | -2.0% | Momentum harmful |
| 3 | NO_VOLUME | -2.76 | 49.9% | 1.08 | -4.1% | Volume slightly helpful |
| 4 | NO_REGIME | -10.94 | 48.8% | 0.89 | -6.0% | Regime routing helps |
| 5 | NO_ML_SIGNAL | -12.76 | 50.2% | 1.05 | -6.8% | ML signal helps |
| 6 | NO_VOLATILITY | -21.10 | 48.8% | 0.90 | -8.9% | Volatility helps |
| 7 | NO_PATTERN | -23.48 | 49.7% | 0.94 | -10.2% | Patterns help |
| 8 | NO_CROSS_ASSET | -30.62 | 49.0% | 0.94 | -9.9% | Cross-asset very important |
| 9 | ALL (baseline) | -37.12 | 49.3% | 0.91 | -11.0% | Full set worst — too noisy |

> **\*Important**: The original NO_TREND result was invalid — see Bug Fix section below.

### TrendMomentumBiasLayer — Parked

The bias layer (soft directional prior) was tested extensively and found to add noise rather than signal:

| Configuration | Sharpe | Bias behavior |
|---|---|---|
| `NO_BIAS_SATURATION` (fixed temp=0.1) | Best | Bias stays neutral → closest to ALL |
| `ALL` (no bias) | Baseline | — |
| `TREND_MOMENTUM_FIRST` (learnable temp) | Worst | Bias recalibrates but degrades performance |

The bias layer is currently parked — it will be rebuilt as an isolated risk-sizing encoder later.

---

## Critical Bug Fix (June 2026)

**`trend` group indices were wrong:**

| Before (bug) | After (fix) |
|---|---|
| `[15, 16]` → `hour_sin`, `hour_cos` (time-of-day) | `[19, 20]` → `htf_trend`, `vol_bucket` (trend indicator) |

The original `NO_TREND` ablation was actually removing **time-of-day features**, not trend features. All six other group indices were verified correct against the pipeline column order.

Fixed in commit `30040b2`. The `NO_TREND` results above are from the **original** (buggy) run and need re-validation.

---

## Running the Harness

### Quick smoke test (fingerprint verification)

```bash
python training/run_real_feature_ablation.py \
  --symbol XAUUSDm \
  --steps 1000 \
  --groups ALL NO_TREND NO_MOMENTUM \
  --n-bars 1000 \
  --verbose
```

### Full ablation study

```bash
python training/run_real_feature_ablation.py \
  --symbol XAUUSDm \
  --steps 30000 \
  --groups ALL NO_TREND NO_MOMENTUM NO_VOLATILITY NO_VOLUME \
           NO_CROSS_ASSET NO_ML_SIGNAL NO_PATTERN NO_REGIME \
  --n-bars 5000 \
  --verbose
```

### Bias layer diagnostics

```bash
python training/run_real_feature_ablation.py \
  --symbol XAUUSDm \
  --steps 5000 \
  --groups NO_BIAS_SATURATION TREND_MOMENTUM_FIRST ALL \
  --n-bars 3000 \
  --verbose
```

---

## Branch

`experiment/xauusd-regime-baseline`

## Status

- ✅ Real feature ablation harness working end-to-end
- ✅ LSTM feature extractor integrated
- ✅ Fingerprint verification prevents silent mask failures
- ✅ Bias layer diagnostics in place
- ✅ Trend column index bug fixed
- 🔄 Re-run full ablation with corrected indices
- 🔄 Clean feature pipeline based on ablation results
- 🔄 Rebuild bias layer as isolated risk-sizing encoder

---

## Key Commits

| Hash | Description |
|---|---|
| `30040b2` | Fix trend group indices [15,16]→[19,20] |
| `286d045` | Add NO_BIAS_SATURATION ablation group |
| `4bb7fa9` | Fix bias layer saturation (temperature-scaled activations) |
| `22f334d` | Add bias layer diagnostics to harness |
| `d69b800` | Upgrade harness to use AdaptiveLSTMFeatureExtractor |
| `84d2fd4` | Real feature ablation harness created |

## Test infrastructure

Unit tests live under `02_Core_Python/tests/` and use `pytest` + `pytest`'s `caplog` fixture. Shared helper fixtures (`clock_mock`, `build_ollama_response`, `assert_cooldown_bounds`) are centralised in `02_Core_Python/tests/conftest.py`. A `loguru -> stdlib` bridge in the same `conftest.py` lets `caplog` see loguru emissions even though `training_analyzer.py` writes logs through `loguru` directly.

The two key reliability contracts are locked end-to-end:

- `test_training_analyzer_cooldown.py` -- the cooldown-arm / suppress / recover lifecycle (7 tests).
- `test_training_analyzer_logging.py`   -- the no-spam logging contract: a Muting WARNING fires exactly once on the first failure, no further warnings fire during the cooldown, an INFO `Ollama recovered` fires on the first post-cooldown success, and the json-decode ValueError path emits its own dedicated 15 s-cooldown WARNING.

Run them with: `cd 02_Core_Python && pytest tests -q`.

### Contract tests

The `02_Core_Python/tests/` directory contains contract tests that lock system-level wiring against regressions:

- **`test_api_server_health.py`** — Locks the `api_server` `/api/health` endpoint contract.
  Verifies that `Server_AGI` writes the expected `live_state.json` schema
  (`api_server.py:_read_live_state:385` reads `ROOT` at call time, no module-level
  cache); that the `/api/health` response exposes `server_running` and
  `brain_initialized` as booleans; and that `training.cycles_completed > 0` coerces
  the state to `healthy` / `degraded`. A malformed-JSON test confirms the reader
  fails soft instead of propagating `JSONDecodeError` to callers.

These run on every push to `main` via `.github/workflows/python-tests.yml`
(matrix `3.12`/`3.13`); adding a new test file under `02_Core_Python/tests/` is
picked up automatically.
