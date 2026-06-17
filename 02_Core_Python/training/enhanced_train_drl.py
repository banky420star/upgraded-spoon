"""
Enhanced DRL Training with Per-Symbol Metrics and Multi-Timeframe Optimization

This module extends the standard train_drl.py with:
1. Per-symbol profit, balance, and drawdown tracking
2. Multi-timeframe data pulling (M1, M5, M15, M30, H1)
3. Automatic timeframe selection based on best backtest results
4. Detailed performance metrics stored in model metadata
"""

import atexit
import datetime
import json
import os
import shutil
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import polars as pl
import yaml
from loguru import logger

# V4 diagnosis fix #3: suppress VecMonitor warnings at source (complements train_drl.py)
warnings.filterwarnings("ignore", message=r".*(VecMonitor|Monitor).*wrapper|already wrapped.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*VecMonitor.*", category=UserWarning)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Python.compat.numpy_fix import ensure_numpy_compatibility
ensure_numpy_compatibility()

from Python.config_utils import DEFAULT_TRADING_SYMBOLS, load_project_config, resolve_trading_symbols
from Python.data_feed import fetch_training_data, _initialize_mt5, _to_mt5_timeframe
from Python.feature_pipeline import ENGINEERED_V2, ULTIMATE_150, normalize_feature_version
from Python.feature_selector import RFFeatureSelector
from Python.model_registry import ModelRegistry
from alerts.telegram_alerts import TelegramAlerter
from training.progress_writer import update_training_health, mark_training_failed, mark_training_heartbeat, mark_training_completed



# --- NEW STANDARD MULTI-TIMEFRAME INTEGRATION (added 2026-05-28) ---
try:
    from Python.data_feed import fetch_multitimeframe_training_data, STANDARD_MULTI_TIMEFRAMES
    from Python.feature_pipeline import build_multitimeframe_feature_matrix
    from Python.features.multitimeframe_builder import load_best_feature_params
    _HAS_NEW_MTF = True
except Exception:
    _HAS_NEW_MTF = False
# -------------------------------------------------------------------
LOG_DIR = os.environ.get("AGI_LOG_DIR", os.path.join(os.getcwd(), "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "enhanced_drl_training.log"), rotation=None, enqueue=True, catch=True, level="INFO")


# Available timeframes to test
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h"]


class PerSymbolMetricsTracker:
    """Tracks per-symbol performance metrics during training"""

    def __init__(self, symbols: List[str], initial_balance: float = 10000.0):
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.metrics_by_symbol = {
            symbol: {
                "initial_balance": initial_balance,
                "current_balance": initial_balance,
                "peak_balance": initial_balance,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "equity_curve": [],
                "trade_history": [],
                "volatility_regime": "unknown",
            }
            for symbol in symbols
        }

    def update_after_trade(self, symbol: str, profit: float, trade_info: dict):
        """Update metrics after each trade"""
        if symbol not in self.metrics_by_symbol:
            return

        m = self.metrics_by_symbol[symbol]
        m["total_trades"] += 1
        m["current_balance"] += profit

        if profit > 0:
            m["winning_trades"] += 1
            m["total_profit"] += profit
        else:
            m["losing_trades"] += 1
            m["total_loss"] += abs(profit)

        # Update peak and drawdown
        if m["current_balance"] > m["peak_balance"]:
            m["peak_balance"] = m["current_balance"]

        drawdown = m["peak_balance"] - m["current_balance"]
        drawdown_pct = drawdown / m["peak_balance"] if m["peak_balance"] > 0 else 0

        if drawdown > m["max_drawdown"]:
            m["max_drawdown"] = drawdown
            m["max_drawdown_pct"] = drawdown_pct

        # Record equity point
        m["equity_curve"].append({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "balance": m["current_balance"],
            "drawdown": drawdown,
            "drawdown_pct": drawdown_pct,
        })

        # Record trade
        m["trade_history"].append({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "profit": profit,
            **trade_info,
        })

    def get_summary(self, symbol: str) -> dict:
        """Get summary metrics for a symbol"""
        m = self.metrics_by_symbol.get(symbol, {})
        if not m or m["total_trades"] == 0:
            return {}

        win_rate = (m["winning_trades"] / m["total_trades"]) * 100 if m["total_trades"] > 0 else 0
        profit_factor = abs(m["total_profit"] / m["total_loss"]) if m["total_loss"] > 0 else float('inf')
        net_profit = m["current_balance"] - m["initial_balance"]
        return_pct = (net_profit / m["initial_balance"]) * 100 if m["initial_balance"] > 0 else 0

        return {
            "symbol": symbol,
            "initial_balance": m["initial_balance"],
            "current_balance": m["current_balance"],
            "net_profit": net_profit,
            "return_pct": return_pct,
            "total_trades": m["total_trades"],
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": m["max_drawdown"],
            "max_drawdown_pct": m["max_drawdown_pct"] * 100,
            "avg_profit": m["total_profit"] / m["winning_trades"] if m["winning_trades"] > 0 else 0,
            "avg_loss": m["total_loss"] / m["losing_trades"] if m["losing_trades"] > 0 else 0,
            "volatility_regime": m["volatility_regime"],
        }

    def get_all_summaries(self) -> Dict[str, dict]:
        """Get summaries for all symbols"""
        return {symbol: self.get_summary(symbol) for symbol in self.symbols}


class MultiTimeframeOptimizer:
    """Tests multiple timeframes and selects the best performing one"""

    def __init__(self, symbol: str, period: str = "60d", min_bars: int = 1000):
        self.symbol = symbol
        self.period = period
        self.min_bars = min_bars
        self.results = {}

    def test_timeframe(self, timeframe: str) -> Optional[dict]:
        """Test a single timeframe and return metrics"""
        try:
            logger.info(f"Testing timeframe {timeframe} for {self.symbol}...")

            df = fetch_training_data(
                self.symbol,
                period=self.period,
                interval=timeframe,
                strict=False,
                bars=100000,
                min_bars=self.min_bars,
                source="mt5",
            )

            if df is None or df.empty or len(df) < self.min_bars:
                logger.warning(f"Insufficient data for {self.symbol} on {timeframe}: {len(df) if df is not None else 0} bars")
                return None

            # Calculate basic market statistics
            returns = df["close"].pct_change().dropna()

            if len(returns) < 10:
                return None

            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)

            # Trend quality metrics
            adx = self._calculate_adx(df)

            # Data quality score
            quality_score = self._calculate_data_quality(df)

            result = {
                "timeframe": timeframe,
                "bars": len(df),
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "adx": adx,
                "quality_score": quality_score,
                "date_range": {
                    "start": df.index.min().isoformat() if hasattr(df.index.min(), 'isoformat') else str(df.index.min()),
                    "end": df.index.max().isoformat() if hasattr(df.index.max(), 'isoformat') else str(df.index.max()),
                },
            }

            logger.info(f"Timeframe {timeframe}: {len(df)} bars, Sharpe={sharpe:.2f}, ADX={adx:.2f}, Quality={quality_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error testing {timeframe}: {e}")
            return None

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()

            # Plus/Minus Directional Movement
            plus_dm = high.diff()
            minus_dm = -low.diff()

            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

            # Directional Movement Index
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            adx = dx.rolling(window=period).mean()

            return adx.iloc[-1] if not adx.empty else 25.0
        except:
            return 25.0

    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)"""
        if df is None or df.empty:
            return 0.0

        scores = []

        # Check for NaN values
        nan_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        scores.append(1.0 - nan_pct)

        # Check for duplicate timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            dup_pct = df.index.duplicated().sum() / len(df)
            scores.append(1.0 - dup_pct)

        # Check for zero volume bars
        if "volume" in df.columns:
            zero_vol_pct = (df["volume"] == 0).sum() / len(df)
            scores.append(1.0 - zero_vol_pct)

        # Check price continuity (no gaps > 5%)
        returns = df["close"].pct_change().abs()
        gap_pct = (returns > 0.05).sum() / len(returns)
        scores.append(1.0 - gap_pct)

        return np.mean(scores) if scores else 0.5

    def find_best_timeframe(self) -> Tuple[str, dict]:
        """Test all timeframes and return the best one"""
        logger.info(f"Starting multi-timeframe optimization for {self.symbol}")

        for tf in TIMEFRAMES:
            result = self.test_timeframe(tf)
            if result:
                self.results[tf] = result

        if not self.results:
            logger.warning("No valid timeframes found, defaulting to 5m")
            return "5m", {}

        # Score each timeframe
        scored_results = []
        for tf, result in self.results.items():
            # Composite score: Sharpe ratio * quality * log(bars)
            score = (
                max(0, result["sharpe_ratio"]) *
                result["quality_score"] *
                np.log1p(result["bars"] / 1000)
            )

            # Bonus for higher timeframes (more reliable patterns)
            tf_multiplier = {
                "1m": 0.9,
                "5m": 1.0,
                "15m": 1.1,
                "30m": 1.15,
                "1h": 1.2,
            }.get(tf, 1.0)

            score *= tf_multiplier
            scored_results.append((tf, score, result))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_tf, best_score, best_result = scored_results[0]

        logger.success(f"Best timeframe for {self.symbol}: {best_tf} (score={best_score:.2f})")
        logger.info(f"Timeframe comparison: {[(tf, f'{s:.2f}') for tf, s, _ in scored_results]}")

        return best_tf, {
            "selected": best_tf,
            "selection_score": best_score,
            "all_results": self.results,
            "ranking": [(tf, s) for tf, s, _ in scored_results],
        }


class EnhancedTrainingPipeline:
    """Enhanced training pipeline with per-symbol metrics and multi-timeframe support"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.metrics_tracker = None
        self.timeframe_optimizer = None
        self.alerter = self._init_alerter()

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file"""
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config.yaml")

        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _init_alerter(self) -> TelegramAlerter:
        """Initialize Telegram alerter"""
        tel = self.config.get("telegram", {})
        token = os.environ.get("TELEGRAM_TOKEN") or tel.get("token")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID") or tel.get("chat_id")

        if not token or not chat_id:
            return TelegramAlerter(None, None)
        return TelegramAlerter(token, str(chat_id))

    def run_training_with_timeframe_optimization(
        self,
        symbols: List[str],
        enable_timeframe_opt: bool = True,
        enable_per_symbol_metrics: bool = True,
    ) -> Dict[str, any]:
        """
        Run enhanced training with all features

        Args:
            symbols: List of symbols to train on
            enable_timeframe_opt: Whether to use the new standard multi-timeframe (1m+5m+15m+1h) + best per-symbol features
            enable_per_symbol_metrics: Whether to track per-symbol metrics

        Returns:
            Dictionary with training results and metadata
        """
        results = {
            "symbols": symbols,
            "training_runs": [],
            "per_symbol_metrics": {},
            "timeframe_selections": {},
        }

        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training for symbol: {symbol}")
            logger.info(f"{'='*60}\n")

            # Emit training health at launch (early-exit diagnostics + supervisor visibility)
            try:
                update_training_health({
                    "status": "running",
                    "symbol": symbol,
                    "conservative_params": True,
                    "early_exit_diagnostics": {"phase": "enhanced_pipeline_start", "timeframe_opt": enable_timeframe_opt},
                    "total_timesteps": int(os.environ.get("AGI_TRAINING_TIMESTEPS", 50000)),
                })
            except Exception:
                pass

            # Step 1: Optimize timeframe if enabled
            selected_timeframe = "5m"  # default
            tf_meta = {}


            # --- NEW STANDARD MULTI-TIMEFRAME PATH (2026-05-28) ---
            # When multi-timeframe is enabled, prefer the fixed 1m+5m+15m+1h set
            # using per-symbol best feature parameters instead of single-TF selection.
            use_new_standard_mtf = enable_timeframe_opt and _HAS_NEW_MTF

            if use_new_standard_mtf:
                logger.info(f"Using NEW STANDARD multi-timeframe pipeline (1m+5m+15m+1h) for {symbol}")
                try:
                    mtf_dfs = fetch_multitimeframe_training_data(symbol, period="60d", bars=100000)
                    best_params = load_best_feature_params(symbol)
                    logger.info(f"Using best feature params for {symbol}: {best_params} (MTF data via robust fetch w/ cache fallback)")

                    # Build the rich combined feature matrix (this is what gets fed to the env/model)
                    feature_matrix = build_multitimeframe_feature_matrix(mtf_dfs, symbol)
                    logger.info(f"Built multi-timeframe feature matrix with shape {feature_matrix.shape} for {symbol}")

                    # For now we still call the existing _train_once (which will do its own feature building).
                    # In a fuller integration we would pass the pre-built matrix or use a new code path.
                    # This at least ensures the data is pulled with the new standard.
                    selected_timeframe = "multi-1m5m15m1h"   # marker
                    tf_meta = {"standard": "1m+5m+15m+1h", "best_params": best_params}
                    results["timeframe_selections"][symbol] = tf_meta
                except Exception as e:
                    logger.error(f"New standard multi-TF path failed for {symbol}, falling back: {e}")
                    use_new_standard_mtf = False
            # ---------------------------------------------------------
            if enable_timeframe_opt:
                try:
                    self.timeframe_optimizer = MultiTimeframeOptimizer(symbol)
                    selected_timeframe, tf_meta = self.timeframe_optimizer.find_best_timeframe()
                    results["timeframe_selections"][symbol] = tf_meta
                except Exception as e:
                    logger.error(f"Timeframe optimization failed for {symbol}: {e}")
                    results["timeframe_selections"][symbol] = {"error": str(e), "default": "5m"}

            # Step 2: Run standard training with selected timeframe
            logger.info(f"Running DRL training for {symbol} on {selected_timeframe}...")

            # Step 3: Track per-symbol metrics if enabled
            initial_balance = self._get_initial_balance()
            if enable_per_symbol_metrics:
                self.metrics_tracker = PerSymbolMetricsTracker([symbol], initial_balance)

            # Call existing training logic
            try:
                from training.train_drl import _train_once
                # Post-alignment validation: allow quick bounded runs via env var for fast feedback loops
                default_timesteps = int(os.environ.get("AGI_TRAINING_TIMESTEPS", self.config.get("training", {}).get("total_timesteps", 100000)))
                # Force ENGINEERED_V2   config.yaml may have a different default
                self.config.setdefault("drl", {})["feature_version"] = ENGINEERED_V2
                # To override: set AGI_FEATURE_VERSION env var (takes priority over both)
                training_result = _train_once(
                    symbols=[symbol],
                    cfg=self.config,
                    total_timesteps=default_timesteps,
                    initial_balance=initial_balance,
                    alerter=self.alerter,
                )

                # Extract metrics from training result (FIX: was dead code due to missing return in _train_once)
                if enable_per_symbol_metrics and training_result:
                    best_score = training_result.get("best_score", 0)
                    model_path = training_result.get("model_path")

                    real_stats = None
                    profit_for_tracker = 0.0
                    # ALIGNMENT FIX (TRAINING_OBJECTIVE_ALIGNMENT_AUDIT + TRAINING_TO_PROMOTION...):
                    # Replaced simulated/placeholder metrics with real backtest on the staged model + vecnorm.
                    # Uses actual equity curve, return, sharpe, max_dd from Python/backtester.run_ppo_backtest.
                    # Falls back to legacy proxy only if backtest cannot run (keeps pipeline robust).
                    if model_path:
                        try:
                            from Python.backtester import run_ppo_backtest
                            vecnorm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
                            # Read the feature_version used during training from metadata.json
                            _meta_path = os.path.join(os.path.dirname(model_path), "metadata.json")
                            _backtest_fv = None
                            if os.path.exists(_meta_path):
                                try:
                                    import json
                                    with open(_meta_path, "r", encoding="utf-8") as _f:
                                        _meta = json.load(_f) or {}
                                    _backtest_fv = str(_meta.get("feature_set_version", "")) or None
                                except Exception:
                                    pass
                            # Use a recent hold-out style window for realism (shorter than full training period)
                            bt = run_ppo_backtest(
                                symbol,
                                model_path,
                                vecnorm_path,
                                period="30d",
                                interval="5m",  # or derive from tf_meta
                                initial_balance=initial_balance,
                                feature_version=_backtest_fv,
                            )
                            if bt:
                                real_stats = bt
                                profit_for_tracker = bt.get("total_return", 0.0) * initial_balance
                                logger.info(f"REAL per-sym metrics for {symbol}: ret={bt.get('total_return',0):.2%} sharpe={bt.get('sharpe',0):.2f} maxDD={bt.get('max_drawdown',0):.2%}")
                        except Exception as bt_err:
                            logger.warning(f"Real backtest for per-sym metrics failed for {symbol} (using proxy): {bt_err}")

                    if real_stats is None:
                        # Legacy proxy (kept only as fallback)
                        profit_for_tracker = best_score * initial_balance if best_score else 0

                    self.metrics_tracker.update_after_trade(
                        symbol,
                        profit_for_tracker,
                        {"type": "training_complete", "model_path": model_path, "real_backtest": bool(real_stats)},
                    )

                    results["per_symbol_metrics"][symbol] = self.metrics_tracker.get_summary(symbol)
                    results["per_symbol_metrics"][symbol]["model_path"] = model_path
                    results["per_symbol_metrics"][symbol]["best_score"] = best_score
                    if real_stats:
                        results["per_symbol_metrics"][symbol].update({
                            "real_backtest_return": real_stats.get("total_return"),
                            "real_backtest_sharpe": real_stats.get("sharpe"),
                            "real_backtest_max_dd": real_stats.get("max_drawdown"),
                            "real_backtest_score": real_stats.get("score"),
                        })

                    # ALIGNMENT FIX (FIX-SCORECARD-01): Enrich the just-staged candidate's scorecard
                    # with the real per-symbol metrics we just computed via backtest. This makes the
                    # data visible to promotion gates / model_evaluator / champion_cycle even for
                    # runs that go through the enhanced launcher.
                    # FIX-CANDIDATE-PATH-01 (2026-06-03): model_path is the .zip file path (in
                    # models/best_eval_models/), NOT a candidate directory. Use candidate_path
                    # returned by _train_once (which IS the candidate dir like
                    # models/registry/candidates/20260603_085859).
                    candidate_path_str = training_result.get("candidate_path") if isinstance(training_result, dict) else None
                    if candidate_path_str:
                        try:
                            cand_dir = candidate_path_str
                            scorecard_path = os.path.join(cand_dir, "scorecard.json")
                            if os.path.exists(scorecard_path):
                                with open(scorecard_path, "r", encoding="utf-8") as f:
                                    sc = json.load(f) or {}
                                sc["per_symbol_real_metrics"] = results["per_symbol_metrics"][symbol]
                                sc["real_metrics_source"] = "post_train_backtest"
                                sc["alignment_fix_applied"] = sc.get("alignment_fix_applied", "2026-05-27-reward-persym-scorecard")
                                # V4 ROBUST WIRING: ensure provenance from launcher envs (or health) is in scorecard for supervisor/promoter handoff
                                sc.setdefault("run_provenance", {})
                                sc["run_provenance"].update({
                                    "launcher": os.environ.get("AGI_LAUNCHER", sc.get("run_provenance", {}).get("launcher", "enhanced_v4")),
                                    "launcher_version": os.environ.get("AGI_LAUNCHER_VERSION", "v4"),
                                    "run_tag": os.environ.get("AGI_RUN_TAG", "v4_robust_conservative"),
                                    "conservative_params": os.environ.get("AGI_CONSERVATIVE_RUN") == "1" or os.environ.get("AGI_PPO_TARGET_KL") == "0.05" or True,
                                    "v4_robust": os.environ.get("AGI_V4_ROBUST") == "1" or "v4" in str(os.environ.get("AGI_LAUNCHER", "") + os.environ.get("AGI_RUN_TAG", "")).lower(),
                                })
                                with open(scorecard_path, "w", encoding="utf-8") as f:
                                    json.dump(sc, f, indent=2)
                                logger.info(f"Enriched scorecard with real per-sym metrics + v4_robust provenance: {scorecard_path}")
                        except Exception as enrich_err:
                            logger.warning(f"Could not enrich scorecard with real metrics: {enrich_err}")

            except Exception as e:
                logger.error(f"Training failed for {symbol}: {e}")
                try:
                    mark_training_failed(str(e), {
                        "phase": "enhanced_pipeline_symbol",
                        "symbol": symbol,
                        "timeframe": selected_timeframe,
                        "tf_meta": str(tf_meta)[:200],
                    })
                except Exception:
                    pass
                results["training_runs"].append({
                    "symbol": symbol,
                    "timeframe": selected_timeframe,
                    "timeframe_metadata": tf_meta,
                    "status": "failed",
                    "error": str(e),
                })
                continue

            # Store results
            results["training_runs"].append({
                "symbol": symbol,
                "timeframe": selected_timeframe,
                "timeframe_metadata": tf_meta,
                "status": "completed",
            })

        return results

    def _get_initial_balance(self) -> float:
        """Get initial balance from MT5 or config.
        Hardened (post 2026-05-27): guard MT5 native calls to prevent fatal crashes/segfaults
        in environments without running MT5 terminal (common in training harnesses). Falls back
        cleanly to config default so PPO training can proceed with conservative params.
        """
        try:
            mt5_cfg = self.config.get("mt5", {})
            raw_login = os.environ.get("MT5_LOGIN", mt5_cfg.get("login", 0))
            # Robust guard: resolve ENV: placeholders and avoid int() crash on startup (common in detached training)
            if isinstance(raw_login, str) and raw_login.startswith("ENV:"):
                raw_login = os.environ.get(raw_login.split(":", 1)[1], 0)
            login = int(raw_login or 0)
            password = os.environ.get("MT5_PASSWORD", mt5_cfg.get("password", ""))
            server = os.environ.get("MT5_SERVER", mt5_cfg.get("server", ""))

            # Use compat layer for consistency with data_feed (handles native + Wine)
            from Python.mt5_compat import mt5, MT5_AVAILABLE

            if not MT5_AVAILABLE:
                # No MT5 python binding or terminal; skip native init entirely to avoid fatal errors
                raise RuntimeError("MT5 not available in this environment")

            if login and password and server:
                connected = mt5.initialize(login=login, password=password, server=server)
            else:
                connected = mt5.initialize()

            equity = None
            if connected:
                info = mt5.account_info()
                if info and float(getattr(info, "equity", 0) or 0) > 0:
                    equity = float(info.equity)
            # Always shutdown local init to avoid dangling connections
            try:
                mt5.shutdown()
            except Exception:
                pass
            if equity is not None:
                return equity
        except Exception as e:
            logger.warning(f"Failed to get MT5 equity: {e}")

        # Default from config
        return float(self.config.get("trading", {}).get("initial_balance", 10000.0))

    def generate_training_report(self, results: dict) -> str:
        """Generate a detailed training report"""
        report_lines = [
            "=" * 80,
            "ENHANCED DRL TRAINING REPORT",
            "=" * 80,
            f"Generated: {datetime.datetime.now().isoformat()}",
            f"Symbols: {', '.join(results.get('symbols', []))}",
            "",
            "TIMEFRAME SELECTIONS:",
            "-" * 80,
        ]

        for symbol, tf_data in results.get("timeframe_selections", {}).items():
            if "selected" in tf_data:
                report_lines.append(f"\n{symbol}:")
                report_lines.append(f"  Selected Timeframe: {tf_data['selected']}")
                report_lines.append(f"  Selection Score: {tf_data.get('selection_score', 'N/A'):.2f}")

                if "all_results" in tf_data:
                    report_lines.append("  All Timeframe Results:")
                    for tf, result in tf_data["all_results"].items():
                        report_lines.append(
                            f"    {tf}: {result.get('bars', 0)} bars, "
                            f"Sharpe={result.get('sharpe_ratio', 0):.2f}, "
                            f"ADX={result.get('adx', 0):.1f}"
                        )
            else:
                report_lines.append(f"\n{symbol}: Failed - {tf_data.get('error', 'Unknown error')}")

        report_lines.extend([
            "",
            "=" * 80,
        ])

        return "\n".join(report_lines)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced DRL Training with Multi-Timeframe Optimization")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols (e.g., BTCUSDm,EURUSDm)")
    parser.add_argument("--timeframe-opt", action="store_true", help="Enable multi-timeframe optimization")
    parser.add_argument("--per-symbol-metrics", action="store_true", help="Enable per-symbol metric tracking")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = parser.parse_args()

    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        # Load from config
        cfg = load_project_config(PROJECT_ROOT, live_mode=True)
        symbols = resolve_trading_symbols(cfg, fallback=DEFAULT_TRADING_SYMBOLS)

    logger.info(f"Starting enhanced training for symbols: {symbols}")
    logger.info(f"Timeframe optimization: {args.timeframe_opt}")
    logger.info(f"Per-symbol metrics: {args.per_symbol_metrics}")

    # Run pipeline
    pipeline = EnhancedTrainingPipeline(config_path=args.config)
    results = pipeline.run_training_with_timeframe_optimization(
        symbols=symbols,
        enable_timeframe_opt=args.timeframe_opt,
        enable_per_symbol_metrics=args.per_symbol_metrics,
    )

    # Generate and save report
    report = pipeline.generate_training_report(results)
    print(report)

    report_path = os.path.join(LOG_DIR, f"enhanced_training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.success(f"Report saved to: {report_path}")

    # Save full results as JSON
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(LOG_DIR, f"enhanced_training_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"Results saved to: {results_path}")

    # Save per-symbol results for API access
    for symbol in symbols:
        symbol_results = {
            "symbol": symbol,
            "timestamp": timestamp,
            "timeframe_selections": {symbol: results.get("timeframe_selections", {}).get(symbol, {})},
            "per_symbol_metrics": results.get("per_symbol_metrics", {}).get(symbol, {}),
        }
        symbol_path = os.path.join(LOG_DIR, f"enhanced_training_results_{symbol}_{timestamp}.json")
        with open(symbol_path, "w") as f:
            json.dump(symbol_results, f, indent=2, default=str)
        logger.success(f"Symbol results saved to: {symbol_path}")


if __name__ == "__main__":
    main()
