import asyncio
import datetime
import json
import os
import subprocess
import sys
import time

from loguru import logger

from Python.config_utils import DEFAULT_TRADING_SYMBOLS, load_project_config, resolve_trading_symbols
from Python.model_evaluator import evaluate_candidate_vs_champion
from Python.model_registry import ModelRegistry
from alerts.telegram_alerts import TelegramAlerter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _append_perpetual_improvement(symbol: str, action: str, old_value, new_value, reason: str):
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "perpetual_improvement.jsonl")
    row = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "action": action,
        "symbol": symbol,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


class AutonomyLoop:
    def __init__(self, brain, interval_sec: int = 6 * 60 * 60):
        self.brain = brain
        self.registry = ModelRegistry()
        self._ensure_symbol_registry_seeded()
        self.interval_sec = int(os.environ.get("AGI_AUTONOMY_INTERVAL_SEC", str(3600)))
        self.train_every_sec = int(os.environ.get("AGI_AUTONOMY_TRAIN_EVERY_SEC", "0"))
        self.train_on_start = os.environ.get("AGI_AUTONOMY_TRAIN_ON_START", "false").lower() == "true"
        self.enable_train = os.environ.get("AGI_AUTONOMY_TRAIN", "true").lower() == "true"
        self.enable_auto_canary = os.environ.get("AGI_AUTONOMY_AUTO_CANARY", "true").lower() == "true"

        self.canary_min_trades = int(os.environ.get("CANARY_MIN_TRADES", "10"))
        self.canary_max_loss = float(os.environ.get("CANARY_MAX_LOSS", "75"))
        self.canary_max_dd = float(os.environ.get("CANARY_MAX_DD", "0.12"))
        self.min_score_delta = float(os.environ.get("AGI_GATE_MIN_SCORE_DELTA", "0.25"))
        self.max_eval_dd = float(os.environ.get("AGI_GATE_MAX_EVAL_DD", "0.20"))
        self.min_eval_sharpe = float(os.environ.get("AGI_GATE_MIN_EVAL_SHARPE", "-0.10"))
        self.min_eval_return = float(os.environ.get("AGI_GATE_MIN_EVAL_RETURN", "-0.02"))
        self.require_walkforward = os.environ.get("AGI_GATE_REQUIRE_WALKFORWARD", "false").lower() == "true"
        self.require_papertrade = os.environ.get("AGI_GATE_REQUIRE_PAPERTRADE", "false").lower() == "true"
        self.min_paper_trades = int(os.environ.get("AGI_GATE_MIN_PAPER_TRADES", "20"))
        self._canary_start_trade_count_by_symbol = {}
        self._canary_set_time_by_symbol = {}
        self._last_evaluated_candidate_by_symbol = {}
        self._last_train_ts = 0.0
        self._last_train_ts_by_symbol = {}
        self._train_locks = {}
        self._symbol_tasks = []
        self.training_state = {
            "active_canary": False,
            "cycles_completed": 0,
            "status": "idle",
            "last_check": 0.0,
        }
        self._training_cycle_last_candles = {}

        self.alerter = self._init_alerter()
        self.eval_config = self._load_evaluation_config()

    def _ensure_symbol_registry_seeded(self):
        symbols = self._load_symbols_cfg()
        if not symbols:
            return
        active = self.registry._read_active()
        global_champion = active.get("champion")
        sym_map = active.setdefault("symbols", {})
        touched = False
        for symbol in symbols:
            cur = sym_map.get(symbol)
            if not isinstance(cur, dict):
                cur = {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}
            cur.setdefault("champion", None)
            cur.setdefault("canary", None)
            cur.setdefault("canary_policy", {})
            cur.setdefault("canary_state", {})
            if cur.get("champion") and not self.registry.candidate_targets_symbol(cur.get("champion"), symbol):
                cur["champion"] = None
            if cur.get("canary") and not self.registry.candidate_targets_symbol(cur.get("canary"), symbol):
                cur["canary"] = None
                cur["canary_state"] = {}
            # Bootstrap with current global champion only when ownership matches the symbol lane.
            if not cur.get("champion") and global_champion and self.registry.candidate_targets_symbol(global_champion, symbol):
                cur["champion"] = global_champion
            sym_map[symbol] = cur
            touched = True
        if touched:
            self.registry._write_active(active)

    def _init_alerter(self):
        token = os.environ.get("TELEGRAM_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
        if os.path.exists(cfg_path):
            try:
                import yaml

                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                tel = cfg.get("telegram", {}) or {}
                token = token or tel.get("token")
                chat_id = chat_id or tel.get("chat_id")
            except Exception:
                pass

        if token in (None, "", "YOUR_BOT_TOKEN_HERE") or chat_id in (None, "", "YOUR_CHAT_ID_HERE"):
            return TelegramAlerter(None, None)
        return TelegramAlerter(token, chat_id)

    def _load_evaluation_config(self) -> dict:
        try:
            cfg = load_project_config(PROJECT_ROOT, live_mode=True)
            return cfg.get("evaluation", {}) or {}
        except Exception:
            return {}

    def _notify(self, message: str):
        try:
            self.alerter.alert(message)
        except Exception:
            pass

    def _update_candidate_metadata(self, candidate_dir: str, report: dict, gates_passed: bool, reasons: list[str]):
        payload = {
            "evaluation": {
                "candidate_score": float(report.get("candidate", {}).get("avg_score", 0.0)),
                "winner": bool(report.get("wins", False)),
                "gates_passed": bool(gates_passed),
                "gates_reasons": reasons,
                "forward_windows": report.get("forward_windows", []),
                "per_symbol": report.get("per_symbol_gates", []),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        }
        self.registry.update_metadata(candidate_dir, payload)

    def _load_symbols_cfg(self):
        try:
            cfg = load_project_config(PROJECT_ROOT, live_mode=True)
            return resolve_trading_symbols(
                cfg,
                env_keys=("AGI_AUTONOMY_SYMBOLS", "AGI_RUNTIME_SYMBOLS"),
                fallback=DEFAULT_TRADING_SYMBOLS,
            )
        except Exception:
            return list(DEFAULT_TRADING_SYMBOLS)

    def _dreamer_train_enabled(self) -> bool:
        try:
            cfg = load_project_config(PROJECT_ROOT, live_mode=False)
        except Exception:
            cfg = {}
        dreamer_cfg = ((cfg.get("drl", {}) or {}).get("dreamer", {}) or {}) if isinstance(cfg, dict) else {}
        return bool(dreamer_cfg.get("enabled", False) and dreamer_cfg.get("train_in_cycle", False))

    def _latest_candidate_dir(self, symbol: str | None = None):
        root = self.registry.candidates_dir
        if not os.path.exists(root):
            return None
        dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if not dirs:
            return None
        dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if not symbol:
            return dirs[0]

        target = str(symbol).replace("/", "_")
        for d in dirs:
            scorecard = os.path.join(d, "scorecard.json")
            if not os.path.exists(scorecard):
                continue
            try:
                with open(scorecard, "r", encoding="utf-8") as f:
                    sc = json.load(f) or {}
                sym = str(sc.get("symbol", "")).replace("/", "_")
                if sym == target:
                    return d
            except Exception:
                continue
        return None

    def _get_champion_dir(self, symbol: str | None = None):
        active = self.registry._read_active()
        if symbol:
            return active.get("symbols", {}).get(symbol, {}).get("champion")
        return active.get("champion")

    def _get_canary_dir(self, symbol: str | None = None):
        active = self.registry._read_active()
        if symbol:
            return active.get("symbols", {}).get(symbol, {}).get("canary")
        return active.get("canary")

    def _resolve_canary_target(self, symbol: str) -> tuple[str | None, str | None]:
        symbol_canary = self._get_canary_dir(symbol=symbol)
        if symbol_canary:
            return symbol_canary, "symbol"

        active = self.registry._read_active()
        global_canary = active.get("canary")
        if not global_canary:
            return None, None

        meta = (active.get("registry_metadata", {}) or {}).get("canary_metadata", {}) or {}
        symbols = meta.get("symbols")
        if not isinstance(symbols, list) or not symbols:
            sym = meta.get("symbol")
            symbols = [sym] if sym else []
        symbol_set = {str(s) for s in symbols if s}
        if str(symbol) in symbol_set:
            return global_canary, "global"
        return None, None

    def _read_candidate_metadata(self, candidate_dir: str) -> dict:
        meta_path = os.path.join(candidate_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _evaluate_release_gates(self, candidate_dir: str, report: dict) -> tuple[bool, list[str]]:
        reasons = []
        cand = report.get("candidate") or {}
        champ = report.get("champion") or {}

        cand_dd = float(cand.get("worst_drawdown", 1.0))
        cand_sharpe = float(cand.get("avg_sharpe", -999.0))
        cand_return = float(cand.get("avg_return", -999.0))
        cand_score = float(cand.get("avg_score", 0.0))
        champ_score = float(champ.get("avg_score", 0.0)) if champ else None

        if cand_dd > self.max_eval_dd:
            reasons.append(f"backtest_dd_fail:{cand_dd:.4f}>{self.max_eval_dd:.4f}")
        if cand_sharpe < self.min_eval_sharpe:
            reasons.append(f"backtest_sharpe_fail:{cand_sharpe:.4f}<{self.min_eval_sharpe:.4f}")
        if cand_return < self.min_eval_return:
            reasons.append(f"backtest_return_fail:{cand_return:.4f}<{self.min_eval_return:.4f}")

        if champ_score is not None:
            delta = cand_score - champ_score
            if delta < self.min_score_delta:
                reasons.append(f"score_delta_fail:{delta:.4f}<{self.min_score_delta:.4f}")

        if not report.get("wins"):
            reasons.append("candidate_win_checks_failed")

        for fw in report.get("forward_windows", []):
            if not bool(fw.get("wins")):
                reasons.append(f"forward_{fw.get('period', 'unknown')}_loss")

        meta = self._read_candidate_metadata(candidate_dir)
        if self.require_walkforward:
            wf = meta.get("walkforward", {}) if isinstance(meta, dict) else {}
            if not bool(wf.get("passed", False)):
                reasons.append("walkforward_fail:metadata.walkforward.passed!=true")

        if self.require_papertrade:
            paper = meta.get("paper_trade", {}) if isinstance(meta, dict) else {}
            paper_passed = bool(paper.get("passed", False))
            paper_trades = int(paper.get("trades", 0))
            if not paper_passed:
                reasons.append("paper_trade_fail:metadata.paper_trade.passed!=true")
            if paper_trades < self.min_paper_trades:
                reasons.append(f"paper_trade_min_trades_fail:{paper_trades}<{self.min_paper_trades}")

        return len(reasons) == 0, reasons

    def _champion_cycle_running(self) -> bool:
        """Return True if champion_cycle.py holds its lock file, indicating an active cycle."""
        lock_path = os.path.join(PROJECT_ROOT, ".tmp", "champion_cycle.lock")
        if not os.path.exists(lock_path):
            return False
        try:
            with open(lock_path, "r", encoding="utf-8") as fh:
                pid = int((fh.read() or "0").strip())
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    return True  # process is still alive — cycle is running
                except OSError:
                    pass  # stale lock
        except Exception:
            pass
        return False

    async def _train_symbol_candidate(self, symbol: str):
        # Do not start autonomous training while champion_cycle.py is actively running.
        # Both paths write to the model registry and must not race each other.
        if self._champion_cycle_running():
            logger.warning(f"Autonomy training skipped for {symbol}: champion_cycle lock is active")
            return

        lock = self._train_locks.setdefault(symbol, asyncio.Lock())
        async with lock:
            started = time.time()
            self._notify(f"Autonomy training started {symbol}: isolated LSTM + Dreamer + PPO")

            lstm_env = os.environ.copy()
            lstm_env["AGI_LSTM_SYMBOLS"] = str(symbol)
            subprocess.check_call([sys.executable, "training/train_lstm.py"], cwd=PROJECT_ROOT, env=lstm_env)
            self._notify(f"LSTM training finished {symbol}")

            if self._dreamer_train_enabled():
                dreamer_env = os.environ.copy()
                dreamer_env["AGI_DREAMER_SYMBOL"] = str(symbol)
                subprocess.check_call([sys.executable, "training/train_dreamer.py"], cwd=PROJECT_ROOT, env=dreamer_env)
                self._notify(f"Dreamer training finished {symbol}")

            drl_env = os.environ.copy()
            drl_env["AGI_DRL_SYMBOL"] = str(symbol)
            subprocess.check_call([sys.executable, "training/train_drl.py"], cwd=PROJECT_ROOT, env=drl_env)

            cand = self._latest_candidate_dir(symbol=symbol)
            if cand:
                self._last_evaluated_candidate_by_symbol[symbol] = cand
                self._maybe_set_canary(cand, symbol)

            elapsed = int(time.time() - started)
            self._last_train_ts = time.time()
            self._last_train_ts_by_symbol[str(symbol)] = self._last_train_ts
            self._notify(f"Autonomy training finished {symbol}. elapsed={elapsed}s")

    async def _train_candidate(self):
        symbols = self._load_symbols_cfg()
        if not symbols:
            return
        await asyncio.gather(*[self._train_symbol_candidate(s) for s in symbols])

    def _maybe_reload_brain(self):
        if hasattr(self.brain, "_load_ppo_from_registry"):
            try:
                self.brain._load_ppo_from_registry()
            except Exception as exc:
                logger.warning(f"brain reload failed: {exc}")

    def _maybe_set_canary(self, candidate_dir: str, symbol: str):
        import yaml

        cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
        if not os.path.exists(cfg_path):
            eval_period = "120d"
        else:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            eval_period = cfg.get("drl", {}).get("eval_period", "120d")

            # keep evaluation window bounded for short-interval market data backtests
            try:
                if isinstance(eval_period, str) and eval_period.endswith("d"):
                    d = int(eval_period[:-1])
                    if d > 60:
                        eval_period = "60d"
            except Exception:
                eval_period = "60d"

        report = evaluate_candidate_vs_champion(
            candidate_dir=candidate_dir,
            champion_dir=self._get_champion_dir(symbol=symbol),
            symbols=[symbol],
            period=eval_period,
            gates=self.eval_config,
            interval="5m",
        )

        if report.get("error"):
            logger.warning(f"Autonomy evaluator error: {report['error']}")
            self._notify(f"Autonomy evaluator error: {report['error']}")
            return

        gates_passed, reasons = self._evaluate_release_gates(candidate_dir, report)
        cand = report.get("candidate") or {}
        summary = (
            f"score={float(cand.get('avg_score', 0.0)):.4f}, "
            f"ret={float(cand.get('avg_return', 0.0)):.4f}, "
            f"dd={float(cand.get('worst_drawdown', 1.0)):.4f}, "
            f"sharpe={float(cand.get('avg_sharpe', -999.0)):.4f}"
        )

        self._update_candidate_metadata(candidate_dir, report, gates_passed, reasons)

        if self.enable_auto_canary and report["wins"] and report["passes_thresholds"] and gates_passed:
            self.registry.set_canary(
                candidate_dir,
                symbol=symbol,
                policy={
                    "min_trades": self.canary_min_trades,
                    "min_realized_pnl": 0.0,
                    "max_drawdown": self.canary_max_dd,
                    "min_runtime_minutes": 30,
                },
            )
            risk = getattr(self.brain, "risk_engine", None)
            by_symbol = getattr(risk, "daily_trades_by_symbol", {}) if risk is not None else {}
            self._canary_start_trade_count_by_symbol[symbol] = int(by_symbol.get(symbol, 0))
            self._canary_set_time_by_symbol[symbol] = time.time()
            self._notify(f"Canary enabled {symbol}: {os.path.basename(candidate_dir)} | {summary}")
            try:
                self.alerter.model(f"Canary set {symbol}: {candidate_dir}")
            except Exception:
                pass
        else:
            detail = ", ".join(reasons) if reasons else "wins_or_thresholds_not_met"
            logger.info(f"candidate not promoted: {detail}")
            self._notify(f"Candidate blocked {symbol}: {detail} | {summary}")

    def _canary_monitor(self, symbol: str):
        canary, scope = self._resolve_canary_target(symbol)
        if not canary:
            return

        risk = getattr(self.brain, "risk_engine", None)
        by_symbol = getattr(risk, "daily_trades_by_symbol", {}) if risk is not None else {}
        trades_now = int(by_symbol.get(symbol, 0))

        if symbol not in self._canary_start_trade_count_by_symbol:
            self._canary_start_trade_count_by_symbol[symbol] = trades_now
            self._canary_set_time_by_symbol[symbol] = time.time()

        trades_since = trades_now - int(self._canary_start_trade_count_by_symbol.get(symbol, 0))

        realized = 0.0
        try:
            from Python.mt5_compat import mt5
            import pytz

            if mt5 is not None and mt5.initialize():
                tz = pytz.timezone("Etc/UTC")
                now_utc = datetime.datetime.now(tz)
                lookback = now_utc - datetime.timedelta(days=7)
                deals = mt5.history_deals_get(lookback, now_utc)
                if deals:
                    realized = sum(
                        deal.profit
                        for deal in deals
                        if deal.entry == mt5.DEAL_ENTRY_OUT and str(getattr(deal, "symbol", "")) == str(symbol)
                    )
        except Exception as exc:
            logger.warning(f"Autonomy MT5 PnL check failed: {exc}")

        dd_pct = float(getattr(risk, "current_dd", 0.0)) / 100.0
        runtime_minutes = 0.0
        if symbol in self._canary_set_time_by_symbol:
            runtime_minutes = max(0.0, (time.time() - float(self._canary_set_time_by_symbol[symbol])) / 60.0)

        self.registry.update_canary_metrics(
            trades=trades_since,
            realized_pnl=realized,
            drawdown=dd_pct,
            runtime_minutes=runtime_minutes,
            symbol=(symbol if scope == "symbol" else None),
        )

        if realized <= -self.canary_max_loss or dd_pct >= self.canary_max_dd:
            self.registry.rollback_to_champion(symbol=(symbol if scope == "symbol" else None))
            self._canary_start_trade_count_by_symbol.pop(symbol, None)
            self._canary_set_time_by_symbol.pop(symbol, None)
            self._maybe_reload_brain()
            self._notify(f"Canary rollback {symbol}. realized={realized:.2f}, dd={dd_pct:.3f}")
            return

        if trades_since >= self.canary_min_trades and realized >= 0:
            try:
                self.registry.promote_canary_to_champion(symbol=(symbol if scope == "symbol" else None))
                self._canary_start_trade_count_by_symbol.pop(symbol, None)
                self._canary_set_time_by_symbol.pop(symbol, None)
                self._maybe_reload_brain()
                self._notify(f"Canary promoted {symbol}. trades={trades_since}, realized={realized:.2f}")
                try:
                    champ = self.registry.load_active_model(
                        prefer_canary=False,
                        symbol=(symbol if scope == "symbol" else None),
                    )
                    self.alerter.model(f"Champion promoted {symbol}: {champ}")
                except Exception:
                    pass
            except Exception as exc:
                logger.warning(f"Canary promotion blocked for {symbol}: {exc}")

    def _evaluate_and_maybe_promote(self, symbol: str, candidate: str):
        if not self._get_canary_dir(symbol=symbol):
            self._maybe_set_canary(candidate, symbol)

    def _symbol_due_for_training(self, symbol: str) -> bool:
        cadence = float(self.train_every_sec if self.train_every_sec > 0 else 24 * 60 * 60)
        last_ts = float(self._last_train_ts_by_symbol.get(str(symbol), 0.0) or 0.0)
        return (time.time() - last_ts) >= cadence

    async def _run_symbol_lane(self, symbol: str):
        if self.enable_train and self.train_on_start and str(symbol) not in self._last_train_ts_by_symbol:
            await self._train_symbol_candidate(symbol)

        _consecutive_errors = 0
        _MAX_BACKOFF_SEC = 300  # cap exponential backoff at 5 minutes

        while True:
            try:
                self._canary_monitor(symbol)

                candidate = self._latest_candidate_dir(symbol=symbol)
                if candidate:
                    last_eval = self._last_evaluated_candidate_by_symbol.get(symbol)
                    if last_eval != candidate:
                        self._evaluate_and_maybe_promote(symbol, candidate)
                        self._last_evaluated_candidate_by_symbol[symbol] = candidate

                if self.enable_train and self._symbol_due_for_training(symbol):
                    await self._train_symbol_candidate(symbol)

                # Successful iteration — reset error counter.
                _consecutive_errors = 0
                await asyncio.sleep(self.interval_sec)

            except Exception as exc:
                # Log before sleeping so failures are always visible in the log.
                _consecutive_errors += 1
                logger.warning(f"Symbol lane error {symbol} (attempt {_consecutive_errors}): {exc}")
                self._notify(f"Symbol lane error {symbol}: {exc}")

                # Exponential backoff: 2^(n-1) seconds, capped at _MAX_BACKOFF_SEC.
                backoff = min(_MAX_BACKOFF_SEC, 2 ** (_consecutive_errors - 1))
                logger.info(f"Symbol lane {symbol}: backing off {backoff}s before retry")
                await asyncio.sleep(backoff)

    def _run_symbol_cycle(self, symbol):
        """Train, backtest and promote a single symbol. Called in parallel."""
        if not self.enable_train:
            return {"symbol": symbol, "skipped": True}

        # ---- candle-count gate ----
        try:
            from Python.mt5_compat import mt5
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 99999)
            total_available = len(rates) if rates is not None else 0
        except Exception as exc:
            logger.warning(f"training_cycle: could not fetch candle count for {symbol}: {exc}")
            total_available = 0

        last_count = int(self._training_cycle_last_candles.get(str(symbol), 0))
        new_candles = total_available - last_count
        if new_candles < 1000:
            logger.info(
                f"training_cycle: {symbol} has only {new_candles} new 5m candles since last cycle; skipping"
            )
            return {"symbol": symbol, "skipped": True}
        self._training_cycle_last_candles[str(symbol)] = total_available

        self.training_state.update(
            {
                "status": f"training_{symbol}",
                "lstm_epoch": None,
                "ppo_timesteps": None,
                "backtest_score": None,
                "backtest_pnl": None,
                "backtest_drawdown": None,
                "promoted": False,
                "error": None,
            }
        )
        self._notify(f"Training cycle started {symbol}")

        # ---- LSTM ----
        try:
            lstm_env = os.environ.copy()
            lstm_env["AGI_LSTM_SYMBOLS"] = str(symbol)
            subprocess.check_call(
                [sys.executable, "training/train_lstm.py"],
                cwd=PROJECT_ROOT,
                env=lstm_env,
            )
            self.training_state["lstm_epoch"] = "completed"
        except Exception as exc:
            logger.warning(f"training_cycle LSTM failed for {symbol}: {exc}")
            self.training_state["error"] = f"lstm_failed:{exc}"
            return {"symbol": symbol, "error": f"lstm_failed:{exc}"}

        # ---- PPO ----
        try:
            ppo_env = os.environ.copy()
            ppo_env["AGI_DRL_SYMBOL"] = str(symbol)
            subprocess.check_call(
                [sys.executable, "training/train_ppo.py"],
                cwd=PROJECT_ROOT,
                env=ppo_env,
            )
            self.training_state["ppo_timesteps"] = "completed"
        except Exception as exc:
            logger.warning(f"training_cycle PPO failed for {symbol}: {exc}")
            self.training_state["error"] = f"ppo_failed:{exc}"
            return {"symbol": symbol, "error": f"ppo_failed:{exc}"}

        # ---- locate candidate ----
        candidate_dir = self._latest_candidate_dir(symbol=symbol)
        if not candidate_dir:
            logger.warning(f"training_cycle: no candidate found after training for {symbol}")
            self.training_state["error"] = "no_candidate"
            return {"symbol": symbol, "error": "no_candidate"}

        # ---- backtest ----
        try:
            report = evaluate_candidate_vs_champion(
                candidate_dir=candidate_dir,
                champion_dir=self._get_champion_dir(symbol=symbol),
                symbols=[symbol],
                period="60d",
                gates=self.eval_config,
                interval="5m",
            )
            cand = report.get("candidate") or {}
            pnl = float(cand.get("avg_return", -999.0))
            dd = float(cand.get("worst_drawdown", 1.0))
            score = float(cand.get("avg_score", 0.0))
            self.training_state["backtest_pnl"] = pnl
            self.training_state["backtest_drawdown"] = dd
            self.training_state["backtest_score"] = score
        except Exception as exc:
            logger.warning(f"training_cycle backtest failed for {symbol}: {exc}")
            self.training_state["error"] = f"backtest_failed:{exc}"
            return {"symbol": symbol, "error": f"backtest_failed:{exc}"}

        # ---- promotion gate ----
        if pnl > 0 and dd <= self.max_eval_dd:
            try:
                self.registry.set_canary(
                    candidate_dir,
                    symbol=symbol,
                    policy={
                        "min_trades": 0,
                        "min_realized_pnl": 0.0,
                        "max_drawdown": self.max_eval_dd,
                        "min_runtime_minutes": 0,
                    },
                )
                self.registry.promote_canary_to_champion(symbol=symbol, force=True)
                self.training_state["promoted"] = True
                self.training_state["active_canary"] = bool(self._get_canary_dir(symbol=symbol))
                self._notify(
                    f"Training cycle promoted champion for {symbol}: {os.path.basename(candidate_dir)}"
                )
            except Exception as exc:
                logger.warning(f"training_cycle promotion failed for {symbol}: {exc}")
                self.training_state["error"] = f"promotion_failed:{exc}"
        else:
            logger.info(
                f"training_cycle: candidate for {symbol} blocked by backtest gates pnl={pnl:.4f} dd={dd:.4f}"
            )
            self.training_state["error"] = f"backtest_gates:pnl={pnl:.4f},dd={dd:.4f}"

        # ---- perpetual improvement log ----
        try:
            _append_perpetual_improvement(
                symbol=symbol,
                action="training_cycle_complete",
                old_value=None,
                new_value=self.training_state.get("backtest_score"),
                reason=(
                    f"pnl={self.training_state.get('backtest_pnl')}, "
                    f"dd={self.training_state.get('backtest_drawdown')}, "
                    f"promoted={self.training_state.get('promoted', False)}"
                ),
            )
        except Exception:
            pass

        return {"symbol": symbol, "promoted": self.training_state.get("promoted", False)}

    def training_cycle(self, symbols, shutdown_event=None):
        """
        Synchronous training orchestration invoked from Server_AGI main loop.
        Checks new 5m candle count, runs LSTM + PPO, backtests the resulting
        candidate, and promotes to champion if backtest gates pass.
        Now runs all symbols in parallel using a thread pool.
        """
        now = time.time()
        self.training_state.update({"status": "checking", "last_check": now})

        max_workers = max(1, int(os.environ.get("AGI_TRAIN_PARALLEL_WORKERS", "0")))
        if max_workers == 0:
            max_workers = max(1, min(len(symbols), 4))

        if len(symbols) == 1 or max_workers == 1:
            for symbol in symbols:
                if shutdown_event is not None and shutdown_event.is_set():
                    logger.info("training_cycle: shutdown requested, stopping early")
                    break
                self._run_symbol_cycle(symbol)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self._run_symbol_cycle, s): s for s in symbols}
                for future in as_completed(futures):
                    if shutdown_event is not None and shutdown_event.is_set():
                        logger.info("training_cycle: shutdown requested, stopping early")
                        break
                    symbol = futures[future]
                    try:
                        result = future.result()
                        logger.info(f"training_cycle completed for {symbol}: {result}")
                    except Exception as exc:
                        logger.warning(f"training_cycle exception for {symbol}: {exc}")

        self.training_state["status"] = "idle"
        self.training_state["cycles_completed"] = int(self.training_state.get("cycles_completed", 0)) + 1

    async def start(self):
        logger.warning("AutonomyLoop started")
        self._notify("AutonomyLoop started")

        symbols = self._load_symbols_cfg()
        self._symbol_tasks = [asyncio.create_task(self._run_symbol_lane(symbol)) for symbol in symbols]

        if self._symbol_tasks:
            await asyncio.gather(*self._symbol_tasks)


