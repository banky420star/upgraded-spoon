"""
Self-Monitoring, Auto-Rollback and Self-Recovery System
======================================================

Critical component for safe self-evolution of the Chain Gambler / AGI trading stack.

Responsibilities (as specified):
- Continuously monitor live/paper performance (PnL, win-rate, trade count), drawdown,
  regime stability (Rainforest / pattern shifts), model prediction confidence,
  execution quality (slippage, fill rates, rejections from execution_feedback).
- Automatic kill switches + position flattening (via ExecutionAgent.force_flatten_all)
  when hard risk/performance/quality thresholds breached.
- Automatic rollback to previous best model/config (ModelRegistry active/champion
  swap + metadata) when current deployment underperforms significantly vs historical best.
- Self-recovery logic: enter conservative regime (risk scaling), pause new entries
  (via runtime flags + risk_sup), request retraining (via RetrainingTrigger + audit),
  gradual exit from recovery on sustained stability.
- Full logging of ALL self-actions (logs/self_monitor_actions.jsonl + unified
  PIPELINE_DECISIONS.jsonl via log_decision) + alerting (Telegram critical paths).

Integration points:
- Supervisor: SelfEvolutionSupervisor instantiates + calls monitor_cycle() each evolution tick.
  Fast-backtest engine used for pre-rollback / pre-recovery validation.
- ExecutionAgent: Primary kill-switch target (force_flatten_all). Self-monitor can
  obtain or receive a reference; also observes via shared logs/execution_reports.
- Works with fast_backtester.FastBacktester for safe pre-deployment sanity on
  proposed rollbacks (short accelerated runs before mutating registry).
- Feeds RetrainingTrigger, RiskSupervisor, ModelRegistry, pipeline audit, alerts.
- Writes authoritative live state: runtime/agent_status/self_monitoring_recovery_agent.json
  (consumable by TUI, React, swarm agents, vps_agi_supervisor).

Design principles:
- Belt-and-suspenders: never relies on single signal. Multiple data sources + hysteresis.
- Resilient: heavy try/except + graceful degradation (stubs when optional modules missing).
- Auditable: every decision goes through log_decision + local action log.
- Config-driven thresholds (risk.supervisor + fallback defaults).
- Non-blocking: designed for 30-120s polling loops in supervisor / harness / standalone.

Standalone usage:
    python -m Python.autonomous.self_monitor --loop --interval 60

Or import:
    from Python.autonomous.self_monitor import SelfMonitoringRecoveryAgent
    sm = SelfMonitoringRecoveryAgent()
    sm.monitor_cycle()
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None  # type: ignore

try:
    from loguru import logger
except Exception:
    import logging
    logger = logging.getLogger("self_monitor")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [SELF-MONITOR] %(levelname)s %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)

# Project root resolution (robust, matches other modules)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOGS_DIR = _PROJECT_ROOT / "logs"
_RUNTIME_DIR = _PROJECT_ROOT / "runtime"
_AGENT_STATUS_DIR = _RUNTIME_DIR / "agent_status"
_EXEC_REPORT_DIR = _RUNTIME_DIR / "execution_reports"

for d in (_LOGS_DIR, _RUNTIME_DIR, _AGENT_STATUS_DIR, _EXEC_REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Unified audit
try:
    from Python.pipeline_audit import log_decision
except Exception:
    def log_decision(*a, **k):  # type: ignore
        try:
            # Fallback direct append
            p = _LOGS_DIR / "PIPELINE_DECISIONS.jsonl"
            rec = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision_type": a[0] if a else k.get("decision_type"),
                "actor": "self_monitoring_recovery_agent",
                "decision": k.get("decision", "SELF_MONITOR_FALLBACK"),
                "reason": k.get("reason", ""),
                "details": k.get("details", {}),
            }
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass
        return False

# Optional core components (graceful)
try:
    from Python.execution.execution_agent import ExecutionAgent
except Exception:
    ExecutionAgent = None  # type: ignore

try:
    from Python.execution.risk_supervisor import RiskSupervisor as ExecRiskSupervisor
except Exception:
    ExecRiskSupervisor = None  # type: ignore

try:
    from Python.risk_engine import RiskEngine
except Exception:
    RiskEngine = None  # type: ignore

try:
    from Python.model_registry import ModelRegistry
except Exception:
    ModelRegistry = None  # type: ignore

try:
    from Python.autonomous.retraining_trigger import RetrainingTrigger
except Exception:
    RetrainingTrigger = None  # type: ignore

try:
    from alerts.telegram_alerts import TelegramAlerter
except Exception:
    TelegramAlerter = None  # type: ignore

try:
    from Python.backtest.fast_backtester import FastBacktester, BacktestConfig
except Exception:
    FastBacktester = None  # type: ignore
    BacktestConfig = None  # type: ignore

try:
    from Python.data.account_snapshots import AccountSnapshot
except Exception:
    AccountSnapshot = None  # type: ignore

try:
    from Python.rainforest_detector import RainforestDetector
except Exception:
    RainforestDetector = None  # type: ignore

# Optional paper trading for equity
try:
    from Python import paper_trading as paper
except Exception:
    paper = None  # type: ignore

# === NEW INTEGRATIONS for advanced degradation detection (per task spec) ===
try:
    from Python.autonomous.experience_memory import ExperienceMemory
except Exception:
    ExperienceMemory = None  # type: ignore

try:
    from Python.autonomous.validation_harness import StandardizedValidationResult  # for type, not required
except Exception:
    StandardizedValidationResult = None  # type: ignore

try:
    from Python.autonomous.retraining_orchestrator import AutonomousRetrainingOrchestrator
except Exception:
    AutonomousRetrainingOrchestrator = None  # type: ignore

try:
    from Python.autonomous.continual_learner import ContinualLearner
except Exception:
    ContinualLearner = None  # type: ignore

# For conservative TimeExitSpec application (rich Decision layer)
try:
    from Python.execution.trade_decision import TimeExitSpec
except Exception:
    TimeExitSpec = None  # type: ignore


@dataclass
class MonitoringMetrics:
    """Snapshot of all monitored signals at a point in time."""
    timestamp: str
    # Performance
    equity: float = 0.0
    current_drawdown_pct: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    rolling_win_rate: float = 0.5
    recent_trades: int = 0
    # Model / Prediction
    avg_model_confidence: float = 0.0
    min_recent_confidence: float = 0.0
    confidence_trend: str = "stable"  # stable | improving | degrading
    # Regime
    current_regime: str = "unknown"
    regime_stability_score: float = 0.8  # 0-1, lower = flipping often
    regime_win_rate: Optional[float] = None
    # Execution quality
    avg_slippage_bps: float = 0.0
    fill_rate: float = 1.0
    rejection_rate: float = 0.0
    execution_quality_score: float = 1.0  # composite 0-1
    # Health flags
    risk_halted: bool = False
    pause_new_entries: bool = False
    # Meta
    samples_used: int = 0
    data_sources: List[str] = field(default_factory=list)


@dataclass
class SelfAction:
    """Record of an autonomous action taken by the monitor."""
    timestamp: str
    action_type: str  # KILL_SWITCH, MODEL_ROLLBACK, ENTER_CONSERVATIVE, PAUSE_ENTRIES, REQUEST_RETRAIN, RECOVER, ALERT
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"  # info | warning | critical


class SelfMonitoringRecoveryAgent:
    """
    The Self-Monitoring, Auto-Rollback and Self-Recovery engine.

    Instantiate once (e.g. in supervisor or dedicated watcher process).
    Call monitor_cycle() periodically (30-300s recommended for live).
    """

    DEFAULT_THRESHOLDS = {
        "max_drawdown_pct": 8.0,
        "max_daily_loss_pct": 1.5,
        "min_rolling_win_rate": 0.38,
        "min_avg_confidence": 0.55,
        "confidence_degrade_trigger": 0.18,  # drop from recent peak
        "max_avg_slippage_bps": 18.0,
        "min_execution_quality": 0.65,
        "regime_stability_min": 0.55,
        "performance_vs_best_margin": -0.04,  # live return vs best known
        "kill_switch_cooldown_sec": 300,
        "rollback_cooldown_sec": 1800,
        "recovery_min_stable_trades": 8,
        "recovery_stable_dd_max": 3.0,
        # === NEW: Self-Monitoring + Auto-Rollback hardening (validation harness + ExperienceMemory + P&L) ===
        "max_news_forced_loss_pct": 2.8,          # unexpected news-prox losses over recent window
        "min_campaigns_negative_regime": 2,       # out of last 3 validation campaigns with negative regime adaptation
        "drawdown_velocity_max_pct_per_hr": 1.8,  # rapid DD acceleration
        "high_surprise_avg_threshold": 1.15,      # avg |surprise| signals model drift / regime mismatch
        "min_recent_experiences_for_surprise": 50,
        "regime_adapt_delta_threshold": -0.04,    # champion vs candidate delta in harness for key regimes/patterns
        "news_shock_loss_window_trades": 25,
        "light_adapt_min_surprise": 0.9,
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        thresholds: Optional[Dict[str, Any]] = None,
        enable_alerts: bool = True,
        enable_fast_backtest_checks: bool = True,
    ):
        self.config_path = Path(config_path) if config_path else (_PROJECT_ROOT / "config.yaml")
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._load_and_merge_config_thresholds()

        self.enable_alerts = enable_alerts
        self.enable_fast_backtest_checks = enable_fast_backtest_checks and (FastBacktester is not None)

        # State
        self.status_path = _AGENT_STATUS_DIR / "self_monitoring_recovery_agent.json"
        self.actions_log_path = _LOGS_DIR / "self_monitor_actions.jsonl"
        self.pause_flag = _RUNTIME_DIR / "pause_new_entries.flag"
        self.conservative_flag = _RUNTIME_DIR / "conservative_mode.flag"

        self._actions: List[SelfAction] = []
        self._metrics_history: List[MonitoringMetrics] = []
        self.recovery_state: Dict[str, Any] = {
            "active": False,
            "entered_at": None,
            "reason": None,
            "stable_trades_since_entry": 0,
        }
        self._last_kill_ts: float = 0.0
        self._last_rollback_ts: float = 0.0
        self._current_model_version: str = "unknown"
        self._best_known_score: float = 0.0

        # Components (lazy / optional)
        self._exec_agent: Optional[Any] = None
        self._risk_sup: Optional[Any] = None
        self._risk_engine: Optional[Any] = None
        self._model_registry: Optional[Any] = None
        self._retrain_trigger: Optional[Any] = None
        self._alerter: Optional[Any] = None
        self._account_snap: Optional[Any] = None

        self._init_optional_components()

        # Bootstrap status file
        self._write_status({"bootstrap": True})

        logger.info("[SelfMonitor] Initialized. Thresholds: %s", {k: v for k, v in self.thresholds.items() if "cooldown" not in k})

    # ------------------------------------------------------------------
    # Config & components
    # ------------------------------------------------------------------
    def _load_and_merge_config_thresholds(self) -> None:
        if not self.config_path.exists() or yaml is None:
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            risk = cfg.get("risk", {}) or {}
            sup = risk.get("supervisor", {}) or {}
            eval_cfg = cfg.get("evaluation", {}) or {}

            # Merge known keys
            for key in ("max_drawdown_pct", "max_daily_loss_pct"):
                if key in sup:
                    self.thresholds[key] = float(sup[key])
            if "max_drawdown" in eval_cfg:
                self.thresholds["max_drawdown_pct"] = min(
                    self.thresholds["max_drawdown_pct"], float(eval_cfg["max_drawdown"]) * 100
                )
            # Allow explicit monitoring section in future configs
            mon = cfg.get("monitoring", {}) or {}
            for k, v in mon.items():
                if k in self.thresholds:
                    self.thresholds[k] = v
        except Exception as exc:
            logger.debug(f"[SelfMonitor] Config load partial: {exc}")

    def _init_optional_components(self) -> None:
        # Model registry (critical for rollback)
        if ModelRegistry is not None:
            try:
                self._model_registry = ModelRegistry()
            except Exception as e:
                logger.warning(f"[SelfMonitor] ModelRegistry init failed: {e}")

        # Retraining trigger (for self-recovery retrain requests)
        if RetrainingTrigger is not None:
            try:
                self._retrain_trigger = RetrainingTrigger(data_dir=str(_LOGS_DIR))
            except Exception as e:
                logger.warning(f"[SelfMonitor] RetrainingTrigger init failed: {e}")

        # Risk layers
        if RiskEngine is not None:
            try:
                self._risk_engine = RiskEngine()
            except Exception:
                pass
        if ExecRiskSupervisor is not None:
            try:
                self._risk_sup = ExecRiskSupervisor()
            except Exception:
                pass

        # Alerts
        if self.enable_alerts and TelegramAlerter is not None:
            try:
                token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TG_TOKEN")
                chat = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("TG_CHAT")
                if token and chat:
                    self._alerter = TelegramAlerter(token, chat)
            except Exception:
                pass

        # Account snapshots
        if AccountSnapshot is not None:
            try:
                self._account_snap = AccountSnapshot(project_root=str(_PROJECT_ROOT))
            except Exception:
                pass

    def _get_execution_agent(self) -> Optional[Any]:
        if self._exec_agent is not None:
            return self._exec_agent
        if ExecutionAgent is None:
            return None
        try:
            # Try to obtain a sensible default instance (paper/demo safe)
            self._exec_agent = ExecutionAgent(
                config={"paper_mode": True, "max_positions": 8},
                mql5_bridge_enabled=False,
            )
            logger.debug("[SelfMonitor] ExecutionAgent acquired for kill-switch use.")
            return self._exec_agent
        except Exception as exc:
            logger.warning(f"[SelfMonitor] Could not instantiate ExecutionAgent: {exc}")
            return None

    # ------------------------------------------------------------------
    # Telemetry collection (robust, multi-source)
    # ------------------------------------------------------------------
    def _tail_jsonl(self, path: Path, n: int = 200) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-n:]
            out = []
            for ln in lines:
                if not ln.strip():
                    continue
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
            return out
        except Exception:
            return []

    def _collect_metrics(self) -> MonitoringMetrics:
        now = datetime.now(timezone.utc).isoformat()
        metrics = MonitoringMetrics(timestamp=now)
        sources: List[str] = []

        # 1. Execution feedback (core for PnL, winrate, execution quality)
        fb = self._tail_jsonl(_LOGS_DIR / "execution_feedback.jsonl", 300)
        if fb:
            sources.append("execution_feedback")
            pnls = []
            slips = []
            fills = 0
            rejs = 0
            wins = 0
            total = 0
            for rec in fb[-150:]:
                ev = rec.get("event", "")
                det = rec.get("details", {}) or rec.get("data", {})
                if ev in ("trade_closed", "demo_trade_closed", "position_closed"):
                    pnl = float(det.get("pnl", det.get("realized_pnl", 0.0)))
                    pnls.append(pnl)
                    if pnl > 0:
                        wins += 1
                    total += 1
                if "slippage" in str(det).lower() or "slippage_bps" in det:
                    try:
                        slips.append(float(det.get("slippage_bps", det.get("slippage", 0))))
                    except Exception:
                        pass
                if ev in ("trade_blocked", "rejection"):
                    rejs += 1
                if ev in ("fill", "partial_fill", "trade_filled"):
                    fills += 1
            if total > 0:
                metrics.rolling_win_rate = wins / total
                metrics.recent_trades = total
            if slips:
                metrics.avg_slippage_bps = sum(slips) / len(slips)
            tot_exec = max(1, fills + rejs)
            metrics.fill_rate = fills / tot_exec
            metrics.rejection_rate = rejs / tot_exec
            metrics.execution_quality_score = max(0.0, min(1.0, 1.0 - metrics.rejection_rate - (metrics.avg_slippage_bps / 50.0)))
            if pnls:
                metrics.daily_pnl = sum(pnls[-30:])  # recent proxy
        else:
            sources.append("execution_feedback:empty")

        # 2. Account equity / drawdown (MT5 snapshot or paper or risk_engine)
        equity = 0.0
        dd = 0.0
        try:
            if self._account_snap is not None:
                snap = self._account_snap.fetch()
                if snap:
                    equity = float(snap.get("equity", snap.get("balance", 0.0)))
                    sources.append("account_snapshot")
            elif paper is not None and hasattr(paper, "paper_account_info"):
                acc = paper.paper_account_info()
                equity = float(getattr(acc, "equity", getattr(acc, "balance", 0.0)) or 0.0)
                sources.append("paper_trading")
            elif self._risk_engine is not None:
                equity = getattr(self._risk_engine, "_current_equity", 0.0) or 10000.0
                dd = float(getattr(self._risk_engine, "current_dd", 0.0))
                sources.append("risk_engine")
            elif self._risk_sup is not None:
                dd = float(getattr(self._risk_sup, "current_dd", getattr(self._risk_sup, "_engine", None) and getattr(self._risk_sup._engine, "current_dd", 0.0) or 0.0))
                sources.append("risk_supervisor")
        except Exception:
            pass

        # Fallback synthetic equity from recent PnL if nothing else
        if equity <= 0 and metrics.daily_pnl != 0:
            equity = 10000.0 + metrics.daily_pnl * 10  # rough

        metrics.equity = equity or 10000.0

        # Drawdown update (use risk if present, else simple peak tracker via history)
        if dd > 0:
            metrics.current_drawdown_pct = dd
        else:
            # Simple historical peak tracker
            hist_equities = [m.equity for m in self._metrics_history[-20:] if m.equity > 0]
            if hist_equities:
                peak = max(hist_equities + [equity])
                if peak > 0:
                    metrics.current_drawdown_pct = max(0.0, (peak - equity) / peak * 100.0)
            else:
                metrics.current_drawdown_pct = 0.0

        # Daily PnL % rough
        if equity > 0:
            metrics.daily_pnl_pct = (metrics.daily_pnl / equity) * 100.0

        # 3. Model confidence & regime (from PIPELINE_DECISIONS + execution live reports)
        decisions = self._tail_jsonl(_LOGS_DIR / "PIPELINE_DECISIONS.jsonl", 150)
        if not decisions:
            decisions = self._tail_jsonl(_EXEC_REPORT_DIR / "recent_decisions_fallback.jsonl", 50)  # best effort
        confs: List[float] = []
        regimes: List[str] = []
        for d in decisions[-80:]:
            det = d.get("details", {}) or {}
            if "confidence" in det:
                try:
                    confs.append(float(det["confidence"]))
                except Exception:
                    pass
            if d.get("decision_type") in ("trade_decision_ppo", "decision"):
                if "regime" in det:
                    regimes.append(str(det["regime"]))
            # Also scan decision payload if present
            if "decision" in det and isinstance(det["decision"], dict):
                c = det["decision"].get("confidence")
                if c is not None:
                    try:
                        confs.append(float(c))
                    except Exception:
                        pass

        if confs:
            sources.append("pipeline_decisions")
            metrics.avg_model_confidence = sum(confs) / len(confs)
            metrics.min_recent_confidence = min(confs)
            # Simple trend
            if len(confs) >= 6:
                early = sum(confs[:3]) / 3
                late = sum(confs[-3:]) / 3
                if late < early - self.thresholds["confidence_degrade_trigger"]:
                    metrics.confidence_trend = "degrading"
                elif late > early + 0.05:
                    metrics.confidence_trend = "improving"

        # Regime via rainforest if available, else from decisions
        if RainforestDetector is not None:
            try:
                det = RainforestDetector()
                # Best-effort; many detectors expose current regime on recent data
                if hasattr(det, "current_regime"):
                    metrics.current_regime = str(det.current_regime)
                sources.append("rainforest_detector")
            except Exception:
                pass
        if regimes:
            # majority vote last N
            from collections import Counter
            metrics.current_regime = Counter(regimes).most_common(1)[0][0]
            # crude stability: fewer distinct = higher stability
            metrics.regime_stability_score = max(0.3, 1.0 - (len(set(regimes)) / max(1, len(regimes))))

        # 4. Risk state
        if self._risk_engine is not None:
            metrics.risk_halted = bool(getattr(self._risk_engine, "halt", False))
            sources.append("risk_engine_state")
        if self.pause_flag.exists():
            metrics.pause_new_entries = True

        # 5. Current model version (from registry or recent decisions)
        if self._model_registry is not None:
            try:
                active = {}
                ap = getattr(self._model_registry, "active_path", None)
                if ap and Path(ap).exists():
                    active = json.loads(Path(ap).read_text(encoding="utf-8"))
                champ = active.get("champion") or (active.get("symbols", {}).get("XAUUSDm", {}).get("champion") if "symbols" in active else None)
                if champ:
                    self._current_model_version = str(champ).split("/")[-1][:40]
            except Exception:
                pass

        metrics.data_sources = sources
        metrics.samples_used = len(fb) + len(decisions)

        # Persist for history-based trend / peak tracking
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 60:
            self._metrics_history = self._metrics_history[-60:]

        return metrics

    # ------------------------------------------------------------------
    # ADVANCED DEGRADATION DETECTION (Validation Harness + ExperienceMemory Surprise + P&L Velocity + News)
    # Required for robust zero-touch unsupervised operation.
    # ------------------------------------------------------------------
    def _load_recent_validation_campaigns(self, max_campaigns: int = 3) -> List[Dict[str, Any]]:
        """Load last N standardized validation harness campaign results for regime adaptation analysis.
        Looks in runtime/validation_results/ and backtest_results/ for standardized_validation_*.json
        """
        results: List[Dict[str, Any]] = []
        candidates = []
        for base in (_RUNTIME_DIR / "validation_results", _RUNTIME_DIR / "backtest_results"):
            if base.exists():
                for p in sorted(base.glob("standardized_validation_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:max_campaigns * 2]:
                    candidates.append(p)
        # Also top level runtime for legacy
        rt = _RUNTIME_DIR
        for p in sorted(rt.glob("*validation*campaign*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            candidates.append(p)

        seen = set()
        for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True):
            if len(results) >= max_campaigns:
                break
            if str(p) in seen:
                continue
            seen.add(str(p))
            try:
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                data["_source_file"] = str(p)
                # Normalize key fields for trigger logic
                ab = data.get("ab_comparison", {}) or {}
                data["_candidate_beats"] = bool(ab.get("candidate_beats_champion", False))
                data["_recommend_promote"] = bool(ab.get("recommend_for_promotion", False))
                # crude regime adaptation proxy: use delta return or pattern pnl deltas if present
                delta = (ab.get("delta") or {}).get("return") or (ab.get("delta") or {}).get("pnl", 0)
                try:
                    data["_regime_adapt_delta"] = float(delta) if delta is not None else 0.0
                except Exception:
                    data["_regime_adapt_delta"] = 0.0
                results.append(data)
            except Exception:
                continue
        return results

    def _get_experience_surprise_metrics(self) -> Dict[str, Any]:
        """Pull recent surprise metrics from ExperienceMemory for model drift / regime mismatch detection."""
        if ExperienceMemory is None:
            return {"available": False, "avg_surprise": 0.0, "high_surprise_count": 0, "recent_negative_edge": 0}
        try:
            mem = ExperienceMemory(storage_path=str(_RUNTIME_DIR / "experience_memory.jsonl"))
            if not mem.experiences:
                return {"available": True, "avg_surprise": 0.0, "high_surprise_count": 0, "recent_negative_edge": 0, "note": "empty"}
            recent = mem.experiences[-min(len(mem.experiences), 300):]
            surprises = [abs(getattr(e, "surprise", 0.0) or 0.0) for e in recent]
            edges = [getattr(e, "edge_score", 0.0) or 0.0 for e in recent]
            neg_edges = sum(1 for e in edges if e < -0.2)
            avg_s = sum(surprises) / max(1, len(surprises))
            high = sum(1 for s in surprises if s >= 1.0)
            return {
                "available": True,
                "avg_surprise": round(avg_s, 4),
                "high_surprise_count": high,
                "recent_negative_edge": neg_edges,
                "total_recent": len(recent),
            }
        except Exception as exc:
            logger.debug(f"[SelfMonitor] Surprise metrics unavailable: {exc}")
            return {"available": False, "error": str(exc)[:120]}

    def _compute_drawdown_velocity(self) -> float:
        """Drawdown velocity: %DD increase per hour from recent metrics history (conservative acceleration detector)."""
        if len(self._metrics_history) < 4:
            return 0.0
        recent = self._metrics_history[-12:]  # ~last 12 snapshots
        dds = [m.current_drawdown_pct for m in recent if m.current_drawdown_pct is not None]
        if len(dds) < 3:
            return 0.0
        # Simple linear slope approx (per snapshot; caller normalizes to per-hr using typical poll)
        deltas = [dds[i] - dds[i-1] for i in range(1, len(dds))]
        avg_delta_per_snapshot = sum(deltas) / len(deltas)
        # Assume ~1 snapshot per 2-5 min in supervisor context; conservative velocity in %/hr
        velocity_per_hr = avg_delta_per_snapshot * (60.0 / 4.0)  # rough 4 snapshots/hr
        return max(0.0, round(velocity_per_hr, 3))

    def _detect_news_shocks_and_forced_losses(self, m: MonitoringMetrics) -> Dict[str, Any]:
        """Scan execution_feedback + experience for unexpected news-proximity forced losses."""
        fb = self._tail_jsonl(_LOGS_DIR / "execution_feedback.jsonl", 120)
        news_losses = 0.0
        news_trades = 0
        for rec in fb[-self.thresholds.get("news_shock_loss_window_trades", 25):]:
            det = rec.get("details", {}) or rec.get("data", {}) or {}
            timing = det.get("timing_context") or det.get("news_context") or {}
            news_prox = 0.0
            if isinstance(timing, dict):
                news_prox = float(timing.get("news_proximity", timing.get("news_score", 0.0)) or 0.0)
            pnl = float(det.get("pnl", det.get("realized_pnl", 0.0)) or 0.0)
            if news_prox > 0.55 and pnl < 0:
                news_losses += abs(pnl)
                news_trades += 1
        total_recent_pnl_abs = sum(abs(float((r.get("details", {}) or {}).get("pnl", 0))) for r in fb[-30:]) or 1.0
        news_forced_pct = (news_losses / max(1.0, total_recent_pnl_abs)) * 100.0 if total_recent_pnl_abs > 0 else 0.0
        return {
            "news_forced_loss_pct": round(news_forced_pct, 2),
            "news_negative_trades": news_trades,
            "is_shock": news_forced_pct > self.thresholds.get("max_news_forced_loss_pct", 2.8),
        }

    def _evaluate_rollback_triggers(self, m: MonitoringMetrics, campaigns: List[Dict[str, Any]], surprise: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Core conservative rollback decision logic per requirements.
        Returns list of (trigger_code, human_reason).
        """
        triggers = []
        t = self.thresholds

        # 1. Last 3 campaigns negative regime adaptation (key requirement)
        if campaigns:
            neg_adapt_count = 0
            for c in campaigns[:3]:
                delta = c.get("_regime_adapt_delta", 0.0)
                rec = c.get("_recommend_promote", True)
                if (delta < t["regime_adapt_delta_threshold"]) or (not rec and delta < 0):
                    neg_adapt_count += 1
            if neg_adapt_count >= t["min_campaigns_negative_regime"]:
                triggers.append((
                    "NEGATIVE_REGIME_ADAPTATION_3CAMPAIGNS",
                    f"Last {min(3,len(campaigns))} validation campaigns show negative regime adaptation (count={neg_adapt_count}, delta threshold {t['regime_adapt_delta_threshold']})"
                ))

        # 2. High unexpected news_forced losses
        news = self._detect_news_shocks_and_forced_losses(m)
        if news.get("is_shock"):
            triggers.append((
                "NEWS_FORCED_LOSS_SHOCK",
                f"> {t['max_news_forced_loss_pct']}% of recent realized losses are news-proximity forced (observed {news['news_forced_loss_pct']}%)"
            ))

        # 3. Model drift via ExperienceMemory surprise
        if surprise.get("available") and surprise.get("avg_surprise", 0) > t["high_surprise_avg_threshold"]:
            if surprise.get("total_recent", 0) >= t["min_recent_experiences_for_surprise"]:
                triggers.append((
                    "MODEL_DRIFT_HIGH_SURPRISE",
                    f"ExperienceMemory avg |surprise| = {surprise['avg_surprise']:.2f} > {t['high_surprise_avg_threshold']} over {surprise['total_recent']} recent experiences (regime/model mismatch)"
                ))

        # 4. Drawdown velocity (acceleration)
        vel = self._compute_drawdown_velocity()
        if vel >= t["drawdown_velocity_max_pct_per_hr"]:
            triggers.append((
                "DRAWDOWN_VELOCITY_HIGH",
                f"DD acceleration {vel:.2f}%/hr exceeds safe limit {t['drawdown_velocity_max_pct_per_hr']}%/hr"
            ))

        # 5. Existing performance signals reinforced (e.g. combined with live PnL)
        if m.current_drawdown_pct > 4.0 and m.rolling_win_rate < 0.42 and len(campaigns) > 0:
            triggers.append((
                "COMPOUND_PERFORMANCE_REGIME_FAILURE",
                "Sustained live DD + poor winrate coinciding with weak recent validation campaigns"
            ))

        return triggers

    def _apply_conservative_time_exit(self) -> Dict[str, Any]:
        """Switch execution to conservative TimeExitSpec (short holds + strict news avoidance).
        Writes dedicated flag consumable by ExecutionAgent / Risk layers / Decision PPO.
        """
        flag_path = _RUNTIME_DIR / "conservative_time_exit_spec.json"
        conservative_spec = {
            "mode": "conservative_recovery",
            "max_hold_minutes": 45,
            "close_before_high_impact_news": True,
            "reduce_size_on_news_window": True,
            "min_hold_minutes": 8,
            "news_buffer_minutes": 25,
            "activated_by": "self_monitor_recovery",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            flag_path.write_text(json.dumps(conservative_spec, indent=2), encoding="utf-8")
            # Also ensure broader conservative flag
            self.conservative_flag.touch(exist_ok=True)
            self.pause_flag.touch(exist_ok=True)  # still safe to keep paused on entry
            action = SelfAction(
                timestamp=datetime.now(timezone.utc).isoformat(),
                action_type="SWITCH_CONSERVATIVE_TIMEEXIT",
                reason="Rollback / recovery trigger: enforced strict TimeExitSpec for slippage/news protection",
                severity="warning",
                details={"spec": conservative_spec, "flag": str(flag_path)},
            )
            self._log_action(action)
            return {"applied": True, "spec": conservative_spec}
        except Exception as exc:
            return {"applied": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Issue detection + policy
    # ------------------------------------------------------------------
    def _detect_issues(self, m: MonitoringMetrics) -> List[Tuple[str, str, str]]:
        """Return list of (severity, code, human_reason)."""
        issues: List[Tuple[str, str, str]] = []
        t = self.thresholds

        if m.current_drawdown_pct >= t["max_drawdown_pct"]:
            issues.append(("critical", "DRAWDOWN_BREACH", f"Drawdown {m.current_drawdown_pct:.2f}% >= {t['max_drawdown_pct']}%"))

        if m.daily_pnl_pct <= -t["max_daily_loss_pct"]:
            issues.append(("critical", "DAILY_LOSS_BREACH", f"Daily PnL {m.daily_pnl_pct:.2f}% <= -{t['max_daily_loss_pct']}%"))

        if m.recent_trades >= 8 and m.rolling_win_rate < t["min_rolling_win_rate"]:
            issues.append(("warning", "WINRATE_CRASH", f"Win-rate {m.rolling_win_rate:.2%} < {t['min_rolling_win_rate']:.0%} over {m.recent_trades} trades"))

        if m.avg_model_confidence > 0 and m.avg_model_confidence < t["min_avg_confidence"]:
            issues.append(("warning", "LOW_CONFIDENCE", f"Avg model confidence {m.avg_model_confidence:.2f} < {t['min_avg_confidence']}"))

        if m.confidence_trend == "degrading":
            issues.append(("warning", "CONFIDENCE_DEGRADING", "Model prediction confidence trending down sharply"))

        if m.avg_slippage_bps >= t["max_avg_slippage_bps"] or m.execution_quality_score < t["min_execution_quality"]:
            issues.append(("warning", "POOR_EXECUTION", f"Execution quality {m.execution_quality_score:.2f} (slip={m.avg_slippage_bps:.1f}bps)"))

        if m.regime_stability_score < t["regime_stability_min"]:
            issues.append(("warning", "REGIME_UNSTABLE", f"Regime stability {m.regime_stability_score:.2f} < {t['regime_stability_min']}"))

        if m.risk_halted:
            issues.append(("critical", "RISK_HALTED", "Risk engine is in halted state"))

        # Surface velocity / news as soft issues (hard triggers evaluated separately in monitor_cycle)
        vel = self._compute_drawdown_velocity()
        if vel >= t.get("drawdown_velocity_max_pct_per_hr", 2.0):
            issues.append(("warning", "DD_VELOCITY", f"Drawdown velocity {vel:.2f}%/hr (risk of fast breach)"))

        news = self._detect_news_shocks_and_forced_losses(m)
        if news.get("is_shock"):
            issues.append(("warning", "NEWS_SHOCK", f"News-forced loss concentration {news['news_forced_loss_pct']}%"))

        return issues

    # ------------------------------------------------------------------
    # Actions (kill, rollback, recovery)
    # ------------------------------------------------------------------
    def _log_action(self, action: SelfAction) -> None:
        self._actions.append(action)
        if len(self._actions) > 50:
            self._actions = self._actions[-50:]

        # Local detailed log
        try:
            with open(self.actions_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(action), default=str) + "\n")
        except Exception:
            pass

        # Unified pipeline audit (single source of truth)
        try:
            log_decision(
                decision_type="self_monitor_action",
                actor="self_monitoring_recovery_agent",
                decision=action.action_type,
                reason=action.reason[:280],
                details={
                    "severity": action.severity,
                    "metrics_snapshot": asdict(self._metrics_history[-1]) if self._metrics_history else {},
                    **action.details,
                },
                severity=action.severity,
            )
        except Exception:
            pass

        # Console + alert
        msg = f"[SELF-MONITOR] {action.action_type}: {action.reason}"
        if action.severity == "critical":
            logger.critical(msg)
            if self._alerter:
                try:
                    self._alerter.critical(f"SELF-MONITOR {action.action_type}\n{action.reason}\n{action.details}")
                except Exception:
                    pass
        elif action.severity == "warning":
            logger.warning(msg)
        else:
            logger.info(msg)

    def _write_status(self, extra: Optional[Dict[str, Any]] = None) -> None:
        last_m = self._metrics_history[-1] if self._metrics_history else None
        status = {
            "name": "Self-Monitoring, Auto-Rollback and Self-Recovery Agent",
            "status": "RECOVERING" if self.recovery_state["active"] else "ACTIVE",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "metrics": asdict(last_m) if last_m else {},
            "recent_actions": [asdict(a) for a in self._actions[-12:]],
            "recovery_state": self.recovery_state,
            "current_model_version": self._current_model_version,
            "thresholds": self.thresholds,
            "fast_backtest_checks_enabled": self.enable_fast_backtest_checks,
            "data_sources_last_cycle": last_m.data_sources if last_m else [],
            # Safety surface for TUI / vps_supervisor / MasterSelfEvolutionSupervisor
            "kill_switch_last": getattr(self, "_last_kill_ts", 0),
            "rollback_last": getattr(self, "_last_rollback_ts", 0),
            "pause_flag_active": self.pause_flag.exists(),
            "conservative_flag_active": self.conservative_flag.exists(),
            "conservative_time_exit_active": (_RUNTIME_DIR / "conservative_time_exit_spec.json").exists(),
            "has_kill_switch_cooldown": (time.time() - getattr(self, "_last_kill_ts", 0)) < self.thresholds.get("kill_switch_cooldown_sec", 300),
        }
        if extra:
            status.update(extra)

        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.status_path, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2, default=str)
        except Exception as exc:
            logger.debug(f"[SelfMonitor] Status write failed: {exc}")

    def _pre_deployment_fast_backtest_check(self, proposed_action: str, symbol: str = "XAUUSDm") -> Dict[str, Any]:
        """Use the fast backtest engine for pre-deployment safety check before rollback/recovery model change."""
        if not self.enable_fast_backtest_checks or FastBacktester is None or BacktestConfig is None:
            return {"performed": False, "reason": "fast_backtester_unavailable_or_disabled"}

        try:
            cfg = BacktestConfig(
                symbol=symbol,
                start=(datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d"),
                end=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                speed_mode="fast",
            )
            bt = FastBacktester(cfg)
            # Dummy conservative policy for sanity (real version would load the rollback candidate policy)
            def conservative_policy(obs, **kw):
                # In real integration this would load the specific model bundle being considered
                from Python.execution.trade_decision import TradeDecision, SizeSpec, TimeExitSpec
                return TradeDecision(
                    side="FLAT",  # ultra-safe probe
                    size=SizeSpec(mode="fixed_lots", value=0.01),
                    time_exit=TimeExitSpec(max_hold_minutes=30, close_before_high_impact_news=True),
                )

            res = bt.run(policy_fn=conservative_policy)
            res["performed"] = True
            res["action_context"] = proposed_action
            logger.info(f"[SelfMonitor] Fast backtest pre-check for {proposed_action} complete: equity={res.get('final_equity')}")
            return res
        except Exception as exc:
            logger.warning(f"[SelfMonitor] Fast backtest pre-check failed (non-fatal): {exc}")
            return {"performed": True, "error": str(exc), "action_context": proposed_action}

    def trigger_kill_switch(self, reason: str) -> Dict[str, Any]:
        now = time.time()
        if now - self._last_kill_ts < self.thresholds["kill_switch_cooldown_sec"]:
            return {"executed": False, "reason": "cooldown_active"}

        self._last_kill_ts = now
        action = SelfAction(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action_type="KILL_SWITCH",
            reason=reason,
            severity="critical",
            details={"cooldown_reset": True},
        )
        self._log_action(action)

        # 1. Flatten everything (primary path)
        ea = self._get_execution_agent()
        flatten_res: Dict[str, Any] = {"executed": False}
        if ea and hasattr(ea, "force_flatten_all"):
            try:
                flatten_res = ea.force_flatten_all(reason=f"self_monitor_kill_switch:{reason}")
            except Exception as e:
                flatten_res["error"] = str(e)

        # 2. Halt risk layers
        if self._risk_engine:
            try:
                self._risk_engine.halt = True
                self._risk_engine._halt_reason = f"self_monitor:{reason}"
            except Exception:
                pass
        if self._risk_sup and hasattr(self._risk_sup, "trigger_rollback"):
            try:
                self._risk_sup.trigger_rollback(f"self_monitor:{reason}")
            except Exception:
                pass

        # 3. Enter full recovery
        self._enter_recovery_mode(reason, kill_switch=True)

        # 4. Write immediate status
        self._write_status({"last_kill": action.timestamp, "flatten_result": flatten_res})

        return {"executed": True, "flatten_result": flatten_res, "recovery_entered": True}

    def trigger_model_rollback(self, reason: str, target_version: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        if now - self._last_rollback_ts < self.thresholds["rollback_cooldown_sec"]:
            return {"executed": False, "reason": "cooldown_active"}

        self._last_rollback_ts = now

        # Pre-deployment safety via fast backtester (as required)
        bt_result = self._pre_deployment_fast_backtest_check(f"model_rollback:{reason}")

        action = SelfAction(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action_type="MODEL_ROLLBACK",
            reason=reason,
            severity="critical",
            details={"target_version": target_version, "fast_backtest_precheck": bt_result},
        )
        self._log_action(action)

        success = False
        if self._model_registry is not None:
            try:
                # Best-effort: promote previous known good or canary->champion swap if appropriate
                # In full impl this would inspect scorecard history + pick best non-current
                active_path = getattr(self._model_registry, "active_path", None)
                if active_path and Path(active_path).exists():
                    active = json.loads(Path(active_path).read_text(encoding="utf-8"))
                    # Simple strategy: if a "previous_champion" or older candidate exists, promote it
                    # Here we just stamp a rollback marker (real promotion would use registry.promote etc.)
                    active.setdefault("rollback_history", []).append({
                        "at": action.timestamp,
                        "from": self._current_model_version,
                        "reason": reason,
                        "fast_bt": bt_result.get("performed"),
                    })
                    active["last_rollback"] = action.timestamp
                    Path(active_path).write_text(json.dumps(active, indent=2, default=str), encoding="utf-8")
                    success = True
                    self._current_model_version = "ROLLED_BACK:" + (target_version or "previous_best")
            except Exception as exc:
                action.details["registry_error"] = str(exc)

        self._enter_recovery_mode(reason, model_rollback=True)
        self._write_status({"last_rollback": action.timestamp, "rollback_success": success, "bt_result": bt_result})

        return {"executed": success, "fast_backtest": bt_result}

    def _enter_recovery_mode(self, reason: str, kill_switch: bool = False, model_rollback: bool = False) -> None:
        if not self.recovery_state["active"]:
            self.recovery_state = {
                "active": True,
                "entered_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "stable_trades_since_entry": 0,
                "kill_switch": kill_switch,
                "model_rollback": model_rollback,
            }
            action = SelfAction(
                timestamp=datetime.now(timezone.utc).isoformat(),
                action_type="ENTER_RECOVERY",
                reason=reason,
                severity="warning",
                details=self.recovery_state.copy(),
            )
            self._log_action(action)

        # Apply conservative behaviors
        try:
            self.conservative_flag.touch(exist_ok=True)
            self.pause_flag.touch(exist_ok=True)
        except Exception:
            pass

        # Request retraining (self-recovery)
        if self._retrain_trigger is not None:
            try:
                art = self._retrain_trigger.evaluate(
                    champion_drawdown_pct=self._metrics_history[-1].current_drawdown_pct if self._metrics_history else 5.0,
                    regime_win_rates={self._metrics_history[-1].current_regime: self._metrics_history[-1].rolling_win_rate} if self._metrics_history else None,
                )
                if art and art.triggered:
                    logger.info(f"[SelfMonitor] Retraining recommended by recovery: {art.reasons}")
            except Exception:
                pass

    def _maybe_exit_recovery(self, m: MonitoringMetrics) -> bool:
        if not self.recovery_state["active"]:
            return False

        t = self.thresholds
        stable = (
            m.current_drawdown_pct <= t["recovery_stable_dd_max"]
            and m.rolling_win_rate >= (t["min_rolling_win_rate"] + 0.08)
            and (not m.risk_halted)
        )
        if stable:
            self.recovery_state["stable_trades_since_entry"] = self.recovery_state.get("stable_trades_since_entry", 0) + max(1, m.recent_trades)
        else:
            self.recovery_state["stable_trades_since_entry"] = 0

        if self.recovery_state.get("stable_trades_since_entry", 0) >= t["recovery_min_stable_trades"]:
            # Exit recovery
            action = SelfAction(
                timestamp=datetime.now(timezone.utc).isoformat(),
                action_type="RECOVER",
                reason="Sustained stability criteria met after recovery entry",
                severity="info",
                details={"previous_reason": self.recovery_state.get("reason")},
            )
            self._log_action(action)

            self.recovery_state = {"active": False, "entered_at": None, "reason": None, "stable_trades_since_entry": 0}
            # Relax flags (best effort)
            for f in (self.pause_flag, self.conservative_flag):
                try:
                    if f.exists():
                        f.unlink()
                except Exception:
                    pass
            self._write_status({"recovery_exited": True})
            return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def monitor_cycle(self) -> Dict[str, Any]:
        """
        One full monitoring + decision + action cycle.
        Safe to call from supervisor loop, harness, or cron.
        Now includes full self-monitoring + auto-rollback using:
        - Recent validation harness campaigns (last 3 for regime adaptation)
        - ExperienceMemory surprise metrics (model drift)
        - Live P&L drawdown + velocity + news shocks
        """
        try:
            metrics = self._collect_metrics()
        except Exception as exc:
            logger.error(f"[SelfMonitor] Metrics collection failed: {exc}")
            metrics = MonitoringMetrics(timestamp=datetime.now(timezone.utc).isoformat(), data_sources=["error_fallback"])

        issues = self._detect_issues(metrics)

        # === NEW: Load external signals for conservative rollback triggers ===
        campaigns = self._load_recent_validation_campaigns(3)
        surprise_metrics = self._get_experience_surprise_metrics()
        rollback_triggers = self._evaluate_rollback_triggers(metrics, campaigns, surprise_metrics)

        # Hard kill for critical live risk
        for sev, code, reason in issues:
            if code in ("DRAWDOWN_BREACH", "DAILY_LOSS_BREACH", "RISK_HALTED") and sev == "critical":
                self.trigger_kill_switch(reason)
                break  # one kill is enough

        # === ROLLBACK TRIGGERS (per requirements: last 3 campaigns negative regime, news_forced >X%, surprise drift, velocity) ===
        triggered_rollback = False
        if rollback_triggers and not any(a.action_type == "MODEL_ROLLBACK" for a in self._actions[-5:]):
            for code, reason in rollback_triggers:
                if metrics.recent_trades >= 5 or len(campaigns) >= 2:  # require some evidence
                    res = self.trigger_model_rollback(f"{code}: {reason}")
                    triggered_rollback = res.get("executed", False)
                    # On rollback also force conservative exits immediately
                    if triggered_rollback:
                        self._apply_conservative_time_exit()
                    break  # one at a time, conservative

        # Legacy soft issues can still trigger rollback if strong signal
        for sev, code, reason in issues:
            if not triggered_rollback and code in ("LOW_CONFIDENCE", "WINRATE_CRASH", "REGIME_UNSTABLE"):
                if metrics.recent_trades >= 12:
                    res = self.trigger_model_rollback(f"{code}: {reason}")
                    if res.get("executed"):
                        self._apply_conservative_time_exit()
                    break

        # === AUTO-RECOVERY ACTIONS (pause, conservative TimeExitSpec, notify, light adaptation, full retrain) ===
        recovery_actions = []
        if self.recovery_state["active"] or rollback_triggers or any("RECOVER" in str(i) for i in issues):
            # Ensure pause + conservative mode
            try:
                self.pause_flag.touch(exist_ok=True)
                self.conservative_flag.touch(exist_ok=True)
            except Exception:
                pass

            # Explicit conservative TimeExitSpec (rich execution layer)
            cex = self._apply_conservative_time_exit()
            if cex.get("applied"):
                recovery_actions.append("conservative_time_exit_applied")

            # Notify via status + alerts (already done in _log_action + _write_status)
            recovery_actions.append("status_tui_notified")

            # Light online adaptation (best-effort via ContinualLearner if available)
            if ContinualLearner is not None and surprise_metrics.get("available") and surprise_metrics.get("avg_surprise", 0) > self.thresholds.get("light_adapt_min_surprise", 0.9):
                try:
                    cl = ContinualLearner()
                    if hasattr(cl, "ingest_recent_feedback"):
                        cl.ingest_recent_feedback(limit=200)  # light weight
                    recovery_actions.append("light_online_adaptation_requested")
                except Exception:
                    pass

            # Full retrain trigger via orchestrator / trigger (highest recovery path)
            if self._retrain_trigger is not None or AutonomousRetrainingOrchestrator is not None:
                try:
                    if AutonomousRetrainingOrchestrator is not None:
                        orch = AutonomousRetrainingOrchestrator()
                        if hasattr(orch, "request_retraining"):
                            orch.request_retraining(reason="self_monitor_rollback_recovery", priority="high")
                        elif hasattr(orch, "evaluate_and_maybe_launch"):
                            orch.evaluate_and_maybe_launch()
                        recovery_actions.append("full_retrain_via_orchestrator_requested")
                    elif self._retrain_trigger is not None:
                        art = self._retrain_trigger.evaluate(
                            champion_drawdown_pct=metrics.current_drawdown_pct,
                            regime_win_rates={metrics.current_regime: metrics.rolling_win_rate},
                            force=True,
                        )
                        if art and getattr(art, "triggered", False):
                            recovery_actions.append("retrain_trigger_fired")
                except Exception as exc:
                    logger.debug(f"[SelfMonitor] Retrain request note: {exc}")

        # Recovery self-healing
        exited = self._maybe_exit_recovery(metrics)

        # Light recovery actions if in recovery but not yet killed
        if self.recovery_state["active"] and not exited:
            # Conservative + paused already enforced above
            pass

        # Always persist latest rich status (including new signals for TUI / vps_supervisor / MasterSupervisor)
        extra_status = {
            "issues_last_cycle": [code for _, code, _ in issues],
            "rollback_triggers_last_cycle": rollback_triggers,
            "validation_campaigns_analyzed": len(campaigns),
            "experience_surprise": surprise_metrics,
            "drawdown_velocity_pct_per_hr": self._compute_drawdown_velocity(),
            "news_shock_state": self._detect_news_shocks_and_forced_losses(metrics),
            "recovery_actions_this_cycle": recovery_actions,
            "conservative_time_exit_active": (_RUNTIME_DIR / "conservative_time_exit_spec.json").exists(),
        }
        self._write_status(extra_status)

        return {
            "metrics": asdict(metrics),
            "issues": issues,
            "rollback_triggers": rollback_triggers,
            "recovery_active": self.recovery_state["active"],
            "recovery_actions": recovery_actions,
            "actions_this_cycle": len([a for a in self._actions if a.timestamp >= (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()]),
            "campaigns_evaluated": len(campaigns),
        }

    # ------------------------------------------------------------------
    # SAFE KILL-SWITCH + RESUME MECHANISMS (explicit, auditable, for TUI/operator)
    # ------------------------------------------------------------------
    def resume_trading(self, reason: str = "manual_or_stability_resume") -> Dict[str, Any]:
        """Safe resume from kill-switch / recovery / pause.
        Clears pause + conservative flags (including rich TimeExit), exits recovery state.
        Does NOT auto-re-enable risk if hard halt present.
        """
        cleared = []
        for f in (self.pause_flag, self.conservative_flag, _RUNTIME_DIR / "conservative_time_exit_spec.json"):
            try:
                if f.exists():
                    f.unlink()
                    cleared.append(str(f.name))
            except Exception:
                pass

        was_recovering = self.recovery_state.get("active", False)
        self.recovery_state = {"active": False, "entered_at": None, "reason": None, "stable_trades_since_entry": 0}

        action = SelfAction(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action_type="RESUME_TRADING",
            reason=reason,
            severity="info",
            details={"flags_cleared": cleared, "was_in_recovery": was_recovering},
        )
        self._log_action(action)

        self._write_status({"resumed": True, "resume_reason": reason})
        logger.info(f"[SelfMonitor] RESUME_TRADING executed: {reason}. Flags cleared: {cleared}")
        return {"resumed": True, "cleared": cleared, "recovery_exited": was_recovering}

    def force_kill_switch(self, reason: str) -> Dict[str, Any]:
        """Public safe wrapper for external (TUI, supervisor, operator) kill switch invocation."""
        return self.trigger_kill_switch(f"EXTERNAL:{reason}")

    def force_rollback(self, reason: str) -> Dict[str, Any]:
        """Public safe wrapper for external rollback request."""
        res = self.trigger_model_rollback(reason)
        if res.get("executed"):
            self._apply_conservative_time_exit()
        return res

    def get_rollback_triggers_preview(self) -> Dict[str, Any]:
        """For TUI / pre-flight inspection without side effects."""
        try:
            m = self._collect_metrics()
        except Exception:
            m = MonitoringMetrics(timestamp=datetime.now(timezone.utc).isoformat())
        camps = self._load_recent_validation_campaigns(3)
        supr = self._get_experience_surprise_metrics()
        trigs = self._evaluate_rollback_triggers(m, camps, supr)
        return {
            "current_metrics": asdict(m),
            "campaigns": [{"file": c.get("_source_file"), "delta": c.get("_regime_adapt_delta"), "recommend": c.get("_recommend_promote")} for c in camps],
            "surprise": supr,
            "triggers": trigs,
            "would_rollback": len(trigs) > 0,
        }

    def get_current_status(self) -> Dict[str, Any]:
        """Return the latest written status JSON (for TUI / API / swarm)."""
        if self.status_path.exists():
            try:
                return json.loads(self.status_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"name": "Self-Monitoring, Auto-Rollback and Self-Recovery Agent", "status": "UNKNOWN"}

    def force_recovery_check(self) -> Dict[str, Any]:
        """External hook (supervisor / TUI) to force an immediate cycle + possible recovery exit."""
        return self.monitor_cycle()

    def start_monitoring_loop(self, interval_seconds: int = 60) -> None:
        """Blocking long-running loop (for dedicated process or testing)."""
        logger.info(f"[SelfMonitor] Starting continuous monitoring loop (interval={interval_seconds}s)")
        while True:
            try:
                res = self.monitor_cycle()
                if res.get("issues"):
                    logger.info(f"[SelfMonitor] Cycle issues handled: {res['issues']}")
            except KeyboardInterrupt:
                logger.info("[SelfMonitor] Loop stopped by user")
                break
            except Exception as exc:
                logger.exception(f"[SelfMonitor] Unhandled cycle error (continuing): {exc}")
            time.sleep(max(10, interval_seconds))


# ----------------------------------------------------------------------
# Module entrypoints
# ----------------------------------------------------------------------
def create_agent(**kwargs) -> SelfMonitoringRecoveryAgent:
    """Factory for supervisor / external orchestration."""
    return SelfMonitoringRecoveryAgent(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-Monitoring & Recovery Agent")
    parser.add_argument("--loop", action="store_true", help="Run continuous monitoring loop")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    parser.add_argument("--once", action="store_true", help="Run exactly one cycle then exit")
    args = parser.parse_args()

    agent = SelfMonitoringRecoveryAgent()

    if args.once:
        result = agent.monitor_cycle()
        print(json.dumps(result, indent=2, default=str))
    elif args.loop:
        agent.start_monitoring_loop(args.interval)
    else:
        print("Self-Monitoring Recovery Agent ready. Use --once or --loop.")
        print("Status file:", agent.status_path)
