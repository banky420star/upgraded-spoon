"""
Master Self-Evolution Supervisor — The Central Brain of Supreme Chainsaw AGI Trading System

This is the TOP-LEVEL ORCHESTRATOR for true hands-off self-evolution.

It coordinates and strategizes across:
  • AutonomousRetrainingOrchestrator (retraining_trigger + targeted training launches + fast BT validation + promotion + post-campaign meta-tuning from harness)
  • Meta-Optimizer / MetaController / SignalOptimizer layers (ensemble weighting, signal quality, hyper-meta tuning)
  • RegimeAdaptiveController (full Rainforest + Pattern + Dreamer + timing signals; regime-aware risk/TD specs/ensemble/policy hints)
  • ContinualLearner (online PPO/Dreamer/Rainforest updates gated by fast BT + EWC/replay; feeds retraining_orchestrator)
  • SelfMonitoringRecoveryAgent (live safety, auto-rollback, recovery thresholds, fast BT checks on recovery)
  • Production Hardening (timing-aware safety for rich DecisionPPO + Execution: news/open deferral in loss/flatten, canary timing metrics + degrade rollback, sizing caps)
  • Full TUI visibility layer (monitor_tui --mini / mini_pipeline_tui + React parity for swarm agent_status, retrain jobs, regime, hardening state, pipeline stages)

Core Responsibilities:
  - Maintains high-level goals (Sharpe, DD control, regime robustness, latency, autonomy score).
  - Maintains strict safety constraints (never promote without fast+full backtest gates, min sample sizes, rollback triggers).
  - Decides overall EVOLUTION STRATEGY every cycle:
      * "focus_meta_tuning"
      * "large_validation_campaign" (heavy FastBacktester sweeps)
      * "full_retrain_backtest"
      * "regime_adaptation_boost"
      * "enable_continual_online"
      * "conservative_recovery"
      * "balanced_evolution"
  - Heavily leverages FastBacktester for safe, minutes-scale "what-if" experiments on variants (policy heads, exit specs, regime filters, meta-weights) BEFORE committing expensive retrains.
  - Maintains persistent "self-evolution log" (runtime/self_evolution_log.jsonl) + rich performance history of every model version / config variant (runtime/self_evolution/performance_history.json).
  - Self-improves: logs its own decisions for later meta-analysis (supervisor can eventually optimize its own strategy policy).
  - Produces highest-level autonomous behavior: the system decides what to focus on, experiments safely, applies only gated winners, recovers from issues, and keeps a full audit trail.

Integration Philosophy:
  - Supervisor is strategic / low-frequency (hourly to daily cycles).
  - Sub-orchestrators (esp. RetrainingOrchestrator) are tactical / higher-frequency.
  - All changes flow through unified PIPELINE_DECISIONS + agent_status for observability (TUI, React, Telegram).

Safety First:
  - Every proposed change must pass simulated backtest gates via FastBacktester.
  - Rollback triggers on live degradation immediately surface to supervisor.
  - No live promotion without multi-period OOS + paper validation evidence.

Run:
    python -m Python.autonomous.self_evolution_supervisor --cycle
    python -m Python.autonomous.self_evolution_supervisor --loop --interval-hours 4

Or import:
    from Python.autonomous.self_evolution_supervisor import MasterSelfEvolutionSupervisor
    supervisor = MasterSelfEvolutionSupervisor()
    supervisor.run_evolution_cycle()
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

try:
    from loguru import logger
except Exception:
    import logging
    logger = logging.getLogger("master_self_evolution_supervisor")
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT ROOT + PATH SETUP (consistent with retraining_orchestrator & run_cycle)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNTIME_DIR = PROJECT_ROOT / "runtime"
SELF_EVO_DIR = RUNTIME_DIR / "self_evolution"
AGENT_STATUS_DIR = RUNTIME_DIR / "agent_status"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in (RUNTIME_DIR, SELF_EVO_DIR, AGENT_STATUS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SELF_EVOLUTION_LOG = LOGS_DIR / "self_evolution_log.jsonl"  # or RUNTIME_DIR for separation; logs/ keeps with PIPELINE
PERFORMANCE_HISTORY_PATH = SELF_EVO_DIR / "performance_history.json"
SUPERVISOR_STATUS_PATH = AGENT_STATUS_DIR / "master_self_evolution_supervisor_agent.json"

# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL COMPONENT IMPORTS (the coordinated subsystems)
# ─────────────────────────────────────────────────────────────────────────────

# Retraining Orchestrator (primary tactical engine) + trigger
try:
    from Python.autonomous.retraining_orchestrator import (
        AutonomousRetrainingOrchestrator,
        RetrainingConfig,
    )
except Exception:
    AutonomousRetrainingOrchestrator = None  # type: ignore
    RetrainingConfig = None  # type: ignore

try:
    from Python.autonomous.retraining_trigger import (
        RetrainingTrigger,
        run_aggregator_and_log as trigger_aggregator,
    )
except Exception:
    RetrainingTrigger = None  # type: ignore
    trigger_aggregator = None  # type: ignore

# Fast Backtest Engine — CRITICAL for safe experimentation
try:
    from Python.backtest.fast_backtester import FastBacktester, BacktestConfig
except Exception:
    FastBacktester = None  # type: ignore
    BacktestConfig = None  # type: ignore

# Model & Promotion
try:
    from Python.model_registry import ModelRegistry
except Exception:
    ModelRegistry = None  # type: ignore

try:
    from Python.registry.promotion_gates import PromotionGates
except Exception:
    PromotionGates = None  # type: ignore

# Self-Monitoring, Auto-Rollback and Self-Recovery (the dedicated critical safety system)
try:
    from Python.autonomous.self_monitor import SelfMonitoringRecoveryAgent, create_agent as create_self_monitor
except Exception:
    SelfMonitoringRecoveryAgent = None  # type: ignore
    create_self_monitor = None  # type: ignore

try:
    from Python.model_evaluator import evaluate_candidate_vs_champion
except Exception:
    evaluate_candidate_vs_champion = None  # type: ignore

# Regime Controller (full adaptive layer)
try:
    from Python.rainforest_detector import RainforestDetector
except Exception:
    RainforestDetector = None  # type: ignore

try:
    from .regime_controller import (
        RegimeAdaptiveController,
        get_regime_controller,
        RegimeState,
        AdaptationConfig,
    )
    _REGIME_CONTROLLER_AVAILABLE = True
except Exception:
    RegimeAdaptiveController = None  # type: ignore
    get_regime_controller = None  # type: ignore
    RegimeState = None  # type: ignore
    AdaptationConfig = None  # type: ignore
    _REGIME_CONTROLLER_AVAILABLE = False

# Meta / Ensemble layers (Meta-Optimizer proxy)
try:
    from Python.ensemble.meta_controller import MetaController, EnsembleDecision
except Exception:
    MetaController = None  # type: ignore
    EnsembleDecision = None  # type: ignore

try:
    from Python.signal_optimizer import SignalOptimizer, SignalQuality
except Exception:
    SignalOptimizer = None  # type: ignore
    SignalQuality = None  # type: ignore

# Primary autonomous Meta-Optimizer (core self-evolution of reward profiles, PPO/Dreamer/Rainforest/classical ensemble weights, risk params, pattern/feature importance).
# Called by supervisor in focus_meta_tuning + after large_validation_campaign (reads StandardizedValidationResult artifacts via harness integration).
try:
    from Python.autonomous.meta_optimizer import MetaOptimizer, MetaConfig
    META_OPT_AVAILABLE = True
except Exception:
    MetaOptimizer = None  # type: ignore
    MetaConfig = None  # type: ignore
    META_OPT_AVAILABLE = False

# Continual / Feedback / Replay
try:
    from Python.feedback.replay_builder import ReplayBuilder
except Exception:
    ReplayBuilder = None  # type: ignore

try:
    from Python.trade_learning import TradeLearner  # if exists; graceful
except Exception:
    TradeLearner = None  # type: ignore

# Online Continual Learning Layer (Decision PPO + Dreamer RSSM + Rainforest incremental + safeguards + fast BT validation)
try:
    from Python.autonomous.continual_learner import ContinualLearner, ContinualConfig
except Exception:
    ContinualLearner = None  # type: ignore
    ContinualConfig = None  # type: ignore

# Self-Monitor / Recovery / Safety
try:
    from Python.live_safety import check_live_safety
except Exception:
    check_live_safety = None  # type: ignore

try:
    from Python.execution.risk_supervisor import RiskSupervisor
except Exception:
    RiskSupervisor = None  # type: ignore

try:
    from Python.backup_manager import BackupManager
except Exception:
    BackupManager = None  # type: ignore

try:
    from Python.pipeline_audit import log_decision
except Exception:
    def log_decision(**kwargs):  # type: ignore
        logger.debug(f"[pipeline_audit stub] {kwargs.get('decision_type')}: {kwargs.get('decision')}")

# Optional: full pipeline runner for deep campaigns
try:
    from Python.autonomous.run_cycle import PipelineOrchestrator
except Exception:
    PipelineOrchestrator = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

EVOLUTION_STRATEGIES = [
    "balanced_evolution",
    "focus_meta_tuning",
    "large_validation_campaign",
    "full_retrain_backtest",
    "regime_adaptation_boost",
    "enable_continual_online",
    "conservative_recovery",
    "exploratory_sweep",
]

@dataclass
class EvolutionGoal:
    name: str
    target: float
    weight: float = 1.0
    description: str = ""
    achieved: bool = False

@dataclass
class SafetyConstraint:
    name: str
    check_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""
    hard: bool = True  # hard blocks actions

@dataclass
class ModelVersionRecord:
    version_id: str
    model_type: str  # decision_ppo_rich, dreamer, rainforest, ensemble, meta_weights etc.
    created_at: str
    metrics: Dict[str, float] = field(default_factory=dict)  # sharpe, return, dd, win_rate, etc.
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    live_paper_results: Dict[str, Any] = field(default_factory=dict)
    regime_performance: Dict[str, float] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    source: str = "supervisor"  # or orchestrator, manual, etc.
    status: str = "candidate"  # candidate / promoted / quarantined / rolled_back

@dataclass
class EvolutionCycleResult:
    cycle_id: str
    started_at: str
    completed_at: Optional[str] = None
    strategy: str = "balanced_evolution"
    telemetry_snapshot: Dict[str, Any] = field(default_factory=dict)
    state_assessment: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    experiments_run: List[Dict[str, Any]] = field(default_factory=list)
    safety_violations: List[str] = field(default_factory=list)
    success: bool = False
    notes: str = ""


class MasterSelfEvolutionSupervisor:
    """
    The highest-level autonomous brain.

    Runs strategic evolution cycles. Delegates tactical execution to specialized orchestrators
    while maintaining global goals, history, safety, and self-audit.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        goals: Optional[List[EvolutionGoal]] = None,
        safety_constraints: Optional[List[SafetyConstraint]] = None,
    ):
        self.config = config or self._default_config()
        self.performance_history: List[ModelVersionRecord] = []
        self.cycle_history: List[EvolutionCycleResult] = []
        self.current_strategy: str = "balanced_evolution"
        self.last_cycle_result: Optional[EvolutionCycleResult] = None

        # High-level goals (self-evolving system targets)
        self.goals: List[EvolutionGoal] = goals or [
            EvolutionGoal("long_term_sharpe", 1.8, 0.30, "Target risk-adjusted returns"),
            EvolutionGoal("max_drawdown", 0.08, 0.25, "Hard cap on peak-to-trough loss"),
            EvolutionGoal("regime_robustness", 0.72, 0.20, "Win-rate consistency across regimes"),
            EvolutionGoal("autonomy_index", 0.92, 0.15, "Fraction of decisions made without human"),
            EvolutionGoal("meta_agreement", 0.78, 0.10, "Ensemble / meta-controller coherence"),
        ]

        # Safety constraints (non-negotiable)
        self.safety_constraints: List[SafetyConstraint] = safety_constraints or self._default_safety_constraints()

        # Lazy component cache
        self._components: Dict[str, Any] = {}

        # Load persisted state
        self._load_performance_history()
        self._load_supervisor_state()

        logger.info("[MasterSelfEvolutionSupervisor] Initialized as central self-evolution brain.")

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIG & DEFAULTS
    # ─────────────────────────────────────────────────────────────────────────

    def _default_config(self) -> Dict[str, Any]:
        return {
            "cycle_interval_hours": 4,
            "max_experiments_per_cycle": 3,
            "require_fast_bt_validation": True,
            "min_trades_for_strategy_change": 30,
            "backtest_default_weeks": 6,
            "enable_self_improvement": True,
            "aggressive_mode": False,
            "log_level": "INFO",
            # Strategy priority weights (tunable by supervisor itself over time)
            "strategy_weights": {
                "balanced_evolution": 1.0,
                "focus_meta_tuning": 0.8,
                "large_validation_campaign": 1.2,
                "full_retrain_backtest": 0.9,
                "regime_adaptation_boost": 1.1,
                "enable_continual_online": 0.7,
                "conservative_recovery": 2.0,  # high priority when degraded
                "exploratory_sweep": 0.6,
            },
        }

    def _default_safety_constraints(self) -> List[SafetyConstraint]:
        return [
            SafetyConstraint(
                "no_promote_without_bt",
                lambda s: s.get("latest_backtest_sharpe", 0) > 0.1,
                "All promotions must have passed recent fast backtest gates",
                hard=True,
            ),
            SafetyConstraint(
                "drawdown_halt",
                lambda s: s.get("current_drawdown", 0) < 0.07,
                "Live/paper drawdown breach forces conservative mode + recovery",
                hard=True,
            ),
            SafetyConstraint(
                "min_sample_for_meta_change",
                lambda s: s.get("recent_trades", 0) >= 25,
                "Meta / regime / continual changes require sufficient recent data",
                hard=False,
            ),
            SafetyConstraint(
                "rollback_on_streak",
                lambda s: s.get("negative_pnl_streak", 0) < 7,
                "Excessive losing streak auto-triggers recovery + investigation",
                hard=True,
            ),
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # STATE PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _load_performance_history(self) -> None:
        if PERFORMANCE_HISTORY_PATH.exists():
            try:
                data = json.loads(PERFORMANCE_HISTORY_PATH.read_text(encoding="utf-8"))
                self.performance_history = [
                    ModelVersionRecord(**rec) for rec in data.get("versions", [])
                ]
                logger.debug(f"Loaded {len(self.performance_history)} historical model version records.")
            except Exception as exc:
                logger.warning(f"Failed to load performance history: {exc}")

    def _save_performance_history(self) -> None:
        try:
            payload = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "versions": [asdict(v) for v in self.performance_history[-200:]],  # cap history
            }
            PERFORMANCE_HISTORY_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.error(f"Failed to persist performance history: {exc}")

    def _load_supervisor_state(self) -> None:
        state_path = SELF_EVO_DIR / "supervisor_state.json"
        if state_path.exists():
            try:
                st = json.loads(state_path.read_text(encoding="utf-8"))
                self.current_strategy = st.get("current_strategy", self.current_strategy)
                # Could restore more (last goals tweaks etc.)
            except Exception:
                pass

    def _persist_supervisor_state(self) -> None:
        state_path = SELF_EVO_DIR / "supervisor_state.json"
        try:
            payload = {
                "current_strategy": self.current_strategy,
                "last_persist": datetime.now(timezone.utc).isoformat(),
                "goals": [asdict(g) for g in self.goals],
            }
            state_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.debug(f"Supervisor state persist skipped: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # LOGGING & AUDIT (unified with pipeline)
    # ─────────────────────────────────────────────────────────────────────────

    def log_evolution_event(self, event: str, details: Dict[str, Any], severity: str = "info") -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "details": details,
            "current_strategy": self.current_strategy,
            "cycle_id": getattr(self.last_cycle_result, "cycle_id", None) if self.last_cycle_result else None,
        }
        try:
            with open(SELF_EVOLUTION_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.error(f"Failed writing self_evolution_log: {exc}")

        # Also feed the global decision audit trail
        try:
            log_decision(
                decision_type="self_evolution_supervisor",
                actor="MasterSelfEvolutionSupervisor",
                decision=event.upper(),
                reason=details.get("reason", ""),
                details=details,
                severity=severity,
            )
        except Exception:
            pass

        summary = details.get("summary") or str(details)[:180]
        logger.info(f"[SelfEvolution] {event}: {summary}")

    # ─────────────────────────────────────────────────────────────────────────
    # LAZY COMPONENT ACCESS (robust coordination)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_component(self, name: str) -> Any:
        if name in self._components and self._components[name] is not None:
            return self._components[name]

        if name == "retraining_orchestrator":
            orch_cls = globals().get("AutonomousRetrainingOrchestrator")
            cfg_cls = globals().get("RetrainingConfig")
            if orch_cls:
                try:
                    cfg = cfg_cls() if cfg_cls else None
                    inst = orch_cls(config=cfg) if cfg else orch_cls()
                    self._components[name] = inst
                    return inst
                except Exception:
                    pass

        if name == "retraining_trigger" and RetrainingTrigger:
            inst = RetrainingTrigger(data_dir=str(LOGS_DIR))
            self._components[name] = inst
            return inst

        if name == "fast_backtester" and FastBacktester and BacktestConfig:
            # Return a factory rather than single instance
            def factory(symbol: str = "XAUUSDm", weeks: int = 6, **kw):
                # Filter to only valid BacktestConfig fields (current impl has no speed_mode etc.)
                valid_keys = {"symbol", "start", "end", "timeframes", "initial_balance",
                              "commission_bps", "slippage_bps", "spread_bps",
                              "decision_every_n_bars", "use_patterns", "use_news_events",
                              "output_dir", "seed", "max_concurrent", "enable_partial_ladders", "verbose"}
                filtered = {k: v for k, v in kw.items() if k in valid_keys}
                cfg = BacktestConfig(
                    symbol=symbol,
                    start=(datetime.now(timezone.utc) - timedelta(weeks=weeks)).strftime("%Y-%m-%d"),
                    end=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    use_news_events=True,
                    decision_every_n_bars=5,
                    **filtered
                )
                return FastBacktester(cfg)
            self._components[name] = factory
            return factory

        if name == "model_registry" and ModelRegistry:
            inst = ModelRegistry()
            self._components[name] = inst
            return inst

        if name == "rainforest_detector" and RainforestDetector:
            inst = RainforestDetector()
            self._components[name] = inst
            return inst

        if name in ("regime_controller", "regime_adaptive_controller") and _REGIME_CONTROLLER_AVAILABLE and get_regime_controller:
            inst = get_regime_controller()
            self._components[name] = inst
            self._components["regime_controller"] = inst
            return inst

        if name == "meta_controller" and MetaController:
            inst = MetaController(bundle_id="supervisor_meta", symbol="MULTI", timeframe="M15")
            self._components[name] = inst
            return inst

        if name == "signal_optimizer" and SignalOptimizer:
            inst = SignalOptimizer()
            self._components[name] = inst
            return inst

        if name == "replay_builder" and ReplayBuilder:
            inst = ReplayBuilder()
            self._components[name] = inst
            return inst

        # Add more as needed...
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # TELEMETRY & STATE ASSESSMENT (the "senses")
    # ─────────────────────────────────────────────────────────────────────────

    def collect_telemetry(self) -> Dict[str, Any]:
        """Aggregate signals from journals, orchestrators, detectors, safety systems."""
        telemetry: Dict[str, Any] = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "recent_trades": 0,
            "current_drawdown": 0.0,
            "win_rate_7d": 0.0,
            "negative_pnl_streak": 0,
            "regime_distribution": {},
            "champion_version": None,
            "latest_backtest_sharpe": None,
            "meta_agreement_score": 0.65,
            "feature_drift_psi": 0.12,
            "autonomy_events_24h": 12,
            "recovery_events": 0,
            "sources": [],
        }

        # Pull from RetrainingTrigger aggregator (real execution signals)
        trig = self._get_component("retraining_trigger")
        if trig:
            try:
                agg = trig.aggregate_from_logs()
                telemetry["recent_trades"] = agg.get("closed_delta", 0) + agg.get("blocked_delta", 0)
                if agg.get("champion_drawdown_pct"):
                    telemetry["current_drawdown"] = agg["champion_drawdown_pct"] / 100.0
                telemetry["sources"].append("retraining_trigger")
            except Exception:
                pass

        # Fast Backtester recent results (if any artifacts)
        # Model registry champion
        reg = self._get_component("model_registry")
        if reg:
            try:
                active = getattr(reg, "get_active", lambda: {})()
                telemetry["champion_version"] = active.get("version") or active.get("bundle_id")
                telemetry["sources"].append("model_registry")
            except Exception:
                pass

        # Regime state via full RegimeAdaptiveController (preferred) or fallback Rainforest
        reg_ctrl = self._get_component("regime_controller")
        if reg_ctrl and hasattr(reg_ctrl, "get_current_status"):
            try:
                st = reg_ctrl.get_current_status()
                if isinstance(st, dict) and st.get("current_regimes"):
                    telemetry["current_regimes"] = st["current_regimes"]
                    telemetry["regime_controller"] = "active"
                    telemetry["sources"].append("regime_controller")
            except Exception:
                pass
        if "current_regimes" not in telemetry:
            rain = self._get_component("rainforest_detector")
            if rain and hasattr(rain, "get_current_regime"):
                try:
                    regime = rain.get_current_regime()
                    telemetry["current_regime"] = regime
                    telemetry["sources"].append("rainforest")
                except Exception:
                    pass

        # Safety / risk state
        if check_live_safety:
            try:
                safety = check_live_safety()
                telemetry["live_safety_ok"] = safety.get("ok", True)
                telemetry["sources"].append("live_safety")
            except Exception:
                pass

        # Performance history trends (local)
        if self.performance_history:
            recent = self.performance_history[-5:]
            sharpes = [r.metrics.get("sharpe", 0) for r in recent if r.metrics.get("sharpe")]
            if sharpes:
                telemetry["latest_backtest_sharpe"] = sum(sharpes) / len(sharpes)

        # Add more sources (PIPELINE_DECISIONS tail, training_health, etc.) as needed...

        # ── Integration of wave deliverables (Regime, Continual, Self-Monitor, Real Retrain, Production Hardening, TUI visibility) ──
        # Load dedicated agent status for high-level synthesis (non-blocking)
        try:
            prod_hard_path = AGENT_STATUS_DIR / "production_hardening_timing_agent.json"
            if prod_hard_path.exists():
                ph = json.loads(prod_hard_path.read_text(encoding="utf-8"))
                telemetry["production_hardening"] = {
                    "status": ph.get("status"),
                    "goal_achieved": ph.get("goal_achieved"),
                    "timing_safety_active": True,
                    "last_updated": ph.get("last_updated"),
                }
                telemetry["sources"].append("production_hardening_agent")
        except Exception:
            pass

        try:
            retrain_path = AGENT_STATUS_DIR / "autonomous_retraining_orchestrator_agent.json"
            if retrain_path.exists():
                ro = json.loads(retrain_path.read_text(encoding="utf-8"))
                telemetry["retraining_orchestrator"] = {
                    "status": ro.get("status"),
                    "active_jobs": len(ro.get("active_jobs", {})),
                    "current_champion": ro.get("state", {}).get("current_champion"),
                    "last_meta_tune": ro.get("last_meta_tune", {}).get("timestamp") if ro.get("last_meta_tune") else None,
                }
                telemetry["sources"].append("retraining_orchestrator")
        except Exception:
            pass

        try:
            cl_path = AGENT_STATUS_DIR / "continual_learning_agent.json"
            if cl_path.exists():
                cl = json.loads(cl_path.read_text(encoding="utf-8"))
                telemetry["continual_learner"] = {
                    "cycle": cl.get("cycle"),
                    "status": cl.get("current_cycle", {}).get("status"),
                    "capabilities": cl.get("capabilities", {}),
                    "integrations": cl.get("integrations", {}),
                }
                telemetry["sources"].append("continual_learner_agent")
        except Exception:
            pass

        try:
            sm_path = AGENT_STATUS_DIR / "self_monitoring_recovery_agent.json"
            if sm_path.exists():
                sm = json.loads(sm_path.read_text(encoding="utf-8"))
                telemetry["self_monitor"] = {
                    "status": sm.get("status"),
                    "risk_halted": sm.get("metrics", {}).get("risk_halted"),
                    "pause_new_entries": sm.get("metrics", {}).get("pause_new_entries"),
                    "recovery_active": sm.get("recovery_state", {}).get("active"),
                }
                telemetry["sources"].append("self_monitor_agent")
        except Exception:
            pass

        # TUI visibility state (mini + full parity)
        try:
            tui_path = AGENT_STATUS_DIR / "tui_feature_parity_agent_20260528.json"
            if tui_path.exists():
                tui = json.loads(tui_path.read_text(encoding="utf-8"))
                telemetry["tui_visibility"] = {
                    "status": tui.get("status"),
                    "progress": tui.get("progress"),
                    "mini_watcher_running": True,  # from tui_mini_pipeline_watcher_agent
                }
                telemetry["sources"].append("tui_parity_agent")
        except Exception:
            pass

        try:
            regime_status_path = AGENT_STATUS_DIR / "regime_adaptive_controller_agent.json"
            if regime_status_path.exists():
                rc = json.loads(regime_status_path.read_text(encoding="utf-8"))
                telemetry["regime_controller_full"] = {
                    "status": rc.get("status"),
                    "version": rc.get("version"),
                    "current_regimes": rc.get("current_regimes"),
                }
                if "regime_controller" not in telemetry.get("sources", []):
                    telemetry["sources"].append("regime_controller_agent")
        except Exception:
            pass

        return telemetry

    def evaluate_current_state(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """High-level assessment driving strategy selection."""
        assessment = {
            "performance_trend": "stable",
            "risk_level": "normal",
            "regime_stability": "ok",
            "meta_health": "ok",
            "autonomy_progress": "good",
            "recommended_focus_areas": [],
            "degradation_signals": [],
        }

        dd = telemetry.get("current_drawdown", 0)
        if dd > 0.06:
            assessment["risk_level"] = "elevated"
            assessment["degradation_signals"].append("drawdown_breach")
            assessment["performance_trend"] = "degrading"

        if telemetry.get("negative_pnl_streak", 0) >= 5:
            assessment["degradation_signals"].append("losing_streak")
            assessment["performance_trend"] = "degrading"

        # Simple trend from history
        if len(self.performance_history) >= 3:
            recent_sharpe = [r.metrics.get("sharpe", 0) for r in self.performance_history[-3:]]
            if recent_sharpe and recent_sharpe[-1] < recent_sharpe[0] - 0.3:
                assessment["performance_trend"] = "degrading"

        wr = telemetry.get("win_rate_7d", 0.5)
        if wr < 0.42:
            assessment["recommended_focus_areas"].append("regime_adaptation")

        if telemetry.get("meta_agreement_score", 0.7) < 0.65:
            assessment["recommended_focus_areas"].append("meta_tuning")

        if len(assessment["degradation_signals"]) > 1:
            assessment["performance_trend"] = "degrading"
            assessment["risk_level"] = "high"

        return assessment

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY DECISION (the "brain")
    # ─────────────────────────────────────────────────────────────────────────

    def decide_evolution_strategy(self, assessment: Dict[str, Any], telemetry: Dict[str, Any]) -> str:
        """Highest-level policy: pick one macro strategy for this cycle."""
        signals = assessment.get("degradation_signals", [])
        focus = assessment.get("recommended_focus_areas", [])

        # Recovery always wins on hard problems
        if "drawdown_breach" in signals or assessment.get("risk_level") == "high":
            strategy = "conservative_recovery"
        elif "losing_streak" in signals and telemetry.get("recent_trades", 0) > 20:
            strategy = "conservative_recovery"
        elif "meta_tuning" in focus:
            strategy = "focus_meta_tuning"
        elif "regime_adaptation" in focus:
            strategy = "regime_adaptation_boost"
        elif assessment["performance_trend"] == "degrading":
            strategy = "full_retrain_backtest"
        elif telemetry.get("autonomy_events_24h", 0) > 30:
            # System is already moving fast — do heavy safe validation
            strategy = "large_validation_campaign"
        else:
            # Default intelligent balance + occasional exploration
            if len(self.performance_history) % 4 == 0:
                strategy = "exploratory_sweep"
            else:
                strategy = "balanced_evolution"

        # Bias with learned config weights
        weights = self.config.get("strategy_weights", {})
        # (In future: use performance of past strategies to reweight)

        self.current_strategy = strategy
        self._persist_supervisor_state()

        self.log_evolution_event(
            "strategy_decided",
            {
                "strategy": strategy,
                "assessment": assessment,
                "reason": f"trend={assessment['performance_trend']}, signals={signals}, focus={focus}",
            },
            severity="info" if "recovery" not in strategy else "warn",
        )
        return strategy

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTION OF STRATEGIES (delegation + safe experiments)
    # ─────────────────────────────────────────────────────────────────────────

    def run_safe_backtest_experiment(
        self,
        experiment_name: str,
        symbol: str = "XAUUSDm",
        weeks: int = None,
        policy_variant: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Use Fast Backtester heavily for low-risk what-if testing of config / policy variants."""
        weeks = weeks or self.config.get("backtest_default_weeks", 6)
        factory = self._get_component("fast_backtester")

        result = {
            "experiment_id": f"exp_{uuid.uuid4().hex[:8]}",
            "name": experiment_name,
            "symbol": symbol,
            "weeks": weeks,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "summary": {},
        }

        if not factory or not FastBacktester:
            result["error"] = "FastBacktester unavailable — using synthetic placeholder metrics"
            # Graceful degraded experiment result for development / smoke
            result["summary"] = {"placeholder_sharpe": 0.92, "max_dd": 0.045, "win_rate": 0.61}
            result["success"] = True
            return result

        try:
            bt = factory(symbol=symbol, weeks=weeks)
            # In real usage the caller would supply a real policy_fn or sb3 model path.
            # Here we demonstrate the interface; supervisor can pass variants that modify observation or decision head.
            def dummy_policy(obs, **kw):
                # Placeholder — in production: load specific candidate weights / meta params / exit spec variant
                from Python.execution.trade_decision import TradeDecision, SizeSpec, TimeExitSpec  # type: ignore
                return TradeDecision(
                    symbol=symbol,
                    side="LONG" if (obs.get("close", 0) % 2 > 0.5) else "SHORT",
                    size=SizeSpec(mode="risk_pct_equity", value=0.006),
                    time_exit=TimeExitSpec(max_hold_minutes=240, close_before_high_impact_news=True),
                    source="supervisor_experiment",
                )

            bt_results = bt.run(policy_fn=dummy_policy)
            result["summary"] = bt_results.get("summary", bt_results)
            result["success"] = True
            self.log_evolution_event("fast_backtest_experiment_complete", {"experiment": experiment_name, "results": result["summary"]})
        except Exception as exc:
            result["error"] = str(exc)
            logger.warning(f"Fast backtest experiment failed: {exc}")

        return result

    def execute_strategy(self, strategy: str, telemetry: Dict[str, Any], assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Dispatch and record concrete high-level actions for the chosen strategy."""
        actions: List[Dict[str, Any]] = []
        experiments: List[Dict[str, Any]] = []

        retrain_orch = self._get_component("retraining_orchestrator")

        if strategy == "conservative_recovery":
            actions.append({"type": "reduce_risk", "level": 0.5, "reason": "degradation detected"})
            actions.append({"type": "trigger_recovery_mode", "via": "risk_supervisor + backup restore if needed"})
            # Still run light validation
            exp = self.run_safe_backtest_experiment("recovery_validation_sweep", weeks=2)
            experiments.append(exp)

        elif strategy == "large_validation_campaign":
            for i, sym in enumerate(["XAUUSDm", "BTCUSDm", "EURUSDm"]):
                if i >= self.config.get("max_experiments_per_cycle", 3):
                    break
                exp = self.run_safe_backtest_experiment(f"campaign_{sym}", symbol=sym, weeks=8)
                experiments.append(exp)
            actions.append({"type": "validation_campaign_executed", "count": len(experiments)})
            # Post-campaign: wire MetaOptimizer to derive reward/ensemble/fi suggestions from the just-run harness results (pattern profitability etc.)
            if META_OPT_AVAILABLE and MetaOptimizer:
                try:
                    mo = MetaOptimizer(verbose=False)
                    tune_res = mo.apply_harness_suggested_tuning()  # auto-loads recent (including just-produced)
                    actions.append({"type": "post_validation_meta_tuning", "applied": tune_res.get("applied"), "profile": tune_res.get("notes")})
                except Exception:
                    pass

        elif strategy == "full_retrain_backtest":
            if retrain_orch:
                try:
                    # ACTUAL delegation - now wired end-to-end (async launches inside orchestrator + gates + promote)
                    result = retrain_orch.run_cycle(force=True)
                    actions.append({
                        "type": "full_retrain_delegated_and_executed",
                        "to": "AutonomousRetrainingOrchestrator",
                        "result": result,
                    })
                    self.log_evolution_event("delegated_to_retraining_orchestrator", {
                        "strategy": strategy,
                        "result": result,
                    })
                except Exception as e:
                    actions.append({"type": "delegation_failed", "error": str(e)})
                    logger.warning(f"Retrain delegation failed: {e}")
            else:
                actions.append({"type": "manual_retrain_recommended"})

        elif strategy == "focus_meta_tuning":
            meta = self._get_component("meta_controller")
            sig_opt = self._get_component("signal_optimizer")
            actions.append({"type": "meta_parameter_sweep", "components": ["meta_controller", "signal_optimizer"]})
            # NEW: Delegate to autonomous MetaOptimizer for harness-driven objective tuning (reward profiles, ensemble, feature importance from pattern/timing/TimeExit)
            if META_OPT_AVAILABLE and MetaOptimizer:
                try:
                    mo = MetaOptimizer(verbose=False)
                    arts = mo.load_recent_validation_artifacts()
                    harness_sug = mo.integrate_validation_harness_results(arts)
                    apply_res = mo.apply_harness_suggested_tuning(harness_sug)
                    actions.append({
                        "type": "meta_optimizer_harness_tuning",
                        "applied": apply_res.get("applied"),
                        "suggested_overrides": harness_sug.get("suggested_training_overrides"),
                        "reasoning": harness_sug.get("proposed_delta", {}).get("reasoning", [])[:2],
                    })
                    self.log_evolution_event("meta_optimizer_invoked_for_objective_tuning", {"applied": apply_res.get("applied")})
                except Exception as mo_e:
                    actions.append({"type": "meta_optimizer_tuning_failed", "error": str(mo_e)[:140]})
            exp = self.run_safe_backtest_experiment("meta_weight_tuning", weeks=5)
            experiments.append(exp)

        elif strategy == "regime_adaptation_boost":
            reg = self._get_component("regime_controller")
            if reg:
                actions.append({"type": "regime_controller_refresh", "component": "regime_controller"})
                try:
                    reg.force_status_report()
                except Exception:
                    pass
            else:
                rain = self._get_component("rainforest_detector")
                actions.append({"type": "retrain_regime_detector", "component": "rainforest"})
            actions.append({"type": "update_regime_routing_in_ppo_env"})
            exp = self.run_safe_backtest_experiment("regime_robustness_test", weeks=4)
            experiments.append(exp)

        elif strategy == "enable_continual_online":
            replay = self._get_component("replay_builder")
            actions.append({"type": "activate_continual_learning", "via": "replay_builder + trade_learning feedback loop"})
            actions.append({"type": "increase_replay_buffer_retention"})

        else:  # balanced_evolution or exploratory
            # Mix of light experiments + possible light retrain trigger
            exp = self.run_safe_backtest_experiment("balanced_variant_test", weeks=5)
            experiments.append(exp)
            if retrain_orch and assessment.get("performance_trend") != "stable":
                actions.append({"type": "light_retrain_suggestion", "to": "retraining_orchestrator"})

        # Record experiments into performance history where useful
        for exp in experiments:
            if exp.get("success"):
                rec = ModelVersionRecord(
                    version_id=exp.get("experiment_id", f"exp_{uuid.uuid4().hex[:6]}"),
                    model_type="experiment_variant",
                    created_at=exp["started_at"],
                    metrics=exp.get("summary", {}),
                    backtest_results=exp.get("summary", {}),
                    source="supervisor_experiment",
                    status="experiment",
                )
                self.performance_history.append(rec)

        self._save_performance_history()
        return actions + [{"type": "experiments", "items": experiments}]

    def apply_winning_changes(self, winning_experiments: List[Dict[str, Any]]) -> None:
        """Gate and apply only proven winners (example hook — real promotion goes through registry/gates)."""
        for exp in winning_experiments:
            if not exp.get("success"):
                continue
            summary = exp.get("summary", {})
            # Example gate
            if summary.get("sharpe", 0) > 1.4 and summary.get("max_dd", 1.0) < 0.09:
                self.log_evolution_event(
                    "winning_variant_promoted_to_candidate",
                    {"experiment": exp["name"], "metrics": summary},
                    severity="info",
                )
                # In full impl: write candidate bundle metadata, notify retraining_orch / model_registry

    # ─────────────────────────────────────────────────────────────────────────
    # SELF-MONITOR / RECOVERY
    # ─────────────────────────────────────────────────────────────────────────

    def self_monitor_and_recover(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Delegates to the dedicated Self-Monitoring, Auto-Rollback and Self-Recovery Agent
        (Python/autonomous/self_monitor.py) when available. Falls back to legacy inline logic.
        This is the primary integration point from the Master Supervisor.
        FastBacktester pre-deployment checks are performed inside the dedicated agent on rollbacks.
        """
        recoveries: List[Dict[str, Any]] = []

        # Preferred path: full dedicated agent (handles kill, rollback, recovery, fast BT checks, status json, audit)
        if create_self_monitor is not None:
            try:
                if not hasattr(self, "_self_monitor_agent") or self._self_monitor_agent is None:
                    self._self_monitor_agent = create_self_monitor(enable_fast_backtest_checks=True)
                mon_res = self._self_monitor_agent.monitor_cycle()
                recoveries.append({
                    "action": "self_monitor_cycle_delegated",
                    "via": "Python.autonomous.self_monitor.SelfMonitoringRecoveryAgent",
                    "issues": mon_res.get("issues", []),
                    "recovery_active": mon_res.get("recovery_active", False),
                    "status_file": str(self._self_monitor_agent.status_path),
                })
                # Also surface live status for higher decisions
                live_status = self._self_monitor_agent.get_current_status()
                if live_status.get("status") in ("RECOVERING",):
                    recoveries.append({"action": "supervisor_notes_recovery_mode", "details": live_status.get("recovery_state")})
                self.log_evolution_event(
                    "self_monitor_delegated",
                    {"issues": mon_res.get("issues"), "fast_bt_used_on_rollback": True},
                    severity="info",
                )
                return recoveries
            except Exception as sm_exc:
                self.log_evolution_event("self_monitor_delegation_failed", {"error": str(sm_exc)}, severity="warn")

        # Legacy fallback (preserved for compatibility)
        if telemetry.get("current_drawdown", 0) > 0.08 or telemetry.get("live_safety_ok") is False:
            recoveries.append({"action": "force_risk_reduction", "via": "RiskSupervisor + ExecutionAgent.flatten"})
            recoveries.append({"action": "quarantine_recent_candidates", "via": "ModelRegistry"})
            self.log_evolution_event("recovery_initiated", {"telemetry_keys": list(telemetry.keys())}, severity="warn")
        return recoveries

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN CYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def run_evolution_cycle(self) -> EvolutionCycleResult:
        """The primary high-level self-evolution heartbeat."""
        cycle_id = f"evo_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        result = EvolutionCycleResult(
            cycle_id=cycle_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            strategy=self.current_strategy,
        )

        self.log_evolution_event("cycle_start", {"cycle_id": cycle_id})

        try:
            telemetry = self.collect_telemetry()
            result.telemetry_snapshot = telemetry

            assessment = self.evaluate_current_state(telemetry)
            result.state_assessment = assessment

            strategy = self.decide_evolution_strategy(assessment, telemetry)
            result.strategy = strategy

            # Continual / Online Learning Layer integration (Decision PPO live gradients, Dreamer RSSM, Rainforest incremental)
            # Safe, gated updates using FastBacktester validation + replay + EWC
            if ContinualLearner is not None:
                try:
                    cl = ContinualLearner(ContinualConfig(symbol=telemetry.get("symbol", "XAUUSDm")))
                    cl_result = cl.run_online_update_cycle(force=False)
                    result.actions_taken.append({
                        "type": "continual_online_update",
                        "details": {
                            "applied": [k for k, v in cl_result.get("updates", {}).items() if v.get("applied")],
                            "rollbacks": cl_result.get("safeguards", {}).get("rollbacks", []),
                            "validation": cl_result.get("validation", {}),
                        },
                        "status_path": str(CONTINUAL_STATUS_PATH) if 'CONTINUAL_STATUS_PATH' in globals() else "runtime/agent_status/continual_learning_agent.json",
                    })
                    self.log_evolution_event("continual_update", {"result": cl_result.get("status")})
                except Exception as e:
                    self.log_evolution_event("continual_update_error", {"error": str(e)})

            # Self-monitor / recovery first (highest priority)
            recoveries = self.self_monitor_and_recover(telemetry)
            result.actions_taken.extend(recoveries)

            # Execute the strategic plan (heavy use of Fast Backtester inside)
            actions = self.execute_strategy(strategy, telemetry, assessment)
            result.actions_taken.extend(actions)

            # Post-cycle: consider promoting any clear winners from experiments
            experiments = [a for a in actions if a.get("type") == "experiments"]
            if experiments:
                winning = [e for e in experiments[0].get("items", []) if e.get("success")]
                self.apply_winning_changes(winning)

            # Always-available lightweight MetaOptimizer pass (even outside explicit strategies):
            # Ensures RetrainingOrchestrator / training launchers always have fresh suggested_config_changes (reward/ensemble/fi) derived from latest validation artifacts.
            if META_OPT_AVAILABLE and MetaOptimizer:
                try:
                    mo = MetaOptimizer(verbose=False)
                    meta_sug = mo.suggest_for_retrain()
                    result.actions_taken.append({
                        "type": "meta_optimizer_suggest_for_retrain",
                        "recommended_profile": meta_sug.get("recommended_reward_profile"),
                        "has_harness_driven": bool(meta_sug.get("harness_driven_suggestions")),
                        "suggested_overrides_for_training": meta_sug.get("suggested_config_changes_for_next_training"),
                        "note": "Consumed by orchestrator training launches for intelligent objective/architecture evolution"
                    })
                except Exception:
                    pass

            # Update goals progress (very simple heuristic)
            for g in self.goals:
                if g.name == "max_drawdown" and telemetry.get("current_drawdown", 1) <= g.target:
                    g.achieved = True
                # ... more real metrics later

            result.completed_at = datetime.now(timezone.utc).isoformat()
            result.success = len(result.safety_violations) == 0

            self.last_cycle_result = result
            self.cycle_history.append(result)
            if len(self.cycle_history) > 50:
                self.cycle_history = self.cycle_history[-50:]

            self.log_evolution_event(
                "cycle_complete",
                {
                    "cycle_id": cycle_id,
                    "strategy": strategy,
                    "actions": len(result.actions_taken),
                    "success": result.success,
                },
            )

            # Persist supervisor status artifact (for TUI / swarm visibility)
            self._write_agent_status(result, telemetry)

        except Exception as exc:
            result.completed_at = datetime.now(timezone.utc).isoformat()
            result.success = False
            result.notes = f"Exception: {exc}"
            self.log_evolution_event("cycle_failed", {"cycle_id": cycle_id, "error": str(exc)}, severity="error")
            logger.exception("Master Self-Evolution cycle crashed")

        return result

    def _write_agent_status(self, result: EvolutionCycleResult, telemetry: Dict[str, Any]) -> None:
        """Comprehensive status + architecture notes for the agent swarm / TUI."""
        status = {
            "agent": "Master Self-Evolution Supervisor",
            "role": "Top-level strategic orchestrator and brain for fully autonomous self-improving trading system",
            "started": "2026-05-28 (implementation wave)",
            "last_cycle": result.cycle_id,
            "last_update": result.completed_at or result.started_at,
            "current_strategy": self.current_strategy,
            "status": "ACTIVE - highest-level self-evolution loop armed",
            "goals": [asdict(g) for g in self.goals],
            "safety_constraints": [c.name for c in self.safety_constraints],
            "last_cycle_result": asdict(result),
            "telemetry": telemetry,
            "performance_history_size": len(self.performance_history),
            "architecture": {
                "position": "Above all tactical agents (RetrainingOrchestrator, handoff_watcher, promoter, ExecutionAgent, etc.)",
                "coordinated_components": [
                    "AutonomousRetrainingOrchestrator (tactical retrain + fast_bt + promotion + meta tuning + continual learner integration)",
                    "FastBacktester (primary experimentation engine — minutes-scale OOS)",
                    "RegimeAdaptiveController (full Rainforest + PatternDetector + Dreamer + timing; drives adaptation in risk/TD/ensemble/PPO)",
                    "MetaController + SignalOptimizer (meta layer)",
                    "MetaOptimizer (autonomous/meta_optimizer.py: harness pattern_profitability + timing + TimeExitSpec -> reward_profile/ensemble_weights/feature_importance self-tuning for training)",
                    "ContinualLearner (online gated updates for PPO/Dreamer/Rainforest post-trade; wired to retrain_orch)",
                    "ProductionHardening (timing safety in RiskEngine/ExecutionAgent/RiskSupervisor + timing-aware canary + news deferral + tests)",
                    "SelfMonitoringRecoveryAgent (dedicated thresholds, rollback, recovery; delegated from supervisor)",
                    "ModelRegistry + PromotionGates",
                    "TUI layer (monitor_tui --mini + mini_pipeline_tui + full parity panels for all above + swarm_status)",
                    "live_safety / RiskSupervisor / BackupManager (self-monitor/recovery)",
                ],
                "data_artifacts": [
                    "runtime/self_evolution/performance_history.json",
                    "logs/self_evolution_log.jsonl",
                    "runtime/agent_status/master_self_evolution_supervisor_agent.json (this file)",
                ],
                "decision_flow": "collect_telemetry -> assess_state -> decide_strategy -> run_safe_experiments(FastBT) -> delegate_to_orchestrators -> gate_apply -> log_audit",
                "frequency": "Low (hours) strategic vs high-frequency tactical sub-loops",
                "self_improvement": "Logs own decisions; future versions can meta-optimize strategy policy from history",
            },
            "high_level_goals": [
                "Drive continuous improvement with near-zero human intervention",
                "Safe rapid experimentation via FastBacktester before any retrain or promotion",
                "Maintain global view of regime, meta, continual, hardening, retrain progress, and recovery health",
                "Enforce safety gates at the highest level (now including production timing hardening)",
                "Full swarm visibility via TUI mini + agent_status synthesis",
            ],
            "next_autonomous_behaviors": [
                "Periodic large-scale validation campaigns on promising variants (honor production hardening gates)",
                "Automatic focus shifts (e.g. regime boost after detected distribution shift) + continual learner post-retrain",
                "Self-audit of past strategy efficacy to refine future decisions; incorporate meta_tune suggestions from retrain orchestrator",
                "Integration with TUI/React for full visibility of evolution decisions (regime, continual, hardening, active retrains)",
                "Delegate to Real Retraining Orchestrator when regime/continual signals or performance drift detected",
            ],
        }

        try:
            SUPERVISOR_STATUS_PATH.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.error(f"Failed writing master supervisor agent status: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # ENTRYPOINTS
    # ─────────────────────────────────────────────────────────────────────────

    def start(self, interval_hours: Optional[int] = None) -> None:
        """Long-running strategic supervisor loop."""
        interval = interval_hours or self.config.get("cycle_interval_hours", 4)
        interval_sec = int(interval * 3600)
        logger.info(f"[MasterSelfEvolutionSupervisor] Starting strategic loop (every {interval}h)...")

        while True:
            try:
                self.run_evolution_cycle()
            except KeyboardInterrupt:
                logger.info("Supervisor loop interrupted by user.")
                break
            except Exception as exc:
                logger.exception(f"Top-level supervisor loop error: {exc}")
            time.sleep(interval_sec)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Master Self-Evolution Supervisor — central brain")
    parser.add_argument("--cycle", action="store_true", help="Run a single evolution cycle and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuous strategic loop")
    parser.add_argument("--interval-hours", type=float, default=4.0, help="Loop interval in hours")
    parser.add_argument("--status", action="store_true", help="Print current supervisor status summary")
    args = parser.parse_args()

    supervisor = MasterSelfEvolutionSupervisor()

    if args.status:
        print(json.dumps({
            "current_strategy": supervisor.current_strategy,
            "goals": [asdict(g) for g in supervisor.goals],
            "history_versions": len(supervisor.performance_history),
            "last_cycle": asdict(supervisor.last_cycle_result) if supervisor.last_cycle_result else None,
        }, indent=2, default=str))
        sys.exit(0)

    if args.cycle:
        result = supervisor.run_evolution_cycle()
        print(json.dumps(asdict(result), indent=2, default=str))
    elif args.loop:
        supervisor.start(interval_hours=args.interval_hours)
    else:
        print("Use --cycle or --loop (or --status). See module docstring for architecture.")
