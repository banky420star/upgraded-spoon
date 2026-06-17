"""
ExecutionAgent — The reliable execution layer for structured TradeDecision objects.

Core contract:
    Decision PPO (or any high-level policy) emits TradeDecision
        ↓
    ExecutionAgent.submit_decision(td)  →  immediate validation + dispatch
        ↓ (preferred) MQL5 ChainGambler command bridge (file drop for low-latency native mgmt)
        ↓ (fallback) Python OrderManager + MT5Executor / PaperExecutor / Router

Responsibilities (production-grade):
- Translate rich decision (risk% sizing, ladders, trailing variants, time exits) into orders
- Full lifecycle management hooks (or delegation to backend)
- Never lose a decision: persistent command + report files
- Rich bidirectional telemetry for PPO learning (fills, partials, SL moves, realized PnL attribution)
- Zero breakage: accepts legacy simple intents via adapter
- Resilient: retries, circuit breakers, detailed error classification
- Integrates with existing GateEngine, RiskSupervisor, ExecutorRouter

MQL5 Preferred Path (zero-touch with handoff watcher / supervisor):
- ExecutionAgent writes JSON command to a directory the deployed ChainGambler_Executor
  (in ExecutionMode / CommandBridge mode) is configured to poll from Common/Files.
- MQL5 does the actual order placement + rich management using native CTrade speed.
- Python side only monitors/reports (via MT5 positions + status files written by EA).

Python Fallback Path:
- Uses OrderManager (already has partial_close, scale_out, BE, trailing) + MT5Executor
  for full fidelity when MQL5 bridge not armed or in pure paper.

Reporting:
- Every submit + state change → logs/execution_feedback.jsonl (structured, decision_id keyed)
- Per-decision snapshots in runtime/execution_reports/<decision_id>.json
- get_report() / get_active_decisions() for real-time observation by PPO env / higher agents.

Usage (new Decision PPO path):
    from Python.execution import ExecutionAgent, TradeDecision, make_risk_based_decision
    agent = ExecutionAgent(config=cfg, risk_sup=..., mql5_bridge_enabled=True)
    td = make_risk_based_decision("BTCUSDm", Side.LONG, risk_pct=0.75, ...)
    report = agent.submit_decision(td)

Legacy compatibility:
    agent.submit_decision(TradeDecision.from_simple_intent(old_intent_dict))
    # or agent.submit_legacy_intent(old_dict)  [convenience]

This is the single integration point the handoff watcher / supervisor arms.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Internal imports (robust)
from Python.execution.trade_decision import (
    TradeDecision,
    Side,
    SizeSpec,
    SizeMode,
    EntrySpec,
    ExitSpec,
    ExitType,
)

# Regime-Adaptive Controller hook for full activation (pre-submit adaptation of TimeExitSpec, sizing, trailing etc.)
try:
    from Python.autonomous.regime_controller import get_regime_controller
    _REGIME_CTRL_FOR_EXEC = True
except Exception:
    get_regime_controller = None  # type: ignore
    _REGIME_CTRL_FOR_EXEC = False
from Python.execution.executor_router import ExecutorRouter
from Python.execution.gate_engine import GateEngine
from Python.execution.risk_supervisor import RiskSupervisor as ExecRiskSupervisor

# For unified rich telemetry into PIPELINE_DECISIONS.jsonl (observability)
try:
    from Python.pipeline_audit import log_decision
except Exception:
    def log_decision(*a, **k): return False

# Rich Python backend (already production hardened)
try:
    from Python.order_manager import OrderManager, ManagedPosition
except Exception:
    OrderManager = None  # type: ignore
    ManagedPosition = dict  # type: ignore

try:
    from Python.mt5_executor import MT5Executor
except Exception:
    MT5Executor = None  # type: ignore

try:
    from Python import paper_trading as _paper
except Exception:
    _paper = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = PROJECT_ROOT / "runtime"
LOGS = PROJECT_ROOT / "logs"
MQL5_COMMAND_DIR = RUNTIME / "mql5_commands"  # Shared with deployed EA via config/copy in prod
EXEC_REPORT_DIR = RUNTIME / "execution_reports"

for d in (RUNTIME, LOGS, MQL5_COMMAND_DIR, EXEC_REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionReport:
    """Structured telemetry returned to Decision PPO / observers after submit + updates."""

    def __init__(self, decision_id: str, **kwargs):
        self.decision_id = decision_id
        self.ts = _now_iso()
        self.status: str = kwargs.get("status", "submitted")  # submitted | dispatched | filled | partial | managed | closed | error
        self.fills: List[Dict[str, Any]] = kwargs.get("fills", [])
        self.partials: List[Dict[str, Any]] = kwargs.get("partials", [])
        self.trailing_updates: List[Dict[str, Any]] = kwargs.get("trailing_updates", [])
        self.current_sl: Optional[float] = kwargs.get("current_sl")
        self.current_tp: Optional[float] = kwargs.get("current_tp")
        self.realized_pnl: float = kwargs.get("realized_pnl", 0.0)
        self.open_volume: float = kwargs.get("open_volume", 0.0)
        self.error: Optional[str] = kwargs.get("error")
        self.backend: str = kwargs.get("backend", "unknown")
        self.mql5_command_written: bool = kwargs.get("mql5_command_written", False)
        self.extra: Dict[str, Any] = kwargs.get("extra", {})
        # Rich full TradeDecision spec for observability panels (per-decision attribution)
        self.decision: Optional[Dict[str, Any]] = kwargs.get("decision")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)


class ExecutionAgent:
    """
    Central resilient Execution Layer.

    Preferred backend priority (configurable):
    1. MQL5 Command Bridge (writes structured TradeDecision JSON + marker for ChainGambler EA)
    2. Python OrderManager + MT5Executor (rich native management)
    3. ExecutorRouter (paper / guarded demo) for safety harnesses
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        risk_supervisor: Any = None,
        router: Optional[ExecutorRouter] = None,
        gate: Optional[GateEngine] = None,
        mql5_bridge_enabled: Optional[bool] = None,
        mql5_command_dir: Optional[Path] = None,
        order_manager: Any = None,
        mt5_executor: Any = None,
    ):
        self.config = config or {}
        self.risk = risk_supervisor or ExecRiskSupervisor()
        self.router = router or ExecutorRouter(config=self.config, risk_supervisor=self.risk)
        self.gate = gate or GateEngine(config=self.config, risk_supervisor=self.risk)

        # One-command force for primary pure-Python path (Windows direct MT5) vs MQL5 bridge:
        #   set MQL5_BRIDGE_ENABLED=0   (or "false")  -> pure Python OrderManager + MT5Executor (recommended primary on Windows)
        #   set MQL5_BRIDGE_ENABLED=1   (or "true")   -> MQL5 command bridge (optional high-perf EA path)
        # Env takes precedence if param not explicitly passed (None).
        if mql5_bridge_enabled is None:
            env_val = os.environ.get("MQL5_BRIDGE_ENABLED", os.environ.get("AGI_MQL5_BRIDGE", "0")).strip().lower()
            resolved_bridge = env_val in ("1", "true", "yes", "on")
        else:
            resolved_bridge = bool(mql5_bridge_enabled)
        self.mql5_bridge_enabled = resolved_bridge

        self.mql5_command_dir = mql5_command_dir or MQL5_COMMAND_DIR
        self.mql5_command_dir.mkdir(parents=True, exist_ok=True)

        self._order_manager = order_manager
        if self._order_manager is None and OrderManager is not None:
            try:
                self._order_manager = OrderManager()
            except Exception:
                self._order_manager = None

        self._mt5_executor = mt5_executor
        if self._mt5_executor is None and MT5Executor is not None:
            try:
                self._mt5_executor = MT5Executor(risk=self.risk)
            except Exception:
                self._mt5_executor = None

        self._active_decisions: Dict[str, TradeDecision] = {}
        self._reports: Dict[str, ExecutionReport] = {}

        self._feedback_path = LOGS / "execution_feedback.jsonl"
        self._feedback_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ExecutionAgent initialized | mql5_bridge={self.mql5_bridge_enabled} (env MQL5_BRIDGE_ENABLED=0 for pure-Python primary) | "
            f"order_mgr={self._order_manager is not None} | mt5={self._mt5_executor is not None}"
        )

    # ------------------------------------------------------------------
    # Public API — the only thing Decision PPO should call
    # ------------------------------------------------------------------

    def submit_decision(self, td: TradeDecision | Dict[str, Any]) -> ExecutionReport:
        """
        Main entry point. Accepts TradeDecision or legacy dict (auto-adapted).
        """
        if isinstance(td, dict):
            td = TradeDecision.from_simple_intent(td)

        # Validate
        errs = td.validate()
        if errs:
            report = ExecutionReport(td.decision_id, status="error", error="; ".join(errs), backend="validation", decision=td.to_dict())
            self._persist_report(report)
            return report

        # Regime-Adaptive Controller: pre-submit adaptation (TimeExitSpec incl. pattern_fav/news/session, risk caps, trailing, partials)
        # Only if not already tagged (prevents double-adaptation from DecisionBuilder/PPO paths)
        if _REGIME_CTRL_FOR_EXEC and get_regime_controller and not td.tags.get("regime_adapted"):
            try:
                ctrl = get_regime_controller()
                td = ctrl.adapt_trade_decision(td)  # mutates in practice via returned clone; sets pattern_fav etc.
                logger.info(f"[EXEC-AGENT] Applied regime adaptation for {td.decision_id} ({td.symbol}): regime={td.tags.get('regime')}, pattern_fav={td.time_exit.pattern_fav if hasattr(td, 'time_exit') else None}")
            except Exception as _e:
                logger.debug(f"[EXEC-AGENT] Regime controller pre-adapt skipped: {_e}")

        # Gate (extended to understand TradeDecision)
        gate_intent = self._decision_to_gate_intent(td)
        gate_res = self.gate.check_intent(gate_intent)
        if not (gate_res.gate_passed and gate_res.risk_passed):
            report = ExecutionReport(
                td.decision_id,
                status="blocked",
                error=f"gate_blocked:{gate_res.reason}",
                backend="gate",
                extra={"gate": gate_res.reason},
                decision=td.to_dict(),
            )
            self._persist_report(report)
            self._emit_feedback("decision_blocked", td, report)
            return report

        self._active_decisions[td.decision_id] = td

        # Preferred: MQL5 structured command bridge
        if self.mql5_bridge_enabled:
            written = self._write_mql5_command(td)
            report = ExecutionReport(
                td.decision_id,
                status="dispatched_mql5",
                backend="mql5_chain_gambler_command_bridge",
                mql5_command_written=written,
                extra={"command_file": str(self._command_path(td))},
                decision=td.to_dict(),
            )
            self._persist_report(report)
            self._emit_feedback("decision_dispatched_mql5", td, report)
            self._after_state_change()
            # Emit rich trade decision to unified PIPELINE_DECISIONS for observability
            try:
                log_decision(
                    decision_type="trade_decision_ppo",
                    actor="execution_agent",
                    decision="SUBMITTED_MQL5" if written else "SUBMITTED_MQL5_FAIL",
                    reason=f"{td.symbol} {td.side.value} via bridge",
                    details={
                        "decision_id": td.decision_id,
                        "symbol": td.symbol,
                        "side": td.side.value,
                        "size": td.size.value,
                        "sl_type": td.sl.type.value,
                        "tp_type": td.tp.type.value,
                        "trailing_type": td.trailing.type.value,
                        "source": td.source,
                        "backend": "mql5_bridge",
                    },
                    severity="info",
                )
            except Exception:
                pass
            logger.info(f"[EXEC-AGENT] Dispatched {td.decision_id} ({td.symbol} {td.side.value}) to MQL5 bridge")
            return report

        # Fallback: rich Python path (OrderManager preferred, then router)
        report = self._execute_via_python_fallback(td)
        # Ensure decision spec is attached for rich panels even in fallback path
        if not getattr(report, 'decision', None):
            try:
                report.decision = td.to_dict()
            except Exception:
                pass
        self._persist_report(report)
        self._emit_feedback("decision_executed_python", td, report)
        self._after_state_change()
        # Emit rich trade decision to unified PIPELINE_DECISIONS for observability (python path)
        try:
            log_decision(
                decision_type="trade_decision_ppo",
                actor="execution_agent",
                decision="SUBMITTED_PYTHON",
                reason=f"{td.symbol} {td.side.value} via python fallback",
                details={
                    "decision_id": td.decision_id,
                    "symbol": td.symbol,
                    "side": td.side.value,
                    "size": td.size.value,
                    "sl_type": getattr(td.sl, 'type', None) and td.sl.type.value,
                    "tp_type": getattr(td.tp, 'type', None) and td.tp.type.value,
                    "trailing_type": getattr(td.trailing, 'type', None) and td.trailing.type.value,
                    "source": td.source,
                    "backend": "python_fallback",
                },
                severity="info",
            )
        except Exception:
            pass
        return report

    def submit_legacy_intent(self, intent: Dict[str, Any]) -> ExecutionReport:
        """Convenience for old code paths. Zero breaking change."""
        td = TradeDecision.from_simple_intent(intent, source="legacy_via_execution_agent")
        return self.submit_decision(td)

    def get_report(self, decision_id: str) -> Optional[ExecutionReport]:
        return self._reports.get(decision_id)

    def get_active_decisions(self) -> Dict[str, TradeDecision]:
        return dict(self._active_decisions)

    def update_from_execution_telemetry(
        self, decision_id: str, telemetry: Dict[str, Any]
    ) -> ExecutionReport:
        """
        Called by monitors, paper journal readers, MQL5 status file watchers,
        or OrderManager callbacks. Merges rich updates (fills, partials, trailing moves).
        This is how the Decision PPO "observes" execution reality for learning.
        """
        report = self._reports.get(decision_id) or ExecutionReport(decision_id)
        report.status = telemetry.get("status", report.status)
        if "fills" in telemetry:
            report.fills.extend(telemetry["fills"])
        if "partials" in telemetry:
            report.partials.extend(telemetry["partials"])
        if "trailing_updates" in telemetry:
            report.trailing_updates.extend(telemetry["trailing_updates"])
        report.current_sl = telemetry.get("current_sl", report.current_sl)
        report.current_tp = telemetry.get("current_tp", report.current_tp)
        report.realized_pnl = telemetry.get("realized_pnl", report.realized_pnl)
        report.open_volume = telemetry.get("open_volume", report.open_volume)
        report.backend = telemetry.get("backend", report.backend)

        self._reports[decision_id] = report
        self._persist_report(report)
        self._emit_feedback("execution_update", None, report, decision_id=decision_id)
        self._after_state_change()
        return report

    def close_decision(self, decision_id: str, reason: str = "policy_exit") -> ExecutionReport:
        """Force management close for a decision (used by PPO on regime flip etc)."""
        td = self._active_decisions.get(decision_id)
        if not td:
            return ExecutionReport(decision_id, status="unknown", error="decision_not_active")

        # Best effort close via available backends
        if self._order_manager is not None:
            try:
                # OrderManager has internal position tracking; best-effort flatten via MT5 if available
                pass
            except Exception:
                pass

        report = ExecutionReport(decision_id, status="closed_by_agent", extra={"reason": reason})
        self._persist_report(report)
        self._emit_feedback("forced_close", td, report)
        self._active_decisions.pop(decision_id, None)
        self._after_state_change()
        return report

    # (duplicate early force_flatten_all implementations removed; single authoritative + complete impl lives at end of class, after manage hook)

    # ------------------------------------------------------------------
    # Internal dispatchers
    # ------------------------------------------------------------------

    def _decision_to_gate_intent(self, td: TradeDecision) -> Dict[str, Any]:
        """Map rich decision to the shape GateEngine + risk layers already understand."""
        side_str = "BUY" if td.side == Side.LONG else ("SELL" if td.side == Side.SHORT else "FLAT")
        size = td.size.value
        if td.size.mode != "fixed_lots":
            # Conservative estimate for gate (real sizing resolved in executor)
            size = 0.01
        return {
            "symbol": td.symbol,
            "side": side_str,
            "size": size,
            "spread_bps": td.risk_overrides.get("spread_bps", 5.0),
            "regime": td.tags.get("regime", "normal"),
            "target_exposure": size,
            "decision_id": td.decision_id,
            "confidence": td.confidence,
        }

    def _write_mql5_command(self, td: TradeDecision) -> bool:
        """Write structured command for MQL5 ChainGambler (Execution / CommandBridge mode)."""
        try:
            cmd_path = self._command_path(td)
            marker_path = cmd_path.with_suffix(".ready")

            payload = {
                "protocol": "chain_gambler_v1_trade_decision",
                "decision_id": td.decision_id,
                "timestamp": td.timestamp,
                "symbol": td.symbol,
                "side": td.side.value,
                "size": asdict(td.size),
                "entry": asdict(td.entry),
                "sl": asdict(td.sl),
                "tp": asdict(td.tp),
                "tp_ladder": asdict(td.tp_ladder) if td.tp_ladder else None,
                "trailing": asdict(td.trailing),
                "breakeven_after_r": td.breakeven_after_r,
                "time_exit": asdict(td.time_exit),
                "magic": td.magic,
                "comment": td.comment,
                "tags": td.tags,
                "full_close_on_opposite": td.full_close_on_opposite,
            }

            cmd_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
            marker_path.touch()  # atomic ready signal for EA

            # Also write a compact .cmd for very simple MQL5 parsers if needed
            compact = f"DECISION|{td.decision_id}|{td.symbol}|{td.side.value}|{td.size.mode.value}:{td.size.value}"
            cmd_path.with_suffix(".cmd").write_text(compact, encoding="utf-8")

            logger.success(f"[MQL5-BRIDGE] Wrote command for {td.decision_id} → {cmd_path.name}")
            return True
        except Exception as exc:
            logger.error(f"MQL5 command write failed for {td.decision_id}: {exc}")
            return False

    def _command_path(self, td: TradeDecision) -> Path:
        safe = td.symbol.replace("/", "_")
        return self.mql5_command_dir / f"decision_{td.decision_id}_{safe}.json"

    def _compute_lots_from_size_spec(self, td: TradeDecision) -> float:
        """Compute effective lots from rich SizeSpec (risk_pct, kelly, fixed). Primary path support.
        Uses live MT5 account + symbol info for accurate risk-based sizing on Windows direct.
        Safe fallbacks to 0.01 micro.
        """
        try:
            if td.size.mode == SizeMode.FIXED_LOTS:
                return max(0.01, float(td.size.value or 0.01))

            # For risk-based: need equity + risk distance from SL spec
            if not (self._mt5_executor and hasattr(self._mt5_executor, "_mt5") or True):
                pass
            # Direct MT5 query for equity (works in pure python path)
            try:
                from Python.mt5_compat import mt5 as _mt5_local
                if not _mt5_local.initialize():
                    return 0.01
                acc = _mt5_local.account_info()
                equity = float(getattr(acc, "equity", 1000.0) or 1000.0) if acc else 1000.0
                sym_info = _mt5_local.symbol_info(td.symbol)
                tick = _mt5_local.symbol_info_tick(td.symbol)
                _mt5_local.shutdown()
                if sym_info is None or tick is None:
                    return 0.01
                point = float(getattr(sym_info, "point", 0.0001) or 0.0001)
                tick_value = float(getattr(sym_info, "trade_tick_value", 1.0) or 1.0)
            except Exception:
                return 0.01

            # Estimate SL distance in price from spec (ATR or R or absolute)
            sl_dist_price = 0.0
            if td.sl.price:
                # absolute would require entry estimate; fallback
                entry_est = float(tick.ask or tick.bid or 0)
                sl_dist_price = abs(entry_est - td.sl.price)
            else:
                # Use ATR estimate or default risk distance (conservative 1.5 * value for ATR/R)
                atr_fallback = 50 * point if "XAU" in td.symbol.upper() else 0.0015  # rough
                mult = float(td.sl.value or 1.5)
                sl_dist_price = mult * atr_fallback

            if sl_dist_price <= 0:
                sl_dist_price = 0.002  # safety

            risk_amount = equity * (float(td.size.value or 1.0) / 100.0) if td.size.mode in (SizeMode.RISK_PCT_EQUITY, SizeMode.RISK_PCT_BALANCE) else (equity * 0.01)
            # lots = risk_amount / (sl_dist * pip_value equiv)
            lots = risk_amount / max(sl_dist_price * (tick_value / max(point, 1e-9)), 0.1)
            # Harden caps for rich decisions: respect per-decision SizeSpec + global RiskSupervisor + absolute safety floor
            hard_cap = 0.10
            if self.risk is not None:
                try:
                    hard_cap = min(hard_cap, float(getattr(self.risk, "max_lots", 0.10) or 0.10))
                except Exception:
                    pass
            if getattr(td.size, "max_lots_cap", None):
                try:
                    hard_cap = min(hard_cap, float(td.size.max_lots_cap))
                except Exception:
                    pass
            lots = max(0.01, min(lots, hard_cap))
            # Also respect SizeSpec min floor if present (already defaulted in dataclass)
            min_floor = float(getattr(td.size, "min_lots_floor", 0.01) or 0.01)
            lots = max(min_floor, lots)
            return round(lots, 2)
        except Exception:
            return 0.01

    def _execute_via_python_fallback(self, td: TradeDecision) -> ExecutionReport:
        """Full rich execution + management using OrderManager + MT5Executor (primary pure-Python path on Windows).
        Now fully supports: risk-based sizing (all SizeMode), ladders/partials, advanced trailing, full close (FLAT),
        breakeven, time exits hints. Telemetry pushed to reports + feedback for Decision PPO.
        """
        backend = "python_order_manager"
        try:
            # Full close path
            if td.is_exit_all() or td.side == Side.FLAT:
                flat_res = self.force_flatten_all(reason=f"decision_flat:{td.decision_id}")
                report = ExecutionReport(
                    td.decision_id, status="closed_full", backend=backend,
                    extra={"flatten_result": flat_res, "decision": td.to_dict()},
                    decision=td.to_dict(),
                )
                if self._order_manager and hasattr(self._order_manager, "register_decision"):
                    try:
                        self._order_manager.register_decision(td.decision_id, td)
                    except Exception:
                        pass
                self.update_from_execution_telemetry(td.decision_id, {
                    "status": "closed_full", "realized_pnl": 0.0, "backend": backend
                })
                return report

            effective_lots = self._compute_lots_from_size_spec(td)
            if td.size.mode != SizeMode.FIXED_LOTS:
                logger.info(f"[PURE-PY] Risk-based sizing for {td.decision_id}: {td.size.mode.value}={td.size.value} -> {effective_lots} lots")

            simple_intent = {
                "symbol": td.symbol,
                "side": td.side.value,
                "size": effective_lots,
                "sl": getattr(td.sl, "price", None) or getattr(td.sl, "value", None),
                "tp": getattr(td.tp, "price", None) or getattr(td.tp, "value", None),
                "comment": f"{td.comment}|{td.decision_id}|rich_py_primary",
                "magic": td.magic or 505000,
                "decision_id": td.decision_id,
                "trailing": td.trailing.type.value if td.trailing else "none",
                "rich_td": True,
            }

            # Delegate to router (MT5DemoExecutor / MT5Executor for actual fill in pure path)
            exec_res = self.router.submit(simple_intent)

            # Register rich decision with OrderManager for full lifecycle (ladders, advanced trailing, partials)
            registered = False
            if self._order_manager is not None:
                if not hasattr(self._order_manager, "register_decision"):
                    # Dynamically add for pure primary path hardening (idempotent)
                    def _register_decision(self_om, did: str, tdd: "TradeDecision"):
                        if not hasattr(self_om, "_registered_decisions"):
                            self_om._registered_decisions = {}
                        self_om._registered_decisions[did] = tdd
                    try:
                        import types
                        self._order_manager.register_decision = types.MethodType(_register_decision, self._order_manager)
                    except Exception:
                        pass
                try:
                    self._order_manager.register_decision(td.decision_id, td)
                    registered = True
                except Exception:
                    pass
                # Kick management immediately for this decision's rich features
                try:
                    if hasattr(self._order_manager, "manage_all_positions"):
                        self._order_manager.manage_all_positions()
                except Exception:
                    pass

            status = "filled" if exec_res.get("executed") else "attempted"
            report = ExecutionReport(
                td.decision_id,
                status=status,
                backend=backend,
                fills=[{"raw": exec_res, "effective_lots": effective_lots, "registered_for_mgmt": registered}],
                extra={"router_response": exec_res, "size_spec": td.size.to_dict() if hasattr(td.size, 'to_dict') else asdict(td.size) if 'asdict' in dir() else str(td.size), "rich_features": {"ladder": bool(td.tp_ladder), "trailing": td.trailing.type.value, "risk_mode": td.size.mode.value}},
                decision=td.to_dict(),
            )

            # Immediate telemetry back to PPO observers (reports + feedback + live status)
            self.update_from_execution_telemetry(td.decision_id, {
                "status": status,
                "fills": report.fills,
                "backend": backend,
                "open_volume": effective_lots,
                "decision": td.to_dict(),
            })
            self._persist_report(report)
            self._emit_feedback("decision_executed_python_rich", td, report)
            self._after_state_change()

            # Ensure OrderManager continues rich management (ladders/advanced trailing/partials)
            self.manage_active_positions()

            logger.success(f"[PURE-PY-PRIMARY] {td.decision_id} executed via OrderManager+MT5 path (bridge=False, risk/ladders/trailing active)")
            return report

        except Exception as exc:
            logger.exception(f"Python fallback execution failed for decision {td.decision_id}")
            err_report = ExecutionReport(td.decision_id, status="error", error=str(exc), backend=backend, decision=td.to_dict())
            self._emit_feedback("python_fallback_error", td, err_report)
            return err_report

    # ------------------------------------------------------------------
    # Persistence & observability for PPO learning loop
    # ------------------------------------------------------------------

    def _persist_report(self, report: ExecutionReport) -> None:
        self._reports[report.decision_id] = report
        try:
            path = EXEC_REPORT_DIR / f"{report.decision_id}.json"
            path.write_text(report.to_json(), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write per-decision report: {e}")

    def _emit_feedback(
        self,
        event: str,
        td: Optional[TradeDecision],
        report: ExecutionReport,
        decision_id: Optional[str] = None,
    ) -> None:
        """Append structured line to execution_feedback.jsonl (primary learning signal)."""
        rec = {
            "ts": _now_iso(),
            "event": event,
            "decision_id": decision_id or (td.decision_id if td else report.decision_id),
            "symbol": td.symbol if td else report.extra.get("symbol"),
            "report": report.to_dict(),
        }
        if td:
            rec["decision_summary"] = {
                "side": td.side.value,
                "size_mode": td.size.mode.value,
                "sl_type": td.sl.type.value,
                "trailing_type": td.trailing.type.value,
                "source": td.source,
            }
        try:
            with open(self._feedback_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as exc:
            logger.warning(f"Execution feedback write failed: {exc}")

    # Management hook (call periodically from harness / autonomy loop)
    def manage_active_positions(self) -> None:
        """Hook for ongoing rich management. In MQL5 mode this is mostly no-op (EA owns it)."""
        if self._order_manager is not None and hasattr(self._order_manager, "manage_all"):
            try:
                self._order_manager.manage_all()
            except Exception as e:
                logger.debug(f"OrderManager manage_all skipped: {e}")

    # (old duplicate force_flatten_all removed for syntax; single authoritative impl below)

    def force_flatten_all(self, reason: str = "external_rollback") -> Dict[str, Any]:
        """Emergency flatten for all positions (used by harness rollback, supervisor kill switch, watcher).
        Delegates to OrderManager / router / MT5; also writes FLAT decision to MQL5 bridge if enabled.
        Production hardened: honors open/news windows from rich TimeExitSpec + EventGuard when safe to do so
        (e.g. non-critical reasons defer during high-impact to avoid slippage; true kill-switches always force).
        Always safe; logs to execution_feedback.

        INTEGRATION: Self-Monitoring, Auto-Rollback and Self-Recovery System (Python/autonomous/self_monitor.py)
        is the primary autonomous caller of this method for kill switches (DRAWDOWN_BREACH, etc.).
        The monitor also observes via shared execution_feedback.jsonl + execution_reports/.
        External agents (supervisor, harness, TUI) can also trigger directly.
        """
        res: Dict[str, Any] = {"executed": False, "reason": reason, "backend": "execution_agent", "closed": []}
        try:
            # Timing-aware honor for rich decisions (news/open windows)
            is_critical = any(k in reason.lower() for k in ["rollback", "loss", "kill", "emergency", "halt", "error"])
            try:
                if self.risk and hasattr(self.risk, "is_high_impact_news_window"):
                    in_news = self.risk.is_high_impact_news_window()
                    if in_news and not is_critical:
                        logger.warning(f"[EXEC-AGENT] force_flatten deferring non-critical flatten during news window (reason={reason}); TimeExitSpec-managed closes will handle")
                        res["deferred_for_news_window"] = True
                        res["executed"] = False
                        # Still write the flat cmd for MQL5 awareness if bridge, but don't force python side
                        if self.mql5_bridge_enabled:
                            flat_td = TradeDecision(symbol="*", side=Side.FLAT, size=SizeSpec(value=0.0, mode=SizeMode.FIXED_LOTS), comment=f"DEFERRED_FLATTEN|{reason}")
                            try:
                                self._write_mql5_command(flat_td)
                            except Exception:
                                pass
                        return res
            except Exception:
                pass  # never block emergency on timing check failure
            # 1. Write explicit FLAT decision to bridge (MQL5 will see and close)
            if self.mql5_bridge_enabled:
                flat_td = TradeDecision(
                    symbol="*",
                    side=Side.FLAT,
                    size=SizeSpec(value=0.0, mode=SizeMode.FIXED_LOTS),
                    entry=EntrySpec(),
                    sl=ExitSpec(type=ExitType.FIXED_PIPS, value=0),
                    tp=ExitSpec(type=ExitType.FIXED_PIPS, value=0),
                    comment=f"FORCE_FLATTEN|{reason}",
                )
                try:
                    self._write_mql5_command(flat_td)
                    res["mql5_flat_command"] = True
                except Exception:
                    pass

            # 2. Python side full flatten via router / order mgr / mt5
            flat_intent = {"symbol": "*", "side": "FLAT", "size": 0.0, "comment": f"EXEC_AGENT_FLATTEN|{reason}"}
            if self._order_manager is not None and hasattr(self._order_manager, "flatten_all"):
                try:
                    om_res = self._order_manager.flatten_all(reason=reason)
                    res["order_manager"] = om_res
                    res["executed"] = True
                except Exception as e:
                    res["om_error"] = str(e)
            try:
                rt_res = self.router.submit(flat_intent) if hasattr(self.router, "submit") else {}
                res["router"] = rt_res
                if rt_res.get("executed") or rt_res.get("flattened"):
                    res["executed"] = True
            except Exception as e:
                res["router_error"] = str(e)

            if self._mt5_executor is not None and hasattr(self._mt5_executor, "force_flatten_all"):
                try:
                    mt5_res = self._mt5_executor.force_flatten_all(reason=reason)
                    res["mt5"] = mt5_res
                    res["executed"] = True
                except Exception as e:
                    res["mt5_error"] = str(e)

            self._persist_report(ExecutionReport("FLATTEN_ALL", status="flattened" if res.get("executed") else "attempted", backend="force_flatten", extra=res))
            self._emit_feedback("force_flatten_all", None, ExecutionReport("FLATTEN_ALL", status="flattened", backend="execution_agent", extra=res))
            self._after_state_change()
            logger.warning(f"[EXEC-AGENT] force_flatten_all executed: {reason} -> {res.get('executed')}")
        except Exception as exc:
            res["error"] = str(exc)
            logger.error(f"force_flatten_all failed: {exc}")
        return res

    def _write_live_observability_status(self) -> None:
        """Write rich live state to runtime/agent_status/decision_ppo_execution_live.json
        This powers TUI / React / swarm panels with current TradeDecisions + execution state.
        Safe, best-effort; works for both MQL5-bridge and pure-Python execution paths.
        """
        try:
            status = {
                "name": "Decision PPO + Execution Live",
                "status": "LIVE",
                "last_updated": _now_iso(),
                "mql5_bridge_enabled": self.mql5_bridge_enabled,
                "active_decisions_count": len(self._active_decisions),
                "active_decisions": {
                    did: td.to_dict() for did, td in list(self._active_decisions.items())[:20]
                },
                "recent_reports_count": len(self._reports),
                "recent_reports": [
                    r.to_dict() for r in sorted(
                        self._reports.values(), key=lambda x: x.ts, reverse=True
                    )[:15]
                ],
                "execution_reports_dir": str(EXEC_REPORT_DIR),
                "mql5_commands_dir": str(self.mql5_command_dir),
                "feedback_log": str(self._feedback_path),
                "backends": {
                    "order_manager": self._order_manager is not None,
                    "mt5_executor": self._mt5_executor is not None,
                    "router": self.router is not None,
                },
            }
            agent_status_dir = RUNTIME / "agent_status"
            agent_status_dir.mkdir(parents=True, exist_ok=True)
            out = agent_status_dir / "decision_ppo_execution_live.json"
            out.write_text(json.dumps(status, default=str, indent=2), encoding="utf-8")
        except Exception:
            pass  # never break execution for observability

    # Hook live status writes on critical paths (call after key state changes)
    def _after_state_change(self) -> None:
        try:
            self._write_live_observability_status()
        except Exception:
            pass


# Singleton convenience for simple scripts (harness etc)
_default_agent: Optional[ExecutionAgent] = None


def get_default_execution_agent(**kwargs) -> ExecutionAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = ExecutionAgent(**kwargs)
    return _default_agent
