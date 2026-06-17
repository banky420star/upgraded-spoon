"""Chain Gambler execution layer — safety boot and trade routing.

Now includes the clean Decision / Execution separation:
- TradeDecision (structured rich output from Decision PPO)
- ExecutionAgent (consumes decisions, drives MQL5 preferred or Python fallback)
"""

from Python.execution.mode_resolver import resolve_mode
from Python.execution.account_verifier import verify_account
from Python.execution.live_gate import live_trading_allowed, demo_trading_allowed
from Python.execution.gate_engine import GateEngine
from Python.execution.risk_supervisor import RiskSupervisor
from Python.execution.executor_router import ExecutorRouter
from Python.execution.paper_executor import PaperExecutor
from Python.execution.mt5_demo_executor import MT5DemoExecutor
from Python.execution.execution_agent import ExecutionAgent, ExecutionReport
from Python.execution.trade_decision import (
    TradeDecision,
    Side,
    SizeSpec,
    ExitSpec,
    TrailingSpec,
    PartialCloseLadder,
    TPLadderLevel,
    TimeExitSpec,
    EntrySpec,
    make_risk_based_decision,
    TRADE_DECISION_JSON_SCHEMA,
    from_ppo_action_meta,
)
from Python.execution.execution_agent import ExecutionAgent, ExecutionReport, get_default_execution_agent

__all__ = [
    "resolve_mode",
    "verify_account",
    "live_trading_allowed",
    "demo_trading_allowed",
    "GateEngine",
    "RiskSupervisor",
    "ExecutorRouter",
    "PaperExecutor",
    "MT5DemoExecutor",
    "ExecutionAgent",
    "ExecutionReport",
    # New Decision/Execution architecture (additive, backward compatible)
    "TradeDecision",
    "Side",
    "SizeSpec",
    "ExitSpec",
    "TrailingSpec",
    "PartialCloseLadder",
    "TPLadderLevel",
    "TimeExitSpec",
    "EntrySpec",
    "make_risk_based_decision",
    "TRADE_DECISION_JSON_SCHEMA",
    "from_ppo_action_meta",
    "ExecutionAgent",
    "ExecutionReport",
    "get_default_execution_agent",
]
