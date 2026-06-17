"""Trade-intent gate engine.

Receives a trade intent, runs pre-flight checks (spread, regime, telemetry,
test health), and returns a structured gate result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Python.execution.mode_resolver import resolve_mode
from Python.execution.account_verifier import verify_account
from Python.execution.live_gate import live_trading_allowed, demo_trading_allowed
from Python.execution.risk_supervisor import RiskSupervisor


@dataclass
class GateResult:
    gate_passed: bool
    risk_passed: bool
    execution_mode: str
    reason: str


class GateEngine:
    """Central gate for every trade intent.

    Usage:
        engine = GateEngine(config, risk_supervisor)
        result = engine.check_intent(intent)
    """

    def __init__(
        self,
        config: dict | None = None,
        risk_supervisor: RiskSupervisor | None = None,
    ):
        self.config = config or {}
        self.risk = risk_supervisor
        self._mode = resolve_mode(self.config)

    def check_intent(
        self,
        intent: dict[str, Any],
        account_state: dict[str, Any] | None = None,
        validation_state: dict[str, Any] | None = None,
        test_state: dict[str, Any] | None = None,
    ) -> GateResult:
        """Evaluate a single trade intent against all gates.

        Intent fields expected:
          symbol, side, size, spread_bps, regime, target_exposure
        """
        symbol = intent.get("symbol", "")
        spread_bps = float(intent.get("spread_bps", 0.0) or 0.0)
        regime = str(intent.get("regime", "")).lower()
        target_exposure = float(intent.get("target_exposure", 0.0) or 0.0)

        # 1. Spread gate
        max_spread_bps = float(
            self.config.get("risk", {}).get("max_spread_bps", 50.0)
        )
        if spread_bps > max_spread_bps:
            return GateResult(
                gate_passed=False,
                risk_passed=False,
                execution_mode=self._mode,
                reason=f"spread_too_high ({spread_bps:.1f} > {max_spread_bps})",
            )

        # 2. Regime gate — block chaos / spread danger
        blocked_regimes = {"chaos_spike", "spread_danger", "black_swan", "halt"}
        if regime in blocked_regimes:
            return GateResult(
                gate_passed=False,
                risk_passed=False,
                execution_mode=self._mode,
                reason=f"regime_blocked ({regime})",
            )

        # 3. Account telemetry gate
        account = account_state or {}
        if not account.get("telemetry_valid", True):
            return GateResult(
                gate_passed=False,
                risk_passed=False,
                execution_mode=self._mode,
                reason="account_telemetry_invalid",
            )

        # 4. Test-state gate
        tests = test_state or {}
        if tests.get("tests_clean") is False:
            return GateResult(
                gate_passed=False,
                risk_passed=False,
                execution_mode=self._mode,
                reason="tests_failing",
            )

        # 5. Risk-supervisor gate (drawdown, daily loss, trade count, etc.)
        risk_ok = True
        risk_reason = "ok"
        if self.risk is not None:
            risk_ok = self.risk.can_trade(symbol)
            if not risk_ok:
                risk_reason = getattr(self.risk, "_halt_reason", "risk_blocked")

        if not risk_ok:
            return GateResult(
                gate_passed=True,  # structural gates passed, risk blocked
                risk_passed=False,
                execution_mode=self._mode,
                reason=risk_reason,
            )

        # 6. Mode-specific gates
        if self._mode == "real_live":
            allowed, reason = live_trading_allowed(
                self.config,
                validation_state or {},
                account,
                tests,
            )
            if not allowed:
                return GateResult(
                    gate_passed=False,
                    risk_passed=False,
                    execution_mode=self._mode,
                    reason=reason,
                )

        elif self._mode == "demo_live":
            risk_state = {
                "halt": self.risk.halt if self.risk else False,
                "daily_pnl": getattr(self.risk, "realized_pnl_today", 0.0) if self.risk else 0.0,
                "max_daily_loss": getattr(self.risk, "max_daily_loss", 1000.0) if self.risk else 1000.0,
                "open_positions": int(intent.get("open_positions", 0)),
                "max_open_positions": getattr(self.risk, "max_open_positions", 6) if self.risk else 6,
            }
            allowed, reason = demo_trading_allowed(
                self.config,
                account,
                risk_state,
            )
            if not allowed:
                return GateResult(
                    gate_passed=False,
                    risk_passed=False,
                    execution_mode=self._mode,
                    reason=reason,
                )

        # All gates cleared
        return GateResult(
            gate_passed=True,
            risk_passed=True,
            execution_mode=self._mode,
            reason="ok",
        )
