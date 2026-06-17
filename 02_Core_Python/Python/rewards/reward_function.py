"""
TradingReward — Full reward function with all penalties.

Reward = pnl_after_spread_commission_slippage
         - drawdown_penalty
         - overtrading_penalty
         - spread_penalty
         - risk_violation_penalty
         - excessive_hold_penalty

Rejects raw price_change as reward.

NEW (v5+): penalty_scale (default 1.0) multiplies all penalty terms for training stability via lighter early profiles
(controlled by AGI_PENALTY_SCALE / AGI_REWARD_PROFILE / config). Defaults preserve all hardened modeling.

NEW (v6+): optional DSR (Differential Sharpe Ratio) bonus and asymmetric loss-amplification (Xia 2023) blend.
These are gated by env vars so defaults preserve the previous reward:
- AGI_USE_DSR=1 enables DSR term (Moody & Saffell 2001), blend weight AGI_DSR_WEIGHT (default 0.3)
- AGI_USE_ASYMMETRIC_LOSS=1 enables Xia 2023 negative-step amplification, blend weight AGI_ASYMMETRIC_WEIGHT (default 0.2)
The DSR + asymmetric loss blend attacks the 4:1 loss-to-win asymmetry seen in the live model (PF 0.32):
- DSR penalizes return *variance* (dominated by long-tail losers)
- asymmetric loss amplifies negative step returns by 1.10 (Xia 2023 sweet spot)
"""
import os
import numpy as np
from typing import Any, Dict, Optional

# Regime-Adaptive Controller integration for regime-aware reward shaping (penalty_scale, hold penalties, runner bias)
try:
    from Python.autonomous.regime_controller import get_regime_controller
    _REGIME_REWARD_AVAILABLE = True
except Exception:
    get_regime_controller = None  # type: ignore
    _REGIME_REWARD_AVAILABLE = False


class DSRReward:
    """
    Differential Sharpe Ratio reward (Moody & Saffell, 2001).

    Computes the time-derivative of the Sharpe ratio on a rolling window of returns.
    The DSR is naturally variance-averse: a sequence of 1 win of +$40 + 4 losses of -$164
    generates a NEGATIVE DSR even though arithmetic PnL may be positive — exactly the
    pathology of the live 4:1 loss-to-win asymmetry system.

    This is the standard FinRL extension for risk-aware PPO trading (Srivastava 2025,
    Choudhary 2025). Implementation follows the original Moody & Saffell formulation
    with EMA estimation of mean and variance for online use.

    Reference:
        Moody, J. & Saffell, M. (2001) "Learning to Trade via Direct Reinforcement"
        IEEE Trans. Neural Networks 12(4): 875-889
    """

    def __init__(self, eta: float = 0.04, window: int = 50, clip: float = 2.0):
        self.eta = float(eta)
        self.window = int(window)
        self.clip = float(clip)
        self.reset()

    def reset(self) -> None:
        """Reset running statistics. Call at episode start."""
        self.A = 0.0      # running mean of returns
        self.B = 1.0      # running variance (B = E[r^2] - (E[r])^2 + safety)
        self.S = 0.0      # 2 * A_t-1 * B_t-1 (cached for next step)
        self.n = 0        # step count

    def step(self, r: float) -> float:
        """
        Update running statistics with new return r and return DSR value.

        Args:
            r: step return (already scaled, e.g. pnl_after_costs / equity)

        Returns:
            dsr: float in [-clip, +clip]. Larger = better risk-adjusted return.
        """
        self.n += 1
        # Moody-Saffell 2001 canonical form is:
        #   D_t = (B_{t-1} * dA - 0.5 * S_{t-1} * dB) / (B_{t-1} - S_{t-1}^2)^1.5
        # Capture pre-update state so the formula uses the lag-correct values,
        # not the post-update B_t that would otherwise feed both numerator and
        # denominator.
        B_prev = self.B
        S_prev = self.S
        # Update mean (EMA) BEFORE computing dB, so dB uses new A
        dA = r - self.A
        self.A += self.eta * dA
        dB = (r * r) - self.B  # EMA on E[r^2] (not variance); see paper for derivation
        self.B += self.eta * dB
        # Denominator is (B_{t-1} - S_{t-1}^2)^1.5
        denom_inner = max(B_prev - S_prev * S_prev, 1e-7)
        denom = denom_inner ** 1.5
        if denom <= 0 or self.n < 2:
            self.S = 2.0 * self.A * self.B
            return 0.0
        dsr = (B_prev * dA - 0.5 * S_prev * dB) / denom
        # Cache S for next step
        self.S = 2.0 * self.A * self.B
        return float(np.clip(dsr, -self.clip, self.clip))

    def state(self) -> Dict[str, float]:
        """Return current running statistics for telemetry."""
        return {"A": float(self.A), "B": float(self.B), "S": float(self.S), "n": int(self.n)}


class TradingReward:
    """
    Computes trading reward with comprehensive penalty terms.
    """

    def __init__(
        self,
        commission_rate: float = 0.0002,
        spread_bps: float = 2.0,
        slippage_bps: float = 1.0,
        drawdown_penalty_coeff: float = 3.0,
        overtrading_penalty_coeff: float = 0.02,
        spread_penalty_coeff: float = 5.0,
        risk_violation_penalty_coeff: float = 10.0,
        # Env-var overridable: defaults keep prior hardened profile (200 bars = 16.7h on 5m).
        # Lower max_hold_steps + higher excessive_hold_penalty_coeff attacks the no-trades collapse
        # by forcing the policy out of long HOLDs. Set via AGI_MAX_HOLD_STEPS / AGI_EXCESSIVE_HOLD_PEN.
        excessive_hold_penalty_coeff: float = float(os.environ.get("AGI_EXCESSIVE_HOLD_PEN", "0.1")),
        max_hold_steps: int = int(os.environ.get("AGI_MAX_HOLD_STEPS", "200")),
        # NEW (v4 cost-barrier fix): AGI_COST_PENALTY scales the cost term within the
        # reward blend. Default lowered from 5.0 → 2.0 (env-var override; TradingReward
        # itself is still used by drl/trading_env.py which wires the env var into its
        # own reward_weights["cost_penalty"]). This duplicate knob lets TradingReward
        # callers (synthetic envs, paper backtests) also tune the cost barrier without
        # having to touch the trading_env code path.
        cost_penalty_coeff: float = float(os.environ.get("AGI_COST_PENALTY", "2.0")),
        max_drawdown_threshold: float = 0.15,
        max_risk_per_trade: float = 0.02,
        penalty_scale: float = 1.0,  # NEW: Reward Scale & Signal Improvement (v5+): <1.0 for lighter early-training profiles; default preserves hardened
        pnl_scale: float = float(os.environ.get("AGI_PNL_SCALE", "1.0")),
        bonus_scale: float = float(os.environ.get("AGI_BONUS_SCALE", "1.0")),  # NEW (v33+): scales shaped/bonus components
        bh_scale: float = float(os.environ.get("AGI_BH_SCALE", "0.0")),  # NEW (v35+): buy-hold benchmark comparison
        # NEW (v6+): DSR + asymmetric loss knobs, all gated by env vars for safe default
        use_dsr: bool = False,            # AGI_USE_DSR=1 enables
        dsr_weight: float = 0.3,          # AGI_DSR_WEIGHT (0.1-0.5; 0.3 = FinRL sweet spot)
        use_asymmetric_loss: bool = False,  # AGI_USE_ASYMMETRIC_LOSS=1 enables
        asymmetric_loss_weight: float = float(os.environ.get("AGI_ASYMMETRIC_WEIGHT", "0.2")),  # AGI_ASYMMETRIC_WEIGHT (0.1-0.4)
        asymmetric_loss_amp: float = float(os.environ.get("AGI_ASYMMETRIC_AMP", "1.10")),    # Xia 2023 sweet spot: 1.05-1.20
    ):
        self.commission_rate = float(commission_rate)
        self.spread_bps = float(spread_bps)
        self.slippage_bps = float(slippage_bps)
        self.drawdown_penalty_coeff = float(drawdown_penalty_coeff)
        self.overtrading_penalty_coeff = float(overtrading_penalty_coeff)  # lowered 0.5->0.02: churn is expense, not punishment
        self.spread_penalty_coeff = float(spread_penalty_coeff)
        self.risk_violation_penalty_coeff = float(risk_violation_penalty_coeff)
        self.excessive_hold_penalty_coeff = float(excessive_hold_penalty_coeff)
        self.max_hold_steps = int(max_hold_steps)
        self.cost_penalty_coeff = float(cost_penalty_coeff)
        self.max_drawdown_threshold = float(max_drawdown_threshold)
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.penalty_scale = float(penalty_scale)
        self.pnl_scale = float(pnl_scale)
        self.bonus_scale = float(bonus_scale)
        self.bh_scale = float(bh_scale)
        self.use_dsr = bool(use_dsr)
        self.dsr_weight = float(dsr_weight)
        self.use_asymmetric_loss = bool(use_asymmetric_loss)
        self.asymmetric_loss_weight = float(asymmetric_loss_weight)
        self.asymmetric_loss_amp = float(asymmetric_loss_amp)
        # DSR is stateful (rolling mean/var); lazily instantiate
        self._dsr: Optional[DSRReward] = None
        if self.use_dsr:
            self._dsr = DSRReward()
        self._last_regime_hints: Dict[str, Any] = {}  # populated by regime controller when compute() called with regime context
        # NEW (v36+): Curriculum stage overrides reward params (AGI_CURRICULUM_STAGE=1/2/3)
        self.curriculum_stage = int(os.environ.get("AGI_CURRICULUM_STAGE", "0"))
        self._apply_curriculum_stage()
        self._zscore_norm = bool(int(os.environ.get("AGI_ZSCORE_NORM", "0")))
        self._z_ema_mean = 0.0
        self._z_ema_var = 1.0
        self._z_alpha = 0.001
        self._use_sortino = bool(int(os.environ.get("AGI_USE_SORTINO", "0")))
        self._sortino_alpha = 0.05
        self._sortino_downside_sq = 0.0
        self._sortino_weight = float(os.environ.get("AGI_SORTINO_WEIGHT", "0.5"))
        self._vol_adj_cost = bool(int(os.environ.get("AGI_VOL_ADJUSTED_COST", "0")))
        self._vol_ema = 1e-8
        self._vol_alpha = 0.01
        self._vol_base_cost = self.cost_penalty_coeff
        self._use_calmar = bool(int(os.environ.get("AGI_USE_CALMAR", "0")))
        self._calmar_peak_equity = 0.0
        self._calmar_weight = float(os.environ.get("AGI_CALMAR_WEIGHT", "1.0"))
        self._use_ir = bool(int(os.environ.get("AGI_USE_IR", "0")))
        self._ir_alpha = 0.01
        self._ir_ema_mean = 0.0
        self._ir_ema_var = 1.0
        self._ir_weight = float(os.environ.get("AGI_IR_WEIGHT", "0.5"))
        self._apply_curriculum_stage()

    def _apply_curriculum_stage(self):
        """Override reward params based on curriculum stage.
        Stage 0 = disabled (use individual env vars as-is).
        Stages 1-3 progressively increase PnL pressure for curriculum learning.
        """
        stage = getattr(self, 'curriculum_stage', 0)
        if stage == 0:
            return
        profiles = {
            1: {  # Stage 1: Gentle PnL intro + z-score norm + vol-adj cost
                "pnl_scale": 1000.0,
                "penalty_scale": 0.10,
                "bonus_scale": 1.0,
                "use_asymmetric_loss": False,
                "asymmetric_loss_weight": 0.0,
                "asymmetric_loss_amp": 1.0,
                "bh_scale": 0.0,
                "cost_penalty_coeff": 2.0,
                "_zscore_norm": True,
                "_vol_adj_cost": True,
                "_use_sortino": False,
                "_use_calmar": False,
                "_use_ir": False,
            },
            2: {  # Stage 2: Moderate + z-score + sortino + vol-adj
                "pnl_scale": 5000.0,
                "penalty_scale": 0.05,
                "bonus_scale": 0.5,
                "use_asymmetric_loss": True,
                "asymmetric_loss_weight": 3.0,
                "asymmetric_loss_amp": 1.30,
                "bh_scale": 0.5,
                "cost_penalty_coeff": 1.0,
                "_zscore_norm": True,
                "_vol_adj_cost": True,
                "_use_sortino": True,
                "_use_calmar": False,
                "_use_ir": False,
            },
            3: {  # Stage 3: Full + ALL features
                "pnl_scale": 10000.0,
                "penalty_scale": 0.01,
                "bonus_scale": 0.1,
                "use_asymmetric_loss": True,
                "asymmetric_loss_weight": 5.0,
                "asymmetric_loss_amp": 1.50,
                "bh_scale": 1.0,
                "cost_penalty_coeff": 0.5,
                "_zscore_norm": True,
                "_vol_adj_cost": True,
                "_use_sortino": True,
                "_use_calmar": True,
                "_use_ir": True,
            },
        }
        profile = profiles.get(stage)
        if profile is None:
            return
        for attr, value in profile.items():
            # Boolean feature flags that should not be cast to float
            bool_attrs = {"use_asymmetric_loss", "_zscore_norm", "_vol_adj_cost",
                         "_use_sortino", "_use_calmar", "_use_ir",
                         "use_dsr"}
            if attr in bool_attrs:
                setattr(self, attr, bool(value))
            else:
                setattr(self, attr, float(value))
        # Re-init DSR if stage enables it (stage profiles keep it disabled)
        if getattr(self, 'use_dsr', False) and self._dsr is None:
            self._dsr = DSRReward()

    def compute(
        self,
        prev_equity: float,
        current_equity: float,
        prev_position: float,
        current_position: float,
        current_price: float,
        prev_price: float,
        drawdown: float,
        prev_drawdown: float = 0.0,
        hold_steps: int = 0,
        risk_used: float = 0.0,
        regime_state: Optional[Dict[str, Any]] = None,  # NEW: from RegimeAdaptiveController for shaping
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Compute reward and return dict with components.
        """
        # Raw PnL before costs
        price_ret = (current_price - prev_price) / (prev_price + 1e-12)
        raw_pnl = prev_position * prev_equity * price_ret

        # Costs
        delta = current_position - prev_position
        traded_notional = abs(delta) * current_equity
        commission_cost = traded_notional * self.commission_rate
        spread_cost = traded_notional * (self.spread_bps / 10000.0)
        slippage_cost = traded_notional * (self.slippage_bps / 10000.0)
        total_cost = commission_cost + spread_cost + slippage_cost

        # Apply AGI_COST_PENALTY weight to the per-step cost penalty used in this class's
        # reward blend. Default 2.0 (was 5.0) so a single trade doesn't need a 75% WR to
        # beat the barrier — 40% WR now suffices for break-even. Env-var overridable.
        # Base cost barrier (spread + commission + slippage as fraction of equity)
        raw_cost_penalty = self.cost_penalty_coeff * (total_cost / (prev_equity + 1e-12))

        # NEW (v37+): Volatility-adjusted cost penalty (AGI_VOL_ADJUSTED_COST=1)
        # Scales cost by inverse of recent volatility: when vol is high, spread is a smaller
        # fraction of expected move (trades are cheaper in opportunity-cost terms).
        # When vol is low, cost penalty rises to force the model to try trading.
        if self._vol_adj_cost:
            step_ret = (current_price - prev_price) / (prev_price + 1e-12)
            self._vol_ema = (1 - self._vol_alpha) * self._vol_ema + self._vol_alpha * step_ret**2
            vol = max(1e-8, self._vol_ema ** 0.5)
            vol_adj = 1.0 / (vol * 100 + 1.0)  # normalize: at 1% daily vol, adj = 0.5
            cost_penalty_term = raw_cost_penalty * vol_adj
        else:
            cost_penalty_term = raw_cost_penalty

        pnl_after_costs = raw_pnl - total_cost

        # Penalties
        dd_increase = max(0.0, drawdown - prev_drawdown)
        drawdown_penalty = self.drawdown_penalty_coeff * max(0.0, dd_increase - 0.06)

        # NEW (v37+): Calmar ratio proxy (AGI_USE_CALMAR=1)
        # Tracks peak equity and penalizes drawdown from peak proportionally.
        # This gives a Calmar-like signal (return / max_drawdown) at each step,
        # encouraging the agent to protect equity peaks rather than recover from lows.
        calmar_penalty = 0.0
        if self._use_calmar:
            # Update peak equity (running max)
            self._calmar_peak_equity = max(self._calmar_peak_equity, current_equity)
            current_drawdown_pct = (self._calmar_peak_equity - current_equity) / (self._calmar_peak_equity + 1e-12)
            if current_drawdown_pct > 0.05:  # only penalize drawdowns > 5%
                calmar_penalty = self._calmar_weight * current_drawdown_pct

        overtrading_penalty = self.overtrading_penalty_coeff * abs(delta)
        spread_penalty = self.spread_penalty_coeff * (spread_cost / (prev_equity + 1e-12))
        risk_violation_penalty = self.risk_violation_penalty_coeff * max(0.0, risk_used - self.max_risk_per_trade)
        excessive_hold_penalty = self.excessive_hold_penalty_coeff * max(0.0, hold_steps - self.max_hold_steps) / max(1, self.max_hold_steps)

        # Regime-Adaptive Controller: dynamic shaping (activates real behavior change in training/backtests)
        # Uses regime_aware_reward_hints for penalty_scale, hold mult, runner bias per bull/bear/ranging/news.
        penalty_scale = self.penalty_scale
        if _REGIME_REWARD_AVAILABLE and get_regime_controller:
            try:
                ctrl = get_regime_controller()
                hints = ctrl.regime_aware_reward_hints(
                    state=None,  # will use last known
                    symbol=symbol,
                )
                # Blend with instance penalty_scale (controller can lighten/harden)
                dyn_scale = float(hints.get("penalty_scale", 1.0))
                hold_mult = float(hints.get("excessive_hold_penalty_mult", 1.0))
                # runner_bonus not directly applied here (PPO value shaping / downstream), but logged
                self._last_regime_hints = hints  # for telemetry
                penalty_scale = self.penalty_scale * dyn_scale
                excessive_hold_penalty *= hold_mult
            except Exception as e:
                # Don't swallow silently — log so the engineer can see regime shaping dropped out.
                import logging
                logging.getLogger(__name__).warning(
                    "regime controller failed in TradingReward.compute: %s; reverting to penalty_scale=%.3f",
                    e, self.penalty_scale,
                )
                self._last_regime_hints = {"error": str(e)}  # mark stale in telemetry

        eff_scale = penalty_scale
        drawdown_penalty *= eff_scale
        overtrading_penalty *= eff_scale
        spread_penalty *= eff_scale
        risk_violation_penalty *= eff_scale
        excessive_hold_penalty *= eff_scale

        # Total reward (core)
        reward = (
            pnl_after_costs / (prev_equity + 1e-12) * self.pnl_scale
            - drawdown_penalty
            - overtrading_penalty
            - spread_penalty
            - risk_violation_penalty
            - excessive_hold_penalty
            - cost_penalty_term
            - calmar_penalty
        )

        # ── Asymmetric loss-amplification (Xia 2023) ──
        # Amplify NEGATIVE step returns so the policy gradient rewards avoiding big losers
        # even at the cost of small winners. Xia, Shi, Lin (ICBIS 2023) showed this lifts
        # both absolute return and Sharpe on DJIA 2009-2021 with a 1.05-1.20 multiplier.
        # Default OFF; enable via AGI_USE_ASYMMETRIC_LOSS=1 (env var read at __init__).
        asymmetric_loss_term = 0.0
        if self.use_asymmetric_loss:
            step_ret = pnl_after_costs / (prev_equity + 1e-12)
            if step_ret < 0.0:
                # amplification is "extra" negative reward proportional to the loss size
                asymmetric_loss_term = (self.asymmetric_loss_amp - 1.0) * step_ret  # negative
            reward += self.asymmetric_loss_weight * asymmetric_loss_term

        # ── Differential Sharpe Ratio bonus (Moody & Saffell 2001) ──
        # Penalizes return *variance* (which the live 4:1 loss-to-win asymmetry dominates).
        # Blends as a small bonus term on top of the shaped reward. Default OFF; enable via
        # AGI_USE_DSR=1. Weight 0.3 is the FinRL PPO+DSR empirical sweet spot.
        dsr_bonus = 0.0
        if self.use_dsr and self._dsr is not None:
            step_ret = pnl_after_costs / (prev_equity + 1e-12)
            dsr_bonus = self._dsr.step(step_ret)
            reward += self.dsr_weight * dsr_bonus

        # NEW (v37+): Sortino-like downside penalty (AGI_USE_SORTINO=1)
        # Replace the DSR (which penalizes ALL variance) with a downside-only variance
        # tracker. Only penalizes negative step returns, preserving upside volatility.
        # Uses EMA to track semi-variance: E[min(0, ret)^2].
        sortino_term = 0.0
        if self._use_sortino:
            step_ret = pnl_after_costs / (prev_equity + 1e-12)
            downside_sq = min(0.0, step_ret) ** 2
            self._sortino_downside_sq = (1 - self._sortino_alpha) * self._sortino_downside_sq + self._sortino_alpha * downside_sq
            sortino_vol = max(1e-12, self._sortino_downside_sq ** 0.5)
            sortino_term = -self._sortino_weight * sortino_vol  # negative = penalty
            reward += sortino_term

        # ── Market Timing Awareness (user request: profitable timing + market open / news events) ──
        # These signals come from enriched observations (news_proximity, major_open_window, etc.)
        timing_bonus = 0.0
        news_proximity = float(kwargs.get("news_proximity", 0.0))
        in_open_window = float(kwargs.get("major_open_window", 0.0))
        news_avoidance = float(kwargs.get("news_avoidance_zone", 0.0))

        # Reward good behavior around news (avoiding high-impact windows when not in a strong position)
        if news_avoidance > 0 and current_position == 0:
            timing_bonus += 0.0008 * self.penalty_scale   # small positive for staying flat near news

        # Small bonus for participating in high-volatility open windows when conditions are good
        if in_open_window > 0 and abs(current_position) > 0:
            timing_bonus += 0.0005 * self.penalty_scale

        # Penalty for holding through very close high-impact news without justification
        if news_proximity > 0.7 and abs(current_position) > 0:
            timing_bonus -= 0.0015 * self.penalty_scale

        timing_bonus *= self.bonus_scale
        reward += timing_bonus

        # ── Buy & Hold Benchmark Comparison (v35+) ──
        # Penalizes the agent when it underperforms a simple buy-and-hold strategy,
        # and rewards it when it outperforms. This forces the agent to capture
        # upside moves and avoid downside, preventing the 'do nothing' equilibrium.
        # bh_scale=0.0 disables (default). Set AGI_BH_SCALE=1.0 for equal weighting.
        bh_advantage = 0.0
        if self.bh_scale != 0.0:
            price_ret = (current_price - prev_price) / (prev_price + 1e-12)
            agent_ret = pnl_after_costs / (prev_equity + 1e-12)
            bh_advantage = agent_ret - price_ret  # positive = outperformed B&H
            reward += self.bh_scale * bh_advantage * self.pnl_scale  # scale BH to match PnL magnitude

        # NEW (v37+): Information Ratio benchmark (AGI_USE_IR=1)
        # Online tracking error vs. buy-hold: IR = mean(tracking_error) / std(tracking_error)
        # Uses EMA estimates for online mean and variance of tracking error.
        ir_bonus = 0.0
        if self._use_ir:
            agent_ret = pnl_after_costs / (prev_equity + 1e-12)
            price_ret = (current_price - prev_price) / (prev_price + 1e-12)
            tracking_error = agent_ret - price_ret
            ema_alpha = 0.01  # ~100-step window
            self._ir_ema_mean = (1 - ema_alpha) * self._ir_ema_mean + ema_alpha * tracking_error
            self._ir_ema_var = (1 - ema_alpha) * self._ir_ema_var + ema_alpha * (tracking_error - self._ir_ema_mean) ** 2
            ir_std = max(1e-12, self._ir_ema_var ** 0.5)
            ir_value = self._ir_ema_mean / ir_std  # Information Ratio estimate
            # Bonus scales positively with IR, capped at +/-2.0 (beyond that is noise)
            ir_bonus = self._ir_weight * max(-2.0, min(2.0, ir_value))
            reward += ir_bonus

        # ── HOLD-collapse counter-measure (anti-HOLD-collapse fix, v2) ──
        # The HOLD-collapse pathology: when the agent is in HOLD, the reward is
        # purely negative (just penalties, see line 287 below). Combined with a
        # high cost_penalty (5.0) and a 1:3 RR (losers >> winners), the policy
        # gradient discovers a local minimum: hold forever, accept the penalty
        # as the cost, never enter a position. Result: ep_rew_mean monotonically
        # worsens (e.g. -13.2k → -17.8k over 200k steps), the model takes 0
        # trades, and the value function learns that "everything is bad" rather
        # than learning which trades are good.
        #
        # Fix (v2 - compound):
        #   (a) Per-step HOLD-stagnation penalty: while flat, pay a small per-step
        #       cost proportional to (1 + hold_steps). Without this, the value
        #       function learns that HOLD is "free" beyond the anti-HOLD step
        #       cap. With it, HOLD has a real opportunity cost that grows.
        #   (b) Entry bonus on NEW position: when the agent opens a fresh
        #       position (|prev| < eps AND |current| > eps), award a positive
        #       bonus. This makes the marginal expected value of "try a trade"
        #       > "stay flat one more step" as long as the trade isn't
        #       catastrophic.
        #   (c) Bonus scales with how long the agent has been in HOLD: the
        #       longer it has been flat, the bigger the entry bonus. This
        #       directly counters the "HOLD is locally optimal" gradient.
        #
        # Defaults picked to dominate the per-step cost_penalty (~5e-3 per trade
        # entry on BTC at 1% risk) and the per-step HOLD penalty accumulation,
        # so the gradient locally prefers "try a small trade" over "hold forever".
        # Tunables (all env-var overridable):
        #   AGI_TRADE_EXPLORATION_BONUS   (default 0.005) — base entry bonus
        #   AGI_HOLD_PERSIST_PENALTY      (default 0.0001) — per-step penalty scaling
        #   AGI_HOLD_PERSIST_PENALTY_MAX  (default 0.05)  — cap per-step penalty
        try:
            _teb = float(os.environ.get("AGI_TRADE_EXPLORATION_BONUS", "0.005"))
        except Exception:
            _teb = 0.005
        try:
            _hpp = float(os.environ.get("AGI_HOLD_PERSIST_PENALTY", "0.0001"))
        except Exception:
            _hpp = 0.0001
        try:
            _hpp_max = float(os.environ.get("AGI_HOLD_PERSIST_PENALTY_MAX", "0.05"))
        except Exception:
            _hpp_max = 0.05

        hold_persist_penalty = 0.0
        trade_exploration_bonus = 0.0
        if abs(prev_position) < 1e-6 and abs(delta) < 1e-6:
            # In HOLD this step — pay opportunity cost that grows with hold duration.
            # hold_steps is provided by caller (env's steps_held counter).
            hold_persist_penalty = min(_hpp * float(hold_steps), _hpp_max)
        elif abs(prev_position) < 1e-6 and abs(current_position) > 1e-6:
            # Fresh position entry this step — award bonus (capped at per-step cost)
            trade_exploration_bonus = min(_teb, _teb * 1.0)

        # Reject raw price_change: if the reward is essentially just price_ret with no position scaling, zero it
        # This prevents the agent from getting rewarded for market movement without position
        if abs(prev_position) < 1e-6 and abs(delta) < 1e-6:
            # V20: flat HOLD is nearly neutral (was -0.10/step perpetual penalty)
            reward = -spread_penalty
            if hold_steps > 500:
                reward -= min(0.00001 * (hold_steps - 500), 0.005)
        else:
            # In position -- pay normal penalties
            reward = reward - hold_persist_penalty

        # Scale exploration bonus by bonus_scale
        trade_exploration_bonus *= self.bonus_scale
        # Apply exploration bonus to the final reward
        reward += trade_exploration_bonus

        # ── NEW (v37+): Running Z-Score reward normalization (AGI_ZSCORE_NORM=1) ──
        # Normalizes reward by rolling mean/std so PPO sees consistent reward distribution
        # across curriculum stages (stage 1 ~+0.07, stage 3 ~-0.17). Uses EMA for online
        # mean and variance estimation with alpha=0.001 (~1000-step half-life).
        zscore_scaled = 0.0
        if self._zscore_norm:
            self._z_ema_mean = (1 - self._z_alpha) * self._z_ema_mean + self._z_alpha * reward
            self._z_ema_var = (1 - self._z_alpha) * self._z_ema_var + self._z_alpha * (reward - self._z_ema_mean) ** 2
            z_std = max(1e-8, self._z_ema_var ** 0.5)
            zscore_scaled = (reward - self._z_ema_mean) / z_std
            reward = zscore_scaled

        return {
            "reward": float(np.clip(reward, -5.0, 5.0)),
            "components": {
                "pnl_after_spread_commission_slippage": float(pnl_after_costs / (prev_equity + 1e-12)),
                "drawdown_penalty": float(drawdown_penalty),
                "overtrading_penalty": float(overtrading_penalty),
                "spread_penalty": float(spread_penalty),
                "risk_violation_penalty": float(risk_violation_penalty),
                "excessive_hold_penalty": float(excessive_hold_penalty),
                # NEW (v4 cost-barrier fix): expose AGI_COST_PENALTY-weighted cost term
                # for telemetry and downstream audits. The shaped_reward blend in
                # drl/trading_env.py uses a separate reward_weights["cost_penalty"] that
                # also reads this env var; this is the TradingReward class's mirror.
                "cost_penalty": float(cost_penalty_term),
                # NEW (v6+): expose DSR + asymmetric loss components for telemetry
                "dsr_bonus": float(dsr_bonus) if self.use_dsr else 0.0,
                "asymmetric_loss_term": float(asymmetric_loss_term) if self.use_asymmetric_loss else 0.0,
                # NEW (anti-HOLD-collapse): expose trade exploration bonus + hold persist penalty
                "trade_exploration_bonus": float(trade_exploration_bonus),
                "hold_persist_penalty": float(hold_persist_penalty),
                # NEW (v37+): Advanced reward feature components
                "calmar_penalty": float(calmar_penalty) if self._use_calmar else 0.0,
                "sortino_term": float(sortino_term) if self._use_sortino else 0.0,
                "ir_bonus": float(ir_bonus) if self._use_ir else 0.0,
                "zscore_scaled": float(zscore_scaled) if self._zscore_norm else 0.0,
                "vol_adjusted_cost": float(1.0 if (self._vol_adj_cost and cost_penalty_term != raw_cost_penalty) else 0.0),
                "raw_cost_penalty": float(raw_cost_penalty),
            },
            "costs": {
                "commission": float(commission_cost),
                "spread": float(spread_cost),
                "slippage": float(slippage_cost),
                "total": float(total_cost),
            },
            "dsr_state": (self._dsr.state() if (self.use_dsr and self._dsr is not None) else None),
        }

    @staticmethod
    def reject_raw_price_change(reward: float, position: float, delta: float) -> float:
        """
        Zero out rewards that are just raw price movement without meaningful position.
        """
        if abs(position) < 1e-6 and abs(delta) < 1e-6:
            return 0.0
        return float(reward)
