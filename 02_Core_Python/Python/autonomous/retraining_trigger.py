"""
RetrainingTrigger — decides when the feedback loop has accumulated
enough evidence to justify a new training cycle.

Triggers:
  - 50 new closed demo trades
  - 100 blocked trades
  - champion drawdown warning
  - regime performance degradation
  - feature drift detected
  - model confidence calibration drift
  - new MT5 data window
  - candidate beats champion in validation

Output: trigger artifact with retraining_trigger_id, triggered, reasons,
next_cycle_command.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger
from Python.pipeline_audit import log_decision  # Unified decisions log for full pipeline audit trail


@dataclass
class TriggerArtifact:
    retraining_trigger_id: str
    triggered: bool
    reasons: List[str] = field(default_factory=list)
    next_cycle_command: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrainingTrigger:
    """
    Stateful trigger that evaluates whether the system should enter
    a new training / champion-promotion cycle.
    """

    def __init__(
        self,
        data_dir: str = "logs",
        thresholds: Optional[Dict[str, Any]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.thresholds = {
            "min_closed_demo_trades": 50,
            "min_blocked_trades": 100,
            "champion_drawdown_pct": 5.0,
            "regime_degradation_win_rate": 0.35,
            "confidence_calibration_mae": 0.15,
            "feature_drift_psi": 0.25,
            "validation_beat_champion_margin": 0.02,
            **(thresholds or {}),
        }

        # Counters (persisted to JSON on evaluate and across instances)
        self.closed_demo_trade_count: int = 0
        self.blocked_trade_count: int = 0
        self.last_trigger_time: Optional[str] = None
        self.state_path = self.data_dir / "retraining_trigger_state.json"
        self._load_persisted_state()

    # ------------------------------------------------------------------
    # Incremental counters
    # ------------------------------------------------------------------
    def increment_closed_demo(self, n: int = 1) -> None:
        self.closed_demo_trade_count += n
        self._save_state()

    def increment_blocked(self, n: int = 1) -> None:
        self.blocked_trade_count += n
        self._save_state()

    def _load_persisted_state(self) -> None:
        """Load counters from disk if present (enables cross-run continuity for harness/aggregator)."""
        try:
            if self.state_path.exists():
                st = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.closed_demo_trade_count = int(st.get("closed_demo_trade_count", 0))
                self.blocked_trade_count = int(st.get("blocked_trade_count", 0))
                self.last_trigger_time = st.get("last_trigger_time")
                logger.debug(f"RetrainingTrigger loaded persisted state: closed={self.closed_demo_trade_count}, blocked={self.blocked_trade_count}")
        except Exception as exc:
            logger.warning(f"Failed to load retrain trigger state: {exc}")

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "closed_demo_trade_count": self.closed_demo_trade_count,
                "blocked_trade_count": self.blocked_trade_count,
                "last_trigger_time": self.last_trigger_time,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug(f"RetrainingTrigger state save skipped: {exc}")

    def aggregate_from_logs(self) -> Dict[str, Any]:
        """
        Lightweight aggregator: scan execution/risk/canary/harness logs for real signals.
        Returns dict usable as kwargs to evaluate() or to seed counters.
        Used by harness periodic checks + external aggregator.
        """
        signals: Dict[str, Any] = {
            "closed_delta": 0,
            "blocked_delta": 0,
            "champion_drawdown_pct": None,
            "canary_artifact": None,
            "risk_events": 0,
        }
        try:
            # 1. execution_feedback.jsonl (from harness rollbacks, blocks, closes)
            fb = self.data_dir / "execution_feedback.jsonl"
            if fb.exists():
                for line in fb.read_text(encoding="utf-8").strip().splitlines()[-200:]:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        ev = rec.get("event", "")
                        if ev in ("trade_closed", "demo_trade_closed"):
                            signals["closed_delta"] += 1
                        elif ev in ("trade_blocked", "blocked", "rollback_triggered"):
                            signals["blocked_delta"] += 1
                        if ev == "rollback_triggered":
                            det = rec.get("details", {}) or {}
                            if "loss" in str(det.get("reason", "")).lower():
                                signals["champion_drawdown_pct"] = 5.5
                    except Exception:
                        pass

            # 2. risk_audit.jsonl -> count !allowed as blocked
            ra = self.data_dir / "risk_audit.jsonl"
            if ra.exists():
                for line in ra.read_text(encoding="utf-8").strip().splitlines()[-300:]:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        if not rec.get("allowed", True):
                            signals["blocked_delta"] += 1
                            signals["risk_events"] += 1
                    except Exception:
                        pass

            # 3. Latest canary artifact (for approved flags + dd)
            canary_files = sorted(self.data_dir.glob("canary_*.json"), reverse=True)
            if canary_files:
                try:
                    cart = json.loads(canary_files[0].read_text(encoding="utf-8"))
                    signals["canary_artifact"] = {
                        "approved_for_champion": cart.get("approved_for_champion", False),
                        "approved_for_real_live": cart.get("approved_for_real_live", False),
                        "trades": cart.get("trades", 0),
                        "net_return": cart.get("net_return", 0.0),
                        "max_drawdown": cart.get("max_drawdown", 0.0),
                        "risk_violations": cart.get("risk_violations", 0),
                    }
                    if cart.get("max_drawdown"):
                        dd_pct = (cart["max_drawdown"] / 5000.0) * 100.0  # rough notional proxy
                        if dd_pct > 1.0:
                            signals["champion_drawdown_pct"] = round(dd_pct, 2)
                except Exception:
                    pass

            # 4. harness state for more closes proxy (if trades_today high)
            hs = self.data_dir / "paper_harness_state.json"
            if hs.exists():
                try:
                    hst = json.loads(hs.read_text(encoding="utf-8"))
                    td = int(hst.get("trades_today", 0))
                    if td > 0:
                        signals["closed_delta"] = max(signals["closed_delta"], td)
                except Exception:
                    pass

        except Exception as exc:
            logger.debug(f"aggregate_from_logs partial: {exc}")
        return signals

    # ------------------------------------------------------------------
    # Evaluation inputs
    # ------------------------------------------------------------------
    def evaluate(
        self,
        champion_drawdown_pct: Optional[float] = None,
        regime_win_rates: Optional[Dict[str, float]] = None,
        feature_drift_psi: Optional[float] = None,
        confidence_calibration_mae: Optional[float] = None,
        new_mt5_data_available: bool = False,
        candidate_beats_champion: Optional[float] = None,
        canary_artifact: Optional[Dict[str, Any]] = None,
    ) -> TriggerArtifact:
        """
        Run all trigger rules and return a TriggerArtifact.
        If no explicit signals passed, auto-aggregates from execution logs (harness feedback).
        """
        now = datetime.now(timezone.utc)
        reasons: List[str] = []

        # Auto-aggregate real execution feedback if caller didn't supply strong signals
        if champion_drawdown_pct is None and canary_artifact is None and not new_mt5_data_available and candidate_beats_champion is None:
            try:
                sigs = self.aggregate_from_logs()
                if sigs.get("closed_delta"):
                    self.increment_closed_demo(int(sigs["closed_delta"]))
                if sigs.get("blocked_delta"):
                    self.increment_blocked(int(sigs["blocked_delta"]))
                if sigs.get("champion_drawdown_pct") is not None and champion_drawdown_pct is None:
                    champion_drawdown_pct = sigs["champion_drawdown_pct"]
                if canary_artifact is None and sigs.get("canary_artifact"):
                    canary_artifact = sigs["canary_artifact"]
            except Exception as exc:
                logger.debug(f"auto-aggregate in evaluate skipped: {exc}")

        # 1. Demo trade volume
        if self.closed_demo_trade_count >= self.thresholds["min_closed_demo_trades"]:
            reasons.append(
                f"closed_demo_trades {self.closed_demo_trade_count} >= {self.thresholds['min_closed_demo_trades']}"
            )

        # 2. Blocked trade volume
        if self.blocked_trade_count >= self.thresholds["min_blocked_trades"]:
            reasons.append(
                f"blocked_trades {self.blocked_trade_count} >= {self.thresholds['min_blocked_trades']}"
            )

        # 3. Champion drawdown warning
        if champion_drawdown_pct is not None:
            if champion_drawdown_pct >= self.thresholds["champion_drawdown_pct"]:
                reasons.append(
                    f"champion_drawdown {champion_drawdown_pct:.2f}% >= {self.thresholds['champion_drawdown_pct']}%"
                )

        # 4. Regime degradation
        if regime_win_rates:
            for regime, wr in regime_win_rates.items():
                if wr < self.thresholds["regime_degradation_win_rate"]:
                    reasons.append(
                        f"regime '{regime}' win_rate {wr:.2f} < {self.thresholds['regime_degradation_win_rate']}"
                    )

        # 5. Feature drift
        if feature_drift_psi is not None:
            if feature_drift_psi >= self.thresholds["feature_drift_psi"]:
                reasons.append(
                    f"feature_drift PSI {feature_drift_psi:.3f} >= {self.thresholds['feature_drift_psi']}"
                )

        # 6. Confidence calibration drift
        if confidence_calibration_mae is not None:
            if confidence_calibration_mae >= self.thresholds["confidence_calibration_mae"]:
                reasons.append(
                    f"confidence_calibration MAE {confidence_calibration_mae:.3f} >= {self.thresholds['confidence_calibration_mae']}"
                )

        # 7. New MT5 data window
        if new_mt5_data_available:
            reasons.append("new_mt5_data_window available")

        # 8. Candidate beats champion in validation
        if candidate_beats_champion is not None:
            margin = self.thresholds["validation_beat_champion_margin"]
            if candidate_beats_champion >= margin:
                reasons.append(
                    f"candidate_beats_champion by {candidate_beats_champion:.3f} >= {margin}"
                )

        # 9. Canary promotion signal
        if canary_artifact:
            if canary_artifact.get("approved_for_champion") and not canary_artifact.get("approved_for_real_live"):
                reasons.append("canary approved_for_champion but not yet for real-live")

        triggered = len(reasons) > 0

        next_cycle_command = ""
        if triggered:
            if candidate_beats_champion is not None:
                next_cycle_command = "run_champion_promotion"
            elif canary_artifact and canary_artifact.get("approved_for_real_live"):
                next_cycle_command = "promote_to_real_live"
            elif new_mt5_data_available or self.closed_demo_trade_count >= self.thresholds["min_closed_demo_trades"]:
                next_cycle_command = "run_retraining"
            else:
                next_cycle_command = "run_evaluation"

        # Unified single source of truth audit for retrain decision (ensures candidate has feedback trail)
        if triggered:
            try:
                log_decision(
                    decision_type="retrain_trigger",
                    actor="retraining_trigger",
                    decision="RETRAIN_TRIGGERED",
                    candidate=None,  # context often in metadata; harness/promoter provide candidate
                    run_id=None,
                    reason="|".join(reasons)[:300],
                    details={
                        "next_cycle_command": next_cycle_command,
                        "closed_demo": self.closed_demo_trade_count,
                        "blocked": self.blocked_trade_count,
                        "metadata": {
                            k: v for k, v in (locals().get("metadata") or {}).items() if k in ["champion_drawdown_pct"]
                        },
                    },
                    severity="info",
                )
            except Exception:
                pass

        artifact = TriggerArtifact(
            retraining_trigger_id=f"trigger_{uuid.uuid4().hex[:8]}",
            triggered=triggered,
            reasons=reasons,
            next_cycle_command=next_cycle_command,
            metadata={
                "evaluated_at": now.isoformat(),
                "closed_demo_trade_count": self.closed_demo_trade_count,
                "blocked_trade_count": self.blocked_trade_count,
                "champion_drawdown_pct": champion_drawdown_pct,
                "feature_drift_psi": feature_drift_psi,
                "confidence_calibration_mae": confidence_calibration_mae,
                "candidate_beats_champion": candidate_beats_champion,
            },
        )

        # Persist
        path = self.data_dir / f"{artifact.retraining_trigger_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(artifact), f, indent=2)

        if triggered:
            logger.info(
                f"RetrainingTrigger {artifact.retraining_trigger_id} triggered: {reasons} -> {next_cycle_command}"
            )
            self.last_trigger_time = now.isoformat()
            # Reset counters after trigger so we don't double-fire
            self.closed_demo_trade_count = 0
            self.blocked_trade_count = 0
            self._save_state()
        else:
            logger.debug(f"RetrainingTrigger not triggered at {now.isoformat()}")

        return artifact

    def get_last_trigger(self) -> Optional[Dict[str, Any]]:
        """Return the most recent trigger artifact from disk."""
        files = sorted(self.data_dir.glob("trigger_*.json"), reverse=True)
        if not files:
            return None
        with open(files[0], "r", encoding="utf-8") as f:
            return json.load(f)


# ------------------------------------------------------------------
# Lightweight aggregator entrypoint (for supervisor, harness, TUI, cron)
# Call periodically: python -m Python.autonomous.retraining_trigger --aggregate
# ------------------------------------------------------------------

def run_aggregator_and_log(data_dir: str = "logs", force_evaluate: bool = False) -> Optional[TriggerArtifact]:
    """
    Lightweight aggregator: instantiates trigger (auto-loads persisted + scans logs),
    forces evaluation (with auto-aggregate inside), and on trigger logs the key
    "RETRAIN RECOMMENDED" message with reasons + next command.
    Returns the artifact (or None on error).
    This is the missing periodic evaluator that closes the execution->retrain loop.
    """
    try:
        trig = RetrainingTrigger(data_dir=data_dir)
        # Force a full scan + evaluate (aggregator pulls latest signals)
        art = trig.evaluate() if not force_evaluate else trig.evaluate()
        if art and art.triggered:
            logger.warning(
                "RETRAIN RECOMMENDED: {} -> {} (closed_demo={}, blocked={}) | artifact={}".format(
                    "; ".join(art.reasons),
                    art.next_cycle_command,
                    art.metadata.get("closed_demo_trade_count", 0),
                    art.metadata.get("blocked_trade_count", 0),
                    art.retraining_trigger_id,
                )
            )
            # Unified audit (already also done inside evaluate, but ensure aggregator context)
            try:
                log_decision(
                    decision_type="retrain_trigger",
                    actor="retraining_trigger",
                    decision="RETRAIN_RECOMMENDED",
                    reason="; ".join(art.reasons)[:400],
                    details={
                        "next_cycle_command": art.next_cycle_command,
                        "trigger_id": art.retraining_trigger_id,
                        "source": "aggregator",
                        "closed_demo": art.metadata.get("closed_demo_trade_count", 0),
                        "blocked": art.metadata.get("blocked_trade_count", 0),
                    },
                    severity="warn",
                )
            except Exception:
                pass
            # Also write a convenience marker for supervisor/TUI
            try:
                marker = Path(data_dir) / "RETRAIN_RECOMMENDED.latest.json"
                marker.write_text(json.dumps(asdict(art), default=str, indent=2), encoding="utf-8")
            except Exception:
                pass
        else:
            logger.debug("RetrainingTrigger aggregator: no recommendation at this check")
            try:
                log_decision(
                    decision_type="retrain_trigger",
                    actor="retraining_trigger",
                    decision="NO_RETRAIN_NEEDED",
                    reason="periodic_aggregate_check",
                    details={"source": "aggregator"},
                    severity="info",
                )
            except Exception:
                pass
        return art
    except Exception as exc:
        logger.error(f"run_aggregator_and_log failed: {exc}")
        return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="RetrainingTrigger aggregator / inspector")
    p.add_argument("--aggregate", action="store_true", help="Run the lightweight periodic evaluator and log RETRAIN RECOMMENDED if warranted")
    p.add_argument("--data-dir", default="logs")
    p.add_argument("--last", action="store_true", help="Print last trigger artifact")
    args = p.parse_args()
    if args.aggregate:
        run_aggregator_and_log(data_dir=args.data_dir)
    elif args.last:
        t = RetrainingTrigger(data_dir=args.data_dir)
        last = t.get_last_trigger()
        print(json.dumps(last, indent=2, default=str) if last else "No triggers yet")
    else:
        print("Use --aggregate or --last")
