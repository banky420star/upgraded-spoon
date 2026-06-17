from __future__ import annotations

import json
from typing import Optional, Any

from Python.mt5_compat import mt5

# Decision PPO: canonical structured command spec for executors (Python/MQL5 ChainGambler)
# This is the clean handoff format from the high-level Decision PPO brain.
DECISION_COMMAND_VERSION = "decision_ppo_v1"


def translate_trade_action(
    symbol: str,
    action_meta: dict,
    exposure: float,
    max_lots: float,
    tick: Optional[mt5.Tick] = None,
) -> Optional[dict]:
    """
    Translate (legacy or Decision PPO rich) action_meta into executor command.
    Decision PPO path: uses full structured spec (lot_spec, tp/sl/trailing/partials/BE).
    Returns both legacy flat fields + 'decision_command' (clean JSON spec for MQL5/Python exec).
    """
    if action_meta is None or tick is None:
        return None
    if abs(float(exposure)) < 0.01:
        return None

    direction = 1 if action_meta.get("direction", 0.0) >= 0.0 else -1
    size = float(action_meta.get("size", 0.0))
    if size <= 0.0:
        return None

    entry_mode = action_meta.get("entry_mode", "market")
    entry_offset_pct = float(action_meta.get("entry_offset_pct", 0.0))
    tp_offset_pct = float(action_meta.get("tp_offset_pct", 0.0))
    sl_offset_pct = float(action_meta.get("sl_offset_pct", 0.0))

    # === Decision PPO rich path ===
    dspec = action_meta.get("decision_spec")
    decision_command = None
    if dspec is not None or "lot_spec" in action_meta or "decision_spec_dict" in action_meta:
        # Build canonical structured command (the primary output for executors)
        if isinstance(dspec, dict):
            spec_dict = dspec
        elif hasattr(dspec, "to_dict"):
            spec_dict = dspec.to_dict()
        else:
            spec_dict = action_meta.get("decision_spec_dict") or {
                "lot_spec": action_meta.get("lot_spec", {}),
                "tp": action_meta.get("tp_spec", {"type": "pct", "value": tp_offset_pct}),
                "sl": action_meta.get("sl_spec", {"type": "pct", "value": sl_offset_pct}),
                "trailing": action_meta.get("trailing_spec", {}),
                "partial_close": action_meta.get("partial_close_spec", {}),
                "breakeven": action_meta.get("breakeven_spec", {}),
                "full_close": action_meta.get("full_close_spec", {}),
            }

        decision_command = {
            "version": DECISION_COMMAND_VERSION,
            "symbol": symbol,
            "direction": int(1 if direction > 0 else -1),
            "lot_spec": spec_dict.get("lot_spec", {"mode": "risk_based", "risk_pct_equity": 0.005}),
            "entry": {"type": entry_mode, "offset_pct": entry_offset_pct},
            "take_profit": spec_dict.get("tp", {"type": "pct", "value": tp_offset_pct}),
            "stop_loss": spec_dict.get("sl", {"type": "pct", "value": sl_offset_pct}),
            "trailing_stop": spec_dict.get("trailing", {"enabled": False}),
            "partial_close": spec_dict.get("partial_close", {"enabled": False}),
            "full_close_conditions": spec_dict.get("full_close", {}),
            "breakeven": spec_dict.get("breakeven", {"enabled": True}),
            "confidence": float(action_meta.get("confidence", spec_dict.get("confidence", 0.6))),
            "risk": spec_dict.get("risk", {}),
            "max_lots_cap": float(max_lots),
            "source": "decision_ppo",
        }

    mid_price = float((tick.ask + tick.bid) / 2.0)
    entry_price = _compute_entry_price(direction, mid_price, entry_offset_pct)

    if direction >= 0:
        tp_price = float(entry_price * (1.0 + tp_offset_pct))
        sl_price = float(entry_price * (1.0 - sl_offset_pct))
    else:
        tp_price = float(entry_price * (1.0 - tp_offset_pct))
        sl_price = float(entry_price * (1.0 + sl_offset_pct))

    lots = round(abs(size * max_lots), 2)
    order_type = "BUY" if direction > 0 else "SELL"

    result = {
        "symbol": symbol,
        "order_type": order_type,
        "entry_mode": entry_mode,
        "volume_lots": lots,
        "entry_price": round(entry_price, 6),
        "tp_price": round(tp_price, 6),
        "sl_price": round(sl_price, 6),
        "exposure": float(exposure),
        # Decision PPO extensions (non-breaking)
        "decision_ppo": bool(action_meta.get("decision_ppo", "decision_spec" in action_meta or "lot_spec" in action_meta)),
        "decision_command": decision_command,  # clean structured spec for MQL5 ChainGambler / Python executor
    }
    return result


def decision_spec_to_executor_command(decision_spec: Any, symbol: str, current_price: float | None = None) -> dict:
    """
    Pure helper: convert a DecisionSpec (or dict) directly into the clean JSON
    command consumed by MQL5 ChainGambler executor or Python order manager.
    This is the preferred path for the high-level Decision PPO brain -> executor.
    """
    if hasattr(decision_spec, "to_dict"):
        spec = decision_spec.to_dict()
    elif isinstance(decision_spec, dict):
        spec = decision_spec
    else:
        spec = {}

    cmd = {
        "version": DECISION_COMMAND_VERSION,
        "symbol": symbol,
        "direction": int(spec.get("direction", 0)),
        "lot_spec": spec.get("lot_spec", {}),
        "entry": spec.get("entry", {"type": "market"}),
        "take_profit": spec.get("tp", {}),
        "stop_loss": spec.get("sl", {}),
        "trailing_stop": spec.get("trailing", {}),
        "partial_close": spec.get("partial_close", {}),
        "full_close_conditions": spec.get("full_close", {}),
        "breakeven": spec.get("breakeven", {}),
        "risk": spec.get("risk", {}),
        "confidence": float(spec.get("confidence", 0.5)),
        "timestamp": None,  # filled by caller
        "source": "decision_ppo_brain",
    }
    if current_price is not None:
        cmd["reference_price"] = float(current_price)
    return cmd


def _compute_entry_price(direction: float, base_price: float, offset_pct: float) -> float:
    if direction >= 0:
        return base_price * (1.0 + offset_pct)
    return base_price * (1.0 - offset_pct)


def serialize_decision_for_mql5(decision_command: dict) -> str:
    """
    Produce compact JSON string suitable for MQL5 file or socket handoff
    to ChainGambler executor (or custom EA listening for Decision PPO commands).
    """
    return json.dumps(decision_command, separators=(",", ":"), default=str)
