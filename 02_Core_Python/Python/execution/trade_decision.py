"""
TradeDecision — Structured high-level output from Decision PPO (or hybrid brains).

This is the clean contract between:
  Decision Layer (PPO policy, ensemble, etc.)  -->  ExecutionAgent

The ExecutionAgent (and its backends) are responsible for turning this into
live or paper orders + full lifecycle management (entries, partials, trailing,
breakeven, time exits, reporting).

Design goals:
- Expressive enough for professional risk/execution (risk-% sizing, multi-level
  TP ladders, multiple trailing strategies, time-based + event exits).
- Serializable to JSON for file-bridge to MQL5, logs, audit, PPO feedback.
- Backward compatible via from_simple_intent() adapter (no breakage of old paths).
- Production-grade: explicit enums, validation, decision_id for traceability.

Used by:
- Future richer PPO heads (output structured action -> TradeDecision)
- ExecutionAgent (primary consumer)
- Paper harness, autonomy loops, MQL5 command bridge
- Feedback systems (execution reports keyed by decision_id)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"  # explicit full close / no new position


class SizeMode(str, Enum):
    FIXED_LOTS = "fixed_lots"
    RISK_PCT_EQUITY = "risk_pct_equity"
    RISK_PCT_BALANCE = "risk_pct_balance"
    KELLY_FRACTION = "kelly_fraction"  # future: value = fraction of computed Kelly


class ExitType(str, Enum):
    FIXED_PIPS = "fixed_pips"
    ATR_MULT = "atr_mult"
    R_MULTIPLE = "r_multiple"  # relative to initial risk
    PRICE_ABSOLUTE = "price_absolute"
    LADDER = "ladder"  # only for TP


class TrailingType(str, Enum):
    NONE = "none"
    BREAKEVEN_ONLY = "breakeven_only"
    FIXED_PIPS = "fixed_pips"
    ATR = "atr"
    STEP_TRAIL = "step_trail"  # move SL every X in profit
    CHANDELIER = "chandelier"  # future


@dataclass
class SizeSpec:
    mode: SizeMode = SizeMode.FIXED_LOTS
    value: float = 0.01  # lots, or %, or Kelly fraction
    max_lots_cap: Optional[float] = None
    min_lots_floor: float = 0.01


@dataclass
class ExitSpec:
    """Unified spec for SL or primary TP (ladders handled separately)."""
    type: ExitType = ExitType.ATR_MULT
    value: float = 2.0  # pips / ATR mult / R
    price: Optional[float] = None  # for PRICE_ABSOLUTE


@dataclass
class TPLadderLevel:
    """One rung in a partial-close ladder."""
    level: float  # e.g. 1.0 for 1R, 2.5 for 2.5R, or pips/ATR depending on type
    close_pct: float  # 0.0 < x <= 1.0 ; portion of *remaining* or original? (documented in ExecutionAgent)
    type: ExitType = ExitType.R_MULTIPLE  # what "level" measures against


@dataclass
class PartialCloseLadder:
    """Full ladder definition for scale-outs."""
    levels: List[TPLadderLevel] = field(default_factory=list)
    # If true, close_pct is of the *original* entry size (common in prop firms).
    # If false, of the *current remaining* position (more aggressive runner).
    of_original_size: bool = True
    # After last ladder level, leave a runner (0% close on final) or flatten.
    runner_after_last: bool = True


@dataclass
class TrailingSpec:
    type: TrailingType = TrailingType.NONE
    trigger: float = 1.0  # R / pips / ATR to activate trailing
    distance: float = 1.0  # trail distance behind (same units as trigger or ATR)
    step: float = 0.5  # for STEP_TRAIL: move SL only after this increment in profit
    atr_period: int = 14  # when type=ATR
    breakeven_buffer: float = 0.0  # pips/points buffer when promoting to BE


@dataclass
class TimeExitSpec:
    max_hold_bars: Optional[int] = None
    max_hold_minutes: Optional[int] = None
    max_hold_hours: Optional[int] = None
    close_at_session_end: bool = False
    close_at_eod: bool = False
    close_before_high_impact_news: bool = False
    # Absolute UTC timestamps for this specific decision (rare but powerful)
    force_close_before: Optional[str] = None  # ISO8601
    # Regime-Adaptive Controller: pattern_fav enables pattern-continuation bias
    # (e.g. longer holds / runner preference / relaxed time exits when pattern favorable per Rainforest+PatternDetector)
    pattern_fav: bool = False
    # Optional vol targeting scalar from regime controller (affects SizeSpec risk_pct_equity scaling)
    vol_target_scale: float = 1.0


@dataclass
class EntrySpec:
    type: Literal["market", "limit", "stop"] = "market"
    price_offset_pct: float = 0.0  # for limit/stop relative to mid
    slippage_tolerance_pips: float = 5.0
    max_slippage_bps: float = 15.0


@dataclass
class TradeDecision:
    """
    The canonical structured decision object emitted by Decision PPO / high-level policy.

    decision_id is the primary correlation key for fills, partials, reports, and
    PPO observation augmentation / reward attribution.
    """

    # Identity & provenance (critical for learning + audit)
    decision_id: str = field(default_factory=lambda: f"td_{uuid.uuid4().hex[:12]}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "decision_ppo"  # or "ensemble", "hybrid_brain", "manual_override"
    model_version: Optional[str] = None
    confidence: float = 0.0  # [0,1] policy confidence / prob

    # Core trade intent
    symbol: str = ""
    side: Side = Side.LONG
    size: SizeSpec = field(default_factory=SizeSpec)

    # Entry handling
    entry: EntrySpec = field(default_factory=EntrySpec)

    # Risk exits (SL always required for production safety)
    sl: ExitSpec = field(default_factory=lambda: ExitSpec(type=ExitType.ATR_MULT, value=1.5))
    tp: ExitSpec = field(default_factory=lambda: ExitSpec(type=ExitType.R_MULTIPLE, value=2.0))

    # Rich TP management
    tp_ladder: Optional[PartialCloseLadder] = None

    # Dynamic management
    trailing: TrailingSpec = field(default_factory=TrailingSpec)
    breakeven_after_r: Optional[float] = 0.8  # promote to BE after this R (0 = immediate after fill)

    # Time & event exits
    time_exit: TimeExitSpec = field(default_factory=TimeExitSpec)

    # Full close / management logic hints for ExecutionAgent
    full_close_on_opposite: bool = True
    full_close_on_regime_shift: bool = False
    max_concurrent_positions: int = 1  # per symbol per magic lane

    # Execution metadata
    magic: Optional[int] = None
    comment: str = "DecisionPPO"
    tags: Dict[str, Any] = field(default_factory=dict)

    # Risk overrides (passed through to risk engines)
    risk_overrides: Dict[str, Any] = field(default_factory=dict)

    # Pattern + timing context (for Decision PPO rich decisions + backtest attribution)
    # Allows policies and backtesters to tag decisions for edge validation (e.g. "engulfing at open" bias on TimeExitSpec)
    pattern_context: Optional[Dict[str, Any]] = None
    timing_context: Optional[Dict[str, Any]] = None

    # --- Serialization helpers (JSON bridge to MQL5, logs, PPO feedback) ---

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Enums -> str for clean JSON
        d["side"] = self.side.value if isinstance(self.side, Side) else self.side
        d["size"]["mode"] = self.size.mode.value if isinstance(self.size.mode, SizeMode) else self.size.mode
        d["sl"]["type"] = self.sl.type.value if isinstance(self.sl.type, ExitType) else self.sl.type
        d["tp"]["type"] = self.tp.type.value if isinstance(self.tp.type, ExitType) else self.tp.type
        if self.trailing:
            d["trailing"]["type"] = self.trailing.type.value
        if self.tp_ladder:
            for lvl in d["tp_ladder"]["levels"]:
                if isinstance(lvl.get("type"), ExitType):
                    lvl["type"] = lvl["type"].value
        d["pattern_context"] = self.pattern_context
        d["timing_context"] = self.timing_context
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeDecision":
        # Robust reconstruction with enum conversion + defaults
        data = dict(data)  # copy

        if "side" in data:
            data["side"] = Side(str(data["side"]).upper())

        if "size" in data and isinstance(data["size"], dict):
            sz = data["size"]
            if "mode" in sz:
                sz["mode"] = SizeMode(str(sz["mode"]))

        if "sl" in data and isinstance(data["sl"], dict):
            sl = data["sl"]
            if "type" in sl:
                sl["type"] = ExitType(str(sl["type"]))

        if "tp" in data and isinstance(data["tp"], dict):
            tp = data["tp"]
            if "type" in tp:
                tp["type"] = ExitType(str(tp["type"]))

        if "trailing" in data and isinstance(data["trailing"], dict):
            tr = data["trailing"]
            if "type" in tr:
                tr["type"] = TrailingType(str(tr["type"]))

        if "tp_ladder" in data and data["tp_ladder"] and isinstance(data["tp_ladder"], dict):
            lad = data["tp_ladder"]
            if "levels" in lad:
                new_levels = []
                for lvl in lad["levels"]:
                    if isinstance(lvl, dict) and "type" in lvl:
                        lvl = dict(lvl)
                        lvl["type"] = ExitType(str(lvl["type"]))
                    new_levels.append(TPLadderLevel(**lvl) if isinstance(lvl, dict) else lvl)
                lad["levels"] = new_levels

        # Rebuild nested dataclasses
        if "size" in data and isinstance(data["size"], dict):
            data["size"] = SizeSpec(**data["size"])
        if "entry" in data and isinstance(data["entry"], dict):
            data["entry"] = EntrySpec(**data["entry"])
        if "sl" in data and isinstance(data["sl"], dict):
            data["sl"] = ExitSpec(**data["sl"])
        if "tp" in data and isinstance(data["tp"], dict):
            data["tp"] = ExitSpec(**data["tp"])
        if "trailing" in data and isinstance(data["trailing"], dict):
            data["trailing"] = TrailingSpec(**data["trailing"])
        if "time_exit" in data and isinstance(data["time_exit"], dict):
            data["time_exit"] = TimeExitSpec(**data["time_exit"])
        if "tp_ladder" in data and data["tp_ladder"] and isinstance(data["tp_ladder"], dict):
            data["tp_ladder"] = PartialCloseLadder(**data["tp_ladder"])

        # pattern/timing contexts are plain dicts or None
        if "pattern_context" not in data:
            data["pattern_context"] = None
        if "timing_context" not in data:
            data["timing_context"] = None

        # Fill defaults for missing optionals
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "TradeDecision":
        return cls.from_dict(json.loads(s))

    # --- Compatibility adapters (critical: do not break existing simple paths) ---

    @classmethod
    def from_simple_intent(
        cls,
        intent: Dict[str, Any],
        decision_id: Optional[str] = None,
        source: str = "legacy_intent_adapter",
    ) -> "TradeDecision":
        """
        Convert old-style simple intent dicts (symbol, side, size, sl, tp, ...)
        into a minimal but valid TradeDecision. Used by ExecutionAgent and
        router for zero-breakage transition.
        """
        side_str = str(intent.get("side", "BUY")).upper()
        side = Side.LONG if side_str in ("BUY", "LONG") else (Side.SHORT if side_str in ("SELL", "SHORT") else Side.FLAT)

        size_val = float(intent.get("size", intent.get("volume", 0.01)) or 0.01)
        size_spec = SizeSpec(mode=SizeMode.FIXED_LOTS, value=size_val)

        sl_val = intent.get("sl")
        tp_val = intent.get("tp")

        sl_spec = ExitSpec(type=ExitType.PRICE_ABSOLUTE, value=0.0, price=float(sl_val) if sl_val is not None else None)
        tp_spec = ExitSpec(type=ExitType.PRICE_ABSOLUTE, value=0.0, price=float(tp_val) if tp_val is not None else None)

        # Best-effort: if only offsets were used in legacy, leave ATR defaults
        if sl_spec.price is None:
            sl_spec = ExitSpec(type=ExitType.ATR_MULT, value=1.8)
        if tp_spec.price is None:
            tp_spec = ExitSpec(type=ExitType.R_MULTIPLE, value=2.5)

        return cls(
            decision_id=decision_id or f"legacy_{uuid.uuid4().hex[:8]}",
            source=source,
            symbol=str(intent.get("symbol", "")),
            side=side,
            size=size_spec,
            sl=sl_spec,
            tp=tp_spec,
            magic=intent.get("magic"),
            comment=str(intent.get("comment", "legacy_intent")),
            tags={"legacy": True, "original_intent": {k: v for k, v in intent.items() if k not in ("sl", "tp")}},
        )

    def is_entry(self) -> bool:
        return self.side in (Side.LONG, Side.SHORT)

    def is_exit_all(self) -> bool:
        return self.side == Side.FLAT

    def validate(self) -> List[str]:
        """Light structural validation. Returns list of error strings (empty = OK)."""
        errs: List[str] = []
        if not self.symbol:
            errs.append("symbol is required")
        if self.size.value <= 0:
            errs.append("size.value must be > 0")
        if self.sl.value <= 0 and self.sl.price is None:
            errs.append("SL spec requires positive value or explicit price")
        if self.confidence < 0 or self.confidence > 1:
            errs.append("confidence must be in [0,1]")
        if self.tp_ladder:
            total_pct = sum(l.close_pct for l in self.tp_ladder.levels)
            if total_pct > 1.01:
                errs.append("TP ladder close_pcts sum > 100%")
        return errs


# Convenience factory for common patterns (used in tests / PPO head adapters)
def make_risk_based_decision(
    symbol: str,
    side: Side,
    risk_pct: float = 1.0,
    atr_sl_mult: float = 1.5,
    tp_r: float = 2.0,
    trailing_type: TrailingType = TrailingType.ATR,
    **kwargs,
) -> TradeDecision:
    return TradeDecision(
        symbol=symbol,
        side=side,
        size=SizeSpec(mode=SizeMode.RISK_PCT_EQUITY, value=risk_pct),
        sl=ExitSpec(type=ExitType.ATR_MULT, value=atr_sl_mult),
        tp=ExitSpec(type=ExitType.R_MULTIPLE, value=tp_r),
        trailing=TrailingSpec(type=trailing_type, trigger=1.0, distance=1.5),
        **kwargs,
    )


# JSON Schema (human + machine) for docs + validation in other languages / MQL5 future
TRADE_DECISION_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TradeDecision",
    "description": "Structured output of Decision PPO for the Execution Layer. See Python/execution/trade_decision.py",
    "type": "object",
    "required": ["symbol", "side"],
    "properties": {
        "decision_id": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "source": {"type": "string"},
        "model_version": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "symbol": {"type": "string"},
        "side": {"type": "string", "enum": ["LONG", "SHORT", "FLAT"]},
        "size": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": [e.value for e in SizeMode]},
                "value": {"type": "number"},
            },
        },
        # ... (full schema abbreviated in code; authoritative is the dataclass + to_dict)
    },
}


# ------------------------------------------------------------------
# Adapter: PPO action_meta (from HybridBrain / decode_action) -> TradeDecision
# Enables Decision PPO rich path in harness, autonomy loops, Server_AGI paper mode.
# Preserves legacy fields; enriches with full spec for ExecutionAgent / MQL5 bridge.
# ------------------------------------------------------------------
def from_ppo_action_meta(
    action_meta: dict[str, Any],
    symbol: str,
    source: str = "decision_ppo",
    model_version: Optional[str] = None,
    confidence: float = 0.75,
) -> "TradeDecision":
    """Convert legacy/rich PPO action_meta dict (direction/size/tp/sl/...) to canonical TradeDecision.
    Safe for both 6D decision_ppo and older formats. Used by paper harness + execution closure.
    """
    if not isinstance(action_meta, dict):
        action_meta = {}
    direction = float(action_meta.get("direction", action_meta.get("target", 0.0)) or 0.0)
    size = float(action_meta.get("size", 0.01) or 0.01)
    side = Side.LONG if direction > 0 else (Side.SHORT if direction < 0 else Side.FLAT)

    tp_pct = float(action_meta.get("tp_offset_pct", action_meta.get("tp", 0.0)) or 0.0)
    sl_pct = float(action_meta.get("sl_offset_pct", action_meta.get("sl", 0.0)) or 0.0)
    entry_mode = str(action_meta.get("entry_mode", "market") or "market").lower()

    # Map offsets to ExitSpec (ATR/R defaults if zero/legacy)
    sl_spec = ExitSpec(
        type=ExitType.ATR_MULT if sl_pct <= 0 else ExitType.R_MULTIPLE,
        value=1.5 if sl_pct <= 0 else max(0.5, abs(sl_pct) * 100),
    )
    tp_spec = ExitSpec(
        type=ExitType.R_MULTIPLE if tp_pct <= 0 else ExitType.R_MULTIPLE,
        value=2.0 if tp_pct <= 0 else max(1.0, abs(tp_pct) * 100),
    )

    size_spec = SizeSpec(mode=SizeMode.FIXED_LOTS, value=max(0.01, size))

    trailing = TrailingSpec(
        type=TrailingType.ATR if action_meta.get("trailing") else TrailingType.BREAKEVEN_ONLY,
        trigger=1.0,
        distance=1.5,
    )

    pat_ctx = action_meta.get("pattern_context") or action_meta.get("patterns")
    tim_ctx = action_meta.get("timing_context") or action_meta.get("timing")

    return TradeDecision(
        decision_id=f"ppo_{uuid.uuid4().hex[:10]}",
        source=source,
        model_version=model_version,
        confidence=float(confidence),
        symbol=str(symbol),
        side=side,
        size=size_spec,
        entry=EntrySpec(type=entry_mode if entry_mode in ("market", "limit", "stop") else "market"),
        sl=sl_spec,
        tp=tp_spec,
        trailing=trailing,
        breakeven_after_r=0.8,
        tags={"from_action_meta": True, "raw": {k: v for k, v in list(action_meta.items())[:8]}},
        risk_overrides={"mtf": True, "best_features": True},
        pattern_context=pat_ctx if isinstance(pat_ctx, dict) else None,
        timing_context=tim_ctx if isinstance(tim_ctx, dict) else None,
    )
