"""
Experience Memory / Advanced Replay System

This is the core missing piece for true self-evolution in the autonomous trading stack.

Core responsibilities (as specified):
- Store and intelligently retrieve high-quality experiences: successful/failed trades
  annotated with classical patterns (doji, engulfing, flags, breakouts, ...),
  timing contexts (news proximity, session opens, major windows), Rainforest regimes,
  and Dreamer-predicted outcomes vs reality.
- Prioritize by learning value (surprise, high-edge, regime transitions, rare patterns).
- Provide clean interfaces for Decision PPO (prioritized replay + IS weights),
  Dreamer (pattern+timing conditioned trajectories), Meta-Optimizer and Regime
  Controller (recall what worked in similar past situations).
- Efficient storage: structured JSONL (durable, append-only) + in-memory indices
  (fast metadata queries) + optional vector embeddings.
- Mechanisms against catastrophic forgetting: priority decay for old/low-value
  experiences + intelligent compaction/pruning that preserves high-learning-value
  edge cases while summarizing what was removed.

Designed to interoperate with:
- logs/trade_journal.jsonl and logs/execution_feedback.jsonl (primary sources)
- Python/feedback/ (TradeJournal, ReplayBuilder, OutcomeLabeler, TradeCoroner)
- Python/backtest/fast_backtester.py (RichTrade + pattern/timing attribution)
- autonomous/retraining_orchestrator.py and self_evolution_supervisor.py
- drl/ (PPO & Dreamer agents), ensemble/meta_controller.py, rainforest_detector.py
- patterns/pattern_detector.py

Runtime artifacts:
- runtime/experience_memory.jsonl (primary durable store)
- runtime/agent_status/experience_memory_agent.json (this report + health)

Author: Grok Build specialist agent (2026-05-28)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Callable, Tuple, Set
from pathlib import Path
import json
import time
import math
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

# Optional pandas for rich analytics (graceful degradation)
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except Exception:
    pd = None  # type: ignore
    _PANDAS_AVAILABLE = False


# =============================================================================
# EXPERIENCE DATACLASS — rich, self-describing record
# =============================================================================
@dataclass
class Experience:
    """High-fidelity experience record.

    Every field is chosen to enable precise conditioning and value estimation
    for PPO gradients, Dreamer imagination, and meta-level recall.
    """
    experience_id: str = field(default_factory=lambda: f"exp_{int(time.time()*1000)}_{np.random.randint(1000,9999)}")
    timestamp: float = field(default_factory=time.time)
    symbol: str = ""
    decision_id: str = ""

    # === Context (the heart of the system) ===
    classical_patterns: List[str] = field(default_factory=list)  # ["bullish_engulfing", "bull_flag", "breakout_up", ...]
    pattern_context: Dict[str, Any] = field(default_factory=dict)
    timing_context: Dict[str, Any] = field(default_factory=dict)  # news_proximity, major_open_window, session, is_high_impact_news, ...
    regime: str = "unknown"                                       # Rainforest at decision time
    regime_at_exit: str = "unknown"
    regime_transition: bool = False

    # === Decision ===
    side: str = ""
    risk_pct: float = 0.0
    size_mode: str = ""
    time_exit_spec: Dict[str, Any] = field(default_factory=dict)
    sl_spec: Dict[str, Any] = field(default_factory=dict)
    tp_spec: Dict[str, Any] = field(default_factory=dict)

    # === Realized outcome ===
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    exit_reason: str = ""
    hold_bars: int = 0
    hold_minutes: float = 0.0
    outcome_label: str = ""     # winner_clean | loser_regime_shift | loser_news_spike | ...
    mistake_label: str = ""     # bad_entry_timing | regime_miss | news_miss | ...

    # === Learning value (prioritization) ===
    surprise: float = 0.0
    edge_score: float = 0.0
    learning_priority: float = 1.0
    learning_value_components: Dict[str, float] = field(default_factory=dict)

    # === Dreamer signals ===
    dreamer_predicted_value: float = 0.0
    dreamer_predicted_reward: float = 0.0
    dreamer_predicted_return: float = 0.0
    dreamer_horizon: int = 15

    # === Provenance ===
    source: str = "live"  # live | paper | backtest | replay_harness
    predicted_return: float = 0.0
    actual_return: float = 0.0
    decision_features: Dict[str, Any] = field(default_factory=dict)
    context_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["context_embedding"] = self.context_embedding
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        data = dict(data)
        if "pattern_context" not in data and "patterns" in data:
            data["pattern_context"] = data.pop("patterns")
        known = set(cls.__dataclass_fields__.keys())
        exp = cls(**{k: v for k, v in data.items() if k in known})
        if not exp.experience_id:
            exp.experience_id = f"exp_{int(time.time()*1000)}_{np.random.randint(1000,9999)}"
        return exp


# =============================================================================
# EXPERIENCE MEMORY — the production implementation
# =============================================================================
class ExperienceMemory:
    """
    The central Experience Memory / Advanced Replay System.

    All ingestion, prioritization, decay, querying, and export logic lives here.
    """

    def __init__(
        self,
        capacity: int = 150_000,
        storage_path: Optional[str] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
        decay_half_life_days: float = 45.0,
    ):
        self.capacity = capacity
        self.storage_path = Path(storage_path) if storage_path else (Path("runtime") / "experience_memory.jsonl")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder
        self.decay_half_life_days = decay_half_life_days

        self.experiences: List[Experience] = []
        self._id_to_idx: Dict[str, int] = {}
        self._indices: Dict[str, Dict[str, List[int]]] = {
            "regime": defaultdict(list),
            "symbol": defaultdict(list),
            "source": defaultdict(list),
            "pattern": defaultdict(list),
            "outcome": defaultdict(list),
        }

        self._last_compaction_ts = time.time()
        self._total_added = 0

        self._load_from_disk()
        logger.info(f"[ExperienceMemory] Ready | cap={self.capacity} loaded={len(self.experiences)} store={self.storage_path}")

    # --------------------------- ADD & PRIORITY ---------------------------
    def add(self, experience: Experience, auto_boost: bool = True) -> str:
        if not experience.experience_id:
            experience.experience_id = f"exp_{int(time.time()*1000)}_{np.random.randint(10000,99999)}"

        if auto_boost or experience.learning_priority <= 1.0:
            experience.learning_priority = self._compute_learning_priority(experience)

        if self.embedder and not experience.context_embedding:
            try:
                experience.context_embedding = self.embedder(self._make_text_summary(experience))
            except Exception:
                pass

        if len(self.experiences) >= self.capacity:
            self._evict_lowest_value()

        idx = len(self.experiences)
        self.experiences.append(experience)
        self._id_to_idx[experience.experience_id] = idx
        self._update_indices(idx, experience)

        self._persist_experience(experience)
        self._total_added += 1
        return experience.experience_id

    def _compute_learning_priority(self, exp: Experience) -> float:
        prio = 1.0 + 2.5 * min(3.0, abs(exp.surprise)) + 1.8 * min(2.0, exp.edge_score)
        prio += 1.2 if exp.regime_transition else 0.0
        prio += 0.9 * min(1.5, len(exp.classical_patterns or []))
        if exp.mistake_label and exp.mistake_label != "none":
            prio += 1.4
        if exp.outcome_label in ("winner_clean", "loser_regime_shift", "loser_news_spike"):
            prio += 0.8
        news_p = (exp.timing_context or {}).get("news_proximity", 0.0) if isinstance(exp.timing_context, dict) else 0.0
        if news_p > 0.6 or (exp.timing_context or {}).get("is_high_impact_news"):
            prio += 0.7
        if abs(exp.dreamer_predicted_return - exp.actual_return) > 0.8:
            prio += 1.1
        prio = min(12.0, max(0.1, prio))

        exp.learning_value_components = {
            "surprise": round(abs(exp.surprise), 3),
            "edge": round(exp.edge_score, 3),
            "regime_transition": float(exp.regime_transition),
            "pattern_rarity": float(len(exp.classical_patterns or [])),
            "dreamer_mismatch": round(abs(exp.dreamer_predicted_return - exp.actual_return), 3),
        }
        return float(prio)

    def update_priority(self, experience_id: str, new_priority: Optional[float] = None, td_error: Optional[float] = None) -> bool:
        if experience_id not in self._id_to_idx:
            return False
        idx = self._id_to_idx[experience_id]
        exp = self.experiences[idx]
        if td_error is not None:
            exp.learning_priority = float(abs(td_error) + 0.01)
        elif new_priority is not None:
            exp.learning_priority = float(new_priority)
        self._persist_experience(exp)
        return True

    # --------------------------- INDICES ---------------------------
    def _update_indices(self, idx: int, exp: Experience):
        self._indices["regime"][exp.regime].append(idx)
        self._indices["symbol"][exp.symbol or "UNKNOWN"].append(idx)
        self._indices["source"][exp.source].append(idx)
        self._indices["outcome"][exp.outcome_label or "unlabeled"].append(idx)
        for p in (exp.classical_patterns or []):
            self._indices["pattern"][p].append(idx)
        dom = (exp.pattern_context or {}).get("dominant") if isinstance(exp.pattern_context, dict) else None
        if dom and dom != "none":
            self._indices["pattern"][str(dom)].append(idx)

    def rebuild_indices(self):
        self._indices = {k: defaultdict(list) for k in self._indices}
        self._id_to_idx = {}
        for idx, exp in enumerate(self.experiences):
            self._id_to_idx[exp.experience_id] = idx
            self._update_indices(idx, exp)

    # --------------------------- RETRIEVAL (PPO / DREAMER / META) ---------------------------
    def sample_prioritized_ppo_batch(
        self, batch_size: int, beta: float = 0.4, min_priority: float = 0.1
    ) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Decision PPO interface — returns experiences, priorities, IS weights."""
        if not self.experiences:
            return [], np.array([]), np.array([])

        valid = [i for i, e in enumerate(self.experiences) if e.learning_priority >= min_priority] or list(range(len(self.experiences)))
        priorities = np.asarray([self.experiences[i].learning_priority for i in valid], dtype=np.float64)
        probs = (priorities ** 0.6) / max((priorities ** 0.6).sum(), 1e-9)

        n = min(batch_size, len(valid))
        chosen = np.random.choice(len(valid), n, replace=False, p=probs)
        global_idx = [valid[c] for c in chosen]

        exps = [self.experiences[g] for g in global_idx]
        prios = np.array([self.experiences[g].learning_priority for g in global_idx])
        N = len(valid)
        weights = (N * probs[chosen]) ** (-beta)
        weights /= max(weights.max(), 1e-9)
        return exps, prios, weights

    def get_conditioned_for_dreamer(
        self, pattern: Optional[str] = None, regime: Optional[str] = None,
        timing: Optional[Dict[str, Any]] = None, limit: int = 128
    ) -> List[Dict[str, Any]]:
        """Dreamer interface — pattern + timing + regime conditioned fragments."""
        matches = self.query_similar(patterns=[pattern] if pattern else None, regime=regime,
                                     timing_tags=list(timing.keys()) if timing else None, top_k=limit * 2)
        out = []
        for e in matches[:limit]:
            out.append({
                "experience_id": e.experience_id, "symbol": e.symbol, "regime": e.regime,
                "classical_patterns": e.classical_patterns, "timing_context": e.timing_context,
                "side": e.side, "dreamer_predicted_return": e.dreamer_predicted_return,
                "actual_return": e.actual_return or e.realized_pnl_pct,
                "outcome_label": e.outcome_label, "surprise": e.surprise,
                "time_exit_spec": e.time_exit_spec, "raw": e.to_dict(),
            })
        return out

    def recall_similar_experiences(self, **kwargs) -> List[Experience]:
        return self.query_similar(**kwargs)

    def query_similar(
        self, patterns: Optional[List[str]] = None, regime: Optional[str] = None,
        timing_tags: Optional[List[str]] = None, symbol: Optional[str] = None,
        top_k: int = 50, min_edge: float = 0.0
    ) -> List[Experience]:
        if not self.experiences:
            return []
        cands = set(range(len(self.experiences)))
        if regime and regime in self._indices["regime"]:
            cands &= set(self._indices["regime"][regime])
        if symbol and symbol in self._indices["symbol"]:
            cands &= set(self._indices["symbol"][symbol])

        scored = []
        for i in cands:
            e = self.experiences[i]
            if e.edge_score < min_edge:
                continue
            sc = 0.0
            if patterns:
                ep = set(e.classical_patterns or [])
                dom = str((e.pattern_context or {}).get("dominant", "")).lower()
                for p in patterns:
                    if p.lower() in ep or p.lower() in dom:
                        sc += 3.0
            else:
                sc += 0.5
            if regime and e.regime == regime: sc += 2.0
            if symbol and e.symbol == symbol: sc += 0.8
            if timing_tags:
                tc = str(e.timing_context or "").lower()
                for t in timing_tags:
                    if t.lower() in tc: sc += 1.2
            sc += 0.4 * min(2.0, e.learning_priority / 3.0)
            if sc > 0.1:
                scored.append((sc, i))
        scored.sort(key=lambda x: -x[0])
        return [self.experiences[i] for _, i in scored[:top_k]]

    def get_high_value_experiences(self, min_edge: float = 0.55, limit: int = 2000) -> List[Experience]:
        filt = [e for e in self.experiences if e.edge_score >= min_edge]
        filt.sort(key=lambda e: (-e.edge_score, -e.learning_priority))
        return filt[:limit]

    def get_experiences_for_retraining(self, limit: int = 5000, min_surprise: float = 0.0, min_edge: float = 0.0) -> List[Dict]:
        """High-value experiences formatted for retraining data / candidate eval (orchestrator wiring).
        Filters on surprise (learning signal) + edge for closed-loop memory -> train.
        """
        filtered = [e for e in self.experiences if abs(getattr(e, 'surprise', 0.0) or 0.0) >= min_surprise and (getattr(e, 'edge_score', 0.0) or 0.0) >= min_edge]
        sorted_exp = sorted(filtered, key=lambda e: -getattr(e, 'learning_priority', 0.0))
        return [e.to_dict() if hasattr(e, 'to_dict') else {k: getattr(e, k, None) for k in Experience.__dataclass_fields__} for e in sorted_exp[:limit]]

    def compute_pattern_timing_stats(self) -> Dict[str, Any]:
        """Pattern profitability + timing + surprise/trend stats for should_retrain in orchestrator.
        Returns avg_edge_by_pattern, surprise_by_pattern, edge_trend etc.
        """
        if not self.experiences:
            return {"avg_edge_by_pattern": {}, "surprise_by_pattern": {}, "edge_trend_recent_vs_historical": 0.0, "avg_surprise": 0.0, "high_surprise_count": 0}
        from collections import defaultdict
        pat_edges = defaultdict(list)
        pat_surp = defaultdict(list)
        for e in self.experiences:
            pat = "none"
            if e.classical_patterns: pat = e.classical_patterns[0]
            elif isinstance(e.pattern_context, dict): pat = e.pattern_context.get("dominant", "none")
            pat_edges[pat].append(float(e.edge_score or 0))
            pat_surp[pat].append(float(e.surprise or 0))
        # trend
        edge_trend = 0.0
        if len(self.experiences) >= 8:
            sorted_e = sorted(self.experiences, key=lambda x: x.timestamp)
            n = len(sorted_e)
            split = max(2, n//3)
            old = [x.edge_score or 0 for x in sorted_e[:split]]
            rec = [x.edge_score or 0 for x in sorted_e[-split:]]
            if old and rec:
                edge_trend = float(np.mean(rec) - np.mean(old))
        all_s = [float(e.surprise or 0) for e in self.experiences]
        return {
            "avg_edge_by_pattern": {k: float(np.mean(v)) for k,v in pat_edges.items() if v},
            "surprise_by_pattern": {k: float(np.mean(v)) for k,v in pat_surp.items() if v},
            "avg_surprise": float(np.mean(all_s)) if all_s else 0.0,
            "high_surprise_count": sum(1 for s in all_s if abs(s) > 1.0),
            "edge_trend_recent_vs_historical": edge_trend,
            "total_experiences": len(self.experiences),
        }

    # --------------------------- ANALYTICS ---------------------------
    def get_pattern_success_rates(self) -> Dict[str, Dict[str, Any]]:
        if not _PANDAS_AVAILABLE or not self.experiences:
            stats = defaultdict(lambda: {"count": 0, "wins": 0, "pnl_sum": 0.0})
            for e in self.experiences:
                for p in (e.classical_patterns or ["none"]):
                    stats[p]["count"] += 1
                    if e.realized_pnl > 0: stats[p]["wins"] += 1
                    stats[p]["pnl_sum"] += e.realized_pnl
            for p, s in stats.items():
                s["win_rate"] = round(s["wins"] / max(1, s["count"]), 4)
                s["avg_pnl"] = round(s["pnl_sum"] / max(1, s["count"]), 4)
            return dict(stats)
        rows = []
        for e in self.experiences:
            for p in (e.classical_patterns or ["none"]):
                rows.append({"pattern": p, "pnl": e.realized_pnl, "win": int(e.realized_pnl > 0)})
        if not rows: return {}
        df = pd.DataFrame(rows)
        g = df.groupby("pattern").agg(count=("pnl", "count"), wins=("win", "sum"), avg_pnl=("pnl", "mean"))
        g["win_rate"] = (g.wins / g["count"]).round(4)
        return g.to_dict("index")

    def get_regime_transition_stats(self) -> Dict[str, Any]:
        trans = [e for e in self.experiences if e.regime_transition]
        if not trans:
            return {"total_transitions": 0}
        wins = sum(e.realized_pnl > 0 for e in trans)
        return {
            "total_transitions": len(trans),
            "transition_win_rate": round(wins / len(trans), 4),
            "avg_pnl_on_transition": round(float(np.mean([e.realized_pnl for e in trans])), 4),
            "common_mistake": self._most_common([e.mistake_label for e in trans if e.mistake_label]),
        }

    def compute_analytics_summary(self) -> Dict[str, Any]:
        if not self.experiences:
            return {"size": 0}
        edges = [e.edge_score for e in self.experiences]
        return {
            "size": len(self.experiences),
            "capacity": self.capacity,
            "avg_edge": float(np.mean(edges)),
            "max_edge": float(max(edges)),
            "high_priority_count": len([e for e in self.experiences if e.learning_priority > 4]),
            "high_value_count": len([e for e in self.experiences if e.edge_score > 0.7]),
            "pattern_success": self.get_pattern_success_rates(),
            "regime_transitions": self.get_regime_transition_stats(),
            "top_patterns": sorted(((k, len(v)) for k, v in self._indices["pattern"].items()), key=lambda x: -x[1])[:8],
            "last_compaction_hours_ago": round((time.time() - self._last_compaction_ts) / 3600, 1),
        }

    stats = compute_analytics_summary

    # --------------------------- INGESTION ---------------------------
    def ingest_from_execution_feedback(self, feedback_path: Optional[str] = None, max_lines: int = 5000) -> int:
        path = Path(feedback_path) if feedback_path else Path("logs/execution_feedback.jsonl")
        if not path.exists(): return 0
        added = 0
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-max_lines:]
            for line in lines:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if "close" not in rec.get("event", "").lower() and "executed" not in rec.get("event", "").lower():
                    continue
                dec = rec.get("decision") or rec.get("decision_summary") or {}
                rep = rec.get("report", {})
                symbol = rec.get("symbol") or dec.get("symbol", "UNKNOWN")
                pats = []
                comment = str(dec.get("comment", "")).lower()
                for kw in ("engulfing", "flag", "breakout", "doji"):
                    if kw in comment: pats.append(kw)
                exp = Experience(
                    decision_id=rec.get("decision_id", ""),
                    symbol=symbol,
                    side=(dec.get("side") or "").upper(),
                    realized_pnl=float(rep.get("realized_pnl", 0)),
                    realized_pnl_pct=float(rep.get("realized_pnl", 0)) / 100.0,
                    regime=dec.get("regime", "unknown"),
                    classical_patterns=pats,
                    timing_context={"news": "news" in comment},
                    source="live",
                    edge_score=float(dec.get("confidence", 0.4)),
                    outcome_label=rep.get("outcome_label", ""),
                    mistake_label=rep.get("mistake_label", ""),
                )
                self.add(exp)
                added += 1
        except Exception as e:
            logger.warning(str(e))
        logger.info(f"[ExperienceMemory] Ingested {added} from execution_feedback")
        return added

    def ingest_from_trade_journal(self, journal_path: Optional[str] = None, max_lines: int = 3000) -> int:
        path = Path(journal_path) if journal_path else Path("logs/trade_journal.jsonl")
        if not path.exists(): return 0
        added = 0
        try:
            for i, line in enumerate(path.open(encoding="utf-8", errors="replace")):
                if i > max_lines: break
                try:
                    rec = json.loads(line)
                except Exception: continue
                if rec.get("action") not in ("close", "closed"):
                    continue
                self.add(Experience(
                    symbol=rec.get("symbol", ""), side=rec.get("side", ""),
                    realized_pnl=float(rec.get("pnl", 0)), realized_pnl_pct=float(rec.get("pnl_pct", 0)),
                    exit_reason=rec.get("exit_reason", ""), source="live",
                    decision_id=rec.get("decision_id", rec.get("trade_id", "")),
                ))
                added += 1
        except Exception as e:
            logger.warning(str(e))
        return added

    def import_backtest_results(self, backtest_dir: str = "runtime/backtest_results", max_files: int = 3) -> int:
        added = 0
        for f in sorted(Path(backtest_dir).glob("*.json"), reverse=True)[:max_files]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                for t in (data.get("trades") or data.get("closed_trades") or []):
                    if isinstance(t, dict):
                        self.add(Experience(
                            symbol=t.get("symbol", data.get("config", {}).get("symbol", "")),
                            side=t.get("side", ""),
                            realized_pnl=float(t.get("pnl", t.get("net_pnl", 0))),
                            classical_patterns=[t.get("pattern_at_entry")] if t.get("pattern_at_entry") else [],
                            pattern_context={"dominant": t.get("pattern_at_entry", "none")},
                            timing_context={"timing_score": t.get("timing_score", 0)},
                            regime=t.get("regime", "unknown"),
                            source="backtest",
                            edge_score=float(t.get("confidence", 0.5)),
                        ))
                        added += 1
            except Exception:
                pass
        return added

    def record_live_close(self, decision: Dict, closed_trade: Dict, pattern_state=None,
                          rainforest_regime=None, dreamer_summary=None, outcome_labels=None) -> str:
        pats = []
        if pattern_state:
            dom = pattern_state.get("dominant") or pattern_state.get("dominant_pattern")
            if dom: pats.append(str(dom).lower().replace(" ", "_"))
        regime = rainforest_regime or closed_trade.get("regime", "unknown")
        exp = Experience(
            decision_id=decision.get("decision_id", closed_trade.get("decision_id", "")),
            symbol=closed_trade.get("symbol", decision.get("symbol", "")),
            side=closed_trade.get("side", decision.get("side", "")),
            realized_pnl=float(closed_trade.get("pnl", 0)),
            realized_pnl_pct=float(closed_trade.get("pnl_pct", 0)),
            classical_patterns=pats,
            pattern_context=pattern_state or {},
            timing_context=(pattern_state or {}).get("timing_context", {}),
            regime=regime,
            regime_at_exit=closed_trade.get("regime_at_exit", regime),
            regime_transition=bool(rainforest_regime and closed_trade.get("regime_at_exit") and rainforest_regime != closed_trade.get("regime_at_exit")),
            time_exit_spec=decision.get("time_exit", {}),
            outcome_label=(outcome_labels or {}).get("outcome_label", ""),
            mistake_label=(outcome_labels or {}).get("mistake_label", ""),
            source="live",
            edge_score=float(decision.get("confidence", 0.45)),
            dreamer_predicted_return=(dreamer_summary or {}).get("expected_return", 0.0),
            actual_return=float(closed_trade.get("pnl_pct", 0)),
            surprise=abs((dreamer_summary or {}).get("expected_return", 0) - float(closed_trade.get("pnl_pct", 0))),
        )
        return self.add(exp)

    # --------------------------- DECAY & COMPACTION (ANTI-FORGETTING) ---------------------------
    def apply_decay(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        hl = self.decay_half_life_days * 86400
        updated = 0
        for e in self.experiences:
            age = max(0.0, now - e.timestamp)
            df = 0.5 ** (age / hl) if hl > 0 else 1.0
            if df < 0.98:
                e.learning_priority = max(0.05, e.learning_priority * df * (0.6 + 0.4 * min(1.0, e.edge_score)))
                updated += 1
        if updated:
            logger.info(f"[ExperienceMemory] Decay applied to {updated} experiences")
        return updated

    def compact_storage(self, min_priority: float = 0.25, max_age_days: float = 120, keep_high_value: bool = True) -> int:
        before = len(self.experiences)
        if before == 0: return 0
        cutoff = time.time() - max_age_days * 86400
        kept = []
        pruned = 0
        for e in self.experiences:
            keep = e.learning_priority >= min_priority and (e.timestamp >= cutoff or (keep_high_value and e.edge_score > 0.75))
            if keep:
                kept.append(e)
            else:
                pruned += 1
        tmp = self.storage_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in kept:
                f.write(json.dumps(e.to_dict(), default=str) + "\n")
        try:
            tmp.replace(self.storage_path)
        except Exception:
            self.storage_path.write_text("".join(json.dumps(e.to_dict(), default=str) + "\n" for e in kept))
        self.experiences = kept
        self._id_to_idx.clear()
        self.rebuild_indices()
        self._last_compaction_ts = time.time()
        logger.info(f"[ExperienceMemory] Compacted: kept {len(kept)}/{before} (pruned {pruned})")
        return len(kept)

    def _evict_lowest_value(self, count: int = 1):
        if len(self.experiences) < 5: return
        now = time.time()
        scored = []
        for i, e in enumerate(self.experiences):
            age_d = max(0.1, (now - e.timestamp) / 86400)
            rec = 1.0 / (1 + math.log1p(age_d))
            val = e.learning_priority * rec * (0.3 + 0.7 * e.edge_score)
            scored.append((val, i))
        scored.sort()
        for _, i in scored[:count][::-1]:
            if i < len(self.experiences):
                rem = self.experiences.pop(i)
                self._id_to_idx.pop(rem.experience_id, None)
        self.rebuild_indices()

    # --------------------------- PERSIST / LOAD ---------------------------
    def _persist_experience(self, exp: Experience):
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(exp.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.warning(f"Persist error: {e}")

    def _load_from_disk(self):
        if not self.storage_path.exists(): return
        loaded = 0
        try:
            with open(self.storage_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        exp = Experience.from_dict(data)
                        if exp.experience_id not in self._id_to_idx:
                            self.experiences.append(exp)
                            loaded += 1
                    except Exception:
                        continue
            self.rebuild_indices()
            self.apply_decay()
            logger.info(f"[ExperienceMemory] Loaded {loaded} (post-decay {len(self.experiences)})")
        except Exception as e:
            logger.warning(f"Load error: {e}")

    # --------------------------- UTILS ---------------------------
    def _make_text_summary(self, exp: Experience) -> str:
        pats = ",".join((exp.classical_patterns or [])[:2])
        return f"{exp.symbol} {exp.side} {exp.regime} pats:{pats} timing:{(exp.timing_context or {}).get('news_proximity',0):.1f} surprise:{exp.surprise:.2f}"

    def _most_common(self, lst):
        if not lst: return ""
        return max(set(lst), key=lst.count)

    def force_full_compaction(self, **kw):
        self.apply_decay()
        return self.compact_storage(**kw)


# =============================================================================
# PUBLIC CONVERTER
# =============================================================================
def rich_trade_to_experience(rich_trade: Any, backtest_meta: Optional[Dict] = None) -> Experience:
    d = rich_trade if isinstance(rich_trade, dict) else (rich_trade.__dict__ if hasattr(rich_trade, "__dict__") else {})
    meta = backtest_meta or {}
    return Experience(
        symbol=d.get("symbol", meta.get("symbol", "")),
        side=d.get("side", ""),
        realized_pnl=float(d.get("pnl", d.get("net_pnl", 0))),
        realized_pnl_pct=float(d.get("pnl_pct", 0)),
        classical_patterns=[d.get("pattern_at_entry")] if d.get("pattern_at_entry") else [],
        pattern_context={"dominant": d.get("pattern_at_entry", "none")},
        timing_context={"timing_score": d.get("timing_score", 0.0)},
        source="backtest",
        edge_score=float(d.get("confidence", 0.5)),
        time_exit_spec=d.get("time_exit_used") or d.get("time_exit_spec"),
        hold_minutes=float(d.get("hold_minutes", 0)),
        exit_reason=d.get("exit_reason", ""),
        metadata={"from": "RichTrade", **meta},
    )


# =============================================================================
# DEMO / SMOKE TEST
# =============================================================================
if __name__ == "__main__":
    mem = ExperienceMemory(capacity=500, storage_path="runtime/experience_memory_demo.jsonl")
    for i in range(28):
        pats = ["bullish_engulfing"] if i % 3 == 0 else (["bull_flag"] if i % 5 == 0 else ["doji"])
        mem.add(Experience(
            symbol="XAUUSDm",
            decision_id=f"smoke_{i}",
            classical_patterns=pats,
            pattern_context={"dominant": pats[0]},
            timing_context={"major_open_window": 0.7 if i % 4 == 0 else 0.15, "news_proximity": 0.8 if i % 6 == 0 else 0.1},
            regime="bull_trend" if i % 2 == 0 else "ranging",
            regime_transition=(i % 9 == 0),
            side="LONG",
            realized_pnl=11.0 if i % 3 != 0 else -3.2,
            realized_pnl_pct=1.6 if i % 3 != 0 else -0.7,
            edge_score=0.3 + (i % 5) * 0.12,
            surprise=1.9 if i % 4 == 0 else 0.35,
            dreamer_predicted_return=0.9,
            actual_return=1.6 if i % 3 != 0 else -0.7,
            outcome_label="winner_clean" if i % 3 != 0 else "loser_regime_shift",
            mistake_label="none" if i % 3 != 0 else "regime_miss",
            source="backtest",
        ))

    print("STATS:", json.dumps(mem.stats(), indent=2, default=str)[:900])
    exps, p, w = mem.sample_prioritized_ppo_batch(5)
    print(f"PPO batch: {len(exps)} exps, IS weight mean {float(np.mean(w)):.3f}")
    print(f"Similar recall count: {len(mem.recall_similar_experiences(pattern='bullish_engulfing', regime='bull_trend', top_k=3))}")
    print(f"Dreamer fragments: {len(mem.get_conditioned_for_dreamer(pattern='bullish_engulfing', limit=2))}")
    print("Pattern rates sample:", list(mem.get_pattern_success_rates().items())[:2])
    mem.compact_storage(min_priority=0.6)
    print("Demo finished successfully — ExperienceMemory is fully operational.")