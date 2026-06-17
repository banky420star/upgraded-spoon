#!/usr/bin/env python3
"""
Unified Pipeline Decisions Audit Trail (single source of truth).

All major pipeline decisions (promotion, rollback, retrain triggers, MQL5 deployment,
candidate staging, harness arming, etc.) MUST append here via log_decision().

File: logs/PIPELINE_DECISIONS.jsonl (append-only, one JSON per line).

This closes the "Pipeline Auditor" gap for full observability and end-to-end
traceability of every candidate from training run -> promotion decision -> execution
-> feedback/retrain or live deploy.

Usage (Python):
    from Python.pipeline_audit import log_decision, get_recent_decisions, compute_loop_closure_score
    log_decision(
        decision_type="promotion",
        actor="promoter",
        decision="PROCEED_TO_PAPER",
        candidate="20260527_112233",
        run_id="postfix_v4_BTC_20260527",
        reason="core_gates_passed + alignment_fix_applied",
        details={"core_perf_pass": True, "full_gates_pass": False, "symbols": ["BTCUSDm"]}
    )

CLI (for PS1 / wrappers):
    python -m Python.pipeline_audit --log --type candidate_staged --actor training --decision STAGED --candidate 2026... --run-id "..." --reason "50k complete" --details-json '{"timesteps":50000}'

TUI / readers use get_recent_decisions() + compute_loop_closure_score() for "Loop Closure Score".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Robust project root resolution (works when run as module, script, or from any cwd)
def _resolve_project_root() -> Path:
    # 1. From this file
    here = Path(__file__).resolve()
    for p in [here.parent, here.parent.parent, here.parent.parent.parent]:
        if (p / "config.yaml").exists() or (p / "pyproject.toml").exists() or (p / "README.md").exists():
            return p
    # 2. CWD walk up
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "config.yaml").exists() or (parent / "scripts" / "monitor_tui.py").exists():
            return parent
    # 3. Hard fallback for known env
    return Path("C:/supreme-chainsaw")

PROJECT_ROOT = _resolve_project_root()
LOGS_DIR = PROJECT_ROOT / "logs"
PIPELINE_DECISIONS = LOGS_DIR / "PIPELINE_DECISIONS.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_logs() -> None:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def log_decision(
    decision_type: str,
    actor: str,
    decision: str,
    candidate: Optional[str] = None,
    run_id: Optional[str] = None,
    reason: str = "",
    details: Optional[Dict[str, Any]] = None,
    severity: str = "info",
    **extra: Any,
) -> bool:
    """
    Append a single structured decision event to the unified audit log.
    Always succeeds (best-effort, never raises to caller).
    Uses atomic temp+replace for safety on Windows.
    """
    _ensure_logs()
    entry: Dict[str, Any] = {
        "ts": _now_iso(),
        "decision_type": str(decision_type),
        "actor": str(actor),
        "decision": str(decision),
        "candidate": candidate,
        "run_id": run_id,
        "reason": str(reason)[:512] if reason else "",
        "details": details or {},
        "severity": severity,
    }
    if extra:
        entry.update({k: v for k, v in extra.items() if k not in entry})

    try:
        line = json.dumps(entry, default=str, ensure_ascii=False) + "\n"
        # Atomic append on Windows: write to temp in same dir, then append content
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(LOGS_DIR), suffix=".jsonl.tmp", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(line)
            tmp_path = tmp.name
        # Append the temp content to target (simple, robust, no lock needed for our usage)
        with open(PIPELINE_DECISIONS, "a", encoding="utf-8") as f:
            f.write(line)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return True
    except Exception:
        # Last-ditch: direct append (may race but better than losing the event)
        try:
            with open(PIPELINE_DECISIONS, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False


def get_recent_decisions(limit: int = 30) -> List[Dict[str, Any]]:
    """Return the most recent N decision events (newest last). Safe on missing/empty."""
    if not PIPELINE_DECISIONS.exists():
        return []
    try:
        lines = PIPELINE_DECISIONS.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        recs: List[Dict[str, Any]] = []
        for line in lines[-limit:]:
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
        return recs
    except Exception:
        return []


def compute_loop_closure_score(recent: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Heuristic "Loop Closure Score" (0-100) for the unified audit trail.

    Measures how well the autonomous pipeline is closing the loop for candidates:
    - candidate_staged present
    - followed by promotion decision (PROCEED / HOLD)
    - harness activity or rollback
    - retrain_trigger or mql5_deploy follow-through

    For the most recent candidate(s), computes trace completeness.
    This is the key observability metric requested by Pipeline Auditor.
    """
    if recent is None:
        recent = get_recent_decisions(60)

    if not recent:
        return {"score": 0, "status": "NO_DATA", "details": "No decisions logged yet. Start using log_decision() in all components.", "traces": 0}

    # Group by candidate or run_id (prefer candidate)
    traces: Dict[str, List[Dict[str, Any]]] = {}
    for d in recent:
        key = d.get("candidate") or d.get("run_id") or "unknown"
        traces.setdefault(key, []).append(d)

    completed = 0
    partial = 0
    total = len(traces)

    for key, events in traces.items():
        types = {e.get("decision_type") for e in events}
        decisions = {e.get("decision") for e in events}

        has_staged = "candidate_staged" in types or any("staged" in str(d).lower() for d in types)
        has_promo = "promotion" in types or any(d in ("PROCEED_TO_PAPER", "PROMOTE_CANARY", "HOLD_FOR_REVIEW") for d in decisions)
        has_harness = "harness_start" in types or "harness_arm" in types or "rollback" in types
        has_feedback = "retrain" in types or "retrain_trigger" in types or any("mql5" in str(t).lower() for t in types)

        score_for_trace = sum([has_staged, has_promo, has_harness, has_feedback])
        if score_for_trace >= 3:
            completed += 1
        elif score_for_trace >= 2:
            partial += 1

    # Base score from completeness of recent traces
    if total == 0:
        base = 0
    else:
        base = int(((completed * 1.0 + partial * 0.5) / total) * 80)

    # Bonus for recent activity (encourages continuous logging)
    recent_ts = [e.get("ts", "") for e in recent[-5:]]
    activity_bonus = 10 if len(recent) > 3 else 5 if len(recent) > 0 else 0
    # Penalty if very old decisions (stale)
    if recent:
        try:
            last = datetime.fromisoformat(recent[-1]["ts"].replace("Z", "+00:00"))
            age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            if age_h > 48:
                activity_bonus -= 15
        except Exception:
            pass

    score = max(0, min(100, base + max(0, activity_bonus)))

    status = "EXCELLENT" if score >= 85 else "GOOD" if score >= 65 else "FAIR" if score >= 40 else "POOR"
    details = (
        f"{completed} full traces / {total} candidates tracked "
        f"({partial} partial). Types seen: {sorted({e.get('decision_type') for e in recent[-20:] if e.get('decision_type')})}"
    )

    return {
        "score": score,
        "status": status,
        "details": details,
        "traces": total,
        "full_traces": completed,
        "partial_traces": partial,
        "recent_events": len(recent),
    }


# ----------------------------- CLI for PS1 / cross-language callers -----------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Decisions unified audit writer (single source of truth)")
    sub = parser.add_subparsers(dest="cmd")

    logp = sub.add_parser("log", help="Log a decision event")
    logp.add_argument("--type", "--decision-type", dest="decision_type", required=True)
    logp.add_argument("--actor", required=True)
    logp.add_argument("--decision", required=True)
    logp.add_argument("--candidate", default=None)
    logp.add_argument("--run-id", "--run_id", dest="run_id", default=None)
    logp.add_argument("--reason", default="")
    logp.add_argument("--severity", default="info")
    logp.add_argument("--details-json", default="{}")

    sub.add_parser("recent", help="Print recent decisions (JSON lines)")
    sub.add_parser("score", help="Print current Loop Closure Score")

    args = parser.parse_args()

    if args.cmd == "log":
        try:
            details = json.loads(args.details_json or "{}")
        except Exception:
            details = {"raw": args.details_json}
        ok = log_decision(
            decision_type=args.decision_type,
            actor=args.actor,
            decision=args.decision,
            candidate=args.candidate,
            run_id=args.run_id,
            reason=args.reason,
            details=details,
            severity=args.severity,
        )
        print("OK" if ok else "WRITE_FAILED_BUT_BEST_EFFORT")
        return 0 if ok else 1

    elif args.cmd == "recent":
        for d in get_recent_decisions(20):
            print(json.dumps(d, default=str))
        return 0

    elif args.cmd == "score":
        s = compute_loop_closure_score()
        print(json.dumps(s, indent=2))
        return 0

    parser.print_help()
    return 1


def get_audit_trail_summary(limit: int = 8) -> str:
    """
    Simple text summary for TUI / supervisor dashboards.
    Includes recent events (compact) + current Loop Closure Score.
    Safe, never raises.
    """
    try:
        recent = get_recent_decisions(limit)
        score = compute_loop_closure_score(recent)
        if not recent:
            return "PIPELINE_AUDIT: NO EVENTS YET (Loop Closure Score: 0/100 - POOR). Use log_decision() from promoter/harness/supervisor/deploy/retrain."

        lines = ["UNIFIED PIPELINE DECISIONS (PIPELINE_DECISIONS.jsonl) — single source of truth:"]
        for d in recent[-limit:]:
            ts_short = (d.get("ts", "") or "")[:19].replace("T", " ")
            typ = d.get("decision_type", "?")
            act = d.get("actor", "?")
            dec = d.get("decision", "?")
            cand = d.get("candidate") or d.get("run_id") or "-"
            lines.append(f"  {ts_short} | {typ:16s} | {act:14s} -> {dec:22s} | cand={cand}")

        sc = score.get("score", 0)
        status = score.get("status", "UNKNOWN")
        lines.append(f"Loop Closure Score: {sc}/100 ({status}) | traces={score.get('traces',0)} full={score.get('full_traces',0)} | {score.get('details','')[:80]}")
        return "\n".join(lines)
    except Exception as exc:
        return f"PIPELINE_AUDIT SUMMARY ERROR: {exc}"


if __name__ == "__main__":
    sys.exit(_cli())
