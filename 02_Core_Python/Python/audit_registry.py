#!/usr/bin/env python3
"""
Registry audit script for Chain Gambler model registry.
Scores all candidates, validates champions against promotion gates,
and reports demotion/promotion recommendations.

Promotion gates:
  - data_source == "mt5"
  - timesteps >= 10000
  - backtest return >= 0 (or evaluation total_return >= 0)
  - max_drawdown <= 0.15
"""

import json
import os
import sys
from pathlib import Path

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Python.model_registry import ModelRegistry


def score_candidate(meta: dict, scorecard: dict, evaluation: dict | None) -> dict:
    """Return a scored summary for a candidate directory."""
    data_source = (meta.get("data_source") or scorecard.get("data_source") or "unknown")
    timesteps = int(meta.get("timesteps") or scorecard.get("timesteps") or 0)

    # Prefer evaluation metrics if present
    total_return = 0.0
    max_drawdown = 0.0
    if evaluation and isinstance(evaluation, dict):
        per_symbol = evaluation.get("per_symbol", [])
        if per_symbol and isinstance(per_symbol, list):
            total_return = float(per_symbol[0].get("total_return", 0.0))
            max_drawdown = float(per_symbol[0].get("max_drawdown", 0.0))
        else:
            total_return = float(evaluation.get("total_return", 0.0))
            max_drawdown = float(evaluation.get("max_drawdown", 0.0))

    # Build gate checks
    gates = {
        "data_source_mt5": data_source == "mt5",
        "timesteps_ge_10k": timesteps >= 10000,
        "backtest_return_non_negative": total_return >= 0.0,
        "max_drawdown_le_15pct": max_drawdown <= 0.15,
    }
    passed = all(gates.values())
    reasons = []
    if not gates["data_source_mt5"]:
        reasons.append(f"data_source_fail:{data_source}!=mt5")
    if not gates["timesteps_ge_10k"]:
        reasons.append(f"timesteps_fail:{timesteps}<10000")
    if not gates["backtest_return_non_negative"]:
        reasons.append(f"backtest_return_fail:{total_return:.4f}<0.0000")
    if not gates["max_drawdown_le_15pct"]:
        reasons.append(f"max_drawdown_fail:{max_drawdown:.4f}>0.1500")

    return {
        "symbol": meta.get("symbol") or scorecard.get("symbol"),
        "data_source": data_source,
        "timesteps": timesteps,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "gates_passed": passed,
        "gate_reasons": reasons,
        "gate_details": gates,
    }


def audit():
    registry = ModelRegistry()
    candidates_dir = registry.candidates_dir
    active = registry._read_active()

    # Hardened integrity audit (sprint improvement)
    integrity_report = registry.audit_integrity()
    print("=" * 80)
    print("CHAIN GAMBLER MODEL REGISTRY AUDIT (with hardened integrity + promotion trail)")
    print("=" * 80)
    print(f"Integrity audit @ {integrity_report.get('timestamp')}")
    print(f"  Lock timeout: {integrity_report.get('lock_timeout')}s")
    if integrity_report.get("failures"):
        print(f"  FAILURES DETECTED: {integrity_report['failures']}")
    else:
        print("  All known champion/canary entries passed integrity (hash+size).")

    # Discover all candidates
    candidate_dirs = []
    if os.path.isdir(candidates_dir):
        for entry in sorted(os.listdir(candidates_dir)):
            path = os.path.join(candidates_dir, entry)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "metadata.json")):
                candidate_dirs.append(path)

    print("=" * 80)
    print("CHAIN GAMBLER MODEL REGISTRY AUDIT")
    print("=" * 80)

    # Score every candidate
    scored = []
    for cdir in candidate_dirs:
        meta = registry.read_metadata(cdir)
        scorecard = registry._read_scorecard(cdir)
        evaluation = meta.get("evaluation") if isinstance(meta, dict) else None
        summary = score_candidate(meta, scorecard, evaluation)
        summary["dir"] = cdir
        scored.append(summary)

    print(f"\nCandidates audited: {len(scored)}\n")
    print(f"{'Directory':<50} {'Symbol':<10} {'Source':<10} {'Timesteps':<12} {'Return':<12} {'Drawdown':<12} {'Safe':<6}")
    print("-" * 120)
    for s in scored:
        dirname = os.path.basename(s["dir"])
        safe = "YES" if s["gates_passed"] else "NO"
        print(
            f"{dirname:<50} {s['symbol'] or '?' :<10} {s['data_source']:<10} "
            f"{s['timesteps']:<12} {s['total_return']:<12.4f} {s['max_drawdown']:<12.4f} {safe:<6}"
        )
        if s["gate_reasons"]:
            for r in s["gate_reasons"]:
                print(f"  -> {r}")

    # Audit current champions
    print("\n" + "=" * 80)
    print("CURRENT CHAMPIONS AUDIT")
    print("=" * 80)

    global_champion = active.get("champion")
    symbols = active.get("symbols", {})

    demotions = []
    promotions = []

    # Global champion
    if global_champion:
        meta = registry.read_metadata(global_champion)
        scorecard = registry._read_scorecard(global_champion)
        evaluation = meta.get("evaluation") if isinstance(meta, dict) else None
        summary = score_candidate(meta, scorecard, evaluation)
        summary["role"] = "global champion"
        summary["dir"] = global_champion
        print(f"\nGlobal champion: {global_champion}")
        print(f"  Safe: {'YES' if summary['gates_passed'] else 'NO'}")
        if summary["gate_reasons"]:
            for r in summary["gate_reasons"]:
                print(f"  -> {r}")
        if not summary["gates_passed"]:
            demotions.append(summary)
    else:
        print("\nGlobal champion: None")

    # Per-symbol champions
    for sym, cfg in symbols.items():
        champ = cfg.get("champion")
        if champ:
            meta = registry.read_metadata(champ)
            scorecard = registry._read_scorecard(champ)
            evaluation = meta.get("evaluation") if isinstance(meta, dict) else None
            summary = score_candidate(meta, scorecard, evaluation)
            summary["role"] = f"{sym} champion"
            summary["dir"] = champ
            print(f"\n{sym} champion: {champ}")
            print(f"  Safe: {'YES' if summary['gates_passed'] else 'NO'}")
            if summary["gate_reasons"]:
                for r in summary["gate_reasons"]:
                    print(f"  -> {r}")
            if not summary["gates_passed"]:
                demotions.append(summary)

            # Look for a safe replacement candidate for this symbol
            symbol_safe = [s for s in scored if s["gates_passed"] and s.get("symbol") == sym]
            if not summary["gates_passed"] and symbol_safe:
                best = max(symbol_safe, key=lambda x: x["timesteps"])
                promotions.append({"symbol": sym, "from": champ, "to": best["dir"], "reason": "safe replacement found"})
        else:
            print(f"\n{sym} champion: None")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if demotions:
        print("\nDEMOTE:")
        for d in demotions:
            print(f"  - {d['role']} ({os.path.basename(d['dir'])}) because: {', '.join(d['gate_reasons'])}")
    else:
        print("\nNo demotions required.")

    if promotions:
        print("\nPROMOTE:")
        for p in promotions:
            print(f"  - {p['symbol']}: {os.path.basename(p['from'])} -> {os.path.basename(p['to'])} ({p['reason']})")
    else:
        print("\nNo promotions recommended.")

    # Write audit report JSON alongside active.json
    report = {
        "audited_at": registry._timestamp_version(),
        "candidates": scored,
        "demotions": demotions,
        "promotions": promotions,
    }
    report_path = os.path.join(registry.root, "audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit report written to: {report_path}")


if __name__ == "__main__":
    audit()
