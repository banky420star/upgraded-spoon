#!/usr/bin/env python3
"""
Mini Pipeline TUI (standalone light version) - Compact watcher for the FULL ingestion-to-champion-execution pipeline.

PREFERRED (integrated "improved TUI" from swarm): 
  python scripts/monitor_tui.py --mini --once     (or -m; 'p' toggles live)
  This is the canonical dense mini full-pipeline watcher (data ingest → rich Decision PPO + TimeExitSpec → ExecutionAgent → loop).

This standalone is a zero-dep ultra-light alternative for tiny terminals.
Covers: ingestion/MTF/timing/patterns, Decision PPO 18-dim rich (TimeExitSpec), ExecutionAgent pure-py telemetry, PIPELINE_DECISIONS, swarm agents, etc.

Focus: Zero-touch autonomous flow with Decision PPO (rich 18-dim TradeDecision),
classical patterns (doji/hammer/engulfing/flags/breakouts), timing (news/opens),
Rainforest + Dreamer conditioning, gates, promotion, pure-Python ExecutionAgent rich execution,
and telemetry back to journal/PIPELINE/retrain.

Single dense screen. Fast refresh. --once for CI/one-shot. Works in small terminals.

Run (standalone):
  .\\.venv312\\Scripts\\python.exe scripts/mini_pipeline_tui.py
  .\\.venv312\\Scripts\\python.exe scripts/mini_pipeline_tui.py --once

Key watched artifacts (all written by autonomous agents/launcher/harness/ExecutionAgent):
- logs/PIPELINE_DECISIONS.jsonl
- runtime/agent_status/*.json (esp. decision_ppo_*, e2e_*, handoff_*)
- runtime/execution_reports/*.json (rich TradeDecision + TimeExitSpec + pattern tags)
- logs/decision_ppo_*.log + logs/drl_joint/PPO_*
- logs/trade_journal.jsonl + logs/execution_feedback.jsonl
- models/registry/candidates/*
- runtime/last_handoff.json + runtime/paper_harness_start.json
- (future) *_timing_insights.json from trade_timing_analyzer
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    print("rich not installed. pip install rich")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

console = Console()

# Watched paths
PIPELINE_DECISIONS = PROJECT_ROOT / "logs" / "PIPELINE_DECISIONS.jsonl"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
AGENT_STATUS_DIR = RUNTIME_DIR / "agent_status"
EXEC_REPORTS_DIR = RUNTIME_DIR / "execution_reports"
CANDIDATES_DIR = PROJECT_ROOT / "models" / "registry" / "candidates"
LAST_HANDOFF = PROJECT_ROOT / "last_handoff.json"
PAPER_HARNESS_START = RUNTIME_DIR / "paper_harness_start.json"
TRAINING_HEALTH = RUNTIME_DIR / "training_health.json"
TRADE_JOURNAL = PROJECT_ROOT / "logs" / "trade_journal.jsonl"
EXEC_FEEDBACK = PROJECT_ROOT / "logs" / "execution_feedback.jsonl"

# Recent decision_ppo training logs (any)
DECISION_PPO_LOGS = list((PROJECT_ROOT / "logs").glob("decision_ppo_*.log")) + \
                    list((PROJECT_ROOT / "logs").glob("decision_ppo_XAU*.log"))

REFRESH_SEC = 2.0
MAX_PIPELINE_EVENTS = 12
MAX_EXEC_REPORTS = 6
MAX_AGENT_STATUSES = 8


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _tail_jsonl(path: Path, n: int = 50) -> List[dict]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        out = []
        for line in lines[-n:]:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
        return out
    except Exception:
        return []


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _load_latest_agent_statuses() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not AGENT_STATUS_DIR.exists():
        return items
    for p in sorted(AGENT_STATUS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:MAX_AGENT_STATUSES]:
        data = _load_json(p)
        if data:
            data["_file"] = p.name
            data["_mtime"] = datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S")
            items.append(data)
    return items


def _load_recent_exec_reports() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not EXEC_REPORTS_DIR.exists():
        return items
    for p in sorted(EXEC_REPORTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:MAX_EXEC_REPORTS]:
        data = _load_json(p)
        if data:
            data["_file"] = p.name
            data["_mtime"] = datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S")
            items.append(data)
    return items


def _get_training_progress() -> Dict[str, Any]:
    """Extract live decision_ppo progress from health + recent log tails."""
    prog: Dict[str, Any] = {"status": "NO ACTIVE RUN", "step": 0, "pct": 0, "symbol": "XAUUSDm", "timesteps": 50000}
    h = _load_json(TRAINING_HEALTH)
    if h:
        prog["status"] = h.get("status", "RUNNING")
        prog["step"] = h.get("current_step", 0)
        prog["pct"] = h.get("progress_pct", 0)
        prog["symbol"] = h.get("symbol", prog["symbol"])
        if "timesteps" in h:
            prog["timesteps"] = h["timesteps"]

    # Fallback: scan latest decision_ppo log for "step=XXXX/50000"
    for logp in sorted(DECISION_PPO_LOGS, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
        try:
            tail = logp.read_text(errors="ignore").splitlines()[-30:]
            for line in reversed(tail):
                if "step=" in line and "/50,000" in line:
                    # Parse "step=10,000/50,000 | pct=20.00"
                    try:
                        parts = line.split("|")
                        for p in parts:
                            if "step=" in p:
                                s = p.split("step=")[1].split("/")[0].replace(",", "").strip()
                                prog["step"] = int(s)
                            if "pct=" in p:
                                prog["pct"] = float(p.split("pct=")[1].split()[0])
                        prog["status"] = "TRAINING (decision_ppo 18-dim + patterns + timing)"
                        break
                    except Exception:
                        pass
        except Exception:
            pass
        if prog["step"] > 0:
            break
    return prog


def _get_latest_candidate() -> Optional[Dict[str, Any]]:
    if not CANDIDATES_DIR.exists():
        return None
    cands = sorted(CANDIDATES_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not cands:
        return None
    latest = cands[0]
    meta = latest / "metrics.json" if (latest / "metrics.json").exists() else None
    info = {"name": latest.name, "path": str(latest), "mtime": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%H:%M")}
    if meta:
        m = _load_json(meta)
        if m:
            info["sharpe"] = m.get("sharpe") or m.get("test_sharpe")
            info["pnl"] = m.get("total_pnl") or m.get("test_pnl")
    return info


def _get_last_handoff() -> Optional[Dict[str, Any]]:
    d = _load_json(LAST_HANDOFF)
    if d:
        d["_file"] = "last_handoff.json"
    return d


def _get_paper_harness_state() -> Optional[Dict[str, Any]]:
    d = _load_json(PAPER_HARNESS_START)
    if d:
        d["_file"] = "paper_harness_start.json"
    return d


def _recent_pipeline_events() -> List[Dict[str, Any]]:
    events = _tail_jsonl(PIPELINE_DECISIONS, 80)
    # Prefer decision_ppo / promotion / execution / handoff events
    prio = []
    for e in reversed(events):
        dt = e.get("decision_type") or e.get("event") or ""
        if any(k in str(dt).lower() for k in ["decision_ppo", "trade_decision", "promotion", "handoff", "execution_agent", "champion", "rich"]):
            prio.append(e)
        if len(prio) >= MAX_PIPELINE_EVENTS:
            break
    if not prio:
        prio = list(reversed(events[-MAX_PIPELINE_EVENTS:]))
    return prio


def _render_header() -> Panel:
    title = Text("SUPREME CHAINSAW - MINI FULL PIPELINE WATCHER", style="bold cyan")
    sub = Text(f"Ingestion -> DecisionPPO (patterns+timing) -> Gates -> Champion Execution (pure-Python rich TradeDecision)  |  {_now()}", style="dim")
    return Panel(Group(title, sub), box=box.ROUNDED, style="cyan", padding=(0,1))


def _render_training_panel(prog: Dict[str, Any], cand: Optional[Dict[str, Any]]) -> Panel:
    t = Table.grid(expand=True)
    t.add_column("k", style="bold yellow", width=18)
    t.add_column("v", style="white")

    t.add_row("Status", prog.get("status", "UNKNOWN"))
    t.add_row("Symbol / Action", f"{prog.get('symbol','?')} | decision_ppo 18-dim")
    step = prog.get("step", 0)
    tot = prog.get("timesteps", 50000)
    pct = prog.get("pct", 0)
    t.add_row("Progress", f"{step:,}/{tot:,}  ({pct:.1f}%)")
    if cand:
        t.add_row("Latest Candidate", f"{cand['name']} @ {cand.get('mtime','?')}  sharpe={cand.get('sharpe')}")
    else:
        t.add_row("Latest Candidate", "none newer than baseline (awaiting training finish + gates)")

    # Pattern / timing note
    t.add_row("Enriched Obs", "MTF(1m+5m+15m+1h) + timing(news/open) + 11 classical patterns (wired in feature_pipeline)")

    return Panel(t, title="TRAINING (Decision PPO + Rainforest/Dreamer + Patterns + Timing)", box=box.ROUNDED, style="green")


def _render_execution_panel(reports: List[Dict[str, Any]], harness: Optional[Dict[str, Any]]) -> Panel:
    t = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE, expand=True)
    t.add_column("Time", width=8)
    t.add_column("Decision", width=22)
    t.add_column("Rich Spec (Size/Exit/Trail/TimeExit)", width=55)
    t.add_column("Backend", width=12)

    if not reports:
        t.add_row("-", "NO RECENT RICH EXEC REPORTS", "Run harness or decision_ppo candidate for live paper trades", "-")
    else:
        for r in reports:
            ts = r.get("_mtime", "?")
            dec = r.get("decision") or r.get("trade_decision") or {}
            did = dec.get("decision_id", r.get("_file","")[:12])
            side = dec.get("side", "?")
            size_spec = dec.get("size", {}) or {}
            size = size_spec.get("risk_pct_equity") or size_spec.get("fixed_lots") or dec.get("size", 0.01)
            exit_spec = dec.get("exit", {}) or {}
            tp = exit_spec.get("tp_type", exit_spec.get("tp", "?"))
            sl = exit_spec.get("sl_type", "?")
            trail = (dec.get("trailing") or {}).get("type", "NONE")
            time_exit = dec.get("time_exit") or dec.get("TimeExitSpec") or {}
            te_flags = []
            if time_exit.get("close_before_high_impact_news"): te_flags.append("news")
            if time_exit.get("close_at_session_end"): te_flags.append("session")
            if time_exit.get("max_hold_minutes"): te_flags.append(f"max{time_exit.get('max_hold_minutes')}m")
            te_str = "+".join(te_flags) if te_flags else "none"
            rich = f"{side} sz={size} SL={sl} TP={tp} trail={trail} time_exit=[{te_str}]"
            backend = r.get("backend", r.get("executor", "python_order_manager"))
            t.add_row(ts, did, rich, str(backend)[:12])

    foot = ""
    if harness:
        foot = f"Harness armed: execution_type={harness.get('execution_type','decision_ppo')}  uses_rich={harness.get('uses_rich_decision',True)}"
    return Panel(Group(t, Text(foot, style="dim")), title="CHAMPION EXECUTION (ExecutionAgent pure-Python + rich TradeDecision + TimeExitSpec)", box=box.ROUNDED, style="magenta")


def _render_pipeline_events(events: List[Dict[str, Any]]) -> Panel:
    t = Table(show_header=True, header_style="bold blue", box=box.SIMPLE, expand=True)
    t.add_column("TS", width=16)
    t.add_column("Type", width=18)
    t.add_column("Actor / Decision", width=28)
    t.add_column("Details (timing/patterns/execution)", width=40)

    if not events:
        t.add_row("-", "NO PIPELINE DECISIONS YET", "Launch training or harness to populate", "")
    else:
        for e in events:
            ts = str(e.get("ts") or e.get("timestamp", ""))[:16]
            dt = str(e.get("decision_type") or e.get("event", ""))[:17]
            actor = str(e.get("actor", ""))[:26]
            det = e.get("details") or e.get("reason") or e.get("decision") or ""
            if isinstance(det, dict):
                # Extract rich bits
                sym = det.get("symbol", "")
                src = det.get("source", "")
                timing = det.get("timing_tags") or det.get("time_exit") or ""
                det = f"{sym} {src} {timing}" if sym or src else str(det)[:60]
            t.add_row(ts, dt, actor, str(det)[:55])
    return Panel(t, title="PIPELINE_DECISIONS.jsonl (full flow: ingestion->train->gates->champion exec)", box=box.ROUNDED, style="blue")


def _render_swarm_panel(statuses: List[Dict[str, Any]]) -> Panel:
    t = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE, expand=True)
    t.add_column("Agent File", width=28)
    t.add_column("Status", width=12)
    t.add_column("Updated", width=9)
    t.add_column("Key Note (patterns/timing/execution/loop)", width=45)

    if not statuses:
        t.add_row("-", "NO AGENTS", "-", "Spawn agents via swarm or they auto-write on task start")
    else:
        for s in statuses:
            name = s.get("_file", "?")[:26]
            st = str(s.get("status", s.get("state", "-")))[:11]
            mt = s.get("_mtime", "?")
            note = ""
            if "decision_ppo" in str(s).lower() or "execution" in str(s).lower():
                note = "decision_ppo + rich TradeDecision + pure py primary"
            elif "pattern" in str(s).lower() or "timing" in str(s).lower():
                note = "PatternDetector (12 patterns) + timing wired to Rainforest/Dreamer/PPO"
            elif "e2e" in name.lower() or "pipeline" in str(s).lower():
                note = "E2E loop armed: watcher->promoter->harness->ExecutionAgent"
            elif "handoff" in name.lower():
                note = "handoff_watcher detecting decision_ppo candidates"
            else:
                note = str(s.get("description", ""))[:44]
            t.add_row(name, st, mt, note[:44])
    return Panel(t, title="SWARM / AGENTS (full pipeline coverage)", box=box.ROUNDED, style="cyan")


def _render_footer() -> Text:
    return Text(
        "Pure-Python primary (no MQL5 needed on Windows+direct MT5).  DecisionPPO rich specs (Size/Exit/Trail/TimeExit + patterns+timing) flow to ExecutionAgent.  "
        "First good candidate from current XAU run will auto-promote + execute rich timed trades.  Ctrl+C to exit.  --once for snapshot.",
        style="dim"
    )


def build_layout() -> Group:
    prog = _get_training_progress()
    cand = _get_latest_candidate()
    reports = _load_recent_exec_reports()
    harness = _get_paper_harness_state()
    events = _recent_pipeline_events()
    swarm = _load_latest_agent_statuses()

    header = _render_header()
    train = _render_training_panel(prog, cand)
    execp = _render_execution_panel(reports, harness)
    pipe = _render_pipeline_events(events)
    swarm_p = _render_swarm_panel(swarm)
    foot = _render_footer()

    return Group(header, train, execp, pipe, swarm_p, foot)


def main(once: bool = False):
    if once:
        console.print(build_layout())
        return

    live = Live(build_layout(), console=console, refresh_per_second=1.0, screen=True)
    live.start()
    try:
        while True:
            time.sleep(REFRESH_SEC)
            live.update(build_layout())
    except KeyboardInterrupt:
        live.stop()
        console.print("\n[cyan]Mini pipeline watcher stopped.[/cyan]")


if __name__ == "__main__":
    once = "--once" in sys.argv or "-1" in sys.argv
    main(once=once)
