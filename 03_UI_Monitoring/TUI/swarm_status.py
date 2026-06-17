#!/usr/bin/env python3
"""
Swarm Status Reporter — Lightweight status sharing for parallel specialized agents.

Agents (human or automated subagents) use this to publish high-level visibility:
  phase, current focus, blockers, etc.

This powers the "Swarm Status" section in monitor_tui.py.

Grok subagent bridge: --sync-grok (or sync_grok_swarm()) pulls the live swarm of
Grok-launched workers (the 30+ specialized background agents on the MT5 pipeline)
from ~/.grok session metas and surfaces them in the exact same shared files / TUI.
Makes the full autonomous swarm observable and coordinated from one place.

Usage (CLI - easy for any agent / shell / PS1):
    python scripts/swarm_status.py --name "MQL5 Execution Lead" \
        --workstream "MQL5 Execution Layer" \
        --phase "Phase 1: Skeleton" \
        --focus "Mapping PPO + LSTM weights to MQL5 CNeuronLSTM" \
        --blockers "Shape mismatch on LSTM; Need sample export weights" \
        --status in_progress --progress 45

    python scripts/swarm_status.py --name "Training Hardener" --status blocked \
        --blockers "KL explosion on first 50k post-fix" --focus "Raising target_kl + LR tuning"

    python scripts/swarm_status.py --list   # show what TUI will see

    python scripts/swarm_status.py --sync-grok   # bridge Grok sub-swarm (the real agents) into visibility

    python scripts/swarm_status.py --clear-stale

Python import (for agents/scripts inside the repo):
    from scripts.swarm_status import report_status, get_active_agents, sync_grok_swarm, get_grok_subagents
    report_status(
        name="Evidence Curator",
        workstream="Go/No-Go & Docs",
        phase="Daily Update",
        current_focus="Synthesizing MQL5 + training evidence into WINDOWS_PRODUCTION_GO_NO_GO_ASSESSMENT.md",
        blockers=[],
        status="in_progress",
        progress=70
    )
    agents = get_active_agents(max_age_seconds=14400)  # last 4 hours

Status file location (shared, no lock contention):
    runtime/agent_status/<slug-of-name>.json

Status values: idle | in_progress | blocked | complete | error
Blockers: list or comma-separated string (CLI normalizes to list)

Keep it lightweight. No dependencies beyond stdlib + optional rich for --list.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).parent.parent.resolve()
STATUS_DIR = REPO_ROOT / "runtime" / "agent_status"
MAX_AGE_DEFAULT = 4 * 3600  # 4 hours — agents are expected to heartbeat reasonably often


def _slug(name: str) -> str:
    """Filesystem-safe slug for the JSON filename."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())
    return s[:80] or "unnamed_agent"


def _ensure_dir() -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    return STATUS_DIR


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_blockers(blockers: Any) -> List[str]:
    if blockers is None:
        return []
    if isinstance(blockers, list):
        return [str(b).strip() for b in blockers if str(b).strip()]
    if isinstance(blockers, str):
        # CLI friendly: "foo, bar; baz" or "foo|bar"
        parts = re.split(r"[,;|\n]+", blockers)
        return [p.strip() for p in parts if p.strip()]
    return [str(blockers)]


def report_status(
    name: str,
    workstream: str = "",
    phase: str = "",
    current_focus: str = "",
    blockers: Any = None,
    status: str = "in_progress",
    progress: Optional[int] = None,
    notes: str = "",
) -> Path:
    """
    Write (or overwrite) the agent's status JSON.
    Returns the path written. Safe for concurrent writers (different files).
    """
    if not name or not name.strip():
        raise ValueError("Agent name is required")

    _ensure_dir()
    slug = _slug(name)
    path = STATUS_DIR / f"{slug}.json"

    payload: Dict[str, Any] = {
        "name": name.strip(),
        "workstream": workstream.strip() if workstream else "",
        "phase": phase.strip() if phase else "",
        "current_focus": current_focus.strip() if current_focus else "",
        "blockers": _normalize_blockers(blockers),
        "status": (status or "in_progress").strip().lower(),
        "last_updated": _now_iso(),
        # TUI Feature Parity: operational fields matching React AgentOperationalStatus / AgentsPanel
        "error_count": 0,
        "current_task": current_focus.strip() if current_focus else "",
        "heartbeat": _now_iso(),
        "last_artifact": "",
    }
    if progress is not None:
        try:
            p = max(0, min(100, int(progress)))
            payload["progress"] = p
        except Exception:
            pass
    if notes:
        payload["notes"] = str(notes).strip()[:500]

    # Atomic-ish write (tmp + rename) to reduce partial reads
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        # Fallback — still try direct write
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise e

    return path


def get_active_agents(max_age_seconds: int = MAX_AGE_DEFAULT) -> List[Dict[str, Any]]:
    """
    Scan the status directory and return currently active agents (recent heartbeats).
    Returns list of dicts sorted by last_updated descending (most recent first).
    Stale or unreadable files are ignored.
    """
    _ensure_dir()
    now = time.time()
    agents: List[Dict[str, Any]] = []

    for p in sorted(STATUS_DIR.glob("*.json")):
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            if not isinstance(data, dict):
                continue

            # Basic normalization / safety
            agent = {
                "name": str(data.get("name", p.stem)),
                "workstream": str(data.get("workstream", "")),
                "phase": str(data.get("phase", "")),
                "current_focus": str(data.get("current_focus", "")),
                "blockers": [str(b) for b in (data.get("blockers") or []) if str(b).strip()],
                "status": str(data.get("status", "unknown")).lower(),
                "last_updated": str(data.get("last_updated", "")),
                "progress": data.get("progress"),
                "notes": str(data.get("notes", "")),
                "file": str(p),
                # TUI parity fields (AgentOperationalStatus shape: error_count, current_task, heartbeat, last_artifact)
                "error_count": int(data.get("error_count", 0) or 0),
                "current_task": str(data.get("current_task", data.get("current_focus", ""))),
                "heartbeat": str(data.get("heartbeat", data.get("last_updated", ""))),
                "last_artifact": str(data.get("last_artifact", "")),
            }

            # Age filter
            ts = agent["last_updated"]
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    age = now - dt.timestamp()
                    if age > max_age_seconds:
                        continue
                    agent["_age_seconds"] = int(age)
                except Exception:
                    # If unparsable timestamp, still include but mark old
                    agent["_age_seconds"] = max_age_seconds + 1
            else:
                agent["_age_seconds"] = max_age_seconds + 1

            # Only include reasonably fresh
            if agent["_age_seconds"] <= max_age_seconds:
                agents.append(agent)
        except Exception:
            # Corrupt / race / permission — skip silently (TUI must be robust)
            continue

    # Sort most recent first
    agents.sort(key=lambda a: a.get("_age_seconds", 999999))
    return agents


def clear_stale(max_age_seconds: int = MAX_AGE_DEFAULT) -> int:
    """Remove stale status files. Returns count removed."""
    _ensure_dir()
    now = time.time()
    removed = 0
    for p in STATUS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            ts = data.get("last_updated", "")
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if now - dt.timestamp() > max_age_seconds:
                    p.unlink(missing_ok=True)
                    removed += 1
        except Exception:
            # Unparsable is considered stale for cleanup
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    return removed


# =============================================================================
# GROK SUBAGENT SWARM BRIDGE — Lightweight visibility for the autonomous swarm
# Grok spawns 30+ specialized subagents tracked in ~/.grok/sessions/.../subagents/*/meta.json
# These are opaque to the project TUI/supervisor. This bridge discovers the current
# supreme-chainsaw MT5 orchestration session + its children and exposes them via the
# same runtime/agent_status/*.json mechanism (or direct query).
#
# Usage (from any agent, supervisor, or cron):
#   python scripts/swarm_status.py --sync-grok
#   python -c "
#     from scripts.swarm_status import sync_grok_swarm, get_grok_subagents
#     print(sync_grok_swarm())          # populates files so TUI sees them
#     print(len(get_grok_subagents()))  # direct inspect
#   "
#
# This fulfills: high-level swarm status surfaced in TUI + supervisor,
# simple shared mechanism (the existing JSONs + auto-sync), observable multi-agent system.
# =============================================================================

GROK_SWARM_MAX_AGE_HOURS = 36.0


def _find_grok_home() -> Optional[Path]:
    """Locate ~/.grok (handles this VPS env + portable)."""
    candidates: List[Path] = [
        Path("C:/Users/Administrator/.grok"),
        Path.home() / ".grok",
    ]
    up = os.environ.get("USERPROFILE")
    if up:
        candidates.append(Path(up) / ".grok")
    gh = os.environ.get("GROK_HOME")
    if gh:
        candidates.append(Path(gh))
    for c in candidates:
        try:
            if c and c.exists() and (c / "sessions").exists():
                return c
        except Exception:
            continue
    return None


def _infer_workstream(text: str) -> str:
    """Heuristic workstream classification from Grok subagent description/prompt."""
    t = (text or "").lower()
    if any(k in t for k in ["mql5", "executor", "execution layer", "deploy_mql5", "chain_gambler"]):
        return "MQL5 Execution Layer"
    if any(k in t for k in ["training", "post-fix", "50k", "v4", "v5", "ppo", "kl explosion", "launch_robust"]):
        return "Training Pipeline"
    if any(k in t for k in ["promot", "gate", "champion", "canary", "handoff", "paper harness", "auto_promote"]):
        return "Champion / Canary / Promotion"
    if any(k in t for k in ["supervisor", "tui", "healthcheck", "ops", "operational", "visibility", "coordination", "swarm", "audit", "pipeline"]):
        return "Ops / Supervisor / Visibility"
    if any(k in t for k in ["hardening", "risk", "production readiness", "monitor"]):
        return "Operational Hardening"
    if any(k in t for k in ["doc", "evidence", "go/no-go", "design"]):
        return "Docs & Evidence"
    return "Grok Subagent Swarm"


def get_grok_subagents(max_age_hours: float = GROK_SWARM_MAX_AGE_HOURS) -> List[Dict[str, Any]]:
    """Return recent Grok subagents from the active supreme-chainsaw/MT5 parent session.

    Output shape compatible with get_active_agents() for unified consumption by TUI etc.
    Only the orchestration session (title containing supreme-chainsaw or MT5) is considered.
    """
    grok_home = _find_grok_home()
    if not grok_home:
        return []
    sessions_root = grok_home / "sessions"
    if not sessions_root.exists():
        return []

    now = time.time()
    max_age_s = max_age_hours * 3600.0
    mt5_sessions: List[tuple] = []

    # Walk: sessions/<encoded-cwd>/<session-uuid>/
    for cwd_dir in sessions_root.iterdir():
        if not cwd_dir.is_dir():
            continue
        for sess_dir in cwd_dir.iterdir():
            if not sess_dir.is_dir():
                continue
            # session ids look like 019e67...
            if not re.match(r"^[0-9a-f]{8}-", sess_dir.name, re.I):
                continue
            summary_p = sess_dir / "summary.json"
            if not summary_p.is_file():
                continue
            try:
                data = json.loads(summary_p.read_text(encoding="utf-8", errors="ignore"))
                title = str(data.get("generated_title") or data.get("session_summary", "")).lower()
                if not ("supreme-chainsaw" in title or ("mt5" in title and ("production" in title or "clone" in title))):
                    continue
                updated = str(data.get("updated_at") or data.get("last_active_at", ""))
                age = 1e9
                if updated:
                    try:
                        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        age = now - dt.timestamp()
                    except Exception:
                        pass
                if age < 48 * 3600:
                    mt5_sessions.append((age, sess_dir))
            except Exception:
                continue

    if not mt5_sessions:
        return []
    mt5_sessions.sort()  # smallest age first
    main_sess = mt5_sessions[0][1]
    subs_dir = main_sess / "subagents"
    if not subs_dir.exists():
        return []

    out: List[Dict[str, Any]] = []
    for meta_p in subs_dir.glob("*/meta.json"):
        if not meta_p.is_file():
            continue
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8", errors="ignore"))
            if not isinstance(meta, dict):
                continue
            sid = str(meta.get("subagent_id", meta_p.parent.name))[:12]
            desc = str(meta.get("description", "")).strip() or str(meta.get("prompt", ""))[:140].strip()
            if not desc:
                continue
            started = str(meta.get("started_at", ""))
            gstatus = str(meta.get("status", "unknown")).lower()
            toolc = int(meta.get("tool_calls", 0) or 0)
            turns = int(meta.get("turns", 0) or 0)
            dur_ms = int(meta.get("duration_ms", 0) or 0)

            age_s = 1e9
            if started:
                try:
                    dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    age_s = now - dt.timestamp()
                except Exception:
                    pass
            if age_s > max_age_s:
                continue

            ws = _infer_workstream(desc + " " + str(meta.get("prompt", "")))
            mapped_status = "complete" if gstatus == "completed" else ("blocked" if gstatus == "failed" else ("in_progress" if gstatus in ("running", "active") else gstatus))

            out.append({
                "name": f"Grok:{sid} {desc[:52]}",
                "workstream": ws,
                "phase": f"{gstatus} (turns={turns}, tools={toolc})",
                "current_focus": desc[:160],
                "blockers": ["Inspect Grok TUI Tasks pane (Ctrl+T) or session meta for details"] if gstatus in ("running", "active") else [],
                "status": mapped_status,
                "last_updated": started or _now_iso(),
                "progress": None,
                "notes": f"Grok subagent | {int(dur_ms/1000)}s | parent={main_sess.name[:8]}",
                "_age_seconds": int(age_s),
                "_source": "grok-meta",
            })
        except Exception:
            continue

    out.sort(key=lambda a: a.get("_age_seconds", 999999))
    return out


def sync_grok_swarm(max_age_hours: float = GROK_SWARM_MAX_AGE_HOURS, prefix: str = "Grok") -> int:
    """Discover Grok MT5 subagents and publish via the shared runtime/agent_status/ files.

    After this, get_active_agents() / --list / monitor_tui Swarm panel will include them.
    Call from supervisor loop, TUI startup, or manually. Idempotent and cheap.
    Returns count of agents published.
    """
    grok_as = get_grok_subagents(max_age_hours=max_age_hours)
    if not grok_as:
        return 0
    _ensure_dir()
    synced = 0
    for ga in grok_as:
        try:
            report_status(
                name=f"{prefix}: {ga['name']}",
                workstream=ga.get("workstream", "Grok Sub-Swarm"),
                phase=ga.get("phase", ""),
                current_focus=ga.get("current_focus", ""),
                blockers=ga.get("blockers", []),
                status=ga.get("status", "in_progress"),
                progress=ga.get("progress"),
                notes=ga.get("notes", "auto-synced from Grok subagent meta"),
            )
            synced += 1
        except Exception:
            pass
    return synced


def _rich_list() -> None:
    """Pretty print active agents (for CLI --list). Requires rich."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
    except ImportError:
        print("rich not installed — falling back to plain output.")
        _plain_list()
        return

    console = Console()
    agents = get_active_agents()
    # Always merge live Grok subagents for complete picture (no file writes needed for --list)
    try:
        grok_ones = get_grok_subagents()
        # de-dup by rough name match
        existing_names = {a["name"].lower() for a in agents}
        for g in grok_ones:
            if g["name"].lower() not in existing_names:
                agents.append(g)
        agents.sort(key=lambda a: a.get("_age_seconds", 999999))
    except Exception:
        pass

    if not agents:
        console.print(Panel(
            "[yellow]No active agents reporting.[/yellow]\n\n"
            "Agents report via:\n"
            "  python scripts/swarm_status.py --name \"...\" --focus \"...\" ...\n"
            "  or from Python: from scripts.swarm_status import report_status\n\n"
            "Grok sub-swarm (the real parallel agents): python scripts/swarm_status.py --sync-grok\n"
            "  (or auto-merged in --list / TUI for the current supreme-chainsaw MT5 session)",
            title="Swarm Status",
            border_style="dim"
        ))
        return

    table = Table(title="Active Swarm Agents (last 4h)", show_lines=False, expand=True)
    table.add_column("Agent", style="bold cyan")
    table.add_column("Workstream", style="blue")
    table.add_column("Phase", style="magenta")
    table.add_column("Focus", style="white")
    table.add_column("Blockers", style="red")
    table.add_column("Status", justify="center")
    table.add_column("Updated", style="dim")

    status_colors = {
        "in_progress": "green",
        "blocked": "red",
        "complete": "bright_green",
        "idle": "yellow",
        "error": "red",
    }

    for a in agents:
        blockers = "\n".join(a["blockers"]) if a["blockers"] else "—"
        status = a["status"]
        color = status_colors.get(status, "white")
        progress = ""
        if a.get("progress") is not None:
            progress = f" {a['progress']}%"

        updated = "just now"
        age = a.get("_age_seconds", 0)
        if age > 300:
            updated = f"{int(age/60)}m ago"

        table.add_row(
            a["name"],
            a["workstream"] or "—",
            a["phase"] or "—",
            Text(a["current_focus"] or "—", style="dim" if not a["current_focus"] else ""),
            Text(blockers, style="red" if a["blockers"] else "dim"),
            f"[{color}]{status}{progress}[/{color}]",
            updated,
        )

    console.print(table)
    console.print(f"[dim]{len(agents)} active agent(s) • runtime/agent_status/ + Grok bridge • heartbeat within last 4h[/dim]")


def _plain_list() -> None:
    agents = get_active_agents()
    try:
        grok_ones = get_grok_subagents()
        existing_names = {a["name"].lower() for a in agents}
        for g in grok_ones:
            if g["name"].lower() not in existing_names:
                agents.append(g)
        agents.sort(key=lambda a: a.get("_age_seconds", 999999))
    except Exception:
        pass
    if not agents:
        print("No active agents reporting (last 4h).")
        print("Use: python scripts/swarm_status.py --name \"Agent Name\" --focus \"...\"")
        return
    print(f"Active Swarm Agents ({len(agents)}):")
    for a in agents:
        src = " [GROK]" if a.get("_source") == "grok-meta" else ""
        print(f"  • {a['name']}{src} | {a['status']} | focus: {a['current_focus'][:60]}")
        if a["blockers"]:
            print(f"    BLOCKERS: {', '.join(a['blockers'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report or query Swarm Agent status for TUI visibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Agent reporting (recommended pattern)
  python scripts/swarm_status.py --name "MQL5 Execution Lead" \\
      --workstream "MQL5 Execution Layer" \\
      --phase "Phase 1" --focus "Weight export + architecture mapping" \\
      --blockers "LSTM shape mismatch, missing sample weights" \\
      --status in_progress --progress 30

  python scripts/swarm_status.py --list

  # Quick blocked report
  python scripts/swarm_status.py --name "Training Pipeline Hardener" \\
      --status blocked --blockers "Post-fix KL explosions — tuning LR/target_kl"

  # Cleanup (run from supervisor occasionally)
  python scripts/swarm_status.py --clear-stale

  # NEW: Make the entire Grok-launched swarm (30+ parallel subagents) visible in TUI/supervisor
  python scripts/swarm_status.py --sync-grok
  python scripts/swarm_status.py --sync-grok --grok-max-age 48
""",
    )
    parser.add_argument("--name", "-n", help="Agent / subagent name (required for reporting)")
    parser.add_argument("--workstream", "-w", default="", help="High-level workstream label")
    parser.add_argument("--phase", default="", help="Current phase or milestone")
    parser.add_argument("--focus", "--current-focus", dest="focus", default="", help="What the agent is focused on right now")
    parser.add_argument("--blockers", "-b", default="", help="Comma or semicolon separated list of current blockers")
    parser.add_argument("--status", default="in_progress", choices=["idle", "in_progress", "blocked", "complete", "error"],
                        help="Overall status (default: in_progress)")
    parser.add_argument("--progress", type=int, default=None, help="Optional 0-100 progress percent")
    parser.add_argument("--notes", default="", help="Optional short note")
    parser.add_argument("--list", "-l", action="store_true", help="List active agents (what the TUI sees)")
    parser.add_argument("--clear-stale", action="store_true", help="Remove status files older than 4h")
    parser.add_argument("--max-age", type=int, default=MAX_AGE_DEFAULT, help="Override max age in seconds for --list")
    parser.add_argument("--sync-grok", action="store_true", help="Bridge recent Grok subagents (supreme-chainsaw MT5 session) into runtime/agent_status/ so TUI + supervisor see the full swarm")
    parser.add_argument("--grok-max-age", type=float, default=GROK_SWARM_MAX_AGE_HOURS, help="Hours back to consider Grok subagents for --sync-grok")

    args = parser.parse_args()

    if args.clear_stale:
        count = clear_stale(args.max_age)
        print(f"Cleared {count} stale agent status file(s).")
        return

    if args.sync_grok:
        count = sync_grok_swarm(args.grok_max_age)
        print(f"Synced {count} Grok subagent(s) (from active MT5/supreme-chainsaw orchestration session) into shared swarm status.")
        print("  Now visible via --list, get_active_agents(), and monitor_tui.py Swarm Status panel.")
        if count == 0:
            print("  (No recent matching Grok session/subagents found — is the main grok-build session active?)")
        return

    if args.list:
        _rich_list() if "rich" in sys.modules or True else _plain_list()  # always try rich
        # The _rich_list already handles import gracefully
        try:
            _rich_list()
        except Exception:
            _plain_list()
        return

    if not args.name:
        parser.print_help()
        print("\n[error] --name is required when reporting status (or use --list / --clear-stale).")
        sys.exit(1)

    path = report_status(
        name=args.name,
        workstream=args.workstream,
        phase=args.phase,
        current_focus=args.focus,
        blockers=args.blockers,
        status=args.status,
        progress=args.progress,
        notes=args.notes,
    )
    print(f"Status reported for '{args.name}' → {path.relative_to(REPO_ROOT)}")
    print("TUI will pick it up on next refresh (usually < 10s).")


if __name__ == "__main__":
    main()
