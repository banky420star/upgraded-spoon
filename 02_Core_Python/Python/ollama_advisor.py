"""
Local Ollama advisor layer for Chain Gambler.

This module gives the trading stack a local LLM council without letting any
language model directly place trades. It is designed for Windows VPS deployment
with Ollama running locally at http://127.0.0.1:11434.

Safe boundary:
- The advisor explains, reviews, summarizes, and proposes configuration patches.
- The deterministic risk engine, promotion gate, and MT5 executor remain the only
  components allowed to approve or execute trading actions.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import requests
import yaml

try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging

    logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATHS = [ROOT / "configs" / "ollama.yaml", ROOT / "config.yaml"]


DEFAULT_COUNCIL = {
    "code_architect": "qwen3-coder-next",
    "strategy_reviewer": "qwen3",
    "failure_detective": "deepseek-r1:14b",
    "chief_analyst": "llama3.3",
    "embedder": "nomic-embed-text",
}


@dataclass
class OllamaSettings:
    """Runtime settings for the local Ollama council."""

    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 90
    temperature: float = 0.15
    num_ctx: int = 8192
    council: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COUNCIL))
    max_prompt_chars: int = 24000
    allow_config_patch_suggestions: bool = True
    allow_live_trade_actions: bool = False


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot satisfy a request."""


class OllamaClient:
    """Small HTTP client for Ollama's local API."""

    def __init__(self, settings: OllamaSettings):
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        """Return local Ollama availability and model list."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=min(self.settings.timeout_seconds, 15),
            )
            response.raise_for_status()
            payload = response.json()
            models = [m.get("name") for m in payload.get("models", []) if m.get("name")]
            return {"ok": True, "base_url": self.base_url, "models": models}
        except Exception as exc:
            return {"ok": False, "base_url": self.base_url, "error": str(exc), "models": []}

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """Generate a non-streaming response from a local Ollama model."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": _limit_text(prompt, self.settings.max_prompt_chars),
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_ctx": self.settings.num_ctx,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get("response", "")).strip()
        except Exception as exc:
            raise OllamaError(f"Ollama generation failed for {model}: {exc}") from exc

    def embed(self, *, text: str, model: Optional[str] = None) -> List[float]:
        """Return an embedding vector using Ollama's embedding endpoint."""
        model_name = model or self.settings.council.get("embedder", DEFAULT_COUNCIL["embedder"])
        payload = {"model": model_name, "prompt": _limit_text(text, self.settings.max_prompt_chars)}
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            vector = data.get("embedding")
            if not isinstance(vector, list):
                raise OllamaError(f"No embedding returned by model {model_name}")
            return [float(x) for x in vector]
        except Exception as exc:
            raise OllamaError(f"Ollama embedding failed for {model_name}: {exc}") from exc


class OllamaAdvisor:
    """High-level local LLM council for trading review and operations support."""

    def __init__(self, settings: Optional[OllamaSettings] = None, client: Optional[OllamaClient] = None):
        self.settings = settings or load_ollama_settings()
        self.client = client or OllamaClient(self.settings)

    def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health and model coverage."""
        health = self.client.health()
        available = set(health.get("models", []))
        required = set(self.settings.council.values())
        missing = sorted(required - available)
        health["required_models"] = sorted(required)
        health["missing_models"] = missing
        health["ready"] = bool(health.get("ok")) and not missing
        return health

    def explain_trade_decision(self, trade_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Explain a proposed or historical trade decision in plain language."""
        prompt = {
            "task": "Explain this trading decision. Focus on risk, regime, signal quality, and why the risk engine should approve or block it.",
            "decision": trade_decision,
            "output_schema": {
                "summary": "string",
                "risk_flags": ["string"],
                "confidence_comment": "string",
                "actionability": "approve | block | monitor",
                "notes": ["string"],
            },
        }
        return self._json_council_call("strategy_reviewer", prompt)

    def review_risk_state(self, risk_state: Dict[str, Any]) -> Dict[str, Any]:
        """Review current risk-engine state without authorizing trades."""
        prompt = {
            "task": "Review this risk state. Identify whether the bot should keep trading, reduce size, pause, or hard-stop.",
            "risk_state": risk_state,
            "hard_rule": "Never suggest bypassing risk limits or increasing risk after drawdown.",
            "output_schema": {
                "status": "healthy | caution | reduce_size | pause | hard_stop",
                "reasons": ["string"],
                "risk_flags": ["string"],
                "recommended_operator_action": "string",
            },
        }
        return self._json_council_call("failure_detective", prompt)

    def analyze_backtest(self, report_path: Union[str, Path]) -> Dict[str, Any]:
        """Analyze a backtest, walk-forward report, or result CSV/text file."""
        path = Path(report_path)
        content = _read_text_safely(path)
        prompt = {
            "task": "Analyze this trading backtest/walk-forward evidence. Find overfit risk, drawdown danger, weak symbols, and promotion readiness.",
            "file": str(path),
            "content": content,
            "promotion_gate": {
                "min_trades": 100,
                "profit_factor_min": 1.3,
                "expectancy_must_be_positive": True,
                "must_pass_walk_forward": True,
                "must_pass_slippage_spread_stress": True,
            },
            "output_schema": {
                "verdict": "promote | reject | retest | paper_trade_only",
                "strengths": ["string"],
                "weaknesses": ["string"],
                "overfit_warnings": ["string"],
                "next_tests": ["string"],
            },
        }
        return self._json_council_call("failure_detective", prompt)

    def suggest_config_patch(
        self,
        config_yaml: Union[str, Dict[str, Any]],
        evidence: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Suggest a safe config patch. Does not write files or change trading state."""
        if not self.settings.allow_config_patch_suggestions:
            return {"allowed": False, "reason": "Config patch suggestions disabled."}

        prompt = {
            "task": "Suggest a safe YAML config patch for this trading system. Do not increase risk caps beyond current values. Prefer safer gates, lower exposure, better pauses, and clearer telemetry.",
            "config_yaml": config_yaml,
            "evidence": evidence or {},
            "output_schema": {
                "summary": "string",
                "patch_yaml": "string",
                "risk_impact": "lower | neutral | higher",
                "requires_backtest": True,
                "notes": ["string"],
            },
        }
        return self._json_council_call("strategy_reviewer", prompt)

    def summarize_daily_session(
        self,
        logs: Union[str, Iterable[str]],
        trade_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize daily logs for Telegram/dashboard operator review."""
        if isinstance(logs, str):
            log_text = logs
        else:
            chunks = []
            for path in logs:
                chunks.append(f"\n--- {path} ---\n{_read_text_safely(Path(path))}")
            log_text = "\n".join(chunks)

        prompt = {
            "task": "Summarize this trading session for the operator. Highlight PnL drivers, blocked trades, errors, risk events, and tomorrow's checklist.",
            "trade_summary": trade_summary or {},
            "logs": log_text,
            "output_schema": {
                "headline": "string",
                "pnl_drivers": ["string"],
                "risk_events": ["string"],
                "execution_issues": ["string"],
                "tomorrow_checklist": ["string"],
            },
        }
        return self._json_council_call("chief_analyst", prompt)

    def code_review(self, diff_or_file: str, focus: str = "safety, bugs, tests, Windows VPS readiness") -> Dict[str, Any]:
        """Review code changes using the code-focused model."""
        prompt = {
            "task": "Review this code for the trading bot.",
            "focus": focus,
            "code_or_diff": diff_or_file,
            "output_schema": {
                "severity": "pass | low | medium | high | critical",
                "issues": [
                    {"title": "string", "severity": "string", "details": "string", "fix": "string"}
                ],
                "tests_to_add": ["string"],
                "windows_vps_notes": ["string"],
            },
        }
        return self._json_council_call("code_architect", prompt)

    def _json_council_call(self, role: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call one council model and parse a JSON response safely."""
        if not self.settings.enabled:
            return {"ok": False, "role": role, "error": "Ollama advisor is disabled."}

        model = self.settings.council.get(role) or DEFAULT_COUNCIL.get(role)
        if not model:
            return {"ok": False, "role": role, "error": f"No model configured for role {role}."}

        system = _advisor_system_prompt(self.settings.allow_live_trade_actions)
        prompt = json.dumps(payload, indent=2, default=str)
        started = time.time()
        try:
            raw = self.client.generate(model=model, prompt=prompt, system=system, json_mode=True)
            parsed = _parse_json_object(raw)
            parsed.setdefault("ok", True)
            parsed.setdefault("role", role)
            parsed.setdefault("model", model)
            parsed["latency_sec"] = round(time.time() - started, 3)
            parsed["live_trade_actions_allowed"] = self.settings.allow_live_trade_actions
            return parsed
        except Exception as exc:
            logger.warning(f"Ollama advisor call failed: role={role} model={model} error={exc}")
            return {
                "ok": False,
                "role": role,
                "model": model,
                "error": str(exc),
                "latency_sec": round(time.time() - started, 3),
                "live_trade_actions_allowed": self.settings.allow_live_trade_actions,
            }


def load_ollama_settings(paths: Optional[List[Path]] = None) -> OllamaSettings:
    """Load Ollama settings from configs/ollama.yaml and config.yaml."""
    merged: Dict[str, Any] = {}
    for path in paths or DEFAULT_CONFIG_PATHS:
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if "ollama" in data:
                merged.update(data.get("ollama") or {})
            elif path.name == "ollama.yaml":
                merged.update(data or {})
        except Exception as exc:
            logger.warning(f"Could not read Ollama config {path}: {exc}")

    council = dict(DEFAULT_COUNCIL)
    council.update(merged.get("council") or {})

    return OllamaSettings(
        enabled=bool(merged.get("enabled", True)),
        base_url=str(merged.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))),
        timeout_seconds=int(merged.get("timeout_seconds", 90)),
        temperature=float(merged.get("temperature", 0.15)),
        num_ctx=int(merged.get("num_ctx", 8192)),
        council=council,
        max_prompt_chars=int(merged.get("max_prompt_chars", 24000)),
        allow_config_patch_suggestions=bool(merged.get("allow_config_patch_suggestions", True)),
        allow_live_trade_actions=bool(merged.get("allow_live_trade_actions", False)),
    )


def _advisor_system_prompt(allow_live_trade_actions: bool) -> str:
    action_rule = "You may discuss live trade actions only as operator-reviewed suggestions."
    if not allow_live_trade_actions:
        action_rule = "Never issue, authorize, or simulate a direct live order command."
    return (
        "You are the local Ollama council for a high-risk algorithmic trading system. "
        "Respond with valid JSON only. Be skeptical, concise, and risk-first. "
        "Do not promise profits. Do not bypass drawdown, spread, margin, event, or kill-switch controls. "
        "Prefer paper trading, walk-forward validation, and stress testing before promotion. "
        f"{action_rule}"
    )


def _limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n...[trimmed for context budget]...\n\n{tail}"


def _read_text_safely(path: Path, max_chars: int = 20000) -> str:
    try:
        if not path.exists():
            return f"File not found: {path}"
        text = path.read_text(encoding="utf-8", errors="replace")
        return _limit_text(text, max_chars)
    except Exception as exc:
        return f"Could not read {path}: {exc}"


def _parse_json_object(raw: str) -> Dict[str, Any]:
    """Parse JSON object from LLM output, tolerating fenced or prefixed text."""
    raw = (raw or "").strip()
    if not raw:
        return {"ok": False, "error": "Empty model response"}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"ok": True, "response": parsed}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {"ok": False, "error": "Could not parse JSON response", "raw_response": raw[:2000]}


def make_advisor() -> OllamaAdvisor:
    """Factory used by scripts and API modules."""
    return OllamaAdvisor()


if __name__ == "__main__":
    advisor = make_advisor()
    print(json.dumps(advisor.health_check(), indent=2))
