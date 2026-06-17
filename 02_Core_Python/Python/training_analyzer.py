"""
Training Learning Process Analyzer

Uses Ollama to generate human-readable descriptions of what the model is learning
at any point during training. Connects training metrics to trading performance.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import requests
import time
from loguru import logger

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

# Circuit-breaker cooldown after an Ollama failure.
COOLDOWN_SECONDS = 60


class TrainingAnalyzer:
    """Analyzes training progress and generates learning process descriptions using Ollama."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.cache = {}
        self.cache_ttl = 30  # seconds
        self._last_analysis = None
        self._analysis_history = []
        self._ollama_cooldown_until = 0.0

    def _call_ollama(self, prompt: str, system: str = "") -> str:
        """Call Ollama to generate a response.

        A circuit-breaker cooldown suppresses log/CPU noise when Ollama is
        unreachable, returns 4xx/5xx, or returns malformed JSON. After one
        failure, future calls within COOLDOWN_SECONDS short-circuit with a
        debug-level log and the same placeholder string. On the next call
        AFTER the cooldown expires, a real attempt is made; on full success
        we reset the cooldown and emit an info-level "recovered" log.
        """
        now = time.time()
        # Circuit-breaker: short-circuit while in cooldown
        if self._ollama_cooldown_until > now:
            remaining = int(self._ollama_cooldown_until - now)
            logger.debug(f"Ollama call suppressed by cooldown ({remaining}s remaining)")
            return f"Analysis unavailable: ollama cooldown ({remaining}s)"

        # Real attempt
        was_previously_failing = self._ollama_cooldown_until > 0.0  # 0.0 means never failed
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 300,
                    }
                },
                timeout=30
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            now = time.time()  # refresh after possibly slow network call
            self._ollama_cooldown_until = now + COOLDOWN_SECONDS
            logger.warning(
                f"Ollama call failed: {e}. Muting further Ollama calls for {COOLDOWN_SECONDS}s."
            )
            return f"Analysis unavailable: ollama unreachable ({type(e).__name__})"
        except (ValueError,) as e:
            now = time.time()  # refresh after possibly slow network call
            self._ollama_cooldown_until = now + min(COOLDOWN_SECONDS, 15)
            logger.warning(f"Ollama response malformed: {e}. Cooldown set to 15s.")
            return "Analysis unavailable: malformed ollama response"
        except Exception as e:
            now = time.time()  # refresh after possibly slow network call
            self._ollama_cooldown_until = now + COOLDOWN_SECONDS
            logger.warning(f"Ollama unexpected error: {type(e).__name__}: {e}. Cooldown {COOLDOWN_SECONDS}s.")
            return f"Analysis unavailable: ollama error ({type(e).__name__})"

        # Full success: parse JSON, re-arm short cooldown if parse fails
        try:
            result = response.json().get("response", "")
        except (ValueError,) as e:
            now = time.time()  # refresh after possibly slow network call
            self._ollama_cooldown_until = now + min(COOLDOWN_SECONDS, 15)
            logger.warning(f"Ollama JSON decode failed: {e}. Cooldown set to 15s.")
            return "Analysis unavailable: malformed ollama response"
        if was_previously_failing:
            logger.info("Ollama recovered — cooldown cleared.")
        self._ollama_cooldown_until = 0.0
        return result

    def _build_training_prompt(self, metrics: Dict[str, Any]) -> tuple[str, str]:
        """Build a prompt for training analysis."""
        system = """You are an expert trading AI analyst. Describe what the model is learning
based on the training metrics provided. Be concise (2-3 sentences) and focus on:
1. What patterns the model is currently learning
2. How performance is trending
3. What trading behaviors are being reinforced

Use trading terminology appropriately but keep it understandable."""

        # Extract key metrics
        symbol = metrics.get("symbol", "Unknown")
        epoch = metrics.get("epoch", 0)
        total_epochs = metrics.get("total_epochs", 100)
        loss = metrics.get("loss", 0)
        val_loss = metrics.get("val_loss", 0)
        reward = metrics.get("avg_reward", 0)
        trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0)
        progress_pct = (epoch / total_epochs * 100) if total_epochs > 0 else 0

        prompt = f"""Training Progress for {symbol}:
- Epoch: {epoch}/{total_epochs} ({progress_pct:.1f}%)
- Training Loss: {loss:.6f}
- Validation Loss: {val_loss:.6f}
- Average Reward: {reward:.4f}
- Total Trades (simulated): {trades}
- Win Rate: {win_rate:.1f}%

What is the model learning at this stage? Describe the learning process and what trading behaviors are being reinforced."""

        return prompt, system

    def analyze_training_progress(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an analysis of the current training progress."""
        cache_key = f"{metrics.get('symbol', 'unknown')}_{metrics.get('epoch', 0)}"
        now = time.time()

        # Check cache
        if cache_key in self.cache:
            cached, timestamp = self.cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return cached

        prompt, system = self._build_training_prompt(metrics)
        description = self._call_ollama(prompt, system)

        # Determine learning stage
        progress = metrics.get("epoch", 0) / max(metrics.get("total_epochs", 100), 1)
        if progress < 0.1:
            stage = "exploration"
            stage_desc = "Model is exploring the action space and learning basic price patterns"
        elif progress < 0.3:
            stage = "pattern_recognition"
            stage_desc = "Model is developing pattern recognition for entry/exit signals"
        elif progress < 0.6:
            stage = "strategy_refinement"
            stage_desc = "Model is refining trading strategy and risk management"
        elif progress < 0.9:
            stage = "optimization"
            stage_desc = "Model is optimizing decision boundaries and fine-tuning"
        else:
            stage = "convergence"
            stage_desc = "Model is converging on optimal policy"

        result = {
            "symbol": metrics.get("symbol", "Unknown"),
            "epoch": metrics.get("epoch", 0),
            "total_epochs": metrics.get("total_epochs", 100),
            "progress_pct": progress * 100,
            "learning_stage": stage,
            "stage_description": stage_desc,
            "ai_description": description.strip(),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_summary": {
                "loss": metrics.get("loss", 0),
                "val_loss": metrics.get("val_loss", 0),
                "reward": metrics.get("avg_reward", 0),
                "win_rate": metrics.get("win_rate", 0),
                "trades": metrics.get("total_trades", 0),
            }
        }

        self.cache[cache_key] = (result, now)
        self._last_analysis = result
        self._analysis_history.append(result)

        # Keep only last 100 analyses
        if len(self._analysis_history) > 100:
            self._analysis_history = self._analysis_history[-100:]

        return result

    def analyze_trading_connection(self, training_metrics: Dict, trading_metrics: Dict) -> Dict[str, Any]:
        """Analyze the connection between training progress and live trading performance."""
        system = """You are an expert trading AI analyst. Compare training metrics with live trading performance.
Describe how the model's training progress relates to its real-world trading decisions.
Be specific about what behaviors from training are appearing in live trading."""

        prompt = f"""Training vs Live Trading Analysis:

TRAINING METRICS:
- Symbol: {training_metrics.get('symbol', 'Unknown')}
- Training Progress: {training_metrics.get('epoch', 0)}/{training_metrics.get('total_epochs', 100)} epochs
- Training Win Rate: {training_metrics.get('win_rate', 0):.1f}%
- Average Training Reward: {training_metrics.get('avg_reward', 0):.4f}

LIVE TRADING METRICS:
- Current P&L: ${trading_metrics.get('pnl', 0):.2f}
- Live Win Rate: {trading_metrics.get('live_win_rate', 0):.1f}%
- Open Positions: {trading_metrics.get('open_positions', 0)}
- Recent Actions: {', '.join(trading_metrics.get('recent_actions', ['HOLD']))}
- Confidence Level: {trading_metrics.get('avg_confidence', 0):.1f}%

How is the model's training reflected in its live trading behavior? What patterns from training
are visible in the current trading decisions?"""

        description = self._call_ollama(prompt, system)

        return {
            "training_symbol": training_metrics.get("symbol"),
            "trading_symbol": trading_metrics.get("symbol"),
            "connection_description": description.strip(),
            "alignment_score": self._calculate_alignment(training_metrics, trading_metrics),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _calculate_alignment(self, training: Dict, trading: Dict) -> float:
        """Calculate alignment between training and trading metrics."""
        scores = []

        # Win rate alignment
        train_wr = training.get("win_rate", 50)
        live_wr = trading.get("live_win_rate", 50)
        if train_wr > 0:
            scores.append(1 - abs(train_wr - live_wr) / max(train_wr, live_wr, 1))

        # Confidence alignment with profitability
        conf = trading.get("avg_confidence", 0.5)
        pnl = trading.get("pnl", 0)
        if pnl > 0 and conf > 0.6:
            scores.append(1.0)
        elif pnl < 0 and conf > 0.6:
            scores.append(0.3)
        else:
            scores.append(0.5)

        return sum(scores) / len(scores) if scores else 0.5

    def get_learning_trajectory(self, symbol: str) -> Dict[str, Any]:
        """Get the learning trajectory for a symbol from history."""
        symbol_history = [
            h for h in self._analysis_history
            if h.get("symbol") == symbol
        ]

        if not symbol_history:
            return {"error": "No history for symbol"}

        # Analyze trajectory
        stages = [h["learning_stage"] for h in symbol_history]
        progressions = len(set(stages))

        # Detect issues
        recent = symbol_history[-5:] if len(symbol_history) >= 5 else symbol_history
        losses = [h["metrics_summary"]["loss"] for h in recent if "loss" in h.get("metrics_summary", {})]
        loss_trend = "stable"
        if len(losses) >= 3:
            if losses[-1] > losses[0] * 1.2:
                loss_trend = "increasing (possible overfitting)"
            elif losses[-1] < losses[0] * 0.8:
                loss_trend = "decreasing (improving)"

        return {
            "symbol": symbol,
            "total_analyses": len(symbol_history),
            "current_stage": stages[-1] if stages else "unknown",
            "stage_progressions": progressions,
            "loss_trend": loss_trend,
            "history": symbol_history[-10:],  # Last 10 analyses
        }

    def generate_training_insights(self) -> List[str]:
        """Generate general insights about the training process."""
        if not self._analysis_history:
            return ["No training data available yet."]

        insights = []

        # Analyze all symbols
        symbols = set(h.get("symbol") for h in self._analysis_history)
        for symbol in symbols:
            traj = self.get_learning_trajectory(symbol)
            if traj.get("loss_trend") == "increasing (possible overfitting)":
                insights.append(f"⚠️ {symbol}: Loss increasing - consider early stopping")
            elif traj.get("stage_progressions", 0) >= 3:
                insights.append(f"✅ {symbol}: Progressed through {traj['stage_progressions']} learning stages")

        return insights if insights else ["Training progressing normally across all symbols"]


# Global analyzer instance
_analyzer = None


def get_analyzer() -> TrainingAnalyzer:
    """Get or create the global training analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = TrainingAnalyzer()
    return _analyzer


def analyze_current_training(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to analyze current training progress."""
    return get_analyzer().analyze_training_progress(metrics)


def analyze_training_trading_connection(training: Dict, trading: Dict) -> Dict[str, Any]:
    """Analyze connection between training and trading."""
    return get_analyzer().analyze_trading_connection(training, trading)


def get_training_description(symbol: str = None) -> Dict[str, Any]:
    """Get a description of what the model is currently learning.

    This can be called from the API to get real-time training insights.
    """
    analyzer = get_analyzer()

    # Try to read current training progress
    try:
        import glob
        progress_files = glob.glob(os.path.join("logs", "ppo_*_progress.json"))

        if progress_files:
            # Get most recent
            progress_files.sort(key=os.path.getmtime, reverse=True)
            with open(progress_files[0], "r") as f:
                progress = json.load(f)

            # Convert to metrics format
            metrics = {
                "symbol": progress.get("symbol", symbol or "Unknown"),
                "epoch": progress.get("timesteps", 0) // 1000,  # Approximate
                "total_epochs": progress.get("target_timesteps", 100000) // 1000,
                "loss": progress.get("loss", 0),
                "avg_reward": progress.get("reward", 0),
                "win_rate": progress.get("win_rate", 0),
                "total_trades": progress.get("episodes", 0),
            }

            return analyzer.analyze_training_progress(metrics)
    except Exception as e:
        logger.debug(f"Could not read training progress: {e}")

    return {
        "error": "No active training found",
        "symbol": symbol or "Unknown",
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    # Test the analyzer
    test_metrics = {
        "symbol": "BTCUSDm",
        "epoch": 45,
        "total_epochs": 100,
        "loss": 0.0234,
        "val_loss": 0.0312,
        "avg_reward": 0.156,
        "win_rate": 58.5,
        "total_trades": 234,
    }

    print("Testing Training Analyzer...")
    result = analyze_current_training(test_metrics)
    print(json.dumps(result, indent=2))
