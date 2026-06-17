"""
Python.autonomous package
Exposes self-evolution components including the Self-Monitoring, Auto-Rollback & Recovery system.

Master component: MasterSelfEvolutionSupervisor (the top-level strategic brain coordinating
Retraining Orchestrator, Meta-Optimizer, RegimeAdaptiveController (Rainforest+Pattern+Dreamer+timing),
Continual Learner, and Self-Monitor/Recovery).
"""

from .self_monitor import (
    SelfMonitoringRecoveryAgent,
    MonitoringMetrics,
    SelfAction,
    create_agent,
)

# Top-level strategic supervisor (central brain)
from .self_evolution_supervisor import (
    MasterSelfEvolutionSupervisor,
    EvolutionGoal,
    SafetyConstraint,
    ModelVersionRecord,
    EvolutionCycleResult,
    EVOLUTION_STRATEGIES,
)

__all__ = [
    "SelfMonitoringRecoveryAgent",
    "MonitoringMetrics",
    "SelfAction",
    "create_agent",
    # Master brain
    "MasterSelfEvolutionSupervisor",
    "EvolutionGoal",
    "SafetyConstraint",
    "ModelVersionRecord",
    "EvolutionCycleResult",
    "EVOLUTION_STRATEGIES",
    # Re-exports of siblings for convenience
    "self_evolution_supervisor",
    "retraining_trigger",
    "run_cycle",
    # Online / Continual Learning Layer (new)
    "continual_learner",
    "ContinualLearner",
    "ContinualConfig",
    # Regime-Adaptive Controller
    "RegimeAdaptiveController",
    "RegimeController",
    "RegimeState",
    "AdaptationConfig",
    "get_regime_controller",
    # Experience Memory / Advanced Replay (core self-evolution primitive)
    "experience_memory",
    "ExperienceMemory",
    "Experience",
    "rich_trade_to_experience",
]

# Online/Continual Learning Layer (Decision PPO policy gradients on live trades + Dreamer RSSM + Rainforest incremental)
try:
    from .continual_learner import (
        ContinualLearner,
        ContinualConfig,
        ExperienceSampler,
        SimpleEWC,
    )
except Exception:
    ContinualLearner = None  # type: ignore
    ContinualConfig = None  # type: ignore
    ExperienceSampler = None  # type: ignore
    SimpleEWC = None  # type: ignore

# Regime-Adaptive Controller (Rainforest + PatternDetector + Dreamer + timing → risk/TradeDecision/ensemble/policy adaptation)
try:
    from .regime_controller import (
        RegimeAdaptiveController,
        RegimeController,
        RegimeState,
        AdaptationConfig,
        get_regime_controller,
    )
except Exception:
    RegimeAdaptiveController = None  # type: ignore
    RegimeController = None  # type: ignore
    RegimeState = None  # type: ignore
    AdaptationConfig = None  # type: ignore
    get_regime_controller = None  # type: ignore

# Experience Memory — Advanced Prioritized Replay for PPO, Dreamer, Meta-Optimizer, Regime Controller
try:
    from .experience_memory import (
        ExperienceMemory,
        Experience,
        rich_trade_to_experience,
    )
except Exception:
    ExperienceMemory = None  # type: ignore
    Experience = None  # type: ignore
    rich_trade_to_experience = None  # type: ignore
