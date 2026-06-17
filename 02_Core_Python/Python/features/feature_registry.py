from dataclasses import dataclass, field
from typing import Literal


FeatureStatus = Literal[
    "enabled",
    "disabled_noise",
    "disabled_redundant",
    "disabled_leakage_risk",
    "disabled_unstable",
    "regime_only",
    "research_only",
    "deprecated",
]


@dataclass
class FeatureMeta:
    feature_name: str
    family: str
    source_columns: list[str] = field(default_factory=list)
    lookback_bars: int = 0
    uses_future_data: bool = False
    allowed_live: bool = True
    leakage_risk: str = "none"          # none | low | medium | high
    status: FeatureStatus = "enabled"
    notes: str = ""


class FeatureRegistry:
    """Catalogue every feature with lineage, risk, and deployment metadata."""

    def __init__(self):
        self._registry: dict[str, FeatureMeta] = {}

    def register(
        self,
        feature_name: str,
        family: str,
        source_columns: list[str] | None = None,
        lookback_bars: int = 0,
        uses_future_data: bool = False,
        allowed_live: bool = True,
        leakage_risk: str = "none",
        status: FeatureStatus = "enabled",
        notes: str = "",
    ) -> FeatureMeta:
        meta = FeatureMeta(
            feature_name=feature_name,
            family=family,
            source_columns=source_columns or [],
            lookback_bars=lookback_bars,
            uses_future_data=uses_future_data,
            allowed_live=allowed_live,
            leakage_risk=leakage_risk,
            status=status,
            notes=notes,
        )
        self._registry[feature_name] = meta
        return meta

    def get(self, feature_name: str) -> FeatureMeta | None:
        return self._registry.get(feature_name)

    def list_features(self) -> list[str]:
        return list(self._registry.keys())

    def get_by_family(self, family: str) -> list[FeatureMeta]:
        return [m for m in self._registry.values() if m.family == family]

    def get_enabled(self) -> list[FeatureMeta]:
        return [m for m in self._registry.values() if m.status == "enabled"]

    def get_live_allowed(self) -> list[FeatureMeta]:
        return [m for m in self._registry.values() if m.allowed_live and m.status == "enabled"]

    def get_high_leakage(self) -> list[FeatureMeta]:
        return [m for m in self._registry.values() if m.leakage_risk == "high"]

    def to_dict(self) -> dict:
        return {
            name: {
                "feature_name": m.feature_name,
                "family": m.family,
                "source_columns": m.source_columns,
                "lookback_bars": m.lookback_bars,
                "uses_future_data": m.uses_future_data,
                "allowed_live": m.allowed_live,
                "leakage_risk": m.leakage_risk,
                "status": m.status,
                "notes": m.notes,
            }
            for name, m in self._registry.items()
        }

    def from_builder(self, builder) -> "FeatureRegistry":
        """Auto-register features produced by a FeatureBuilder."""
        from .build_features import FeatureBuilder
        if not isinstance(builder, FeatureBuilder):
            raise TypeError("builder must be a FeatureBuilder instance")
        # Map known prefixes to metadata
        prefix_rules = {
            "open_rel": ("price_action", ["open", "close"], 0),
            "high_rel": ("price_action", ["high", "close"], 0),
            "low_rel": ("price_action", ["low", "close"], 0),
            "close_ret_1": ("price_action", ["close"], 1),
            "gap_ratio": ("price_action", ["open", "close"], 1),
            "ret_": ("price_action", ["close"], None),
            "log_ret_": ("price_action", ["close"], None),
            "candle_body_pct": ("price_action", ["open", "high", "low", "close"], 0),
            "upper_wick_pct": ("price_action", ["open", "high", "low", "close"], 0),
            "lower_wick_pct": ("price_action", ["open", "high", "low", "close"], 0),
            "range_ratio": ("price_action", ["high", "low", "close"], 0),
            "ema_": ("trend", ["close"], None),
            "ema_slope_": ("trend", ["close"], 5),
            "rsi_": ("momentum", ["close"], None),
            "macd": ("momentum", ["close"], 26),
            "atr_": ("volatility", ["high", "low", "close"], None),
            "atr_pct": ("volatility", ["high", "low", "close"], 14),
            "bb_width": ("volatility", ["close"], 20),
            "realized_vol_": ("volatility", ["close"], None),
            "volume_zscore": ("volume", ["volume"], 20),
            "volume_ma_ratio": ("volume", ["volume"], 20),
            "log_volume": ("volume", ["volume"], 0),
            "spread": ("spread", ["high", "low"], 0),
            "spread_zscore": ("spread", ["high", "low"], 20),
            "session_": ("session", ["time"], 0),
            "distance_to_swing_": ("market_structure", ["high", "low", "close"], 20),
            "recent_loss_streak": ("trade_memory", ["trade_memory"], 0, False, "low"),
            "recent_win_rate": ("trade_memory", ["trade_memory"], 0, False, "low"),
            "recent_slippage_avg": ("trade_memory", ["trade_memory"], 0, False, "low"),
            "hour_sin": ("calendar", ["time"], 0),
            "hour_cos": ("calendar", ["time"], 0),
            "dow_sin": ("calendar", ["time"], 0),
            "dow_cos": ("calendar", ["time"], 0),
            "month_sin": ("calendar", ["time"], 0),
            "month_cos": ("calendar", ["time"], 0),
            "m15_": ("higher_timeframe", ["open", "high", "low", "close", "volume"], 15),
            "h1_": ("higher_timeframe", ["open", "high", "low", "close", "volume"], 60),
            "h4_": ("higher_timeframe", ["open", "high", "low", "close", "volume"], 240),
            "d1_": ("higher_timeframe", ["open", "high", "low", "close", "volume"], 1440),
            "cross_": ("cross", [], 0),
            "close_ma_rel_": ("trend", ["close"], None),
            "volume_rel_": ("volume", ["volume"], None),
            "close_z_": ("trend", ["close"], None),
            "momentum_": ("momentum", ["close"], None),
            "ema_rel_": ("trend", ["close"], None),
            "breakout_high_": ("market_structure", ["close", "high"], None),
            "breakout_low_": ("market_structure", ["close", "low"], None),
            "slope_": ("trend", ["close"], None),
            "volume_z_": ("volume", ["volume"], None),
            "range_mean_": ("volatility", ["high", "low", "close"], None),
            "range_std_": ("volatility", ["high", "low", "close"], None),
        }
        for name in builder.feature_names:
            matched = False
            for prefix, rule in prefix_rules.items():
                if name.startswith(prefix) or name == prefix.rstrip("_"):
                    family, sources = rule[0], rule[1]
                    lb = rule[2] if len(rule) > 2 else 0
                    allowed = rule[3] if len(rule) > 3 else True
                    risk = rule[4] if len(rule) > 4 else "none"
                    # infer lookback from suffix if None
                    if lb is None:
                        parts = name.split("_")
                        try:
                            lb = int(parts[-1])
                        except ValueError:
                            lb = 0
                    self.register(
                        feature_name=name,
                        family=family,
                        source_columns=sources,
                        lookback_bars=lb,
                        allowed_live=allowed,
                        leakage_risk=risk,
                    )
                    matched = True
                    break
            if not matched:
                self.register(feature_name=name, family="unknown")
        return self
