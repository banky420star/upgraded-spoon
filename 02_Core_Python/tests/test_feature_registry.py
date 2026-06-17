import pytest

feature_mod = pytest.importorskip(
    "Python.features.feature_registry",
    reason="FeatureRegistry requires heavy deps; install to enable tests",
)


def test_feature_registry_register_and_retrieve():
    registry = feature_mod.FeatureRegistry()
    feat = registry.register(
        feature_name="test_ema",
        family="trend",
        source_columns=["close"],
        lookback_bars=14,
    )
    assert feat.feature_name == "test_ema"
    assert feat.family == "trend"
    assert feat.leakage_risk in ("none", "low", "medium", "high")
    assert feat.lookback_bars == 14
    # Verify retrieval
    retrieved = registry.get("test_ema")
    assert retrieved is not None
    assert retrieved.feature_name == "test_ema"


def test_feature_registry_list_and_family():
    registry = feature_mod.FeatureRegistry()
    registry.register("ema_14", "trend", source_columns=["close"], lookback_bars=14)
    registry.register("rsi_14", "momentum", source_columns=["close"], lookback_bars=14)
    names = registry.list_features()
    assert len(names) == 2
    assert "ema_14" in names
    assert "rsi_14" in names
    # Family filtering
    trend_feats = registry.get_by_family("trend")
    assert len(trend_feats) == 1
    assert trend_feats[0].feature_name == "ema_14"
