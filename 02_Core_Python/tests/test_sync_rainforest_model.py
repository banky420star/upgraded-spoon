"""Tests for sync_rainforest_model.py -- symbol filtering and model consolidation."""

import json
import os
import pathlib
import sys
import tempfile
import time

import joblib as jb
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "Python"))

from training.sync_rainforest_model import find_latest_model_dir, sync_model

def _make_model_dir(base, name, dataset_id, *, oob_score=0.85, corrupt_meta=False, no_meta=False, is_file=False):
    if is_file:
        p = base / name
        p.write_text("not a dir")
        return p
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    if not no_meta:
        meta = {"dataset_id": dataset_id, "oob_score": oob_score, "trained_at": time.time(), "n_estimators": 100, "max_depth": 12}
        meta_path = d / "meta.json"
        if corrupt_meta: meta_path.write_text("this is not valid json")
        else: meta_path.write_text(json.dumps(meta))
    X, y = make_classification(n_samples=50, n_features=4, n_informative=3, n_redundant=0, n_classes=3, random_state=42)
    clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    clf.fit(X, y)
    jb.dump({"model": clf}, d / "model.pkl")
    return d

class TestFindLatestNoFilter:
    def test_returns_latest_by_mtime(self, tmp_path):
        _make_model_dir(tmp_path, "alpha", "mt5_BTCUSDm")
        time.sleep(0.05)
        _make_model_dir(tmp_path, "bravo", "mt5_XAUUSDm")
        result = find_latest_model_dir(tmp_path)
        assert result.name == "bravo"

    def test_skips_files(self, tmp_path):
        _make_model_dir(tmp_path, "a_dir", "mt5_BTCUSDm")
        _make_model_dir(tmp_path, "a_file", "ignored", is_file=True)
        result = find_latest_model_dir(tmp_path)
        assert result.name == "a_dir"

    def test_exits_when_no_dirs(self, tmp_path):
        with pytest.raises(SystemExit) as exc: find_latest_model_dir(tmp_path)
        assert exc.value.code == 1

class TestFindWithSymbolFilter:
    def test_returns_latest_matching_symbol(self, tmp_path):
        _make_model_dir(tmp_path, "btc_old", "mt5_BTCUSDm", oob_score=0.80)
        time.sleep(0.05)
        _make_model_dir(tmp_path, "xau", "mt5_XAUUSDm", oob_score=0.90)
        time.sleep(0.05)
        _make_model_dir(tmp_path, "btc_new", "mt5_BTCUSDm", oob_score=0.85)
        result = find_latest_model_dir(tmp_path, symbol="BTCUSDm")
        assert result.name == "btc_new"

    def test_case_insensitive_match(self, tmp_path):
        _make_model_dir(tmp_path, "only", "mt5_EURUSDm", oob_score=0.90)
        assert find_latest_model_dir(tmp_path, symbol="eurusdm").name == "only"
        assert find_latest_model_dir(tmp_path, symbol="EURUSDm").name == "only"

    def test_exits_when_no_symbol_match(self, tmp_path):
        _make_model_dir(tmp_path, "btc", "mt5_BTCUSDm")
        with pytest.raises(SystemExit) as exc: find_latest_model_dir(tmp_path, symbol="XAUUSDm")
        assert exc.value.code == 1

    def test_exits_on_empty_dir_with_symbol(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            find_latest_model_dir(tmp_path, symbol="BTCUSDm")
        assert exc.value.code == 1

class TestCorruptMeta:
    def test_skips_dir_with_corrupt_meta(self, tmp_path):
        _make_model_dir(tmp_path, "good", "mt5_BTCUSDm")
        _make_model_dir(tmp_path, "bad", "mt5_BTCUSDm", corrupt_meta=True)
        assert find_latest_model_dir(tmp_path, symbol="BTCUSDm").name == "good"

    def test_skips_dir_without_meta(self, tmp_path):
        _make_model_dir(tmp_path, "no_meta", "mt5_BTCUSDm", no_meta=True)
        _make_model_dir(tmp_path, "has_meta", "mt5_BTCUSDm")
        assert find_latest_model_dir(tmp_path, symbol="BTCUSDm").name == "has_meta"

class TestSyncModel:
    def test_sync_creates_consolidated_pickle(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rf_dir = tmp_path / "models" / "rainforest"
        rf_dir.mkdir(parents=True)
        _make_model_dir(rf_dir, "mymodel", "mt5_BTCUSDm", oob_score=0.7777)
        sync_model("BTCUSDm")
        out = tmp_path / "models" / "rainforest_BTCUSDm.pkl"
        assert out.is_file()
        data = jb.load(str(out))
        expected_keys = {"model", "classes", "feature_importances", "oob_score", "trained_at", "n_estimators", "max_depth"}
        assert set(data.keys()) == expected_keys
        assert data["oob_score"] == 0.7777
        assert isinstance(data["model"], RandomForestClassifier)
        assert len(data["classes"]) == 3
        assert len(data["feature_importances"]) == 4

    def test_sync_exits_if_no_model_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc: sync_model("BTCUSDm")
        assert exc.value.code == 1

    def test_sync_exits_if_missing_model_pkl(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rf_dir = tmp_path / "models" / "rainforest"
        rf_dir.mkdir(parents=True)
        d = rf_dir / "somemodel"
        d.mkdir()
        meta = {"dataset_id": "mt5_BTCUSDm", "oob_score": 0.85}
        (d / "meta.json").write_text(__import__("json").dumps(meta))
        with pytest.raises(SystemExit) as exc:
            sync_model("BTCUSDm")
        assert exc.value.code == 1

class TestCLI:
    def test_requires_symbol(self):
        from training.sync_rainforest_model import main
        with pytest.raises(SystemExit):
            sys.argv = ["sync_rainforest_model.py"]
            main()


class TestDiscoverSymbols:
    """``discover_symbols`` scan of rainforest model directories."""

    def test_returns_sorted_unique_symbols(self, tmp_path):
        rf_dir = tmp_path / "rainforest"
        rf_dir.mkdir()
        _make_model_dir(rf_dir, "btc1", "mt5_BTCUSDm")
        _make_model_dir(rf_dir, "eur1", "mt5_EURUSDm")
        _make_model_dir(rf_dir, "btc2", "mt5_BTCUSDm")  # duplicate
        _make_model_dir(rf_dir, "xau1", "mt5_XAUUSDm")

        from training.sync_rainforest_model import discover_symbols
        result = discover_symbols(str(tmp_path))
        assert result == ["BTCUSDm", "EURUSDm", "XAUUSDm"]

    def test_empty_when_no_dirs(self, tmp_path):
        from training.sync_rainforest_model import discover_symbols
        assert discover_symbols(str(tmp_path)) == []

    def test_empty_when_no_meta_json(self, tmp_path):
        from training.sync_rainforest_model import discover_symbols
        d = tmp_path / "rainforest" / "somemodel"
        d.mkdir(parents=True)
        (d / "model.pkl").write_text("dummy")
        # no meta.json -> should be skipped
        assert discover_symbols(str(tmp_path)) == []


class TestSyncAll:
    """``sync_all`` end-to-end."""

    def test_syncs_all_symbols(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create model dirs for 3 symbols
        rf_dir = tmp_path / "models" / "rainforest"
        rf_dir.mkdir(parents=True)

        _make_model_dir(rf_dir, "btc", "mt5_BTCUSDm", oob_score=0.81)
        _make_model_dir(rf_dir, "eur", "mt5_EURUSDm", oob_score=0.92)
        _make_model_dir(rf_dir, "xau", "mt5_XAUUSDm", oob_score=0.87)

        from training.sync_rainforest_model import sync_all
        sync_all()

        for sym in ["BTCUSDm", "EURUSDm", "XAUUSDm"]:
            out = tmp_path / "models" / f"rainforest_{sym}.pkl"
            assert out.is_file(), f"Missing {out}"

    def test_exits_when_no_symbols(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models" / "rainforest").mkdir(parents=True)

        from training.sync_rainforest_model import sync_all
        with pytest.raises(SystemExit) as exc:
            sync_all()
        assert exc.value.code == 1
