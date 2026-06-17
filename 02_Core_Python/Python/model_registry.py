import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from filelock import FileLock, Timeout as FileLockTimeout
from loguru import logger

from Python.config_utils import load_project_config

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INTEGRITY_TARGETS = {
    "model": "ppo_trading.zip",
    "vec_normalize": "vec_normalize.pkl",
    "metadata": "metadata.json",
    "scorecard": "scorecard.json",
}


class ModelRegistry:
    """
    File-based model registry.
    Layout:
      models/
        registry/
          active.json
          candidates/<version>/

    active.json structure:
      {
        "champion": <path or null>,
        "canary": <path or null>,
        "symbols": {
          "EURUSDm": {"champion": <path or null>, "canary": <path or null>, "canary_policy": {...}, "canary_state": {...}},
          ...
        },
        "registry_metadata": {
          "git_commit": "...",
          "champion_metadata": {...},
          "canary_metadata": {...}
        }
      }
    """

    def __init__(self, root=None, registry_config: dict | None = None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.root = root or os.path.join(base, "models", "registry")
        os.makedirs(self.root, exist_ok=True)

        self.registry_config = registry_config or self._load_registry_config()

        self.active_path = os.path.join(self.root, "active.json")
        # Hardened locking: configurable timeout (env > config > 30s default for VPS contention safety)
        # Prevents indefinite blocks on inference/API during slow training writes or disk hiccups.
        raw_timeout = (
            os.environ.get("AGI_REGISTRY_LOCK_TIMEOUT")
            or (self.registry_config.get("lock_timeout") if isinstance(self.registry_config, dict) else None)
            or 30.0
        )
        try:
            self._lock_timeout = float(raw_timeout)
        except Exception:
            self._lock_timeout = 30.0
        self._lock = FileLock(f"{self.active_path}.lock", timeout=self._lock_timeout)
        self.champion_dir = os.path.join(self.root, "champion")
        self.canary_dir = os.path.join(self.root, "canary")
        self.candidates_dir = os.path.join(self.root, "candidates")
        self.per_symbol_dir = os.path.join(self.root, "per_symbol")

        for d in (self.champion_dir, self.canary_dir, self.candidates_dir, self.per_symbol_dir):
            os.makedirs(d, exist_ok=True)

        if not os.path.exists(self.active_path):
            self._write_active({"champion": None, "canary": None, "symbols": {}})

        self._explicit_registry_config = registry_config
        self.registry_config = registry_config if registry_config is not None else self._load_registry_config()
        self._canary_policy_cfg = self.registry_config.get("canary_policy", {}) or {}
        self._canary_default_overrides = self._canary_policy_cfg.get("default", {}) or {}
        self._canary_symbol_overrides = self._canary_policy_cfg.get("per_symbol", {}) or {}

    def _normalize_active(self, payload: dict) -> dict:
        out = payload if isinstance(payload, dict) else {}
        if "champion" not in out:
            out["champion"] = None
        if "canary" not in out:
            out["canary"] = None
        if "champion_bundle_id" not in out:
            out["champion_bundle_id"] = None
        if "canary_bundle_id" not in out:
            out["canary_bundle_id"] = None
        if "canary_policy" not in out or not isinstance(out.get("canary_policy"), dict):
            out["canary_policy"] = {}
        if "canary_state" not in out or not isinstance(out.get("canary_state"), dict):
            out["canary_state"] = {}
        if "champion_history" not in out or not isinstance(out.get("champion_history"), list):
            out["champion_history"] = []
        if "symbols" not in out or not isinstance(out.get("symbols"), dict):
            out["symbols"] = {}
        for sym, cfg in list(out["symbols"].items()):
            if not isinstance(cfg, dict):
                out["symbols"][sym] = {
                    "champion": None,
                    "canary": None,
                    "champion_bundle_id": None,
                    "canary_bundle_id": None,
                    "canary_policy": {},
                    "canary_state": {},
                    "champion_history": [],
                }
                continue
            if "champion" not in cfg:
                cfg["champion"] = None
            if "canary" not in cfg:
                cfg["canary"] = None
            if "champion_bundle_id" not in cfg:
                cfg["champion_bundle_id"] = None
            if "canary_bundle_id" not in cfg:
                cfg["canary_bundle_id"] = None
            if "canary_policy" not in cfg or not isinstance(cfg.get("canary_policy"), dict):
                cfg["canary_policy"] = {}
            if "canary_state" not in cfg or not isinstance(cfg.get("canary_state"), dict):
                cfg["canary_state"] = {}
            if "champion_history" not in cfg or not isinstance(cfg.get("champion_history"), list):
                cfg["champion_history"] = []
            out["symbols"][sym] = cfg
        return out

    def _safe_acquire_lock(self, operation: str = "registry_op") -> None:
        """Acquire the FileLock with bounded retries + backoff + clear logging.

        This is the core file locking hardening: tolerates transient contention on VPS
        (e.g. training write + live inference load + API read overlapping) without
        hanging the caller forever. Logs on each retry for observability.
        """
        max_retries = 4
        base_backoff = 0.2
        for attempt in range(1, max_retries + 1):
            try:
                # acquire() honors the timeout set on the FileLock instance
                self._lock.acquire()
                return
            except FileLockTimeout as exc:
                if attempt == max_retries:
                    logger.error(
                        f"Registry lock timeout after {max_retries} attempts for {operation} "
                        f"(timeout={self._lock_timeout}s). Possible long-held writer or disk stall."
                    )
                    raise RuntimeError(
                        f"Failed to acquire model registry lock for {operation} after retries"
                    ) from exc
                backoff = base_backoff * (2 ** (attempt - 1))
                logger.warning(
                    f"Registry lock contended ({operation}); retry {attempt}/{max_retries} "
                    f"in {backoff:.2f}s (timeout={self._lock_timeout}s)"
                )
                time.sleep(backoff)
            except Exception as exc:
                logger.error(f"Unexpected registry lock error during {operation}: {exc}")
                raise

    def _release_lock(self) -> None:
        try:
            if self._lock.is_locked:
                self._lock.release()
        except Exception:
            pass  # best effort

    def _read_active(self):
        self._safe_acquire_lock("read_active")
        try:
            try:
                with open(self.active_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return self._normalize_active(payload)
            except json.JSONDecodeError as exc:
                logger.warning(f"active.json is corrupt ({exc}); resetting to empty registry state")
                return self._normalize_active({})
            except Exception as exc:
                logger.warning(f"Failed to read active.json ({exc}); using empty registry state")
                return self._normalize_active({})
        finally:
            self._release_lock()

    def _load_registry_config(self) -> dict:
        try:
            cfg = load_project_config(PROJECT_ROOT, live_mode=False)
            return cfg.get("registry", {}) or {}
        except Exception:
            return {}

    def _write_active(self, payload: dict):
        self._safe_acquire_lock("write_active")
        try:
            normalized = self._normalize_active(payload)
            normalized["registry_metadata"] = self._build_registry_metadata(normalized)

            tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, dir=self.root, suffix=".json", encoding="utf-8")
            try:
                json.dump(normalized, tmp, indent=2)
            finally:
                tmp.close()

            if os.path.exists(self.active_path):
                backup_path = f"{self.active_path}.bak"
                try:
                    shutil.copy2(self.active_path, backup_path)
                except Exception:
                    logger.warning("Unable to backup active registry file.")

            shutil.move(tmp.name, self.active_path)
        finally:
            self._release_lock()

    def _build_registry_metadata(self, active: dict) -> dict:
        meta = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        commit = self._current_git_commit_hash()
        if commit:
            meta["git_commit"] = commit

        for role in ("champion", "canary"):
            path = active.get(role)
            meta[f"{role}_metadata"] = self._gather_candidate_metadata(path)
        return meta

    def _gather_candidate_metadata(self, candidate_dir: str | None) -> dict:
        if not candidate_dir or not os.path.isdir(candidate_dir):
            return {}
        meta = dict(self.read_metadata(candidate_dir) or {})
        scorecard = self._read_scorecard(candidate_dir)
        if scorecard:
            meta["scorecard"] = scorecard
            if "timesteps" in scorecard:
                meta.setdefault("training_timesteps", int(scorecard.get("timesteps") or 0))
        integrity = self._integrity_snapshot(candidate_dir)
        if integrity:
            meta["integrity"] = integrity
        return meta

    def _read_scorecard(self, candidate_dir: str | None) -> dict:
        if not candidate_dir:
            return {}
        scorecard_path = os.path.join(candidate_dir, "scorecard.json")
        if not os.path.exists(scorecard_path):
            return {}
        try:
            with open(scorecard_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _integrity_snapshot(self, candidate_dir: str | None) -> dict:
        """Capture sha256 + size for defense-in-depth integrity (improved check).

        Size + hash makes tampering or partial writes far more detectable on real
        VPS disk scenarios (e.g. interrupted training writes).
        """
        targets = INTEGRITY_TARGETS
        snapshot = {}
        if not candidate_dir:
            return snapshot
        for label, fname in targets.items():
            path = os.path.join(candidate_dir, fname)
            if os.path.exists(path):
                try:
                    size = os.path.getsize(path)
                    h = self._file_hash(path)
                    snapshot[label] = {"hash": h, "size": size}
                except Exception:
                    snapshot[label] = {"hash": self._file_hash(path), "size": None}
        return snapshot

    def _clear_active_entry(self, active: dict, role_key: str, symbol: str | None = None):
        normalized_role = "canary" if "canary" in role_key else "champion"
        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            cur[normalized_role] = None
            if normalized_role == "canary":
                cur["canary_state"] = {}
            symbols[symbol] = cur
            return
        active[normalized_role] = None

    def _validate_candidate_integrity(self, candidate_dir: str | None) -> bool:
        """Validate integrity using recorded hash (+ size if present in new snapshots).

        Improved checks: supports legacy string-hash snapshots and new {hash,size} format.
        On mismatch, logs detailed reason (file, expected vs actual) for auditability
        before any promotion or load in live conditions.
        """
        if not candidate_dir or not os.path.isdir(candidate_dir):
            return False
        recorded = self.read_metadata(candidate_dir).get("integrity")
        if not isinstance(recorded, dict) or not recorded:
            # No integrity recorded (older artifacts) — allow but warn at call sites
            return True

        for label, expected in recorded.items():
            if not expected:
                continue
            fname = INTEGRITY_TARGETS.get(label)
            if not fname:
                continue  # Unknown from older snapshot — skip gracefully
            path = os.path.join(candidate_dir, fname)
            if not os.path.exists(path):
                logger.warning(f"Integrity fail: missing file {label}={fname} for {candidate_dir}")
                return False

            actual_hash = self._file_hash(path)
            actual_size = os.path.getsize(path) if os.path.exists(path) else None

            # Support legacy (str) and new (dict) recorded formats
            if isinstance(expected, dict):
                exp_hash = expected.get("hash")
                exp_size = expected.get("size")
                if exp_hash and str(actual_hash) != str(exp_hash):
                    logger.error(
                        f"Integrity HASH mismatch for {label} in {candidate_dir}: "
                        f"expected={exp_hash[:16]}... actual={actual_hash[:16] if actual_hash else 'None'}..."
                    )
                    return False
                if exp_size is not None and actual_size != exp_size:
                    logger.error(
                        f"Integrity SIZE mismatch for {label} in {candidate_dir}: "
                        f"expected={exp_size} actual={actual_size}"
                    )
                    return False
            else:
                # Legacy string hash only
                if not actual_hash or str(actual_hash) != str(expected):
                    logger.error(
                        f"Integrity (legacy) HASH mismatch for {label} in {candidate_dir}"
                    )
                    return False

        # Full presence check for all current targets (stricter than recorded only)
        for label, fname in INTEGRITY_TARGETS.items():
            path = os.path.join(candidate_dir, fname)
            if not os.path.exists(path):
                logger.warning(f"Integrity fail: required target {label}={fname} missing in {candidate_dir}")
                return False
        return True

    def _file_hash(self, path: str) -> str:
        hash_obj = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    hash_obj.update(chunk)
        except Exception:
            return ""
        return hash_obj.hexdigest()

    def _current_git_commit_hash(self) -> str | None:
        try:
            output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True)
            return output.strip()
        except Exception:
            return None

    def _timestamp_version(self):
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _metadata_targets_symbol(payload: dict | None, symbol: str | None) -> bool:
        if not symbol:
            return True
        meta = payload if isinstance(payload, dict) else {}
        symbol_str = str(symbol)
        single = str(meta.get("symbol", "") or "").strip()
        scoped = {str(item).strip() for item in (meta.get("symbols", []) or []) if str(item).strip()}
        if single:
            return single == symbol_str
        if scoped:
            return symbol_str in scoped
        return True

    def candidate_targets_symbol(self, candidate_dir: str | None, symbol: str | None) -> bool:
        if not symbol or not candidate_dir:
            return True
        meta = self.read_metadata(candidate_dir)
        if meta and self._metadata_targets_symbol(meta, symbol):
            return True
        scorecard = self._read_scorecard(candidate_dir)
        if scorecard:
            return self._metadata_targets_symbol(scorecard, symbol)
        return True

    def new_candidate_dir(self, tag: str = "candidate") -> str:
        ver = f"{tag}_{self._timestamp_version()}"
        path = os.path.join(self.candidates_dir, ver)
        os.makedirs(path, exist_ok=True)
        return path

    def get_symbol_active(self, symbol: str) -> dict:
        active = self._read_active()
        return dict(
            active.get(
                "symbols",
                {},
            ).get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}})
        )

    def _default_canary_policy(self) -> dict:
        base = {
            "min_trades": 10,
            "min_realized_pnl": 0.0,
            "max_drawdown": 0.12,
            "min_runtime_minutes": 30,
        }
        base.update(self._canary_default_overrides)
        return base

    def _max_history(self) -> int:
        try:
            return max(1, int((self.registry_config.get("ensemble", {}) or {}).get("history_limit", 3) or 3))
        except Exception:
            return 3

    def _append_history(self, history: list, candidate_dir: str | None):
        if not candidate_dir:
            return list(history or [])
        items = list(history or [])
        entry = {
            "path": candidate_dir,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self._gather_candidate_metadata(candidate_dir),
        }
        items = [item for item in items if str(item.get("path")) != str(candidate_dir)]
        items.insert(0, entry)
        return items[: self._max_history()]

    def _policy_for_symbol(self, symbol: str | None) -> dict:
        policy = dict(self._default_canary_policy())
        if symbol and symbol in self._canary_symbol_overrides:
            overrides = self._canary_symbol_overrides.get(symbol) or {}
            policy.update(overrides)
        return policy

    def _merge_canary_policy(self, policy: dict | None, symbol: str | None = None) -> dict:
        out = self._policy_for_symbol(symbol)
        if isinstance(policy, dict):
            for k in out.keys():
                if k in policy:
                    out[k] = policy[k]
        return out

    def _canary_passes(self, policy: dict, state: dict) -> tuple[bool, str]:
        trades = int(state.get("trades", 0))
        realized = float(state.get("realized_pnl", 0.0))
        dd = float(state.get("drawdown", 0.0))
        runtime = float(state.get("runtime_minutes", 0.0))

        if trades < int(policy.get("min_trades", 10)):
            return False, f"trades {trades} < min_trades {int(policy.get('min_trades', 10))}"
        if realized < float(policy.get("min_realized_pnl", 0.0)):
            return False, f"realized_pnl {realized:.2f} < min_realized_pnl {float(policy.get('min_realized_pnl', 0.0)):.2f}"
        if dd > float(policy.get("max_drawdown", 0.12)):
            return False, f"drawdown {dd:.4f} > max_drawdown {float(policy.get('max_drawdown', 0.12)):.4f}"
        if runtime < float(policy.get("min_runtime_minutes", 30)):
            return False, f"runtime_minutes {runtime:.1f} < min_runtime_minutes {float(policy.get('min_runtime_minutes', 30)):.1f}"
        return True, "ok"

    def load_active_model(self, prefer_canary: bool = True, symbol: str | None = None) -> str | None:
        active = self._read_active()
        updated = False

        def resolve_path(path: str | None, role: str, sym: str | None = None) -> str | None:
            nonlocal updated
            if not path:
                return None
            if sym and not self.candidate_targets_symbol(path, sym):
                logger.error("Candidate symbol mismatch for {} ({} -> {}). Clearing entry.", role, sym, path)
                self._clear_active_entry(active, role, sym)
                updated = True
                return None
            if self._validate_candidate_integrity(path):
                return path
            logger.error("Candidate integrity mismatch for %s (%s). Clearing entry.", role, path)
            self._clear_active_entry(active, role, sym)
            updated = True
            return None

        symbols = active.get("symbols", {})
        symbol_entry = symbols.get(symbol, {}) if symbol else {}

        if symbol:
            canary_path = symbol_entry.get("canary")
            champion_path = symbol_entry.get("champion")
            if prefer_canary:
                resolved = resolve_path(canary_path, "symbol_canary", symbol)
                if resolved:
                    if updated:
                        self._write_active(active)
                        updated = False
                    return resolved
            resolved = resolve_path(champion_path, "symbol_champion", symbol)
            if resolved:
                if updated:
                    self._write_active(active)
                    updated = False
                return resolved

        if prefer_canary:
            resolved = resolve_path(active.get("canary"), "canary", symbol)
            if resolved:
                if updated:
                    self._write_active(active)
                    updated = False
                return resolved
        resolved = resolve_path(active.get("champion"), "champion", symbol)
        if resolved:
            if updated:
                self._write_active(active)
                updated = False
            return resolved

        if updated:
            self._write_active(active)
        return None

    def set_canary(self, version_dir: str, symbol: str | None = None, policy: dict | None = None, bundle_id: str | None = None):
        active = self._read_active()
        merged = self._merge_canary_policy(policy, symbol)
        if symbol:
            if not self.candidate_targets_symbol(version_dir, symbol):
                raise RuntimeError(f"Cannot set canary for {symbol}: artifact is not tagged for that symbol.")
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            cur["canary"] = version_dir
            cur["canary_bundle_id"] = bundle_id
            cur["canary_policy"] = merged
            cur["canary_state"] = {"passed": False, "reason": "no_metrics"}
            symbols[symbol] = cur
            self._write_active(active)
            logger.warning(f"Canary set for {symbol}: {version_dir} bundle={bundle_id}")
            self._append_promotion_audit("set_canary", self._enrich_promotion_details({
                "symbol": symbol,
                "role": "canary",
                "path": version_dir,
                "bundle_id": bundle_id,
                "policy": merged,
            }))
            return

        active["canary"] = version_dir
        active["canary_bundle_id"] = bundle_id
        active["canary_policy"] = merged
        active["canary_state"] = {"passed": False, "reason": "no_metrics"}
        self._write_active(active)
        logger.warning(f"Canary set: {version_dir} bundle={bundle_id}")
        self._append_promotion_audit("set_canary", self._enrich_promotion_details({
            "symbol": None,
            "role": "canary",
            "path": version_dir,
            "bundle_id": bundle_id,
            "policy": merged,
        }))

    def update_canary_metrics(
        self,
        trades: int,
        realized_pnl: float,
        drawdown: float,
        runtime_minutes: float,
        symbol: str | None = None,
    ) -> dict:
        active = self._read_active()
        state = {
            "trades": int(trades),
            "realized_pnl": float(realized_pnl),
            "drawdown": float(drawdown),
            "runtime_minutes": float(runtime_minutes),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            policy = self._merge_canary_policy(cur.get("canary_policy"), symbol)
            passed, reason = self._canary_passes(policy, state)
            state["passed"] = bool(passed)
            state["reason"] = reason
            cur["canary_policy"] = policy
            cur["canary_state"] = state
            symbols[symbol] = cur
            self._write_active(active)
            return state

        policy = self._merge_canary_policy(active.get("canary_policy"))
        passed, reason = self._canary_passes(policy, state)
        state["passed"] = bool(passed)
        state["reason"] = reason
        active["canary_policy"] = policy
        active["canary_state"] = state
        self._write_active(active)
        return state

    def can_promote_canary(self, symbol: str | None = None) -> tuple[bool, str]:
        active = self._read_active()
        if symbol:
            cur = dict(active.get("symbols", {}).get(symbol, {"champion": None, "canary": None, "canary_state": {}}))
            if not cur.get("canary"):
                return False, f"No canary to promote for {symbol}."
            state = cur.get("canary_state", {}) if isinstance(cur.get("canary_state"), dict) else {}
            if bool(state.get("passed", False)):
                return True, "ok"
            return False, str(state.get("reason", "canary survival checks not satisfied"))

        if not active.get("canary"):
            return False, "No canary to promote."
        state = active.get("canary_state", {}) if isinstance(active.get("canary_state"), dict) else {}
        if bool(state.get("passed", False)):
            return True, "ok"
        return False, str(state.get("reason", "canary survival checks not satisfied"))

    def promote_canary_to_champion(self, symbol: str | None = None, force: bool = False):
        active = self._read_active()
        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            if not cur.get("canary"):
                raise RuntimeError(f"No canary to promote for {symbol}.")
            if not force:
                ok, reason = self.can_promote_canary(symbol=symbol)
                if not ok:
                    raise RuntimeError(f"Canary promotion blocked for {symbol}: {reason}")
            old_champion = cur.get("champion")
            cur["champion_history"] = self._append_history(cur.get("champion_history", []), old_champion)
            cur["champion"] = cur["canary"]
            cur["champion_bundle_id"] = cur.get("canary_bundle_id")
            cur["canary"] = None
            cur["canary_bundle_id"] = None
            cur["canary_state"] = {}
            symbols[symbol] = cur
            self._write_active(active)
            logger.success(f"Promoted {symbol} champion: {cur['champion']} bundle={cur['champion_bundle_id']}")
            self._append_promotion_audit("promote_canary_to_champion", self._enrich_promotion_details({
                "symbol": symbol,
                "old_champion": old_champion,
                "new_champion": cur["champion"],
                "bundle_id": cur.get("champion_bundle_id"),
                "forced": force,
            }))
            return

        if not active.get("canary"):
            raise RuntimeError("No canary to promote.")
        if not force:
            ok, reason = self.can_promote_canary()
            if not ok:
                raise RuntimeError(f"Canary promotion blocked: {reason}")
        old_champion = active.get("champion")
        active["champion_history"] = self._append_history(active.get("champion_history", []), old_champion)
        active["champion"] = active["canary"]
        active["champion_bundle_id"] = active.get("canary_bundle_id")
        active["canary"] = None
        active["canary_bundle_id"] = None
        active["canary_state"] = {}
        self._write_active(active)
        logger.success(f"Promoted champion: {active['champion']} bundle={active['champion_bundle_id']}")
        self._append_promotion_audit("promote_canary_to_champion", self._enrich_promotion_details({
            "symbol": None,
            "old_champion": old_champion,
            "new_champion": active["champion"],
            "bundle_id": active.get("champion_bundle_id"),
            "forced": force,
        }))

    def clear_canary(self, symbol: str | None = None):
        active = self._read_active()
        if symbol:
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            cur["canary"] = None
            cur["canary_state"] = {}
            symbols[symbol] = cur
            self._write_active(active)
            logger.warning(f"Canary cleared for {symbol}")
            self._append_promotion_audit("clear_canary", self._enrich_promotion_details({"symbol": symbol}))
            return

        active["canary"] = None
        active["canary_state"] = {}
        self._write_active(active)
        logger.warning("Canary cleared")
        self._append_promotion_audit("clear_canary", self._enrich_promotion_details({"symbol": None}))

    def rollback_to_champion(self, symbol: str | None = None):
        self.clear_canary(symbol=symbol)

    def register_candidate(self, candidate_dir: str, metadata: dict | None = None):
        meta = dict(self.read_metadata(candidate_dir) or {})
        meta.update(metadata or {})
        integrity = self._integrity_snapshot(candidate_dir)
        if integrity:
            meta["integrity"] = integrity
        scorecard = self._read_scorecard(candidate_dir)
        if scorecard:
            meta["scorecard"] = scorecard
            meta.setdefault("training_timesteps", int(scorecard.get("timesteps", 0) or 0))
        meta["registered_at"] = datetime.now(timezone.utc).isoformat()
        meta_path = os.path.join(candidate_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Candidate registered: {candidate_dir}")

    def update_metadata(self, candidate_dir: str, patch: dict):
        meta = self.read_metadata(candidate_dir)
        if not isinstance(meta, dict):
            meta = {}
        meta.update(patch or {})
        self.register_candidate(candidate_dir, meta)

    def read_metadata(self, version_dir: str) -> dict:
        meta_path = os.path.join(version_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_recent_champions(self, symbol: str | None = None) -> list[dict]:
        active = self._read_active()
        if symbol:
            return list((active.get("symbols", {}).get(symbol, {}) or {}).get("champion_history", []) or [])
        return list(active.get("champion_history", []) or [])

    def validate_candidate_for_promotion(self, candidate_dir: str) -> tuple[bool, list[str]]:
        """Validate a candidate against champion promotion gates.

        Gates:
          - data_source must be "mt5"
          - timesteps >= 10000
          - backtest return >= 0 (non-negative)
          - max_drawdown <= 15%
        """
        meta = self.read_metadata(candidate_dir)
        scorecard = self._read_scorecard(candidate_dir)
        evaluation = meta.get("evaluation") if isinstance(meta, dict) else None

        data_source = (meta.get("data_source") or scorecard.get("data_source") or "unknown")
        timesteps = int(meta.get("timesteps") or scorecard.get("timesteps") or 0)

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

        reasons = []
        if data_source != "mt5":
            reasons.append(f"data_source_fail:{data_source}!=mt5")
        if timesteps < 10000:
            reasons.append(f"timesteps_fail:{timesteps}<10000")
        if total_return < 0.0:
            reasons.append(f"backtest_return_fail:{total_return:.4f}<0.0000")
        if max_drawdown > 0.15:
            reasons.append(f"max_drawdown_fail:{max_drawdown:.4f}>0.1500")

        passed = len(reasons) == 0
        return passed, reasons

    def promote_with_gates(self, candidate_dir: str, symbol: str | None = None) -> tuple[bool, str]:
        """Promote candidate directly to champion only if all promotion gates pass."""
        passed, reasons = self.validate_candidate_for_promotion(candidate_dir)
        if not passed:
            reason_str = ", ".join(reasons)
            logger.error(f"Promotion gates failed for {candidate_dir}: {reason_str}")
            return False, reason_str

        active = self._read_active()
        if symbol:
            if not self.candidate_targets_symbol(candidate_dir, symbol):
                msg = f"Candidate {candidate_dir} is not tagged for symbol {symbol}"
                logger.error(msg)
                return False, msg
            symbols = active.setdefault("symbols", {})
            cur = dict(symbols.get(symbol, {"champion": None, "canary": None, "canary_policy": {}, "canary_state": {}}))
            cur["champion_history"] = self._append_history(cur.get("champion_history", []), candidate_dir)
            cur["champion"] = candidate_dir
            symbols[symbol] = cur
            self._write_active(active)
            logger.success(f"Promoted {symbol} champion with gates: {candidate_dir}")
            self._append_promotion_audit("promote_with_gates", self._enrich_promotion_details({
                "symbol": symbol,
                "candidate": candidate_dir,
                "via": "gates",
            }))
            return True, "ok"

        # Global champion promotion
        active["champion_history"] = self._append_history(active.get("champion_history", []), candidate_dir)
        active["champion"] = candidate_dir
        self._write_active(active)
        logger.success(f"Promoted global champion with gates: {candidate_dir}")
        self._append_promotion_audit("promote_with_gates", self._enrich_promotion_details({
            "symbol": None,
            "candidate": candidate_dir,
            "via": "gates",
        }))
        return True, "ok"

    # ── Per-symbol accessors (added to satisfy test suite) ───────────────

    def _ensure_symbol_entry(self, active: dict, symbol: str) -> dict:
        """Ensure active['symbols'][symbol] exists and return it."""
        symbols = active.setdefault("symbols", {})
        if symbol not in symbols:
            symbols[symbol] = {
                "champion": None,
                "canary": None,
                "canary_policy": {},
                "canary_state": {},
                "champion_history": [],
            }
        return symbols[symbol]

    def set_champion(self, symbol: str, path: str):
        """Set per-symbol champion directly."""
        active = self._read_active()
        cur = self._ensure_symbol_entry(active, symbol)
        cur["champion"] = path
        self._write_active(active)
        logger.info(f"Champion set for {symbol}: {path}")

    def get_champion(self, symbol: str | None = None) -> str | None:
        """Get champion path for a symbol (or global if symbol is None)."""
        active = self._read_active()
        if symbol:
            symbols = active.get("symbols", {})
            if symbol in symbols:
                return symbols[symbol].get("champion")
            return active.get("champion")
        return active.get("champion")

    def get_canary(self, symbol: str | None = None) -> str | None:
        """Get canary path for a symbol (or global if symbol is None)."""
        active = self._read_active()
        if symbol:
            symbols = active.get("symbols", {})
            if symbol in symbols:
                return symbols[symbol].get("canary")
            return active.get("canary")
        return active.get("canary")

    def get_active_model(self, symbol: str | None = None, prefer_canary: bool = True) -> str | None:
        """Alias for load_active_model with keyword-friendly arg order."""
        return self.load_active_model(prefer_canary=prefer_canary, symbol=symbol)

    def get_all_symbols(self) -> dict:
        """Return a copy of the per-symbol registry map."""
        active = self._read_active()
        return dict(active.get("symbols", {}))

    # ── Bundle ID tracking ───────────────────────────────────────────────

    def set_champion_bundle_id(self, symbol: str | None, bundle_id: str | None):
        """Attach a bundle_id to the current champion entry."""
        active = self._read_active()
        if symbol:
            cur = self._ensure_symbol_entry(active, symbol)
            cur["champion_bundle_id"] = bundle_id
        else:
            active["champion_bundle_id"] = bundle_id
        self._write_active(active)
        logger.info(f"Champion bundle_id set for {symbol or 'global'}: {bundle_id}")

    def set_canary_bundle_id(self, symbol: str | None, bundle_id: str | None):
        """Attach a bundle_id to the current canary entry."""
        active = self._read_active()
        if symbol:
            cur = self._ensure_symbol_entry(active, symbol)
            cur["canary_bundle_id"] = bundle_id
        else:
            active["canary_bundle_id"] = bundle_id
        self._write_active(active)
        logger.info(f"Canary bundle_id set for {symbol or 'global'}: {bundle_id}")

    def get_champion_bundle_id(self, symbol: str | None = None) -> str | None:
        """Get champion bundle_id for a symbol (or global if symbol is None)."""
        active = self._read_active()
        if symbol:
            symbols = active.get("symbols", {})
            if symbol in symbols:
                return symbols[symbol].get("champion_bundle_id")
            return active.get("champion_bundle_id")
        return active.get("champion_bundle_id")

    def get_canary_bundle_id(self, symbol: str | None = None) -> str | None:
        """Get canary bundle_id for a symbol (or global if symbol is None)."""
        active = self._read_active()
        if symbol:
            symbols = active.get("symbols", {})
            if symbol in symbols:
                return symbols[symbol].get("canary_bundle_id")
            return active.get("canary_bundle_id")
        return active.get("canary_bundle_id")

    # ── Hardened integrity audit + promotion logging (sprint improvements) ─

    def audit_integrity(self, symbol: str | None = None) -> dict:
        """Public self-audit for registry integrity (callable pre-cycle or from audit_registry.py).

        Returns summary with per-role status and any failures. Safe to call anytime.
        """
        active = self._read_active()
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "root": self.root,
            "lock_timeout": self._lock_timeout,
            "symbols_checked": [],
            "global": {},
            "failures": [],
        }

        def _check_role(role: str, path: str | None, sym: str | None = None) -> dict:
            if not path:
                return {"present": False}
            ok = self._validate_candidate_integrity(path)
            meta = self._gather_candidate_metadata(path)  # includes integrity + scorecard
            entry = {"path": path, "valid": bool(ok), "meta_summary": {k: meta.get(k) for k in ("symbol", "training_timesteps", "integrity") if k in meta}}
            if not ok:
                report["failures"].append(f"{role}{f'[{sym}]' if sym else ''}:{path}")
            return entry

        # Global
        report["global"]["champion"] = _check_role("global_champion", active.get("champion"))
        report["global"]["canary"] = _check_role("global_canary", active.get("canary"))

        # Per symbol
        for sym, entry in (active.get("symbols") or {}).items():
            sym_report = {
                "champion": _check_role("champion", entry.get("champion"), sym),
                "canary": _check_role("canary", entry.get("canary"), sym),
            }
            report["symbols_checked"].append({"symbol": sym, "entries": sym_report})

        if report["failures"]:
            logger.warning(f"Registry integrity audit found failures: {report['failures']}")
        else:
            logger.info("Registry integrity audit passed for all known entries")
        return report

    _PROMOTION_AUDIT_LOG = os.path.join(
        # Lazy init in method to avoid import-time side effects
        "",  # placeholder; resolved at runtime in _append_promotion_audit
    )

    def _append_promotion_audit(self, event: str, details: dict) -> None:
        """Structured promotion audit trail (clearer logging improvement).

        Appends one JSON line per promotion/canary transition for easy forensics on
        first real MT5 cycles and post-incident review. Complements (does not replace)
        loguru logs and bundle promotion jsonl.
        """
        try:
            log_dir = os.path.join(self.root)
            log_path = os.path.join(log_dir, "promotion_audit.jsonl")
            os.makedirs(log_dir, exist_ok=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "details": details,
                "git_commit": self._current_git_commit_hash(),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to append promotion audit log: {exc}")

    def _enrich_promotion_details(self, base: dict, active_before: dict | None = None) -> dict:
        d = dict(base or {})
        d.setdefault("lock_timeout", self._lock_timeout)
        return d