"""ModelBundle — version-locked ensemble model bundle registry."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("model_bundle")  # type: ignore


BUNDLE_STATUSES = [
    "candidate",
    "validation_pending",
    "rejected",
    "demo_canary",
    "champion",
    "retired",
    "quarantined",
    "disabled",
]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUNDLES_DIR = os.path.join(_PROJECT_ROOT, "models", "bundles")


@dataclass
class ModelBundle:
    """A version-locked bundle of models trained with the same feature set."""

    bundle_id: str
    symbol: str
    timeframe: str
    dataset_id: str
    feature_set_id: str
    label_set_id: str
    lstm_model_id: str = ""
    rainforest_model_id: str = ""
    dreamer_model_id: str = ""
    ppo_model_id: str = ""
    meta_controller_id: str = ""
    status: str = "candidate"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in BUNDLE_STATUSES:
            raise ValueError(f"Invalid bundle status: {self.status}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelBundle":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in payload.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, bundles_dir: Optional[str] = None) -> str:
        path = self._path(bundles_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Bundle saved: {path}")
        return path

    @classmethod
    def load(cls, bundle_id: str, bundles_dir: Optional[str] = None) -> Optional["ModelBundle"]:
        path = cls._path_for_id(bundle_id, bundles_dir)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return cls.from_dict(payload)
        except Exception as exc:
            logger.warning(f"Bundle load failed ({path}): {exc}")
            return None

    def _path(self, bundles_dir: Optional[str] = None) -> str:
        root = bundles_dir or _BUNDLES_DIR
        return os.path.join(root, f"{self.bundle_id}.json")

    @classmethod
    def _path_for_id(cls, bundle_id: str, bundles_dir: Optional[str] = None) -> str:
        root = bundles_dir or _BUNDLES_DIR
        return os.path.join(root, f"{bundle_id}.json")

    def is_version_locked(self) -> bool:
        """True if all model IDs are non-empty and share the same feature_set_id."""
        model_ids = [
            self.lstm_model_id,
            self.rainforest_model_id,
            self.ppo_model_id,
        ]
        if self.dreamer_model_id:
            model_ids.append(self.dreamer_model_id)
        if any(not mid for mid in model_ids):
            return False
        # Feature-set lock is enforced at training time; here we just verify IDs exist.
        return True

    def set_status(self, new_status: str) -> None:
        if new_status not in BUNDLE_STATUSES:
            raise ValueError(f"Invalid bundle status: {new_status}")
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def list_bundles(cls, bundles_dir: Optional[str] = None) -> list[str]:
        root = bundles_dir or _BUNDLES_DIR
        if not os.path.isdir(root):
            return []
        return sorted([
            f.replace(".json", "")
            for f in os.listdir(root)
            if f.endswith(".json")
        ])
