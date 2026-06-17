"""Quarantine a model bundle (unsafe but not permanently rejected)."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("quarantine")  # type: ignore

from Python.ensemble.model_bundle import ModelBundle

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUARANTINE_LOG = os.path.join(_PROJECT_ROOT, "models", "registry", "quarantine_log.jsonl")


def quarantine_bundle(
    bundle_id: str,
    reason: str,
    bundles_dir: Optional[str] = None,
) -> dict:
    """Mark a bundle as quarantined and log the reason.

    Returns:
        {
            "bundle_id": str,
            "action": "quarantined",
            "reason": str,
            "timestamp": str,
        }
    """
    bundle = ModelBundle.load(bundle_id, bundles_dir=bundles_dir)
    if bundle is None:
        logger.warning(f"Quarantining nonexistent bundle {bundle_id}; creating stub")
        bundle = ModelBundle(
            bundle_id=bundle_id,
            symbol="UNKNOWN",
            timeframe="5m",
            dataset_id="",
            feature_set_id="",
            label_set_id="",
        )

    bundle.set_status("quarantined")
    bundle.metadata["quarantine_reason"] = reason
    bundle.metadata["quarantined_at"] = datetime.now(timezone.utc).isoformat()
    bundle.save(bundles_dir=bundles_dir)

    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "bundle_id": bundle_id,
        "action": "quarantined",
        "reason": reason,
        "timestamp": timestamp,
    }

    _append_quarantine_log(result)
    logger.warning(f"Bundle {bundle_id} quarantined: {reason}")
    return result


def _append_quarantine_log(entry: dict) -> None:
    import json
    os.makedirs(os.path.dirname(_QUARANTINE_LOG), exist_ok=True)
    with open(_QUARANTINE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
