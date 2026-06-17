"""Reject a model bundle and log the reasons."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("reject")  # type: ignore

from Python.ensemble.model_bundle import ModelBundle

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REJECTION_LOG = os.path.join(_PROJECT_ROOT, "models", "registry", "rejection_log.jsonl")


def reject_bundle(
    bundle_id: str,
    reasons: list[str],
    bundles_dir: Optional[str] = None,
) -> dict:
    """Mark a bundle as rejected and log the reasons.

    Returns:
        {
            "bundle_id": str,
            "action": "rejected",
            "reasons": list[str],
            "timestamp": str,
        }
    """
    bundle = ModelBundle.load(bundle_id, bundles_dir=bundles_dir)
    if bundle is None:
        logger.warning(f"Rejecting nonexistent bundle {bundle_id}; creating stub")
        bundle = ModelBundle(
            bundle_id=bundle_id,
            symbol="UNKNOWN",
            timeframe="5m",
            dataset_id="",
            feature_set_id="",
            label_set_id="",
        )

    bundle.set_status("rejected")
    bundle.metadata["rejected_reasons"] = list(reasons)
    bundle.metadata["rejected_at"] = datetime.now(timezone.utc).isoformat()
    bundle.save(bundles_dir=bundles_dir)

    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "bundle_id": bundle_id,
        "action": "rejected",
        "reasons": list(reasons),
        "timestamp": timestamp,
    }

    _append_rejection_log(result)
    logger.error(f"Bundle {bundle_id} rejected: {reasons}")
    return result


def _append_rejection_log(entry: dict) -> None:
    import json
    os.makedirs(os.path.dirname(_REJECTION_LOG), exist_ok=True)
    with open(_REJECTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
