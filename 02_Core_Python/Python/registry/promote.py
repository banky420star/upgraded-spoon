"""Promote a model bundle through promotion gates."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("promote")  # type: ignore

from Python.ensemble.model_bundle import ModelBundle
from Python.registry.promotion_gates import PromotionGates

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROMOTION_LOG = os.path.join(_PROJECT_ROOT, "models", "registry", "promotion_log.jsonl")


def promote_bundle(
    bundle_id: str,
    validation_report: dict,
    gates_config: Optional[dict] = None,
    bundles_dir: Optional[str] = None,
) -> dict:
    """Run promotion gates and promote bundle to demo_canary or reject with reasons.

    Returns:
        {
            "bundle_id": str,
            "action": "promoted_to_demo_canary" | "rejected",
            "passed": bool,
            "reasons": list[str],
            "timestamp": str,
        }
    """
    gates = PromotionGates(config=gates_config)
    passed, reasons = gates.evaluate(bundle_id, validation_report)

    bundle = ModelBundle.load(bundle_id, bundles_dir=bundles_dir)
    if bundle is None:
        # If no bundle file exists yet, create a stub so the lifecycle is tracked
        bundle = ModelBundle(
            bundle_id=bundle_id,
            symbol=validation_report.get("symbol", "UNKNOWN"),
            timeframe=validation_report.get("timeframe", "5m"),
            dataset_id=validation_report.get("metadata", {}).get("dataset_id", ""),
            feature_set_id=validation_report.get("metadata", {}).get("feature_set_id", ""),
            label_set_id=validation_report.get("metadata", {}).get("label_set_id", ""),
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "bundle_id": bundle_id,
        "action": "promoted_to_demo_canary" if passed else "rejected",
        "passed": passed,
        "reasons": list(reasons),
        "timestamp": timestamp,
    }

    if passed:
        bundle.set_status("demo_canary")
        logger.success(f"Bundle {bundle_id} promoted to demo_canary")
    else:
        bundle.set_status("rejected")
        logger.error(f"Bundle {bundle_id} rejected: {reasons}")

    bundle.save(bundles_dir=bundles_dir)
    _append_log(result)
    return result


def _append_log(entry: dict) -> None:
    os.makedirs(os.path.dirname(_PROMOTION_LOG), exist_ok=True)
    import json
    with open(_PROMOTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
