import pandas as pd
from loguru import logger


class LabelValidator:
    """Ensure label columns never leak into the live feature matrix."""

    LABEL_PREFIXES = ("target_", "future_", "label_", "next_")

    def validate(self, feature_df: pd.DataFrame, label_df: pd.DataFrame) -> dict:
        """Return a validation report; raise RuntimeError if leakage found."""
        report = {
            "label_columns": list(label_df.columns),
            "feature_count": len(feature_df.columns),
            "leakage_detected": False,
            "details": [],
        }

        # 1. No overlap
        overlap = set(feature_df.columns) & set(label_df.columns)
        if overlap:
            report["leakage_detected"] = True
            msg = f"LabelValidator: label columns found in features: {sorted(overlap)}"
            report["details"].append(msg)
            logger.error(msg)

        # 2. No label-like prefixes in features
        forbidden = [c for c in feature_df.columns if c.lower().startswith(self.LABEL_PREFIXES)]
        if forbidden:
            report["leakage_detected"] = True
            msg = f"LabelValidator: forbidden prefixes in feature columns: {forbidden}"
            report["details"].append(msg)
            logger.error(msg)

        # 3. Length / index alignment
        if len(feature_df) != len(label_df):
            report["details"].append(
                f"Length mismatch: features={len(feature_df)} labels={len(label_df)}"
            )
        else:
            aligned = feature_df.index.equals(label_df.index)
            if not aligned:
                report["details"].append("Index mismatch between features and labels")

        if report["leakage_detected"]:
            raise RuntimeError("LabelValidator: label leakage into features detected.\n" + "\n".join(report["details"]))

        logger.info("LabelValidator: features are clean — no label leakage.")
        return report
