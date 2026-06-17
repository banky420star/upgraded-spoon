import pandas as pd
from loguru import logger


class LeakageDetector:
    """Ensure no label/future data leaks into the live feature set."""

    FORBIDDEN_PREFIXES = ("target_", "future_", "label_", "next_")

    def __init__(self):
        self.violations: list[str] = []

    def assert_no_leakage(
        self,
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame | None = None,
        strict: bool = True,
    ) -> None:
        """Raise RuntimeError if any leakage is detected."""
        self.violations.clear()

        # 1. Check for forbidden prefixes in feature columns
        forbidden = [c for c in feature_df.columns if c.lower().startswith(self.FORBIDDEN_PREFIXES)]
        if forbidden:
            self.violations.append(f"Forbidden label-like columns in features: {forbidden}")

        # 2. Check overlap with label DataFrame
        if label_df is not None:
            overlap = set(feature_df.columns) & set(label_df.columns)
            if overlap:
                self.violations.append(f"Feature/label column overlap: {sorted(overlap)}")

        # 3. Heuristic: columns that contain shifted forward-looking names
        suspicious = [c for c in feature_df.columns if "_future" in c.lower() or "_lead" in c.lower()]
        if suspicious:
            self.violations.append(f"Suspicious future-looking columns: {suspicious}")

        if self.violations:
            msg = "LeakageDetector: LEAKAGE DETECTED!\n" + "\n".join(self.violations)
            logger.error(msg)
            if strict:
                raise RuntimeError(msg)
        else:
            logger.info("LeakageDetector: no leakage detected.")

    def check_future_data_in_features(self, feature_df: pd.DataFrame, strict: bool = True) -> None:
        """Standalone check for future-data usage inside feature columns."""
        suspicious = [c for c in feature_df.columns if "_future" in c.lower() or "_lead" in c.lower()]
        if suspicious:
            msg = f"LeakageDetector: future-data columns found: {suspicious}"
            logger.error(msg)
            if strict:
                raise RuntimeError(msg)
