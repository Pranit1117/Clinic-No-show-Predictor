"""
Data validation module using Great Expectations-style checks.
Validates schema, ranges, and business rules before feature engineering.
"""

import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    failures: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.passed:
            return "✅ All validation checks passed."
        return "❌ Validation failed:\n" + "\n".join(f"  - {f}" for f in self.failures)


def validate_raw_data(df: pd.DataFrame) -> ValidationResult:
    """Run all validation checks on the raw DataFrame."""
    failures = []

    # ── Schema checks ────────────────────────────────────────────────────────
    required_columns = [
        "PatientId", "AppointmentID", "Gender", "ScheduledDay",
        "AppointmentDay", "Age", "Neighbourhood", "Scholarship",
        "Hipertension", "Diabetes", "Alcoholism", "Handcap",
        "SMS_received", "No-show",
    ]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        failures.append(f"Missing columns: {missing_cols}")

    # ── Nulls check ───────────────────────────────────────────────────────────
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        failures.append(f"Null values found:\n{null_counts[null_counts > 0]}")

    # ── Age range ─────────────────────────────────────────────────────────────
    if "Age" in df.columns:
        invalid_ages = df[(df["Age"] < 0) | (df["Age"] > 110)]
        if len(invalid_ages) > 0:
            failures.append(f"Age out of range [0, 110]: {len(invalid_ages)} rows")

    # ── Date logic: AppointmentDay must be >= ScheduledDay ───────────────────
    if "ScheduledDay" in df.columns and "AppointmentDay" in df.columns:
        sched = pd.to_datetime(df["ScheduledDay"]).dt.tz_localize(None)
        appt = pd.to_datetime(df["AppointmentDay"]).dt.tz_localize(None)
        bad_dates = df[appt < sched]
        if len(bad_dates) > 0:
            failures.append(f"AppointmentDay < ScheduledDay: {len(bad_dates)} rows (will be dropped)")

    # ── Binary column checks ──────────────────────────────────────────────────
    for col in ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "SMS_received"]:
        if col in df.columns:
            unique_vals = set(df[col].unique())
            if not unique_vals.issubset({0, 1}):
                failures.append(f"Non-binary values in {col}: {unique_vals}")

    # ── Target column check ───────────────────────────────────────────────────
    if "No-show" in df.columns:
        valid_targets = {"Yes", "No"}
        actual = set(df["No-show"].unique())
        if not actual.issubset(valid_targets):
            failures.append(f"Unexpected target values: {actual}")

    # ── Minimum size ──────────────────────────────────────────────────────────
    if len(df) < 1000:
        failures.append(f"Dataset too small: {len(df)} rows (expected >= 1000)")

    result = ValidationResult(passed=len(failures) == 0, failures=failures)
    logger.info(str(result))
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    df = pd.read_csv(Path("data/raw/appointments.csv"))
    result = validate_raw_data(df)
    print(result)
    sys.exit(0 if result.passed else 1)
