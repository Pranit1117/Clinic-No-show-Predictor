"""
Feature engineering module.
Transforms raw appointment data into model-ready features.
All transformations are documented with business rationale.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

RAW_PATH = Path("data/external/appointments_enriched.parquet")
PROCESSED_PATH = Path("data/processed/features.parquet")


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Remove impossible ages
    - Remove rows where appointment precedes scheduling
    - Standardize column names
    """
    logger.info("Cleaning raw data...")
    initial_count = len(df)

    # Drop impossible ages
    df = df[(df["Age"] >= 0) & (df["Age"] <= 110)].copy()

    # Parse dates (remove timezone for consistency)
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"]).dt.tz_localize(None)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"]).dt.tz_localize(None)

    # Drop rows where appointment is before scheduling (data entry errors)
    df = df[df["AppointmentDay"] >= df["ScheduledDay"]].copy()

    # Binary target
    df["no_show"] = (df["No-show"] == "Yes").astype(int)

    logger.info(f"Cleaned: {initial_count:,} → {len(df):,} rows ({initial_count - len(df):,} dropped)")
    return df


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features — biggest signal drivers.
    Longer lead time = higher no-show rate.
    """
    logger.info("Engineering temporal features...")

    # Lead time (days between scheduling and appointment)
    df["lead_time_days"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days

    # Cap extreme lead times (>60 days are rare, may be rescheduled)
    df["lead_time_days"] = df["lead_time_days"].clip(0, 60)

    # Day of week (0=Mon, 6=Sun) — Monday has highest no-show rates
    df["appt_day_of_week"] = df["AppointmentDay"].dt.dayofweek
    df["appt_is_monday"] = (df["appt_day_of_week"] == 0).astype(int)

    # Hour of scheduling (proxy for how urgent the patient felt)
    df["schedule_hour"] = df["ScheduledDay"].dt.hour

    # Was the appointment same day? (emergency/urgent appointments rarely no-show)
    df["is_same_day"] = (df["lead_time_days"] == 0).astype(int)

    # Week of year (seasonality)
    df["appt_week"] = df["AppointmentDay"].dt.isocalendar().week.astype(int)

    return df


def engineer_patient_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Patient-level rolling no-show rate.
    CRITICAL: Must be calculated without data leakage.
    We sort by date and compute expanding mean on PAST appointments only.
    """
    logger.info("Engineering patient history features...")
    df = df.sort_values(["PatientId", "AppointmentDay"]).copy()

    # Rolling no-show rate (expanding window, shift by 1 to avoid leakage)
    df["patient_noshow_rate"] = (
        df.groupby("PatientId")["no_show"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.20)  # Fill first appointment with population mean
    )

    # Total past appointments for this patient
    df["patient_appt_count"] = (
        df.groupby("PatientId").cumcount()
    )

    # Is this patient a repeat no-shower? (>=2 past no-shows)
    df["patient_is_repeat_noshower"] = (
        df.groupby("PatientId")["no_show"]
        .transform(lambda x: (x.shift(1).expanding().sum() >= 2).astype(int))
        .fillna(0)
    )

    return df


def engineer_health_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine chronic conditions into a composite risk score."""
    logger.info("Engineering health/clinical features...")

    df["chronic_condition_count"] = (
        df["Hipertension"].astype(int)
        + df["Diabetes"].astype(int)
        + df["Alcoholism"].astype(int)
    )

    # Handicap level (0-4 in dataset, higher = more severe)
    df["has_handicap"] = (df["Handcap"] > 0).astype(int)

    # Scholarship (Brazilian welfare program — proxy for socioeconomic status)
    df["scholarship"] = df["Scholarship"].astype(int)

    return df


def engineer_sms_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMS intervention features.
    Did they receive a reminder? How many days before appointment?
    """
    logger.info("Engineering SMS features...")
    df["sms_received"] = df["SMS_received"].astype(int)

    # Interaction: sms sent but still high lead time (reminder too early)
    df["sms_late_reminder"] = (
        (df["sms_received"] == 1) & (df["lead_time_days"] > 14)
    ).astype(int)

    return df


def engineer_neighbourhood_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neighbourhood-level no-show rate (target encoding — done in encode.py).
    Here we just compute the raw rate for reference.
    """
    logger.info("Engineering neighbourhood features...")
    neigh_rate = df.groupby("Neighbourhood")["no_show"].transform("mean")
    df["neighbourhood_noshow_rate_raw"] = neigh_rate
    return df


def engineer_appointment_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    How many appointments are scheduled on the same day?
    Overloaded days = higher no-show risk.
    """
    logger.info("Engineering appointment load feature...")
    load = df.groupby("AppointmentDay")["AppointmentID"].transform("count")
    df["daily_appointment_load"] = load
    # Normalize to percentile rank
    df["daily_load_percentile"] = df["daily_appointment_load"].rank(pct=True)
    return df


def select_final_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select the final feature set for modeling.
    Returns (X, y).
    """
    feature_cols = [
        # Temporal
        "lead_time_days", "appt_day_of_week", "appt_is_monday",
        "schedule_hour", "is_same_day", "appt_week",
        # Patient history
        "patient_noshow_rate", "patient_appt_count", "patient_is_repeat_noshower",
        # Clinical
        "Age", "chronic_condition_count", "has_handicap", "scholarship",
        # Intervention
        "sms_received", "sms_late_reminder",
        # Geography
        "neighbourhood_noshow_rate_raw",
        # Load
        "daily_load_percentile",
        # External enrichment
        "is_near_holiday", "rain_flag",
        # Demographics
        "Gender",
        "Neighbourhood",
    ]
    
    available = [c for c in feature_cols if c in df.columns]
    logger.info(f"Selected {len(available)} features: {available}")

    X = df[available].copy()
    y = df["no_show"].copy()
    return X, y


def build_features() -> Tuple[pd.DataFrame, pd.Series]:
    """Full feature engineering pipeline."""
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = clean_raw(df)
    df = engineer_temporal_features(df)
    df = engineer_patient_history_features(df)
    df = engineer_health_features(df)
    df = engineer_sms_features(df)
    df = engineer_neighbourhood_features(df)
    df = engineer_appointment_load(df)

    X, y = select_final_features(df)

    # Save processed features
    processed = X.copy()
    processed["no_show"] = y
    processed.to_parquet(PROCESSED_PATH, index=False)
    logger.info(f"Features saved to {PROCESSED_PATH} — shape: {processed.shape}")

    logger.info(f"\nClass distribution:\n{y.value_counts(normalize=True).round(3)}")
    return X, y


if __name__ == "__main__":
    X, y = build_features()
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
