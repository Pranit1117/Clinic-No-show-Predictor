"""
Unit tests for feature engineering functions.
Every transformation should have a test — this is what separates
a notebook hacker from a production engineer.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Minimal valid raw appointments DataFrame."""
    today = datetime(2016, 5, 1)
    return pd.DataFrame({
        "PatientId": [1, 1, 2, 3, 4],
        "AppointmentID": [101, 102, 103, 104, 105],
        "Gender": ["F", "F", "M", "F", "M"],
        "ScheduledDay": [
            today, today, today,
            today - timedelta(days=5),
            today - timedelta(days=10),
        ],
        "AppointmentDay": [
            today + timedelta(days=7),
            today + timedelta(days=14),
            today + timedelta(days=3),
            today,
            today + timedelta(days=30),
        ],
        "Age": [34, 34, 22, 65, 0],
        "Neighbourhood": ["JARDIM CAMBURI", "JARDIM CAMBURI", "ITARARÉ", "CENTRO", "MARIA ORTIZ"],
        "Scholarship": [0, 0, 1, 0, 0],
        "Hipertension": [0, 0, 0, 1, 0],
        "Diabetes": [0, 0, 0, 1, 0],
        "Alcoholism": [0, 0, 0, 0, 1],
        "Handcap": [0, 0, 0, 0, 2],
        "SMS_received": [0, 1, 0, 1, 0],
        "No-show": ["No", "Yes", "No", "No", "Yes"],
        "is_near_holiday": [0, 0, 1, 0, 0],
        "rain_flag": [0, 1, 0, 0, 1],
    })


# ── Import the modules under test ─────────────────────────────────────────────

from src.features.build_features import (
    clean_raw,
    engineer_temporal_features,
    engineer_patient_history_features,
    engineer_health_features,
    engineer_sms_features,
)


# ── clean_raw tests ────────────────────────────────────────────────────────────

class TestCleanRaw:
    def test_creates_binary_target(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        assert "no_show" in df.columns
        assert set(df["no_show"].unique()).issubset({0, 1})

    def test_target_encoding_correct(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        # "Yes" → 1, "No" → 0
        yes_rows = df[df["No-show"] == "Yes"]
        no_rows = df[df["No-show"] == "No"]
        assert (yes_rows["no_show"] == 1).all()
        assert (no_rows["no_show"] == 0).all()

    def test_invalid_ages_dropped(self, sample_raw_df):
        # Age=0 is in fixture — should be kept (newborns are valid patients)
        df = clean_raw(sample_raw_df.copy())
        assert len(df[df["Age"] < 0]) == 0
        assert len(df[df["Age"] > 110]) == 0

    def test_dates_parsed_to_datetime(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        assert pd.api.types.is_datetime64_any_dtype(df["ScheduledDay"])
        assert pd.api.types.is_datetime64_any_dtype(df["AppointmentDay"])

    def test_no_negative_lead_time_appointments_remain(self, sample_raw_df):
        """AppointmentDay must be >= ScheduledDay."""
        # Inject a bad row
        bad_row = sample_raw_df.iloc[0].copy()
        bad_row["AppointmentDay"] = bad_row["ScheduledDay"] - timedelta(days=1)
        df_with_bad = pd.concat([sample_raw_df, pd.DataFrame([bad_row])], ignore_index=True)
        df_cleaned = clean_raw(df_with_bad.copy())
        leads = (
            pd.to_datetime(df_cleaned["AppointmentDay"]) -
            pd.to_datetime(df_cleaned["ScheduledDay"])
        ).dt.days
        assert (leads >= 0).all(), "Found appointments before scheduling date"


# ── engineer_temporal_features tests ──────────────────────────────────────────

class TestTemporalFeatures:
    def setup_method(self):
        pass

    def test_lead_time_non_negative(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_temporal_features(df)
        assert (df["lead_time_days"] >= 0).all()

    def test_lead_time_capped_at_60(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_temporal_features(df)
        assert (df["lead_time_days"] <= 60).all()

    def test_is_same_day_flag(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_temporal_features(df)
        same_day = df[df["is_same_day"] == 1]
        assert (same_day["lead_time_days"] == 0).all()

    def test_day_of_week_range(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_temporal_features(df)
        assert df["appt_day_of_week"].between(0, 6).all()

    def test_monday_flag_consistent(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_temporal_features(df)
        monday_rows = df[df["appt_is_monday"] == 1]
        assert (monday_rows["appt_day_of_week"] == 0).all()


# ── engineer_patient_history_features tests ───────────────────────────────────

class TestPatientHistoryFeatures:
    def test_no_data_leakage(self, sample_raw_df):
        """
        Patient history rate must NEVER use the current row's outcome.
        Verified by checking first appointment of each patient has rate=0.20 (population mean).
        """
        df = clean_raw(sample_raw_df.copy())
        df = engineer_patient_history_features(df)

        # First appointment for each patient (patient_appt_count == 0)
        first_appts = df[df["patient_appt_count"] == 0]
        assert (first_appts["patient_noshow_rate"] == 0.20).all(), \
            "First appointments must use population mean, not current row"

    def test_rate_bounded_0_1(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_patient_history_features(df)
        assert df["patient_noshow_rate"].between(0.0, 1.0).all()

    def test_cumcount_is_sequential(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_patient_history_features(df)
        # Patient 1 has 2 appointments → counts should be 0, 1
        pat1 = df[df["PatientId"] == 1].sort_values("AppointmentDay")
        assert list(pat1["patient_appt_count"]) == [0, 1]


# ── engineer_health_features tests ───────────────────────────────────────────

class TestHealthFeatures:
    def test_chronic_count_range(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_health_features(df)
        assert df["chronic_condition_count"].between(0, 3).all()

    def test_chronic_count_sums_correctly(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_health_features(df)
        expected = (
            df["Hipertension"].astype(int) +
            df["Diabetes"].astype(int) +
            df["Alcoholism"].astype(int)
        )
        pd.testing.assert_series_equal(df["chronic_condition_count"], expected, check_names=False)

    def test_has_handicap_binary(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_health_features(df)
        assert set(df["has_handicap"].unique()).issubset({0, 1})


# ── engineer_sms_features tests ───────────────────────────────────────────────

class TestSMSFeatures:
    def test_sms_late_reminder_logic(self, sample_raw_df):
        """
        sms_late_reminder = 1 only when sms_received=1 AND lead_time > 14 days.
        """
        df = clean_raw(sample_raw_df.copy())
        df = engineer_temporal_features(df)
        df = engineer_sms_features(df)

        late = df[df["sms_late_reminder"] == 1]
        assert (late["sms_received"] == 1).all()
        assert (late["lead_time_days"] > 14).all()

    def test_no_sms_means_no_late_reminder(self, sample_raw_df):
        df = clean_raw(sample_raw_df.copy())
        df = engineer_temporal_features(df)
        df = engineer_sms_features(df)

        no_sms = df[df["sms_received"] == 0]
        assert (no_sms["sms_late_reminder"] == 0).all()
