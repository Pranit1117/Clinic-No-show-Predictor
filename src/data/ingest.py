"""
Data ingestion module.
Downloads the Kaggle dataset and enriches with external features.
"""

import os
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

import holidays

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/appointments.csv")
EXTERNAL_PATH = Path("data/external/")


def download_kaggle_dataset() -> None:
    """
    Downloads the no-show dataset from Kaggle.
    Requires KAGGLE_USERNAME and KAGGLE_KEY in environment or .env
    
    Manual fallback:
        1. Go to https://www.kaggle.com/datasets/joniarroba/noshowappointments
        2. Download KaggleV2-May-2016.csv
        3. Place it at data/raw/appointments.csv
    """
    try:
        import kaggle  # type: ignore
        logger.info("Downloading dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "joniarroba/noshowappointments",
            path=str(RAW_PATH.parent),
            unzip=True,
        )
        # Rename to standard name
        for f in RAW_PATH.parent.glob("*.csv"):
            f.rename(RAW_PATH)
            break
        logger.info(f"Dataset saved to {RAW_PATH}")
    except ImportError:
        logger.warning("kaggle package not installed. Using manual path.")
        if not RAW_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at {RAW_PATH}. "
                "Download manually from: https://www.kaggle.com/datasets/joniarroba/noshowappointments"
            )


def load_raw_data() -> pd.DataFrame:
    """Load raw appointments CSV into a DataFrame."""
    logger.info(f"Loading raw data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    logger.info(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


def add_holiday_flags(df: pd.DataFrame, country: str = "BR") -> pd.DataFrame:
    """
    Add is_near_holiday flag: 1 if appointment is within 2 days of a public holiday.
    Uses the `holidays` library — no API key needed.
    """
    logger.info("Adding holiday proximity flags...")
    br_holidays = holidays.country_holidays(country)

    def _is_near_holiday(date: datetime, window_days: int = 2) -> int:
        for delta in range(-window_days, window_days + 1):
            check = (date + pd.Timedelta(days=delta)).date()
            if check in br_holidays:
                return 1
        return 0

    appointment_dates = pd.to_datetime(df["AppointmentDay"])
    df["is_near_holiday"] = appointment_dates.apply(_is_near_holiday)
    logger.info(f"Holiday flag: {df['is_near_holiday'].sum():,} appointments near a holiday")
    return df


def fetch_weather_data(date_str: str, city: str = "Vitoria,BR") -> dict:
    """
    Fetch historical weather for a given date.
    NOTE: Free OpenWeatherMap tier only supports current/forecast.
    For historical data, use the paid History API OR use a synthetic rain flag
    based on month (wet season: Oct–Jan in Espirito Santo, Brazil).
    """
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key:
        return {"rain_flag": 0}
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        rain_flag = 1 if "rain" in data.get("weather", [{}])[0].get("main", "").lower() else 0
        return {"rain_flag": rain_flag}
    except Exception:
        return {"rain_flag": 0}


def add_synthetic_rain_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthetic rain flag based on wet season months in Espirito Santo, Brazil.
    Wet season: October – January (months 10, 11, 12, 1)
    This is a realistic approximation when historical weather API is unavailable.
    """
    logger.info("Adding synthetic rain flag (wet season proxy)...")
    appointment_months = pd.to_datetime(df["AppointmentDay"]).dt.month
    df["rain_flag"] = appointment_months.isin([10, 11, 12, 1]).astype(int)
    return df


def ingest() -> pd.DataFrame:
    """Full ingestion pipeline: download → enrich → save."""
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXTERNAL_PATH.mkdir(parents=True, exist_ok=True)

    download_kaggle_dataset()
    df = load_raw_data()
    df = add_holiday_flags(df)
    df = add_synthetic_rain_flag(df)

    # Save enriched raw data
    enriched_path = EXTERNAL_PATH / "appointments_enriched.parquet"
    df.to_parquet(enriched_path, index=False)
    logger.info(f"Enriched data saved to {enriched_path} — {len(df):,} rows")
    return df


if __name__ == "__main__":
    ingest()
