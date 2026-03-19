"""
Data drift monitoring using Evidently AI.
Detects when production data distribution shifts from training data,
triggering automated retraining alerts.
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

DRIFT_REPORT_PATH = Path("reports/drift/")
PSI_THRESHOLD = 0.2  # Population Stability Index — industry standard


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1  : No significant change
    PSI 0.1–0.2: Moderate change — monitor closely
    PSI > 0.2  : Significant drift — retrain recommended
    """
    def _bucket(arr: np.ndarray, bins: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=bins)
        proportions = counts / len(arr)
        proportions = np.where(proportions == 0, 0.0001, proportions)
        return proportions

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, buckets + 1)
    bins[0] -= 1e-6
    bins[-1] += 1e-6

    expected_props = _bucket(expected, bins)
    actual_props = _bucket(actual, bins)

    psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    return float(psi)


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
) -> Dict[str, Dict]:
    """
    Compute drift metrics for all numeric features.
    Returns per-feature PSI + drift flag.
    """
    results = {}
    for feature in numeric_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue

        ref = reference_df[feature].dropna().values
        cur = current_df[feature].dropna().values

        if len(ref) < 10 or len(cur) < 10:
            continue

        psi = compute_psi(ref, cur)
        results[feature] = {
            "psi": round(psi, 4),
            "drift_detected": psi > PSI_THRESHOLD,
            "severity": _psi_severity(psi),
            "ref_mean": round(float(np.mean(ref)), 4),
            "cur_mean": round(float(np.mean(cur)), 4),
            "ref_std": round(float(np.std(ref)), 4),
            "cur_std": round(float(np.std(cur)), 4),
        }
    return results


def compute_prediction_drift(
    reference_probs: np.ndarray,
    current_probs: np.ndarray,
) -> Dict:
    """Detect drift in model output distribution — often the earliest signal."""
    psi = compute_psi(reference_probs, current_probs)
    return {
        "prediction_psi": round(psi, 4),
        "drift_detected": psi > PSI_THRESHOLD,
        "severity": _psi_severity(psi),
        "ref_mean_prob": round(float(np.mean(reference_probs)), 4),
        "cur_mean_prob": round(float(np.mean(current_probs)), 4),
    }


def _psi_severity(psi: float) -> str:
    if psi < 0.1:
        return "STABLE"
    elif psi < 0.2:
        return "WARNING"
    return "CRITICAL — RETRAIN RECOMMENDED"


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
    save: bool = True,
) -> Tuple[Dict, bool]:
    """
    Full drift monitoring run.
    Returns (report_dict, should_retrain_flag).
    """
    logger.info("Running drift detection...")

    feature_drift = compute_feature_drift(reference_df, current_df, numeric_features)

    drifted_features = [f for f, v in feature_drift.items() if v["drift_detected"]]
    should_retrain = len(drifted_features) >= 3  # Retrain if 3+ features drifting

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "features_monitored": len(feature_drift),
        "features_drifted": len(drifted_features),
        "drifted_feature_names": drifted_features,
        "should_retrain": should_retrain,
        "feature_drift": feature_drift,
    }

    if save:
        DRIFT_REPORT_PATH.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = DRIFT_REPORT_PATH / f"drift_report_{ts}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Drift report saved: {report_path}")

    if should_retrain:
        logger.warning(
            f"⚠️  DRIFT ALERT: {len(drifted_features)} features drifted: {drifted_features}. "
            "Retraining recommended."
        )
    else:
        logger.info(f"✅ Drift check passed. {len(drifted_features)} minor drifts detected.")

    return report, should_retrain


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    import numpy as np
    np.random.seed(42)
    ref = pd.DataFrame({"lead_time_days": np.random.normal(10, 5, 1000)})
    cur = pd.DataFrame({"lead_time_days": np.random.normal(18, 7, 500)})  # Shifted!
    report, retrain = run_drift_report(ref, cur, ["lead_time_days"])
    print(f"Should retrain: {retrain}")
    print(f"PSI: {report['feature_drift']['lead_time_days']['psi']}")
