"""
Prediction module — loads the registered MLflow model and runs inference.
"""

import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


DECISION_THRESHOLD = 0.40
MODEL_NAME = "noshow-predictor"
MODEL_STAGE = "Production"  # or "Staging" for testing


def load_model(model_name: str = MODEL_NAME, stage: str = MODEL_STAGE):
    """Load model from MLflow Model Registry."""
    mlflow.set_tracking_uri("mlruns")
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def predict_single(features: Dict[str, Any], model=None) -> Dict[str, Any]:
    """
    Run inference on a single patient/appointment record.
    Returns probability, risk tier, and recommended intervention.
    """
    if model is None:
        model = load_model()

    df = pd.DataFrame([features])
    prob = model.predict_proba(df)[0][1]
    prediction = int(prob >= DECISION_THRESHOLD)

    risk_tier = _get_risk_tier(prob)
    intervention = _get_intervention(prob, features)

    return {
        "no_show_probability": round(float(prob), 4),
        "will_no_show": bool(prediction),
        "risk_tier": risk_tier,
        "recommended_intervention": intervention,
        "confidence": _get_confidence_label(prob),
    }


def predict_batch(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Run inference on a batch of appointments."""
    if model is None:
        model = load_model()

    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= DECISION_THRESHOLD).astype(int)

    results = df.copy()
    results["no_show_probability"] = probs
    results["will_no_show"] = preds
    results["risk_tier"] = [_get_risk_tier(p) for p in probs]
    return results


def _get_risk_tier(prob: float) -> str:
    if prob >= 0.65:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def _get_confidence_label(prob: float) -> str:
    distance = abs(prob - 0.5)
    if distance >= 0.3:
        return "HIGH CONFIDENCE"
    elif distance >= 0.15:
        return "MODERATE CONFIDENCE"
    return "LOW CONFIDENCE — MONITOR"


def _get_intervention(prob: float, features: Dict[str, Any]) -> str:
    """
    Rule-based intervention recommendation on top of model prediction.
    This is the 'last mile' business logic layer.
    """
    if prob < 0.40:
        return "No action needed"
    if prob >= 0.65:
        if features.get("patient_is_repeat_noshower", 0):
            return "PRIORITY CALL — repeat no-shower with high risk"
        if features.get("lead_time_days", 0) > 14:
            return "CALL + offer reschedule to shorter lead time"
        return "CALL patient immediately"
    # Medium risk
    if not features.get("sms_received", 0):
        return "SEND SMS reminder"
    return "SEND SMS + follow-up call 24h before"
