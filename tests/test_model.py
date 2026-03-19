"""
Unit tests for model evaluation and prediction logic.
"""

import pytest
import numpy as np
import pandas as pd
from src.models.evaluate import fbeta_score, compute_business_metric, get_optimal_threshold
from src.models.predict import _get_risk_tier, _get_confidence_label, _get_intervention


class TestFBetaScore:
    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        assert fbeta_score(y_true, y_pred, beta=2.0) == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        assert fbeta_score(y_true, y_pred, beta=2.0) == pytest.approx(0.0)

    def test_beta_2_weights_recall_more(self):
        # High recall, low precision model
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred_high_recall  = np.array([1, 1, 1, 1, 1, 1])  # catches all positives, many FP
        y_pred_high_prec    = np.array([1, 0, 0, 0, 0, 0])  # very precise, misses many

        f2_recall = fbeta_score(y_true, y_pred_high_recall, beta=2)
        f2_prec   = fbeta_score(y_true, y_pred_high_prec, beta=2)
        assert f2_recall > f2_prec, "F2 should prefer high recall over high precision"


class TestBusinessMetric:
    def test_returns_all_keys(self):
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        result = compute_business_metric(y_true, y_pred)
        assert "estimated_revenue_saved_per_day" in result
        assert "estimated_net_value_per_day" in result
        assert "estimated_revenue_lost_per_day" in result
        assert "intervention_cost_per_day" in result

    def test_perfect_model_maximizes_revenue(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred_perfect = np.array([1, 1, 1, 0, 0, 0])
        y_pred_bad = np.array([0, 0, 0, 0, 0, 0])
        r_perfect = compute_business_metric(y_true, y_pred_perfect)
        r_bad = compute_business_metric(y_true, y_pred_bad)
        assert r_perfect["estimated_revenue_saved_per_day"] > r_bad["estimated_revenue_saved_per_day"]

    def test_revenue_values_positive(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        result = compute_business_metric(y_true, y_pred)
        assert result["estimated_revenue_saved_per_day"] >= 0
        assert result["intervention_cost_per_day"] >= 0


class TestRiskTier:
    def test_high_risk(self):
        assert _get_risk_tier(0.70) == "HIGH"
        assert _get_risk_tier(0.65) == "HIGH"

    def test_medium_risk(self):
        assert _get_risk_tier(0.50) == "MEDIUM"
        assert _get_risk_tier(0.40) == "MEDIUM"

    def test_low_risk(self):
        assert _get_risk_tier(0.20) == "LOW"
        assert _get_risk_tier(0.0) == "LOW"

    def test_boundary_values(self):
        assert _get_risk_tier(0.399) == "LOW"
        assert _get_risk_tier(0.400) == "MEDIUM"
        assert _get_risk_tier(0.649) == "MEDIUM"
        assert _get_risk_tier(0.650) == "HIGH"


class TestIntervention:
    def test_low_prob_no_action(self):
        result = _get_intervention(0.20, {})
        assert "No action" in result

    def test_high_prob_repeat_noshower_gets_call(self):
        result = _get_intervention(0.80, {"patient_is_repeat_noshower": 1})
        assert "CALL" in result.upper()

    def test_high_prob_long_lead_suggests_reschedule(self):
        result = _get_intervention(0.70, {"patient_is_repeat_noshower": 0, "lead_time_days": 20})
        assert "reschedule" in result.lower()

    def test_medium_risk_no_sms_gets_sms(self):
        result = _get_intervention(0.50, {"sms_received": 0})
        assert "SMS" in result


class TestOptimalThreshold:
    def test_returns_float_in_range(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0, 1, 200)
        threshold = get_optimal_threshold(y_true, y_prob)
        assert 0.0 <= threshold <= 1.0

    def test_optimal_threshold_beats_default_f2(self):
        """For imbalanced data, optimal threshold should improve over default 0.5."""
        from src.models.evaluate import fbeta_score
        np.random.seed(0)
        y_true = np.array([1]*30 + [0]*170)
        y_prob = np.clip(np.random.normal(0.35, 0.15, 200), 0, 1)

        threshold_optimal = get_optimal_threshold(y_true, y_prob)
        threshold_default = 0.5

        y_pred_opt = (y_prob >= threshold_optimal).astype(int)
        y_pred_def = (y_prob >= threshold_default).astype(int)

        f2_opt = fbeta_score(y_true, y_pred_opt, beta=2.0)
        f2_def = fbeta_score(y_true, y_pred_def, beta=2.0)
        assert f2_opt >= f2_def, "Optimal threshold should be at least as good as default"
