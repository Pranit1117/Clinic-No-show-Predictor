"""
Model training module.
Trains multiple models, tunes hyperparameters with Optuna, logs to MLflow.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
import shap

from pathlib import Path
from typing import Any, Dict, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.features.encode import build_preprocessor
from src.models.evaluate import evaluate_model, compute_business_metric

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/features.parquet")
RANDOM_STATE = 42
CV_FOLDS = 5
DECISION_THRESHOLD = 0.40  # Tuned for F2 score

NUMERIC_FEATURES = [
    "lead_time_days", "appt_day_of_week", "appt_is_monday", "schedule_hour",
    "is_same_day", "appt_week", "patient_noshow_rate", "patient_appt_count",
    "patient_is_repeat_noshower", "Age", "chronic_condition_count", "has_handicap",
    "scholarship", "sms_received", "sms_late_reminder", "neighbourhood_noshow_rate_raw",
    "daily_load_percentile", "is_near_holiday", "rain_flag",
]

CATEGORICAL_FEATURES = ["Gender"]
TARGET_ENCODE_FEATURES = ["Neighbourhood"]


def load_features() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_PATH)
    y = df.pop("no_show")
    X = df
    logger.info(f"Loaded features: {X.shape} | No-show rate: {y.mean():.2%}")
    return X, y


def get_available_numeric(X: pd.DataFrame) -> list[str]:
    return [c for c in NUMERIC_FEATURES if c in X.columns]


def get_available_categorical(X: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL_FEATURES if c in X.columns]


def build_baseline_model() -> LogisticRegression:
    """Logistic Regression — every project needs a baseline to beat."""
    return LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )


def build_lgbm(params: Dict[str, Any] | None = None) -> lgb.LGBMClassifier:
    defaults = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "scale_pos_weight": 4,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if params:
        defaults.update(params)
    return lgb.LGBMClassifier(**defaults)


def build_xgb(params: Dict[str, Any] | None = None) -> xgb.XGBClassifier:
    defaults = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "scale_pos_weight": 4,
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if params:
        defaults.update(params)
    return xgb.XGBClassifier(**defaults)


def build_catboost(cat_features: list[str]) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        cat_features=cat_features,
        random_state=RANDOM_STATE,
        verbose=0,
    )


def tune_lgbm_with_optuna(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50
) -> Dict[str, Any]:
    """Bayesian hyperparameter search for LightGBM using Optuna."""
    logger.info(f"Tuning LightGBM with Optuna ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 6.0),
        }
        model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best AUC-PR: {study.best_value:.4f} | Params: {study.best_params}")
    return study.best_params


def build_stacking_ensemble(
    lgbm_params: Dict[str, Any], xgb_params: Dict[str, Any]
) -> StackingClassifier:
    """
    Stacked ensemble: LightGBM + XGBoost + RandomForest → Logistic Regression meta-learner.
    This is the production model.
    """
    estimators = [
        ("lgbm", build_lgbm(lgbm_params)),
        ("xgb", build_xgb(xgb_params)),
        ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )


def train_and_log_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    extra_params: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Train a model, evaluate it, and log everything to MLflow."""
    with mlflow.start_run(run_name=model_name, nested=True):
        # Log params
        if extra_params:
            mlflow.log_params(extra_params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("decision_threshold", DECISION_THRESHOLD)

        # Train
        model.fit(X_train, y_train)

        # Probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_prob)
        business = compute_business_metric(y_test, y_pred)
        metrics.update(business)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(
            f"[{model_name}] AUC-PR={metrics['auc_pr']:.4f} | "
            f"F2={metrics['f2_score']:.4f} | "
            f"Revenue Saved/Day=${metrics.get('estimated_revenue_saved_per_day', 0):,.0f}"
        )
        return metrics


def train_pipeline() -> None:
    """Full training pipeline — trains all models and logs to MLflow."""
    X, y = load_features()

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    numeric_cols = get_available_numeric(X)
    categorical_cols = get_available_categorical(X)

    mlflow.set_experiment("noshow-prediction")

    with mlflow.start_run(run_name="training-run"):
        mlflow.log_param("dataset_size", len(X))
        mlflow.log_param("noshow_rate", float(y.mean()))
        mlflow.log_param("features", numeric_cols + categorical_cols)

        # ── 1. Baseline: Logistic Regression ─────────────────────────────────
        logger.info("\n─── Baseline: Logistic Regression ───")
        from sklearn.preprocessing import StandardScaler, OrdinalEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        preprocessor = build_preprocessor(numeric_cols, categorical_cols, [])
        baseline_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", build_baseline_model()),
        ])
        train_and_log_model(
            baseline_pipeline, "LogisticRegression",
            X_train, X_test, y_train, y_test,
        )

        # ── 2. LightGBM (Optuna-tuned) ────────────────────────────────────────
        logger.info("\n─── LightGBM with Optuna tuning ───")
        best_lgbm_params = tune_lgbm_with_optuna(X_train[numeric_cols], y_train, n_trials=30)
        lgbm_model = build_lgbm(best_lgbm_params)
        lgbm_pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols, [])),
            ("model", lgbm_model),
        ])
        train_and_log_model(
            lgbm_pipeline, "LightGBM_Optuna",
            X_train, X_test, y_train, y_test,
            extra_params=best_lgbm_params,
        )

        # ── 3. XGBoost ────────────────────────────────────────────────────────
        logger.info("\n─── XGBoost ───")
        xgb_model = build_xgb()
        xgb_pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols, [])),
            ("model", xgb_model),
        ])
        train_and_log_model(
            xgb_pipeline, "XGBoost",
            X_train, X_test, y_train, y_test,
        )

        # ── 4. Stacking Ensemble (Production Model) ───────────────────────────
        logger.info("\n─── Stacking Ensemble (Production Model) ───")
        stacking = build_stacking_ensemble(best_lgbm_params, {})
        stacking_pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols, [])),
            ("model", stacking),
        ])
        stacking_metrics = train_and_log_model(
            stacking_pipeline, "StackingEnsemble_PRODUCTION",
            X_train, X_test, y_train, y_test,
        )

        # ── Register best model ───────────────────────────────────────────────
        logger.info("\n─── Registering best model ───")
        mlflow.log_metric("best_auc_pr", stacking_metrics["auc_pr"])
        mlflow.sklearn.log_model(
            stacking_pipeline,
            artifact_path="production_model",
            registered_model_name="noshow-predictor",
        )
        logger.info("✅ Production model registered in MLflow Model Registry")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    train_pipeline()
