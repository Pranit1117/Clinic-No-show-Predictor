"""
Encoding module: handles categorical encoding consistently across train/val/test.
Uses scikit-learn Pipeline to prevent leakage.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import category_encoders as ce


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    target_encode_features: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Target-encodes high-cardinality categoricals (Neighbourhood)
    - Ordinal-encodes low-cardinality categoricals (Gender)
    - Scales numerics
    - Imputes any remaining nulls

    NOTE: Target encoding is fit on TRAINING data only.
    This transformer is always fit inside cross-validation or train split.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    low_cardinality_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat_low", low_cardinality_pipeline, categorical_features),
            # High cardinality (Neighbourhood) handled separately via TargetEncoder
            # in the full pipeline — see train.py
        ],
        remainder="drop",
    )
    return preprocessor
