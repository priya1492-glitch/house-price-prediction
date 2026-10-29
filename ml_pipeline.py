"""Preprocessing, model construction, evaluation and prediction helpers.

Everything that has to be identical between training time and prediction time
lives here, so the website and ``train_model.py`` can never drift apart.

Data-leakage note
-----------------
Imputation, scaling and one-hot encoding are steps *inside* a scikit-learn
``Pipeline``. They are therefore fitted only on the training data (and, during
cross-validation, only on each training fold) and merely *applied* to
validation/test rows or to a visitor's form input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

LINEAR_NAME = "Linear Regression"
FOREST_NAME = "Random Forest"


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categoricals."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # Unknown categories at prediction time become all-zero rows
            # instead of raising an error.
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_COLUMNS),
            ("cat", categorical, CATEGORICAL_COLUMNS),
        ]
    )


def build_model(kind: str, *, n_estimators: int = 300, random_state: int = 42):
    """Return an unfitted model (preprocessing + regressor).

    Prices are right-skewed, so both models learn ``log(1 + price)`` and the
    predictions are converted back to the dataset's currency automatically.
    """
    if kind == LINEAR_NAME:
        estimator = LinearRegression()
    elif kind == FOREST_NAME:
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {kind!r}")

    pipeline = Pipeline(
        steps=[("preprocess", build_preprocessor()), ("model", estimator)]
    )
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1
    )


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and R² on the original price scale."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def to_frame(rows) -> pd.DataFrame:
    """Build a DataFrame with exactly the model's columns, in the right order."""
    if isinstance(rows, dict):
        rows = [rows]
    frame = pd.DataFrame(rows)
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    return frame[FEATURE_COLUMNS]


def predict_price(model, clean_input: dict) -> float:
    """Predict the price for one validated input dictionary."""
    value = float(model.predict(to_frame(clean_input))[0])
    return max(value, 0.0)


def forest_feature_importance(fitted_model) -> list[dict]:
    """Random Forest importances, with one-hot columns merged per feature.

    Returns a list of ``{"feature": column, "importance": share}`` sorted from
    most to least important. The importances sum to 1.
    """
    pipeline = fitted_model.regressor_
    preprocess = pipeline.named_steps["preprocess"]
    forest = pipeline.named_steps["model"]
    names = preprocess.get_feature_names_out()
    totals: dict[str, float] = {}
    for name, value in zip(names, forest.feature_importances_):
        # Names contain the transformer prefix and, for categoricals, a value.
        stage, _, rest = name.partition("__")
        column = rest
        if stage == "cat":
            column = next(
                (c for c in CATEGORICAL_COLUMNS if rest.startswith(c + "_")), rest
            )
        totals[column] = totals.get(column, 0.0) + float(value)
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return [{"feature": k, "importance": v} for k, v in ranked]
