"""Tests for validation, preprocessing, training and prediction behaviour."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from demo_data import generate_synthetic_houses
from features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET
from formatting import format_money
from ml_pipeline import (
    FOREST_NAME,
    LINEAR_NAME,
    build_model,
    build_preprocessor,
    forest_feature_importance,
    predict_price,
    regression_metrics,
    to_frame,
)
from train_model import DatasetError, load_bengaluru_csv, main, train_and_save
from validation import validate_house_input

CATEGORIES = {
    "location": ["Whitefield", "Sarjapur Road", "Other"],
    "area_type": ["Super built-up", "Built-up", "Plot"],
}
GOOD = {
    "bhk": "2", "total_sqft": "1500", "bath": "2", "balcony": "1",
    "is_ready_to_move": "1", "location": "Whitefield", "area_type": "Super built-up",
}


@pytest.fixture(scope="module")
def data():
    return generate_synthetic_houses(n_rows=300, seed=3)


# ------------------------------------------------------------ validation --- #
def test_valid_input_is_converted_to_numbers():
    clean, errors = validate_house_input(GOOD, CATEGORIES)
    assert errors == {}
    assert clean["total_sqft"] == 1500.0
    assert clean["bhk"] == 2 and isinstance(clean["bhk"], int)
    assert clean["location"] == "Whitefield"
    assert list(clean) and set(clean) == set(FEATURE_COLUMNS)


def test_thousands_separators_and_whitespace_are_accepted():
    clean, errors = validate_house_input({**GOOD, "total_sqft": " 1,500 "}, CATEGORIES)
    assert errors == {} and clean["total_sqft"] == 1500.0


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e999", "", "  ", "twelve", None])
def test_non_numeric_values_are_rejected(bad):
    _, errors = validate_house_input({**GOOD, "total_sqft": bad}, CATEGORIES)
    assert "total_sqft" in errors


def test_out_of_range_and_fractional_values():
    _, errors = validate_house_input({**GOOD, "bath": "17", "bhk": "1.5"}, CATEGORIES)
    assert "bath" in errors and "bhk" in errors


def test_unknown_location_and_area_type_are_rejected():
    _, errors = validate_house_input({**GOOD, "location": "Nowhere"}, CATEGORIES)
    assert "location" in errors


def test_control_characters_are_stripped_and_length_capped():
    clean, errors = validate_house_input({**GOOD, "location": "Whitefield\x00\n"}, CATEGORIES)
    assert errors == {} and clean["location"] == "Whitefield"
    _, errors = validate_house_input({**GOOD, "location": "A" * 500}, CATEGORIES)
    assert "location" in errors


def test_every_field_reported_when_empty():
    _, errors = validate_house_input({}, CATEGORIES)
    assert set(errors) == set(FEATURE_COLUMNS)


# ---------------------------------------------------------- preprocessing --- #
def test_synthetic_data_is_reproducible_and_has_missing_values():
    a = generate_synthetic_houses(100, seed=5)
    b = generate_synthetic_houses(100, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert set(FEATURE_COLUMNS + [TARGET]) <= set(a.columns)
    assert a.isna().sum().sum() > 0


def test_preprocessor_handles_missing_values(data):
    X = data[FEATURE_COLUMNS]
    out = build_preprocessor().fit_transform(X)
    dense = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    assert not np.isnan(dense).any()
    assert dense.shape[0] == len(X)


def test_preprocessor_is_fitted_on_training_data_only(data):
    """Scaler statistics must come from the training rows, not the test rows."""
    X_train, X_test = train_test_split(data[FEATURE_COLUMNS], test_size=0.3, random_state=0)
    pre = build_preprocessor().fit(X_train)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    imputed = pre.named_transformers_["num"].named_steps["impute"].transform(X_train[NUMERIC_COLUMNS])
    assert np.allclose(scaler.mean_, imputed.mean(axis=0))
    assert not np.allclose(scaler.mean_, X_test[NUMERIC_COLUMNS].mean().to_numpy())


def test_unseen_category_does_not_crash(data):
    model = build_model(LINEAR_NAME).fit(data[FEATURE_COLUMNS], data[TARGET])
    row = {
        **{c: 1 for c in NUMERIC_COLUMNS},
        "location": "NeverSeenBefore",
        "area_type": "UnknownType",
    }
    assert predict_price(model, row) > 0


def test_missing_values_at_prediction_time_are_imputed(data):
    model = build_model(LINEAR_NAME).fit(data[FEATURE_COLUMNS], data[TARGET])
    row = {c: np.nan for c in NUMERIC_COLUMNS}
    row.update(total_sqft=1500.0, location="Whitefield", area_type="Built-up")
    assert np.isfinite(predict_price(model, row))


def test_to_frame_orders_columns_and_rejects_missing():
    shuffled = dict(
        reversed(
            list(
                {
                    **{c: 1 for c in NUMERIC_COLUMNS},
                    "location": "Whitefield",
                    "area_type": "Built-up",
                }.items()
            )
        )
    )
    assert list(to_frame(shuffled).columns) == FEATURE_COLUMNS
    with pytest.raises(ValueError):
        to_frame({"total_sqft": 1})


# ---------------------------------------------------------------- models --- #
@pytest.mark.parametrize("name", [LINEAR_NAME, FOREST_NAME])
def test_models_learn_something(data, name):
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS], data[TARGET], test_size=0.25, random_state=0
    )
    model = build_model(name, n_estimators=40).fit(X_train, y_train)
    metrics = regression_metrics(y_test, model.predict(X_test))
    assert metrics["r2"] > 0.5
    assert metrics["rmse"] >= metrics["mae"] > 0


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("Magic")


def test_regression_metrics_known_values():
    m = regression_metrics([100, 200, 300], [110, 190, 330])
    assert m["mae"] == pytest.approx((10 + 10 + 30) / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert regression_metrics([1, 2, 3], [1, 2, 3])["r2"] == pytest.approx(1.0)


def test_feature_importance_is_merged_and_sums_to_one(data):
    model = build_model(FOREST_NAME, n_estimators=30).fit(data[FEATURE_COLUMNS], data[TARGET])
    importance = forest_feature_importance(model)
    assert {i["feature"] for i in importance} == set(FEATURE_COLUMNS)
    assert sum(i["importance"] for i in importance) == pytest.approx(1.0)
    assert [i["importance"] for i in importance] == sorted(
        (i["importance"] for i in importance), reverse=True
    )


# -------------------------------------------------------------- training --- #
def test_training_writes_artifacts_and_metrics(trained_dir):
    models = trained_dir["models"]
    assert (models / "house_price_model.joblib").is_file()
    metrics = json.loads((models / "metrics.json").read_text(encoding="utf-8"))

    assert set(metrics["models"]) == {LINEAR_NAME, FOREST_NAME}
    for result in metrics["models"].values():
        assert {"mae", "rmse", "r2", "cv_rmse_mean"} <= set(result)
    meta = metrics["meta"]
    best = min(metrics["models"], key=lambda n: metrics["models"][n]["cv_rmse_mean"])
    assert meta["best_model"] == best
    assert meta["dataset_source"] == "synthetic"
    assert meta["n_train"] + meta["n_test"] == meta["n_rows"]
    assert meta["features"] == FEATURE_COLUMNS
    assert metrics["sample_predictions"] and metrics["categories"]["location"]


def test_training_creates_chart_images(trained_dir):
    names = {p.name for p in trained_dir["charts"].glob("*.png")}
    assert {"price_distribution.png", "area_vs_price.png", "model_comparison.png",
            "feature_importance.png", "actual_vs_predicted.png",
            "correlation_heatmap.png"} <= names


def test_saved_model_predicts_like_the_app_expects(trained_dir):
    import joblib

    model = joblib.load(trained_dir["models"] / "house_price_model.joblib")
    clean, errors = validate_house_input(GOOD, CATEGORIES)
    assert not errors
    assert 10_000 < predict_price(model, clean) < 200_000_000


def test_training_rejects_tiny_datasets(tmp_path):
    with pytest.raises(DatasetError):
        train_and_save(
            generate_synthetic_houses(20), model_dir=tmp_path, charts_dir=None,
            dataset_source="synthetic", dataset_label="x",
        )


# --------------------------------------------------------------- dataset --- #
def test_missing_dataset_gives_helpful_message(tmp_path):
    with pytest.raises(DatasetError) as info:
        load_bengaluru_csv(tmp_path / "bengaluru_clean.csv")
    assert "Bengaluru" in str(info.value) and "--demo" in str(info.value)


def test_cli_exits_cleanly_when_dataset_missing(tmp_path, capsys):
    code = main(["--data", str(tmp_path / "nope.csv"), "--model-dir", str(tmp_path / "m")])
    assert code == 1
    assert "Dataset not found" in capsys.readouterr().err
    assert not (tmp_path / "m").exists()  # nothing fake was produced


def test_wrong_csv_columns_are_reported(tmp_path):
    path = tmp_path / "bengaluru.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(DatasetError) as info:
        load_bengaluru_csv(path)
    assert "missing required column" in str(info.value)


def test_bengaluru_style_csv_trains_end_to_end(tmp_path):
    """A Bengaluru CSV with extra columns trains end to end."""
    frame = generate_synthetic_houses(200, seed=11)
    frame.insert(0, "Id", range(1, len(frame) + 1))
    frame["ExtraColumn"] = "ignored"
    path = tmp_path / "bengaluru.csv"
    frame.to_csv(path, index=False)

    payload = train_and_save(
        load_bengaluru_csv(path), model_dir=tmp_path / "out", charts_dir=None,
        dataset_source="bengaluru", dataset_label="test", cv_folds=3, n_estimators=20,
    )
    assert payload["meta"]["dataset_source"] == "bengaluru"


# ------------------------------------------------------------ formatting --- #
def test_money_formatting():
    assert format_money(1234567.8, "USD") == "$1,234,568"
    assert format_money(1234567, "INR") == "₹12,34,567"
    assert format_money(999, "INR") == "₹999"
    assert format_money(-5000, "USD") == "-$5,000"
    assert format_money(None) == "—"
    assert format_money(float("nan")) == "—"
