"""Train, evaluate and save the house price models.

Usage
-----
    python train_model.py                 # train on data/bengaluru_clean.csv
    python train_model.py --data PATH     # train on a CSV stored elsewhere
    python train_model.py --demo          # train on SYNTHETIC demo data (UI testing only)

What it does
------------
1. Loads the Bengaluru dataset and keeps the model columns listed in ``features.py``.
2. Splits it into training (80 %) and test (20 %) sets.
3. Compares Linear Regression and Random Forest with 5-fold cross-validation
   on the *training* set only. The model with the lowest cross-validated RMSE
   is selected.
4. Evaluates both models on the untouched test set (MAE, RMSE, R²).
5. Saves the selected model (preprocessing + regressor) with joblib, plus a
   ``metrics.json`` file and PNG charts used by the dashboard.

Training is deliberately separate from the website: the web app never trains.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from config import CURRENCY_SYMBOLS, Config
from features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    NUMERIC_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
)
from ml_pipeline import (
    FOREST_NAME,
    LINEAR_NAME,
    build_model,
    forest_feature_importance,
    regression_metrics,
)


class DatasetError(Exception):
    """Raised when the dataset is missing or has the wrong format."""


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_bengaluru_csv(path: Path) -> pd.DataFrame:
    """Load the cleaned Bengaluru housing CSV and validate its model columns."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(
            f"Dataset not found at: {path}\n\n"
            "To train on the Bengaluru data:\n"
            "  1. Place the cleaned Bengaluru CSV at the configured data path.\n"
            f"  2. Current path: {path}\n"
            "  3. Run: python train_model.py\n\n"
            "Just want to try the website? Run: python train_model.py --demo\n"
            "(this uses SYNTHETIC data and is clearly labelled as a demo)."
        )
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # unreadable / not a CSV
        raise DatasetError(f"Could not read {path} as a CSV file: {exc}") from exc

    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in frame.columns]
    if missing:
        raise DatasetError(
            f"{path.name} is missing required column(s): {', '.join(missing)}. "
            "Use the cleaned Bengaluru housing CSV with price_inr and the property feature columns."
        )
    return frame.drop_duplicates(subset=FEATURE_COLUMNS + [TARGET]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialise {type(obj)}")


def _atomic_write(path: Path, writer) -> None:
    """Write via a temp file so a crash never leaves a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        writer(tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def train_and_save(
    frame: pd.DataFrame,
    *,
    model_dir: Path,
    charts_dir: Path | None,
    dataset_source: str,
    dataset_label: str,
    currency: str = "INR",
    test_size: float = Config.TEST_SIZE,
    cv_folds: int = Config.CV_FOLDS,
    random_state: int = Config.RANDOM_STATE,
    n_estimators: int = 300,
    verbose: bool = False,
) -> dict:
    """Train both models, pick the best one and write all artifacts.

    ``dataset_source`` is ``"bengaluru"`` for real data or ``"synthetic"`` for the
    demo data; the website uses it to label results honestly.
    """
    data = frame[FEATURE_COLUMNS + [TARGET]].copy()
    data = data[data[TARGET].notna() & (data[TARGET] > 0)].reset_index(drop=True)
    if len(data) < max(50, cv_folds * 5):
        raise DatasetError(
            f"Only {len(data)} usable rows found - at least 50 are needed to train."
        )

    locality_names = data["location"].astype("string").str.strip()
    areas = pd.to_numeric(data["total_sqft"], errors="coerce")
    prices = pd.to_numeric(data[TARGET], errors="coerce")
    locality_mask = (
        locality_names.notna()
        & locality_names.ne("")
        & locality_names.str.lower().ne("other")
        & areas.gt(0)
        & np.isfinite(areas)
        & prices.gt(0)
        & np.isfinite(prices)
    ).fillna(False)
    locality_data = pd.DataFrame(
        {
            "location": locality_names[locality_mask],
            "price_per_sqft": prices[locality_mask] / areas[locality_mask],
        }
    )
    locality_summary = (
        locality_data.groupby("location")
        .agg(
            avg_price_per_sqft=("price_per_sqft", "mean"),
            listing_count=("price_per_sqft", "size"),
        )
        .sort_values(["listing_count", "location"], ascending=[False, True])
        .head(6)
    )
    top_localities = [
        {
            "location": str(location),
            "avg_price_per_sqft": float(row["avg_price_per_sqft"]),
        }
        for location, row in locality_summary.iterrows()
    ]

    X, y = data[FEATURE_COLUMNS], data[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ---- Compare both models: cross-validation on the training set only ---- #
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name in (LINEAR_NAME, FOREST_NAME):
        model = build_model(name, n_estimators=n_estimators, random_state=random_state)
        cv_rmse = -cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
        )
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        results[name] = {
            **regression_metrics(y_test, test_pred),
            "train_r2": regression_metrics(y_train, train_pred)["r2"],
            "cv_rmse_mean": float(cv_rmse.mean()),
            "cv_rmse_std": float(cv_rmse.std()),
        }
        fitted[name] = model
        if verbose:
            r = results[name]
            print(
                f"  {name:<18} CV RMSE {r['cv_rmse_mean']:>10,.0f} (+/- {r['cv_rmse_std']:,.0f})"
                f" | test MAE {r['mae']:>9,.0f} | RMSE {r['rmse']:>9,.0f} | R2 {r['r2']:.3f}"
            )

    best_name = min(results, key=lambda n: results[n]["cv_rmse_mean"])
    best_model = fitted[best_name]
    best_test_pred = best_model.predict(X_test)

    # ---- Dashboard payload -------------------------------------------------- #
    rng = np.random.default_rng(random_state)
    counts, edges = np.histogram(data[TARGET], bins=20)
    scatter_idx = rng.choice(len(data), size=min(400, len(data)), replace=False)
    scatter = [
        {"x": float(data.loc[i, "total_sqft"]), "y": float(data.loc[i, TARGET])}
        for i in scatter_idx
        if pd.notna(data.loc[i, "total_sqft"])
    ]
    sample_idx = rng.choice(len(X_test), size=min(10, len(X_test)), replace=False)
    samples = []
    for i in sample_idx:
        actual, predicted = float(y_test.iloc[i]), float(best_test_pred[i])
        samples.append(
            {
                "location": str(X_test.iloc[i]["location"]),
                "living_area": float(X_test.iloc[i]["total_sqft"]),
                "actual": actual,
                "predicted": predicted,
                "error": predicted - actual,
                "pct_error": (predicted - actual) / actual * 100,
            }
        )

    categories = {
        column: sorted(data[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_COLUMNS
    }
    feature_summary = [
        {
            "feature": spec["name"],
            "label": spec["label"],
            "mean": float(data[spec["name"]].mean()),
            "min": float(data[spec["name"]].min()),
            "max": float(data[spec["name"]].max()),
            "missing": int(data[spec["name"]].isna().sum()),
        }
        for spec in NUMERIC_FEATURES
    ]

    payload = {
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_source": dataset_source,
            "dataset_label": dataset_label,
            "currency": currency,
            "currency_symbol": CURRENCY_SYMBOLS.get(currency, "$"),
            "n_rows": int(len(data)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": test_size,
            "cv_folds": cv_folds,
            "random_state": random_state,
            "best_model": best_name,
            "selection_rule": f"lowest mean {cv_folds}-fold cross-validated RMSE on the training set",
            "sklearn_version": sklearn.__version__,
            "features": FEATURE_COLUMNS,
        },
        "categories": categories,
        "dataset_stats": {
            "rows": int(len(data)),
            "price_mean": float(data[TARGET].mean()),
            "price_median": float(data[TARGET].median()),
            "price_min": float(data[TARGET].min()),
            "price_max": float(data[TARGET].max()),
            "price_std": float(data[TARGET].std()),
            "avg_living_area": float(data["total_sqft"].mean()),
            "median_bhk": float(data["bhk"].median()),
            "feature_summary": feature_summary,
        },
        "top_localities": top_localities,
        "models": results,
        "price_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
        "scatter": scatter,
        "feature_importance": forest_feature_importance(fitted[FOREST_NAME]),
        "sample_predictions": samples,
    }

    # ---- Save artifacts (metrics last: its presence signals "trained") ------ #
    model_dir = Path(model_dir)
    model_path = model_dir / Config.MODEL_FILENAME
    metrics_path = model_dir / Config.METRICS_FILENAME
    _atomic_write(model_path, lambda p: joblib.dump(best_model, p, compress=3))
    _atomic_write(
        metrics_path,
        lambda p: Path(p).write_text(
            json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
        ),
    )

    if charts_dir is not None:
        try:
            save_charts(data, payload, y_test.to_numpy(), best_test_pred, Path(charts_dir))
        except Exception as exc:  # charts are optional; never fail training
            print(f"  (Skipped Matplotlib charts: {exc})", file=sys.stderr)

    return payload


# --------------------------------------------------------------------------- #
# Static Matplotlib / Seaborn charts (shown on the dashboard when available)
# --------------------------------------------------------------------------- #
def save_charts(data, payload, y_test, y_pred, charts_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import seaborn as sns

    charts_dir.mkdir(parents=True, exist_ok=True)
    navy, amber, grey = "#12305c", "#e8a33d", "#6b7a90"
    sym = payload["meta"]["currency_symbol"]
    sns.set_theme(style="whitegrid", context="notebook")

    def money_axis(ax, axis="x"):
        fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{sym}{v / 1000:,.0f}k")
        (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(charts_dir / name, dpi=110)
        plt.close(fig)

    # 1. Price distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data[TARGET], bins=30, kde=True, color=navy, ax=ax)
    ax.set(title="Distribution of house prices", xlabel="Sale price", ylabel="Houses")
    money_axis(ax)
    finish(fig, "price_distribution.png")

    # 2. Living area vs price
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=data, x="total_sqft", y=TARGET, alpha=0.5, color=navy, ax=ax)
    ax.set(title="Property area vs price", xlabel="Property area (sq ft)", ylabel="Price")
    money_axis(ax, "y")
    finish(fig, "area_vs_price.png")

    # 3. Model comparison
    rows = [
        {"Model": name, "Metric": metric, "Value": values[key]}
        for name, values in payload["models"].items()
        for metric, key in (("MAE", "mae"), ("RMSE", "rmse"))
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=pd.DataFrame(rows), x="Model", y="Value", hue="Metric",
                palette=[navy, amber], ax=ax)
    ax.set(title="Test-set error by model (lower is better)", ylabel=f"Error ({sym})", xlabel="")
    money_axis(ax, "y")
    finish(fig, "model_comparison.png")

    # 4. Feature importance (Random Forest)
    top = payload["feature_importance"][:10]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=[t["importance"] * 100 for t in top],
        y=[FEATURE_LABELS.get(t["feature"], t["feature"]) for t in top],
        color=navy, ax=ax,
    )
    ax.set(title="Random Forest feature importance", xlabel="Importance (%)", ylabel="")
    finish(fig, "feature_importance.png")

    # 5. Actual vs predicted
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, y_pred, alpha=0.55, color=navy, s=18)
    lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color=amber, linewidth=2, label="Perfect prediction")
    ax.set(title=f"Actual vs predicted ({payload['meta']['best_model']})",
           xlabel="Actual price", ylabel="Predicted price")
    money_axis(ax, "x")
    money_axis(ax, "y")
    ax.legend()
    finish(fig, "actual_vs_predicted.png")

    # 6. Correlation heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6))
    corr = data[NUMERIC_COLUMNS + [TARGET]].corr(numeric_only=True)
    sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation between numeric features and price")
    finish(fig, "correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the house price models.")
    parser.add_argument("--data", type=Path, default=Config.DATA_PATH,
                        help="Path to cleaned Bengaluru CSV (default: %(default)s)")
    parser.add_argument("--demo", action="store_true",
                        help="Train on SYNTHETIC demo data instead of the Bengaluru file "
                             "(for trying the website only - results are not real)")
    parser.add_argument("--currency", choices=sorted(CURRENCY_SYMBOLS),
                        default=Config.DATASET_CURRENCY,
                        help="Currency of the dataset's prices (Bengaluru data is INR)")
    parser.add_argument("--model-dir", type=Path, default=Config.MODEL_DIR)
    parser.add_argument("--test-size", type=float, default=Config.TEST_SIZE)
    parser.add_argument("--cv-folds", type=int, default=Config.CV_FOLDS)
    parser.add_argument("--seed", type=int, default=Config.RANDOM_STATE)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        if args.demo:
            from demo_data import generate_synthetic_houses

            frame = generate_synthetic_houses()
            source, label = "synthetic", "SYNTHETIC demo data (not real house prices)"
            print("=" * 72)
            print("DEMO MODE: training on artificial, computer-generated data.")
            print("The resulting model and metrics are NOT real Bengaluru results.")
            print("=" * 72)
        else:
            frame = load_bengaluru_csv(args.data)
            source, label = "bengaluru", "Bengaluru housing dataset (cleaned CSV)"

        print(f"Training on {len(frame):,} rows ...")
        payload = train_and_save(
            frame,
            model_dir=args.model_dir,
            charts_dir=Config.CHARTS_DIR,
            dataset_source=source,
            dataset_label=label,
            currency=args.currency,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            random_state=args.seed,
            verbose=True,
        )
    except DatasetError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    best = payload["meta"]["best_model"]
    print(f"\nSelected model: {best} ({payload['meta']['selection_rule']})")
    print(f"Saved model   -> {args.model_dir / Config.MODEL_FILENAME}")
    print(f"Saved metrics -> {args.model_dir / Config.METRICS_FILENAME}")
    print("Start the website with:  python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
