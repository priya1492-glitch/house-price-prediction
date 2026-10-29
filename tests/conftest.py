"""Shared test fixtures.

Tests never need the Bengaluru file: they train a tiny model on SYNTHETIC data
inside a temporary folder, so your real ``models/`` folder is never touched.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from demo_data import generate_synthetic_houses  # noqa: E402
from train_model import train_and_save  # noqa: E402

VALID_FORM = {
    "bhk": "2",
    "total_sqft": "1600",
    "bath": "2",
    "balcony": "1",
    "is_ready_to_move": "1",
    "location": "Whitefield",
    "area_type": "Super built-up",
}


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    """A folder containing a quickly trained DEMO model + metrics."""
    folder = tmp_path_factory.mktemp("models")
    charts = tmp_path_factory.mktemp("charts")
    frame = generate_synthetic_houses(n_rows=400, seed=1)
    train_and_save(
        frame,
        model_dir=folder,
        charts_dir=charts,
        dataset_source="synthetic",
        dataset_label="SYNTHETIC demo data (test)",
        cv_folds=3,
        n_estimators=30,
    )
    return {"models": folder, "charts": charts}


def _make_app(model_dir, charts_dir, **extra):
    overrides = {
        "TESTING": True,
        "CSRF_ENABLED": False,
        "MODEL_DIR": Path(model_dir),
        "CHARTS_DIR": Path(charts_dir),
        "SECRET_KEY": "test-key",
    }
    overrides.update(extra)
    return create_app(overrides)


@pytest.fixture()
def client(trained_dir):
    return _make_app(trained_dir["models"], trained_dir["charts"]).test_client()


@pytest.fixture()
def csrf_client(trained_dir):
    app = _make_app(trained_dir["models"], trained_dir["charts"], CSRF_ENABLED=True)
    return app.test_client()


@pytest.fixture()
def untrained_client(tmp_path):
    return _make_app(tmp_path / "empty_models", tmp_path / "charts").test_client()


@pytest.fixture()
def valid_form():
    return dict(VALID_FORM)
