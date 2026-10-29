"""Application configuration.

All settings can be overridden with environment variables, so nothing
sensitive or machine-specific is hardcoded in the source code.

Environment variables
---------------------
HPP_SECRET_KEY     Secret key for Flask sessions / CSRF tokens. If it is not
                   set, a random key is generated every time the app starts
                   (fine for local development; set it explicitly in production).
HPP_DEBUG          "1"/"true" enables Flask debug mode when running ``python app.py``.
HPP_HOST           Host for ``python app.py`` (default 127.0.0.1).
HPP_PORT           Port for ``python app.py`` (default 5000).
HPP_DATA_PATH      Path to the Bengaluru CSV file (default data/bengaluru_clean.csv).
HPP_MODEL_DIR      Directory for trained artifacts (default models/).
HPP_CURRENCY       Currency of the dataset used for training: INR (default) or USD.
                   The Bengaluru dataset is recorded in Indian rupees.
"""

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CURRENCY_SYMBOLS = {"USD": "$", "INR": "₹"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_currency() -> str:
    code = os.environ.get("HPP_CURRENCY", "INR").strip().upper()
    return code if code in CURRENCY_SYMBOLS else "INR"


class Config:
    """Default configuration used by the Flask app and the training script."""

    SECRET_KEY = os.environ.get("HPP_SECRET_KEY") or secrets.token_hex(32)
    DEBUG = _env_flag("HPP_DEBUG", False)
    HOST = os.environ.get("HPP_HOST", "127.0.0.1")
    PORT = int(os.environ.get("HPP_PORT", "5000"))

    DATA_PATH = Path(
        os.environ.get("HPP_DATA_PATH", BASE_DIR / "data" / "bengaluru_clean.csv")
    )
    MODEL_DIR = Path(os.environ.get("HPP_MODEL_DIR", BASE_DIR / "models"))
    MODEL_FILENAME = "house_price_model.joblib"
    METRICS_FILENAME = "metrics.json"
    CHARTS_DIR = BASE_DIR / "static" / "generated"

    DATASET_CURRENCY = _env_currency()

    # Reject unreasonably large request bodies (the forms are tiny).
    MAX_CONTENT_LENGTH = 64 * 1024

    # Set to False only in automated tests.
    CSRF_ENABLED = True

    # Training defaults (used by train_model.py)
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
