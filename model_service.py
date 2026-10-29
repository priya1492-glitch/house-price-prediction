"""Safe, cached loading of the trained model and its metrics file.

The website never trains anything. It only *loads* what ``train_model.py``
saved in the ``models/`` folder, and it reloads automatically if the files are
replaced (for example after re-training), so the server does not need a restart.

Security note: ``joblib`` files are pickles, so only load model files you
created yourself. This app only ever reads from its own ``models/`` folder.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import joblib

from ml_pipeline import predict_price

READY, MISSING, ERROR = "ready", "missing", "error"

NOT_TRAINED_MESSAGE = (
    "The prediction model has not been trained yet. Add the Bengaluru dataset "
    "to the data/ folder and run 'python train_model.py' "
    "(or 'python train_model.py --demo' to try the site with synthetic demo data), "
    "then reload this page."
)
LOAD_ERROR_MESSAGE = (
    "The saved model could not be loaded. Please re-run 'python train_model.py' "
    "to create a fresh model, then reload this page."
)


class ModelService:
    """Loads ``model.joblib`` + ``metrics.json`` lazily and caches them."""

    def __init__(self, model_dir, model_filename, metrics_filename, logger=None):
        self.model_path = Path(model_dir) / model_filename
        self.metrics_path = Path(model_dir) / metrics_filename
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._signature = None
        self._model = None
        self._metrics = None
        self.state = MISSING
        self.message = NOT_TRAINED_MESSAGE

    # -- internal ---------------------------------------------------------- #
    def _current_signature(self):
        try:
            return (
                self.model_path.stat().st_mtime_ns,
                self.metrics_path.stat().st_mtime_ns,
            )
        except OSError:
            return None

    def refresh(self) -> str:
        """Reload artifacts if they changed on disk. Returns the state."""
        with self._lock:
            signature = self._current_signature()
            if signature == self._signature and (
                signature is not None or self.state == MISSING
            ):
                return self.state

            self._signature = signature
            self._model = self._metrics = None

            if signature is None:
                self.state, self.message = MISSING, NOT_TRAINED_MESSAGE
                return self.state

            try:
                metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
                if not metrics.get("meta") or not metrics.get("categories"):
                    raise ValueError("metrics.json is missing required sections")
                model = joblib.load(self.model_path)
            except Exception:  # corrupt / incompatible files - never leak details
                self.logger.exception("Could not load the trained model artifacts")
                self.state, self.message = ERROR, LOAD_ERROR_MESSAGE
                return self.state

            self._model, self._metrics = model, metrics
            self.state, self.message = READY, ""
            return self.state

    # -- public ------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        return self.refresh() == READY

    @property
    def metrics(self):
        self.refresh()
        return self._metrics

    @property
    def meta(self) -> dict:
        metrics = self.metrics
        return metrics["meta"] if metrics else {}

    @property
    def is_demo(self) -> bool:
        return self.meta.get("dataset_source") == "synthetic"

    @property
    def categories(self) -> dict:
        metrics = self.metrics
        return metrics["categories"] if metrics else {}

    def predict(self, clean_input: dict) -> float:
        if not self.ready:
            raise RuntimeError("Model is not available")
        return predict_price(self._model, clean_input)

    def typical_error(self):
        """Test-set MAE of the saved model (or ``None``)."""
        metrics = self.metrics
        if not metrics:
            return None
        best = metrics["meta"].get("best_model")
        return metrics.get("models", {}).get(best, {}).get("mae")
