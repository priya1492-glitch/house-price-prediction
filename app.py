"""Flask web application for the House Price Prediction project.

Run (development):
    python app.py
    # or
    flask --app app run --debug

The app starts fine without a dataset or a trained model: the informational
pages always work, and the prediction/dashboard pages explain what to do.
"""

from __future__ import annotations

import hmac
import secrets

from flask import Flask, jsonify, render_template, request, session

from config import CURRENCY_SYMBOLS, Config
from features import (
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    NUMERIC_FEATURES,
)
from formatting import format_money
from model_service import READY, ModelService
from validation import validate_house_input

FORM_GROUPS = [
    ("size", "Property size", "How large is the property?"),
    ("rooms", "Rooms", "Bedrooms, bathrooms and balconies."),
    ("location", "Location & availability", "Where the property is and when it is available."),
]

CHART_IMAGES = [
    ("price_distribution.png", "Price distribution (Seaborn)"),
    ("area_vs_price.png", "Property area vs price (Seaborn)"),
    ("model_comparison.png", "Model comparison (Seaborn)"),
    ("feature_importance.png", "Random Forest feature importance (Seaborn)"),
    ("actual_vs_predicted.png", "Actual vs predicted prices (Matplotlib)"),
    ("correlation_heatmap.png", "Correlation heatmap (Seaborn)"),
]


def create_app(overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if overrides:
        app.config.update(overrides)

    service = ModelService(
        app.config["MODEL_DIR"],
        app.config["MODEL_FILENAME"],
        app.config["METRICS_FILENAME"],
        logger=app.logger,
    )
    app.extensions["model_service"] = service

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def currency_code() -> str:
        code = service.meta.get("currency") or app.config.get("DATASET_CURRENCY", "INR")
        return code if code in CURRENCY_SYMBOLS else "INR"

    def csrf_token() -> str:
        token = session.get("_csrf")
        if not token:
            token = secrets.token_hex(16)
            session["_csrf"] = token
        return token

    def csrf_ok() -> bool:
        if not app.config.get("CSRF_ENABLED", True):
            return True
        expected = session.get("_csrf", "")
        sent = request.form.get("csrf_token", "")
        return bool(expected) and hmac.compare_digest(expected, sent)

    def default_values() -> dict:
        return {spec["name"]: spec["default"] for spec in NUMERIC_FEATURES}

    def display_input_value(name, value):
        for spec in NUMERIC_FEATURES:
            if spec["name"] == name:
                choices = {str(option): label for option, label in spec.get("choices", [])}
                if str(value) in choices:
                    return choices[str(value)]
                break
        return f"{value:,.1f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else value

    def form_context(**extra):
        """Everything predict.html needs to render the form."""
        categories = service.categories
        field_options = {
            spec["name"]: [
                {"value": value, "label": value}
                for value in categories.get(spec["name"], [])
            ]
            for spec in CATEGORICAL_FEATURES
        }
        field_options.update(
            {
                spec["name"]: [
                    {"value": value, "label": label}
                    for value, label in spec["choices"]
                ]
                for spec in NUMERIC_FEATURES
                if spec.get("choices")
            }
        )
        groups = []
        for key, title, subtitle in FORM_GROUPS:
            fields = [s for s in NUMERIC_FEATURES if s["group"] == key]
            fields += [s for s in CATEGORICAL_FEATURES if s["group"] == key]
            groups.append({"key": key, "title": title, "subtitle": subtitle, "fields": fields})
        context = {
            "groups": groups,
            "field_options": field_options,
            "values": default_values(),
            "errors": {},
            "result": None,
            "error_message": None,
        }
        context.update(extra)
        return context

    # ------------------------------------------------------------------ #
    # Template helpers
    # ------------------------------------------------------------------ #
    app.jinja_env.filters["money"] = format_money

    @app.context_processor
    def inject_globals():
        service.refresh()
        return {
            "csrf_token": csrf_token,
            "model_state": service.state,
            "model_ready": service.state == READY,
            "model_meta": service.meta,
            "is_demo": service.is_demo,
            "cur": currency_code(),
            "currency_symbol": CURRENCY_SYMBOLS.get(currency_code(), "$"),
            "feature_labels": FEATURE_LABELS,
            "site_name": "HousePrice AI",
        }

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response

    # ------------------------------------------------------------------ #
    # Pages
    # ------------------------------------------------------------------ #
    @app.get("/")
    def index():
        state = service.refresh()
        metrics = service.metrics if state == READY else None
        categories = service.categories if state == READY else {}
        home_stats = None
        top_localities = []

        if metrics:
            dataset_stats = metrics.get("dataset_stats") or {}
            model_meta = metrics.get("meta") or {}
            models = metrics.get("models") or {}
            locations = categories.get("location", [])
            listing_count = dataset_stats.get("rows", model_meta.get("n_rows"))
            median_price = dataset_stats.get("price_median")
            if listing_count is not None and median_price is not None:
                home_stats = {
                    "listings": listing_count,
                    "localities": len(locations),
                    "models": len(models),
                    "median_price": median_price,
                }
            stored_localities = metrics.get("top_localities")
            if isinstance(stored_localities, list):
                top_localities = stored_localities

        bath_spec = next(spec for spec in NUMERIC_FEATURES if spec["name"] == "bath")
        return render_template(
            "index.html",
            home_stats=home_stats,
            locations=categories.get("location", []),
            top_localities=top_localities,
            quick_bath_max=bath_spec["max"],
        )

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.get("/contact")
    def contact():
        return render_template("contact.html")

    @app.get("/health")
    def health():
        service.refresh()
        return jsonify(status="ok", model=service.state)

    # -- Prediction ---------------------------------------------------- #
    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        state = service.refresh()
        if state != READY:
            return (
                render_template(
                    "predict.html", **form_context(error_message=service.message)
                ),
                503,
            )

        if request.method == "GET":
            return render_template("predict.html", **form_context())

        # POST -----------------------------------------------------------
        submitted = {k: request.form.get(k, "") for k in FEATURE_LABELS}
        if not csrf_ok():
            return (
                render_template(
                    "predict.html",
                    **form_context(
                        values=submitted,
                        error_message="Your session expired. Please try submitting the form again.",
                    ),
                ),
                400,
            )

        clean, errors = validate_house_input(request.form, service.categories)
        if errors:
            return (
                render_template(
                    "predict.html",
                    **form_context(
                        values=submitted,
                        errors=errors,
                        error_message="Please fix the highlighted fields and try again.",
                    ),
                ),
                400,
            )

        try:
            price = service.predict(clean)
        except Exception:
            app.logger.exception("Prediction failed")
            return (
                render_template(
                    "predict.html",
                    **form_context(
                        values=submitted,
                        error_message="Sorry, the prediction could not be completed. Please try again later.",
                    ),
                ),
                500,
            )

        result = {
            "price": price,
            "typical_error": service.typical_error(),
            "model": service.meta.get("best_model"),
            "inputs": [
                {
                    "label": FEATURE_LABELS[name],
                    "value": display_input_value(name, clean[name]),
                }
                for name in FEATURE_LABELS
                if name in clean
            ],
        }
        return render_template("predict.html", **form_context(values=submitted, result=result))

    @app.post("/api/predict")
    def api_predict():
        """JSON version of the prediction endpoint."""
        if service.refresh() != READY:
            return jsonify(error=service.message), 503
        if not request.is_json:
            return jsonify(error="Send the house details as JSON (Content-Type: application/json)."), 415
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="The request body must be a JSON object."), 400

        clean, errors = validate_house_input(payload, service.categories)
        if errors:
            return jsonify(error="Invalid input.", fields=errors), 400
        try:
            price = service.predict(clean)
        except Exception:
            app.logger.exception("API prediction failed")
            return jsonify(error="The prediction could not be completed."), 500

        code = currency_code()
        return jsonify(
            prediction=round(price, 2),
            formatted=format_money(price, code),
            currency=code,
            model=service.meta.get("best_model"),
            dataset_source=service.meta.get("dataset_source"),
            typical_error=service.typical_error(),
        )

    # -- Dashboard ----------------------------------------------------- #
    @app.get("/dashboard")
    def dashboard():
        if service.refresh() != READY:
            return render_template("dashboard.html", metrics=None, chart_data=None, images=[])

        metrics = service.metrics
        models = metrics["models"]
        edges = metrics["price_histogram"]["edges"]
        code = currency_code()
        chart_data = {
            "symbol": CURRENCY_SYMBOLS.get(code, "$"),
            "histogram": {
                "labels": [
                    f"{format_money(lo / 1000, code)}k-{format_money(hi / 1000, code)}k"
                    for lo, hi in zip(edges[:-1], edges[1:])
                ],
                "counts": metrics["price_histogram"]["counts"],
            },
            "scatter": metrics["scatter"],
            "models": {
                "names": list(models),
                "mae": [m["mae"] for m in models.values()],
                "rmse": [m["rmse"] for m in models.values()],
                "r2": [m["r2"] for m in models.values()],
            },
            "importance": [
                {
                    "label": FEATURE_LABELS.get(item["feature"], item["feature"]),
                    "value": round(item["importance"] * 100, 2),
                }
                for item in metrics["feature_importance"][:10]
            ],
        }
        charts_dir = app.config["CHARTS_DIR"]
        images = [
            {"file": f"generated/{name}", "title": title}
            for name, title in CHART_IMAGES
            if (charts_dir / name).is_file()
        ]
        return render_template(
            "dashboard.html", metrics=metrics, chart_data=chart_data, images=images
        )

    # ------------------------------------------------------------------ #
    # Error pages (never reveal internal details)
    # ------------------------------------------------------------------ #
    @app.errorhandler(404)
    def not_found(_error):
        return render_template("404.html"), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return (
            render_template(
                "error.html",
                code=405,
                title="Method not allowed",
                message="That page does not accept this kind of request.",
            ),
            405,
        )

    @app.errorhandler(413)
    def too_large(_error):
        return (
            render_template(
                "error.html",
                code=413,
                title="Request too large",
                message="The data you sent was too large.",
            ),
            413,
        )

    @app.errorhandler(500)
    def server_error(_error):
        return (
            render_template(
                "error.html",
                code=500,
                title="Something went wrong",
                message="An unexpected error occurred. Please try again later.",
            ),
            500,
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
