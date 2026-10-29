"""Tests for the Flask routes, forms and error handling."""

import json

import pytest

PAGES = ["/", "/predict", "/dashboard", "/about", "/contact"]
NAV_LINKS = ["/", "/predict", "/dashboard", "/about", "/contact"]


# ---------------------------------------------------------------- pages --- #
@pytest.mark.parametrize("path", PAGES)
def test_pages_load_without_a_trained_model(untrained_client, path):
    """Informational pages must work even if no dataset/model exists."""
    response = untrained_client.get(path)
    # /predict is 503 without a model but still shows a helpful page
    assert response.status_code == (503 if path == "/predict" else 200)
    assert b"<html" in response.data


@pytest.mark.parametrize("path", PAGES)
def test_pages_load_with_a_trained_model(client, path):
    assert client.get(path).status_code == 200


def test_navigation_links_work(client):
    html = client.get("/").get_data(as_text=True)
    for link in NAV_LINKS:
        assert f'href="{link}"' in html
        assert client.get(link).status_code == 200


def test_home_page_quick_estimate_and_data_sections(client, trained_dir):
    html = client.get("/").get_data(as_text=True)
    metrics = json.loads((trained_dir["models"] / "metrics.json").read_text(encoding="utf-8"))

    assert 'id="quick-estimate-form"' in html
    assert 'name="csrf_token"' in html
    for field in ("total_sqft", "bhk", "location", "bath", "balcony",
                  "is_ready_to_move", "area_type"):
        assert f'name="{field}"' in html
    assert "Listings analyzed" in html
    assert "Bengaluru localities" in html
    assert "ML models compared" in html
    assert "Median listing price" in html
    assert "Popular localities" in html
    assert "hero.svg" not in html
    for item in metrics["top_localities"]:
        assert item["location"].lower() != "other"
        assert item["avg_price_per_sqft"] > 0
        assert item["location"] in html


def test_home_page_disables_quick_estimate_without_model(untrained_client):
    html = untrained_client.get("/").get_data(as_text=True)
    assert "Quick estimates are unavailable" in html
    assert 'id="quick-estimate-form"' not in html
    assert "Listings analyzed" not in html
    assert "Popular localities" not in html


def test_quick_estimate_reuses_csrf_protected_prediction_route(csrf_client):
    page = csrf_client.get("/").get_data(as_text=True)
    token = page.split('name="csrf_token" value="')[1].split('"')[0]
    response = csrf_client.post(
        "/predict",
        data={
            "csrf_token": token,
            "total_sqft": "1600",
            "bhk": "2",
            "bath": "2",
            "balcony": "1",
            "is_ready_to_move": "1",
            "location": "Whitefield",
            "area_type": "Super built-up",
        },
    )
    assert response.status_code == 200
    assert b"result-price" in response.data


def test_home_page_supports_metrics_without_locality_breakdown(trained_dir, tmp_path):
    from conftest import _make_app

    models = tmp_path / "models"
    models.mkdir()
    (models / "house_price_model.joblib").write_bytes(
        (trained_dir["models"] / "house_price_model.joblib").read_bytes()
    )
    metrics = json.loads((trained_dir["models"] / "metrics.json").read_text(encoding="utf-8"))
    metrics.pop("top_localities", None)
    (models / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    html = _make_app(models, tmp_path / "charts").test_client().get("/").get_data(as_text=True)
    assert "Listings analyzed" in html
    assert "Popular localities" not in html


def test_health_endpoint(client, untrained_client):
    assert client.get("/health").get_json() == {"status": "ok", "model": "ready"}
    assert untrained_client.get("/health").get_json()["model"] == "missing"


def test_unknown_page_shows_custom_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert b"find that page" in response.data


def test_wrong_method_is_handled(client):
    response = client.post("/about")
    assert response.status_code == 405
    assert b"Traceback" not in response.data


def test_static_assets_exist(client):
    for path in ("/static/css/style.css", "/static/js/main.js",
                 "/static/js/dashboard.js", "/static/images/hero.svg"):
        assert client.get(path).status_code == 200


# --------------------------------------------------------- not trained --- #
def test_predict_page_explains_missing_model(untrained_client):
    response = untrained_client.get("/predict")
    assert b"has not been trained yet" in response.data
    assert b"train_model.py" in response.data


def test_predict_post_without_model_is_a_clear_error(untrained_client, valid_form):
    response = untrained_client.post("/predict", data=valid_form)
    assert response.status_code == 503
    assert b"has not been trained yet" in response.data


def test_dashboard_without_model_shows_setup_help(untrained_client):
    response = untrained_client.get("/dashboard")
    assert b"No trained model yet" in response.data


def test_corrupt_model_does_not_crash_or_leak(tmp_path, trained_dir):
    from conftest import _make_app

    models = tmp_path / "models"
    models.mkdir()
    (models / "house_price_model.joblib").write_bytes(b"not a real model")
    (models / "metrics.json").write_text(
        (trained_dir["models"] / "metrics.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    client = _make_app(models, tmp_path / "charts").test_client()
    response = client.get("/predict")
    assert response.status_code == 503
    assert b"could not be loaded" in response.data
    assert b"Traceback" not in response.data
    assert client.get("/").status_code == 200


# ----------------------------------------------------------- predicting --- #
def test_predict_form_page(client):
    html = client.get("/predict").get_data(as_text=True)
    for field in ("bhk", "total_sqft", "bath", "balcony", "is_ready_to_move",
                  "location", "area_type"):
        assert f'name="{field}"' in html
    assert "Demo model" in html  # demo banner is visible


def test_valid_prediction(client, valid_form):
    response = client.post("/predict", data=valid_form)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "result-price" in html
    assert "₹" in html
    assert "synthetic data" in html  # honest labelling of the demo model


def test_bigger_house_costs_more(client, valid_form):
    small, large = dict(valid_form), dict(valid_form)
    small.update(total_sqft="900", bhk="1", bath="1")
    large.update(total_sqft="3000", bhk="4", bath="4")
    price = lambda payload: client.post("/api/predict", json=payload).get_json()["prediction"]  # noqa: E731
    assert price(large) > price(small)


@pytest.mark.parametrize(
    "field, value",
    [
        ("total_sqft", ""),
        ("total_sqft", "abc"),
        ("total_sqft", "-5"),
        ("total_sqft", "999999"),
        ("bhk", "2.5"),
        ("bath", "17"),
        ("balcony", "4"),
        ("is_ready_to_move", "3"),
        ("location", ""),
        ("location", "Atlantis"),
        ("area_type", "Unknown"),
    ],
)
def test_invalid_input_is_rejected(client, valid_form, field, value):
    valid_form[field] = value
    response = client.post("/predict", data=valid_form)
    assert response.status_code == 400
    html = response.get_data(as_text=True)
    assert "Please fix the highlighted fields" in html
    assert "result-price" not in html


def test_missing_fields_are_rejected(client):
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert b"is required" in response.data


def test_submitted_values_are_kept_after_an_error(client, valid_form):
    valid_form["total_sqft"] = "not-a-number"
    valid_form["bhk"] = "3"
    html = client.post("/predict", data=valid_form).get_data(as_text=True)
    assert 'value="3"' in html


def test_input_is_escaped_in_the_page(client, valid_form):
    valid_form["total_sqft"] = '"><script>alert(1)</script>'
    html = client.post("/predict", data=valid_form).get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html


# ------------------------------------------------------------------ CSRF --- #
def test_csrf_blocks_posts_without_a_token(csrf_client, valid_form):
    response = csrf_client.post("/predict", data=valid_form)
    assert response.status_code == 400
    assert b"session expired" in response.data


def test_csrf_accepts_posts_with_the_page_token(csrf_client, valid_form):
    page = csrf_client.get("/predict").get_data(as_text=True)
    token = page.split('name="csrf_token" value="')[1].split('"')[0]
    valid_form["csrf_token"] = token
    assert csrf_client.post("/predict", data=valid_form).status_code == 200


# ------------------------------------------------------------------- API --- #
def test_api_prediction(client, valid_form):
    response = client.post("/api/predict", json=valid_form)
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] > 0
    assert body["currency"] == "INR"
    assert body["formatted"].startswith("₹")
    assert body["dataset_source"] == "synthetic"


def test_api_rejects_bad_json(client):
    assert client.post("/api/predict", data="nope", content_type="text/plain").status_code == 415
    assert client.post("/api/predict", data="[1,2]", content_type="application/json").status_code == 400
    response = client.post("/api/predict", json={"total_sqft": 1500})
    assert response.status_code == 400
    assert "fields" in response.get_json()


def test_api_without_model(untrained_client, valid_form):
    response = untrained_client.post("/api/predict", json=valid_form)
    assert response.status_code == 503
    assert "error" in response.get_json()


# ------------------------------------------------------------- dashboard --- #
def test_dashboard_content(client, trained_dir):
    html = client.get("/dashboard").get_data(as_text=True)
    for text in ("Linear Regression", "Random Forest", "MAE", "RMSE", "R²",
                 "Sample predictions", "chart-hist", "dashboard-data"):
        assert text in html
    data = json.loads(html.split('id="dashboard-data">')[1].split("</script>")[0])
    assert len(data["models"]["names"]) == 2
    assert data["importance"]
    assert data["scatter"]


def test_dashboard_lists_generated_figures(client):
    html = client.get("/dashboard").get_data(as_text=True)
    assert "generated/price_distribution.png" in html


def test_model_is_reloaded_after_retraining(trained_dir, tmp_path):
    """The app should notice replaced artifacts without a restart."""
    import shutil

    from conftest import _make_app

    models = tmp_path / "models"
    app = _make_app(models, tmp_path / "charts")
    client = app.test_client()
    assert client.get("/health").get_json()["model"] == "missing"
    shutil.copytree(trained_dir["models"], models)
    assert client.get("/health").get_json()["model"] == "ready"
