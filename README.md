# Bengaluru House Price Prediction

A Flask and scikit-learn demo that estimates Bengaluru property prices from a cleaned housing dataset.

- **Dataset:** cleaned Bengaluru housing CSV, with prices in Indian rupees
- **Inputs:** location, area type, BHK, property area, bathrooms, balconies and readiness
- **Models:** Linear Regression and Random Forest, selected by cross-validated RMSE
- **Website:** prediction form, JSON API, model dashboard, dataset statistics and charts

Predictions are estimates from historical data, not professional or current-market valuations.

## GitHub and clean clones

The repository intentionally excludes CSV datasets, trained model files, and generated chart images. This keeps the public repository small and avoids redistributing data without permission. A fresh clone can run the test suite without those files, or use synthetic demo data to preview predictions:

```powershell
python train_model.py --demo
python app.py
```

For predictions based on Bengaluru data, provide a cleaned dataset as described in [`data/README.md`](data/README.md) and train the model locally. GitHub Actions runs the test suite on pushes and pull requests.

To publish the source, create an empty repository on GitHub and push this project to it. Do not commit private datasets, `.env` files, or generated model artifacts.

## Run the website

Use Python 3.10 or newer. From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
python app.py
```

Open <http://127.0.0.1:5000>.

The default dataset path is `data/bengaluru_clean.csv`. The supplied dataset is expected there. To train from another path:

```powershell
python train_model.py --data "C:\path\to\bengaluru_clean.csv"
```

The trained model and dashboard metrics are saved under `models/`; chart images are saved under `static/generated/`. Training is separate from the website startup.

## Demo mode

To preview the website without the Bengaluru CSV, run:

```powershell
python train_model.py --demo
python app.py
```

Demo-mode data is synthetic and clearly labeled. It must not be interpreted as real market pricing.

## Dashboard and API

The dashboard reports data and model metrics in INR. The JSON prediction endpoint accepts the same fields as the website form:

```http
POST /api/predict
Content-Type: application/json
```

```json
{
  "location": "Whitefield",
  "area_type": "Super built-up",
  "bhk": 2,
  "total_sqft": 1297,
  "bath": 2,
  "balcony": 1,
  "is_ready_to_move": 1
}
```

`is_ready_to_move` is `1` for ready to move and `0` for under construction. `GET /health` reports whether the saved model is ready.

## Tests

```powershell
python -m pytest
```

The tests train a small synthetic model in a temporary directory, so they do not overwrite the website's real trained model.

## Project files

- `app.py` — Flask routes and page rendering
- `features.py` — model input schema
- `train_model.py` — dataset loading, model training, evaluation and charts
- `ml_pipeline.py` — preprocessing, model construction and prediction
- `validation.py` — form and API validation
- `model_service.py` — saved model loading
- `templates/` and `static/` — website interface and assets
- `tests/` — route, validation, model and training tests

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `HPP_SECRET_KEY` | Random per start | Flask session and CSRF secret; set a persistent value for deployment |
| `HPP_HOST` | `127.0.0.1` | Development server bind address |
| `HPP_PORT` | `5000` | Development server port |
| `HPP_DATA_PATH` | `data/bengaluru_clean.csv` | Bengaluru training CSV |
| `HPP_MODEL_DIR` | `models/` | Trained model and metrics location |
| `HPP_CURRENCY` | `INR` | Dataset currency |

`python app.py` starts Flask's development server; use a production WSGI server for deployment.
