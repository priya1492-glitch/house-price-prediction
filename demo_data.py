"""Synthetic Bengaluru-shaped data for tests and interface previews.

Generated prices are artificial and must not be presented as real valuations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import TARGET

_LOCATIONS = [
    "Whitefield",
    "Sarjapur Road",
    "Electronic City",
    "Yelahanka",
    "Marathahalli",
    "Uttarahalli",
    "Hebbal",
    "Jayanagar",
    "HSR Layout",
    "Other",
]
_LOCATION_MULTIPLIERS = np.array([1.2, 1.08, 0.88, 0.95, 1.12, 0.82, 1.18, 1.3, 1.25, 0.78])
_AREA_TYPES = ["Super built-up", "Built-up", "Plot", "Carpet"]


def generate_synthetic_houses(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Return artificial Bengaluru-style records with a rupee price target."""
    rng = np.random.default_rng(seed)
    location_idx = rng.integers(0, len(_LOCATIONS), n_rows)
    location = np.array(_LOCATIONS)[location_idx]
    area_type = rng.choice(_AREA_TYPES, n_rows, p=[0.72, 0.20, 0.07, 0.01])
    bhk = np.clip(np.round(rng.normal(2.5, 0.9, n_rows)), 1, 6).astype(int)
    total_sqft = np.clip(rng.normal(1500, 450, n_rows), 400, 5000).round(1)
    bath = np.clip(np.round(bhk + rng.normal(-0.2, 0.7, n_rows)), 1, 7).astype(int)
    balcony = rng.integers(0, 4, n_rows)
    ready = rng.choice([0, 1], n_rows, p=[0.21, 0.79])
    area_type_multiplier = np.select(
        [area_type == "Super built-up", area_type == "Built-up", area_type == "Plot"],
        [1.0, 1.06, 0.92],
        default=1.02,
    )
    price = (
        total_sqft * 5_000
        + bhk * 250_000
        + bath * 100_000
        + balcony * 70_000
    )
    price *= _LOCATION_MULTIPLIERS[location_idx] * area_type_multiplier
    price *= np.where(ready == 1, 1.04, 0.94)
    price *= rng.lognormal(0, 0.08, n_rows)

    frame = pd.DataFrame(
        {
            "location": location,
            "area_type": area_type,
            "bhk": bhk,
            "total_sqft": total_sqft,
            "bath": bath,
            "balcony": balcony,
            "is_ready_to_move": ready,
            TARGET: price.round(-3),
        }
    )
    frame["price_lakhs"] = frame[TARGET] / 100_000

    for column, share in [("bath", 0.02), ("balcony", 0.02)]:
        frame.loc[rng.random(n_rows) < share, column] = np.nan
    frame.loc[rng.random(n_rows) < 0.01, "location"] = np.nan
    return frame
