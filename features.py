"""Single source of truth for the Bengaluru model's input features."""

TARGET = "price_inr"

NUMERIC_FEATURES = [
    {
        "name": "bhk",
        "label": "Bedrooms (BHK)",
        "type": int,
        "min": 1,
        "max": 16,
        "default": 2,
        "help": "Number of bedrooms in the property.",
        "group": "rooms",
    },
    {
        "name": "total_sqft",
        "label": "Property area (sq ft)",
        "type": float,
        "min": 300,
        "max": 30400,
        "default": 1297,
        "help": "Total property area in square feet.",
        "group": "size",
    },
    {
        "name": "bath",
        "label": "Bathrooms",
        "type": int,
        "min": 1,
        "max": 16,
        "default": 2,
        "help": "Number of bathrooms.",
        "group": "rooms",
    },
    {
        "name": "balcony",
        "label": "Balconies",
        "type": int,
        "min": 0,
        "max": 3,
        "default": 1,
        "help": "Number of balconies.",
        "group": "rooms",
    },
    {
        "name": "is_ready_to_move",
        "label": "Availability",
        "type": int,
        "min": 0,
        "max": 1,
        "default": 1,
        "help": "Choose whether the property is ready for occupancy.",
        "group": "location",
        "choices": [(1, "Ready to move"), (0, "Under construction")],
    },
]

CATEGORICAL_FEATURES = [
    {
        "name": "location",
        "label": "Bengaluru location",
        "help": "Area or neighborhood in the Bengaluru dataset.",
        "group": "location",
    },
    {
        "name": "area_type",
        "label": "Area type",
        "help": "How the property area is classified.",
        "group": "location",
    },
]

NUMERIC_COLUMNS = [feature["name"] for feature in NUMERIC_FEATURES]
CATEGORICAL_COLUMNS = [feature["name"] for feature in CATEGORICAL_FEATURES]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_LABELS = {
    feature["name"]: feature["label"]
    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES
}
