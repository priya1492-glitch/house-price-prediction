"""Validation and sanitising of user-supplied house details."""

import math
from typing import Any, Mapping

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _clean_text(value: Any, max_length: int = 64) -> str:
    """Strip whitespace and control characters, and cap the length."""
    text = "" if value is None else str(value)
    text = "".join(ch for ch in text if ch.isprintable())
    return text.strip()[:max_length]


def _parse_number(raw: Any, kind: type):
    """Convert ``raw`` to an int/float. Raises ValueError on bad input."""
    text = _clean_text(raw, 32).replace(",", "")
    if text == "":
        raise ValueError("missing")
    number = float(text)  # may raise ValueError
    if math.isnan(number) or math.isinf(number):
        raise ValueError("not finite")
    if kind is int:
        if number != int(number):
            raise ValueError("not whole")
        return int(number)
    return number


def validate_house_input(raw: Mapping[str, Any], categories=None):
    """Validate a mapping of raw form/JSON values.

    Parameters
    ----------
    raw:
        The submitted values (e.g. ``request.form`` or a parsed JSON dict).
    categories:
        Mapping of categorical feature names to allowed values. If ``None``,
        categorical values are only checked for being non-empty.

    Returns
    -------
    (clean, errors):
        ``clean`` maps every feature name to a validated Python value.
        ``errors`` maps field names to human-readable messages. ``clean`` is
        only safe to use when ``errors`` is empty.
    """
    clean: dict = {}
    errors: dict = {}

    for spec in NUMERIC_FEATURES:
        name, label = spec["name"], spec["label"]
        try:
            value = _parse_number(raw.get(name), spec["type"])
        except (ValueError, TypeError, OverflowError):
            text = _clean_text(raw.get(name), 32)
            if text == "":
                errors[name] = f"{label} is required."
            elif spec["type"] is int:
                errors[name] = f"{label} must be a whole number."
            else:
                errors[name] = f"{label} must be a valid number."
            continue

        if value < spec["min"] or value > spec["max"]:
            errors[name] = (
                f"{label} must be between {spec['min']:,} and {spec['max']:,}."
            )
            continue
        clean[name] = value

    for spec in CATEGORICAL_FEATURES:
        name, label = spec["name"], spec["label"]
        value = _clean_text(raw.get(name))
        allowed = set(categories.get(name, [])) if categories is not None else None
        if value == "":
            errors[name] = f"{label} is required."
        elif allowed is not None and value not in allowed:
            errors[name] = f"Please choose a {label.lower()} from the list."
        else:
            clean[name] = value

    return clean, errors
