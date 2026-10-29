"""Small display helpers (currency formatting)."""

from config import CURRENCY_SYMBOLS


def _group_indian(whole: str) -> str:
    """Group digits the Indian way: 12,34,567."""
    if len(whole) <= 3:
        return whole
    head, tail = whole[:-3], whole[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def format_money(value, currency: str = "USD", decimals: int = 0) -> str:
    """Format a number as money, e.g. ``$180,000`` or ``₹12,34,567``.

    No currency conversion is performed: the number is shown in the currency
    the dataset was recorded in.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "—"
    symbol = CURRENCY_SYMBOLS.get(currency, "$")
    sign = "-" if number < 0 else ""
    whole, _, frac = f"{abs(number):.{decimals}f}".partition(".")
    whole = _group_indian(whole) if currency == "INR" else f"{int(whole):,}"
    return f"{sign}{symbol}{whole}" + (f".{frac}" if frac else "")
