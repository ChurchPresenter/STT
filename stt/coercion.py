"""Safe numeric coercion for untrusted request input.

Extracted so the API's many `int(...)` conversions of request JSON/args stop
turning bad input (missing keys, non-numeric strings, wrong types) into 500s or,
worse, values pushed to the transcription worker that crash it. Stdlib-only.
"""

from typing import Optional


def coerce_int(value: object, default: int, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    """Coerce ``value`` to an int, falling back to ``default`` on any failure.

    - Non-numeric / None / wrong-type ``value`` → ``default`` (returned as-is,
      NOT clamped — callers pass an in-range default).
    - A coercible ``value`` is clamped to ``[lo, hi]`` when those bounds are given.
    """
    try:
        n = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


#: Strings a checkbox or show-control surface sends for "on". A JSON body sends
#: a real bool, but a form or query string can only send text, and "false"/"0"
#: are truthy strings in Python — the trap this function exists to avoid.
_TRUE_STRINGS = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "off", "n", "f", ""})


def coerce_bool(value: object, default: bool = False) -> bool:
    """Coerce ``value`` to a bool, falling back to ``default`` on any failure.

    Real bools and numbers pass through. Strings are matched case-insensitively
    against the usual on/off spellings; anything unrecognised returns
    ``default`` rather than guessing, so a typo can't silently read as "on".
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    return default


def coerce_float(value: object, default: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    """Coerce ``value`` to a float, falling back to ``default`` on any failure.

    Same contract as :func:`coerce_int`: a bad value returns ``default`` as-is;
    a coercible value is clamped to ``[lo, hi]`` when bounds are given.
    """
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n
