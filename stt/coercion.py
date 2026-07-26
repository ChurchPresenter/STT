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
