"""Read request parameters however a client chose to send them.

Control surfaces (Bitfocus Companion, stream decks, show-control systems, a bare
curl in a run-of-show doc) all send the same intent through different channels: a
JSON body, a form-encoded body, or a query string. Routes that only ever call
``request.get_json(silent=True)`` reject the other two silently — ``silent=True``
turns a body with an unexpected or missing ``Content-Type`` into ``None``, the
route sees no parameters, and returns 400 as though the operator sent nothing.
From the far end that is indistinguishable from a button that did nothing.

Stdlib-only and pure so it can be unit-tested without the monolith; the routes in
speech_to_text.py pass ``request.get_json(silent=True)``, ``request.form`` and
``request.args`` in explicitly.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

# Lowest precedence first: later sources overwrite earlier ones, so an explicit
# JSON body always wins over a query string someone left in a bookmarked URL.
_PRECEDENCE_LOW_TO_HIGH = ("query", "form", "json")


def _usable(value: Any) -> bool:
    """Whether a value carries an actual instruction.

    ``None`` and blank strings do not. A surface that sends every field on every
    press — leaving the ones it isn't changing empty — must not thereby blank a
    setting, so an empty value is treated as "not sent" rather than "set to
    nothing". ``False`` and ``0`` are real values and are kept.
    """
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _clean(value: Any) -> Any:
    """Strip surrounding whitespace from string values, leave others alone.

    Control-surface configs accumulate stray whitespace — the Companion button
    that prompted this had a trailing space inside its URL — and no endpoint
    reading these params treats leading/trailing whitespace as meaningful. A
    language code of ``"ru "`` should switch to Russian, not fail validation.
    """
    return value.strip() if isinstance(value, str) else value


def merge_request_params(json_body: Optional[Mapping[str, Any]],
                         form: Optional[Mapping[str, Any]] = None,
                         query: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge the three ways a client can send parameters into one mapping.

    Precedence is JSON > form > query. Values that carry no instruction (``None``,
    blank strings) are dropped rather than overwriting a lower-precedence source,
    and string values are whitespace-stripped.

    A ``json_body`` that isn't a mapping (a client sent a JSON array or bare
    string) is ignored rather than raising — the other sources may still carry
    what was meant.
    """
    sources: Dict[str, Optional[Mapping[str, Any]]] = {
        "json": json_body, "form": form, "query": query,
    }
    merged: Dict[str, Any] = {}
    for name in _PRECEDENCE_LOW_TO_HIGH:
        source = sources[name]
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if _usable(value):
                merged[str(key)] = _clean(value)
    return merged
