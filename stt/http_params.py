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

import json
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


def parse_json_body(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse a raw request body as a JSON object, or None if it isn't one.

    Needed because a client can send perfectly good JSON under the wrong
    Content-Type (``text/plain``, ``application/x-www-form-urlencoded``, none at
    all). Flask then gives ``get_json(silent=True) is None`` *and* an empty
    ``request.form`` — the body is well-formed and completely invisible, and the
    route answers 400 as though nothing was sent. Callers use this as a last
    resort on the raw bytes.

    Only a JSON *object* counts: an array or scalar carries no named parameters.
    """
    if not raw or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def merge_request_params(json_body: Optional[Mapping[str, Any]],
                         form: Optional[Mapping[str, Any]] = None,
                         query: Optional[Mapping[str, Any]] = None,
                         *, keep_blank: bool = False) -> Dict[str, Any]:
    """Merge the three ways a client can send parameters into one mapping.

    Precedence is JSON > form > query. String values are whitespace-stripped.

    ``keep_blank`` decides what a blank value means, and the right answer differs
    by endpoint:

    * ``False`` (default) — blank means "not sent". Correct for a control surface
      that posts a fixed set of fields on every press: an empty one must not blank
      a live setting.
    * ``True`` — blank is a real value. Required by settings endpoints where an
      empty string is how you *clear* a field (a remote endpoint, a voice
      override). Dropping it there would make clearing impossible.

    A ``json_body`` that isn't a mapping (a client sent a JSON array or bare
    string) is ignored rather than raising — the other sources may still carry
    what was meant.

    Note for callers: this unions the query string into the parameters, so it must
    not be used by a route that merges the body wholesale into config. A URL can
    carry ``?key=<access_token>``, which would then be persisted as a config key.
    Only use it where the route reads named fields.
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
            if keep_blank or _usable(value):
                merged[str(key)] = _clean(value)
    return merged
