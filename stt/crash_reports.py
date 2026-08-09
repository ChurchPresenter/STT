"""What leaves the machine in a crash report, and what never should.

Two jobs, both running inside Sentry's ``before_send`` hook:

**Scrubbing.** The UI promises that no transcription content is sent. Request
bodies carry transcript text (``/api/translate``), glossary and dictionary
entries and file paths; the query string carries the ``?key=`` access token;
``sys.argv`` carries the install path, which usually contains the operator's
username; subprocess span descriptions carry the full command line, which for
ffmpeg names the input device and for media jobs the file being processed.
All of that is stripped here. Stack traces, versions and OS context stay.

**Dropping websocket-upgrade signals.** ``engineio`` completes a websocket
handshake by *raising* out of the WSGI app — ``ConnectionError`` under
werkzeug, ``StopIteration`` under gunicorn — because neither server has a way
to say "this request is no longer HTTP" through a return value. The exception
is caught by the server one frame up and is the success path, but Flask's
Sentry integration sees an exception escaping the WSGI app and reports it as
an unhandled error. Every browser that opens the live-transcription page
produces one. They are indistinguishable, by type alone, from a real
``ConnectionError`` in our own network code, so the match is anchored to the
engineio frame that does the raising rather than to the exception type.

Stdlib-only: events come in as the plain dicts Sentry hands to the hook.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: The engineio frame that raises to hand the socket over to the WSGI server,
#: as a path (forward slashes; Windows paths are normalised before comparing)
#: and as the dotted module name, since Sentry fills both.
_HANDOVER_PATH = "engineio/async_drivers/_websocket_wsgi.py"
_HANDOVER_MODULE = "engineio.async_drivers._websocket_wsgi"

#: What that frame raises, per server: werkzeug wants ``ConnectionError``,
#: gunicorn wants ``StopIteration``. Neither means anything went wrong.
_HANDOVER_EXCEPTIONS = frozenset({"ConnectionError", "StopIteration"})

#: Request fields that can hold user content or the access token.
_REQUEST_FIELDS_TO_DROP = ("data", "query_string", "cookies", "headers", "env")


def _frame_is_websocket_handover(frame: Dict[str, Any]) -> bool:
    """True if this frame is engineio's socket hand-over to the WSGI server."""
    for key in ("filename", "abs_path"):
        if (frame.get(key) or "").replace("\\", "/").endswith(_HANDOVER_PATH):
            return True
    return (frame.get("module") or "") == _HANDOVER_MODULE


def is_websocket_handover(event: Dict[str, Any]) -> bool:
    """True if ``event`` is engineio signalling a completed websocket upgrade.

    Requires both halves of the signature — the handover exception type *and*
    the engineio frame raising it — so a genuine ``ConnectionError`` from, say,
    a peer request or a model download is still reported.
    """
    values = ((event.get("exception") or {}).get("values")) or ()
    if not values:
        return False
    # Sentry orders chained exceptions oldest-first; the one that escaped is last.
    raised = values[-1]
    if raised.get("type") not in _HANDOVER_EXCEPTIONS:
        return False
    frames = ((raised.get("stacktrace") or {}).get("frames")) or ()
    return any(_frame_is_websocket_handover(f) for f in frames)


def scrub_event(event: Dict[str, Any], hint: Any = None) -> Optional[Dict[str, Any]]:
    """Sentry ``before_send``/``before_send_transaction`` hook.

    Returns the event with user content removed, or ``None`` to drop it.
    """
    if is_websocket_handover(event):
        return None

    request = event.get("request")
    if request:
        for key in _REQUEST_FIELDS_TO_DROP:
            request.pop(key, None)
    # ArgvIntegration attaches the command line, which carries the install path
    # (frequently including the operator's username).
    extra = event.get("extra")
    if extra:
        extra.pop("sys.argv", None)
    # Subprocess spans are named after the full command line, which for ffmpeg
    # carries the input device name and for media jobs the file path.
    for span in event.get("spans") or ():
        if span.get("op", "").startswith("subprocess"):
            span["description"] = (span.get("description", "").split() or ["subprocess"])[0]
            span.pop("data", None)
    return event
