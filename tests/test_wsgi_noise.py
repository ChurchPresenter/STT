"""The werkzeug records the server refuses to log — and therefore to report.

Werkzeug logs a failed request line at ERROR, and sentry-sdk's logging
integration turns any ERROR record into a Sentry issue. A device on the LAN
opening https:// against the plaintext port thus filed a crash report against
this application every time it retried, and each one grouped separately:
werkzeug builds the message with the client IP and the timestamp inside the
format string, so no two occurrences share a template and ignoring one issue
does nothing about the next.

The filter is asserted through the real logging machinery rather than by
calling filter() alone, because the property that matters is *where* it runs:
Logger.filter() happens before Logger.handle(), which is the path sentry-sdk
patches. A filter attached to a handler instead would silence the console and
still send the report.
"""

import logging

import pytest

from stt.log_gate import is_benign_wsgi_message
from tests.conftest import extract_definitions

_FILTER = extract_definitions(
    "speech_to_text.py", ["_SuppressBenignWSGINoise"],
    extra_globals={"logging": logging,
                   "is_benign_wsgi_message": is_benign_wsgi_message},
)["_SuppressBenignWSGINoise"]


def record(message, *args):
    return logging.LogRecord("werkzeug", logging.ERROR, __file__, 1, message, args, None)


# The literal message from the Sentry issue, TLS bytes and all.
TLS_HANDSHAKE = ("192.168.2.254 - - [30/Aug/2026 10:30:33] code 400, message "
                 "Bad request syntax ('\\x16\\x03\\x00\\x00S\\x01\\x00\\x00O\\x03\\x00')")


@pytest.mark.parametrize("message", [
    TLS_HANDSHAKE,
    "192.168.2.254 - - [30/Aug/2026 10:30:33] code 400, message Bad request version ('\\x00/\\x00')",
    "10.0.0.9 - - [01/Jan/2026 00:00:00] code 400, message Bad HTTP/0.9 request type ('GET')",
    "192.168.1.5 - - [30/Aug/2026 10:30:33] write() before start_response",
])
def test_benign_records_are_dropped(message):
    assert _FILTER().filter(record(message)) is False


@pytest.mark.parametrize("message", [
    "192.168.2.254 - - [30/Aug/2026 10:30:33] code 500, message Internal Server Error",
    "Error on request:",
    "192.168.2.254 - - [30/Aug/2026 10:30:33] code 404, message Not Found",
])
def test_real_errors_still_pass(message):
    """Silencing the werkzeug logger wholesale was the tempting fix. It would
    have taken these with it."""
    assert _FILTER().filter(record(message)) is True


def test_a_record_whose_arguments_do_not_format_is_kept():
    """getMessage() raises on a bad format string. An unreadable record is
    still a record; dropping it would hide the very thing worth seeing."""
    assert _FILTER().filter(record("code %d, message %s", "not-an-int")) is True


def test_the_filter_runs_before_call_handlers():
    """The property that makes this suppress the Sentry event and not just the
    console line. Logger.handle() consults Logger.filter() before calling
    callHandlers(), and callHandlers is the method sentry-sdk monkeypatches to
    turn ERROR records into issues — so a filter on the logger stops the crash
    report. A filter on a *handler* would not: handlers run inside
    callHandlers, after the report has already been queued."""
    logger = logging.getLogger("werkzeug-test-noise")
    logger.setLevel(logging.ERROR)
    logger.addFilter(_FILTER())
    reported = []
    logger.callHandlers = lambda rec: reported.append(rec)  # type: ignore[method-assign]
    try:
        logger.error(TLS_HANDSHAKE)
        assert reported == []
        logger.error("code 500, message Internal Server Error")
        assert len(reported) == 1
    finally:
        logger.filters.clear()
