"""Which log records are not worth keeping.

Two kinds, both of which used to reach Sentry as errors or crowd out the log.

**Records that repeat an unchanged value.**

Some endpoints are polled on a timer and log a value that almost never moves.
`/api/audio-devices` is the worst offender: an external monitor hits it once a
minute, and each hit emitted `Listed 0 devices using ffmpeg` at INFO. Sentry
Logs are not sampled the way traces are, so that shipped ~1,440 identical
records a day and crowded out everything worth reading.

A ChangeGate remembers the last value seen per key and reports whether the
current one differs, so the caller can log at INFO on a transition and drop to
DEBUG for the steady state. The count still surfaces the moment a device
appears or disappears, which is the only time it carries information.

**Records that report someone else's mistake.** Werkzeug logs a failed request
line at ERROR, and sentry-sdk turns any ERROR record into an issue. A client
opening ``https://`` against the plaintext port therefore filed a crash report
against us — one per occurrence, ungroupable, because werkzeug bakes the client
IP and the timestamp into the message template. ``is_benign_wsgi_message``
names those. A logger filter using it runs in ``Logger.filter()``, before
``Logger.handle()`` reaches the ``callHandlers`` that sentry-sdk patches, so it
suppresses the crash report as well as the console line — and does it without
silencing the werkzeug logger wholesale.

Stdlib-only and free of runtime config, per the stt/ module conventions.
"""

import threading
from typing import Any, Dict, Tuple


class ChangeGate:
    """Tracks the last value seen per key; True only when it changes.

    The first call for a key always reports a change, so a value is logged once
    at startup rather than being silently swallowed. Instances are safe to share
    across Flask worker threads.
    """

    def __init__(self) -> None:
        self._seen: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def changed(self, key: str, value: Any) -> bool:
        """Report whether `value` differs from the last value seen for `key`."""
        with self._lock:
            missing = key not in self._seen
            if missing or self._seen[key] != value:
                self._seen[key] = value
                return True
            return False

    def reset(self, key: str = "") -> None:
        """Forget one key, or every key when `key` is empty."""
        with self._lock:
            if key:
                self._seen.pop(key, None)
            else:
                self._seen.clear()


#: Werkzeug messages that never indicate a fault in this application.
BENIGN_WSGI_MESSAGES: Tuple[str, ...] = (
    # The harmless AssertionError the werkzeug dev server logs for Socket.IO
    # polling/transport requests under async_mode="threading": the request
    # finishes without the normal WSGI response path. The connection works.
    "write() before start_response",
    # A client speaking TLS to the plaintext port, or otherwise sending
    # something that is not a request line. stdlib http.server cannot parse it,
    # answers 400, and logs that at ERROR. Nothing here went wrong.
    "code 400, message Bad request syntax",
    "code 400, message Bad request version",
    "code 400, message Bad HTTP/0.9 request type",
)


def is_benign_wsgi_message(message: str) -> bool:
    """True when a werkzeug log record reports no fault of ours.

    Substring matching, not equality: werkzeug prefixes the client address and
    a timestamp, and http.server appends the offending bytes.
    """
    return any(benign in message for benign in BENIGN_WSGI_MESSAGES)
