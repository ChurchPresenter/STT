"""Suppress log records that repeat an unchanged value.

Some endpoints are polled on a timer and log a value that almost never moves.
`/api/audio-devices` is the worst offender: an external monitor hits it once a
minute, and each hit emitted `Listed 0 devices using ffmpeg` at INFO. Sentry
Logs are not sampled the way traces are, so that shipped ~1,440 identical
records a day and crowded out everything worth reading.

A ChangeGate remembers the last value seen per key and reports whether the
current one differs, so the caller can log at INFO on a transition and drop to
DEBUG for the steady state. The count still surfaces the moment a device
appears or disappears, which is the only time it carries information.

Stdlib-only and free of runtime config, per the stt/ module conventions.
"""

import threading
from typing import Any, Dict


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
