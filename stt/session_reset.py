"""Noticing that the session rolled over, so per-session caches can be dropped.

Several web-process caches are keyed by database row id — the translation
cache, the TTS spoken-tracker's high-water mark, the entries cache. Every
session gets its own database, and row ids restart from the same low numbers in
each one, so a cache carried across a rollover addresses the wrong rows.

Until now each consumer that cared hand-rolled the same three lines (remember
the id, compare, reset). Only one did: the translation backfill. The rest were
written when a new database could only appear via Stop → Start, which reloads
the process's world anyway. A mid-run "reset session" breaks that assumption
for all of them at once, and the consequences are quiet rather than loud — the
TTS tracker in particular just stops emitting, because its baseline is a
monotonic high-water mark that new low ids never exceed.

So the rule lives in one tested place instead of being repeated by hand.

Stdlib-only, and deliberately ignorant of what a "session id" is: it is
whatever the caller uses to identify one, compared for equality.
"""

from __future__ import annotations

import threading
from typing import Any, Optional


class SessionTracker:
    """Remembers the last session seen and reports when it changes.

    Thread-safe: the emit loops that consult it run concurrently, and two of
    them noticing the same rollover must not both be told they are first —
    ``changed`` is the trigger for a reset, so it has to fire exactly once per
    rollover per tracker.
    """

    def __init__(self, initial: Optional[Any] = None) -> None:
        self._seen: Optional[Any] = initial
        self._started = initial is not None
        self._lock = threading.Lock()

    def changed(self, session_id: Optional[Any]) -> bool:
        """Whether ``session_id`` differs from the last one seen.

        The first call only records — it does not report a change, because
        starting up is not a rollover and callers would otherwise clear caches
        needlessly on every boot.

        ``None`` carries no information and is **not recorded**: the worker
        publishes it for the window between retiring one database and opening
        the next. Recording it would both report a spurious change on the way
        in and — worse — hide the real one on the way out, because the new id
        would then be compared against ``None`` instead of against the session
        it replaced. Skipping it leaves the last real id in place, so the
        rollover is caught exactly once, when the new id arrives.
        """
        with self._lock:
            if session_id is None:
                return False

            if not self._started:
                self._started = True
                self._seen = session_id
                return False

            previous = self._seen
            self._seen = session_id
            return session_id != previous

    @property
    def current(self) -> Optional[Any]:
        """The last session id seen, or None before the first call."""
        with self._lock:
            return self._seen

    def reset(self) -> None:
        """Forget what was seen, so the next call records rather than compares."""
        with self._lock:
            self._seen = None
            self._started = False
