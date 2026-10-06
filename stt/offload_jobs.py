"""Offloaded caption translations running beside the live translation loop.

The loop translated the cycle's captions one after another, so one call to a paired
server that took its full 15s timeout held every caption behind it, including ones
already cached and needing no call at all. That is how a busy server on 2026-08-30
delayed every caption rather than only the slow ones.

:class:`OffloadJobs` runs each offloaded call on its own thread. The loop never waits:
it hands a caption over, carries on emitting what it has, and picks up the answer on
a later cycle. Only the call runs on the thread. Everything that changes state (the
translation cache, the retry bookkeeping, the session database) stays on the loop's
own thread, which is what keeps those free of new races.

Two rules make a late answer safe to apply:

* **Keyed by session.** Caption ids restart in every session's database, so an answer
  that lands after a session change must not be applied to the new session's caption
  with the same id. :meth:`take` only returns answers for the session asked about.
* **A cap on calls in flight.** The server generates one caption at a time anyway;
  more parallel calls only queue there and spend more timeouts here.

Stdlib-only; the thread starter is injectable so tests run jobs synchronously.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Hashable, Optional, Set, Tuple

# Calls to the paired server at once. Matches the loop's per-cycle translation budget,
# so a backlog drains at the rate it always did, just without one call blocking the next.
DEFAULT_MAX_IN_FLIGHT = 3

Key = Tuple[Hashable, int]  # (session id, caption id)


def _start_daemon(target: Callable[[], None]) -> None:
    # Daemon, not a ThreadPoolExecutor: an executor's workers are joined at exit, so a
    # call stuck in its 15s timeout would hold up a server shutdown.
    threading.Thread(target=target, daemon=True, name="OffloadCaption").start()


class OffloadJobs:
    """Captions handed to background calls, and the answers waiting to be applied."""

    def __init__(self, max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
                 start: Callable[[Callable[[], None]], None] = _start_daemon) -> None:
        self._max = max(1, int(max_in_flight))
        self._start = start
        self._lock = threading.Lock()
        self._in_flight: Set[Key] = set()
        self._done: Dict[Key, Any] = {}

    def slots(self) -> int:
        """How many more captions may be handed over right now."""
        with self._lock:
            return max(0, self._max - len(self._in_flight))

    def busy(self, session: Hashable, caption_id: int) -> bool:
        """Whether this caption is in flight or has an answer waiting to be applied."""
        key = (session, caption_id)
        with self._lock:
            return key in self._in_flight or key in self._done

    def submit(self, session: Hashable, caption_id: int, call: Callable[[], Any]) -> bool:
        """Run ``call`` in the background; False if the caption is already handled or
        no slot is free. An exception from ``call`` is kept as the answer, so the loop
        sees the failure rather than the caption waiting for ever.
        """
        key = (session, caption_id)
        with self._lock:
            if key in self._in_flight or key in self._done or len(self._in_flight) >= self._max:
                return False
            self._in_flight.add(key)

        def run() -> None:
            try:
                result: Any = call()
            except Exception as exc:  # handed to the loop, which decides what it means
                result = exc
            with self._lock:
                self._in_flight.discard(key)
                self._done[key] = result

        try:
            self._start(run)
        except Exception:
            with self._lock:
                self._in_flight.discard(key)
            raise
        return True

    def take(self, session: Hashable, caption_id: int) -> Tuple[bool, Optional[Any]]:
        """(True, answer) once the caption's call has finished, else (False, None)."""
        key = (session, caption_id)
        with self._lock:
            if key in self._done:
                return True, self._done.pop(key)
        return False, None

    def drop_other_sessions(self, session: Hashable) -> int:
        """Forget answers from any other session; returns how many were dropped.

        Calls still in flight for an old session finish into ``_done`` and are dropped
        by the next call to this. Nothing ever reads them.
        """
        with self._lock:
            stale = [k for k in self._done if k[0] != session]
            for key in stale:
                del self._done[key]
            return len(stale)

    def in_flight(self) -> int:
        with self._lock:
            return len(self._in_flight)
