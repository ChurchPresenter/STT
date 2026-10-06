"""Whether a paired translation server is busy working or simply not answering.

When an offloaded caption fails, the machine that sent it (A) has to decide how long
the caption keeps waiting. The right answer depends on something A cannot see: whether
the server (B) is working through a queue or is gone. A timeout looks the same either
way. On 2026-08-30 B was busy behind sermon-summary chunks and every caption it timed
out on was still answered within a minute; a B that has hung or lost its network
never will be.

So B says what it is doing. :class:`WorkTracker` counts the translate and summarise
requests B is in the middle of and when it last finished one, and B puts that
snapshot in its heartbeat reply. A asks for one fresh straight after a timeout and
:func:`peer_is_working` reads it:

* **working**: requests in flight, and one *finished* recently, so B is making
  progress. A request merely starting is not progress: a hung B keeps receiving new
  captions, so "started recently" would describe it as busy for ever. The caption keeps waiting, within a cap.
* **not working**: nothing in flight (the caption never got there), or requests stuck
  with no progress. Waiting will not help.
* **unknown**: a B that predates this sends no snapshot. A keeps the old rule.

Stdlib-only, with the clock injected, so all of it is testable without a network.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

# How recently B must have finished or started something to count as making progress.
# A summary chunk takes ~17s on .52, a caption ~1.5s; twice the longest unit of work
# leaves room for one more chunk in the queue ahead of the caption.
PROGRESS_WINDOW_SECONDS = 45.0

# The key the snapshot travels under in B's heartbeat reply.
LOAD_KEY = "load"


class WorkTracker:
    """B's side: which requests are running, and when one last finished."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._started: Dict[int, float] = {}
        self._next = 0
        self._last_done: Optional[float] = None

    def begin(self) -> int:
        """Note a request starting; returns the token to pass to :meth:`end`."""
        with self._lock:
            self._next += 1
            self._started[self._next] = self._clock()
            return self._next

    def end(self, token: int) -> None:
        """Note a request finishing, successfully or not: either way B got through it."""
        with self._lock:
            if self._started.pop(token, None) is not None:
                self._last_done = self._clock()

    @contextmanager
    def working(self) -> Iterator[None]:
        token = self.begin()
        try:
            yield
        finally:
            self.end(token)

    def snapshot(self) -> Dict[str, Any]:
        """What B reports: ages in seconds, so the two machines' clocks never meet."""
        with self._lock:
            now = self._clock()
            oldest = min(self._started.values()) if self._started else None
            return {
                "in_flight": len(self._started),
                "oldest_started_age_s": None if oldest is None else round(now - oldest, 1),
                "last_done_age_s": None if self._last_done is None else round(now - self._last_done, 1),
            }


def parse_load(reply: Any) -> Optional[Dict[str, Optional[float]]]:
    """The snapshot from B's heartbeat reply, or None for a B that sends none."""
    if not isinstance(reply, dict):
        return None
    load = reply.get(LOAD_KEY)
    if not isinstance(load, dict):
        return None
    in_flight = load.get("in_flight")
    if not isinstance(in_flight, int) or isinstance(in_flight, bool) or in_flight < 0:
        return None
    out: Dict[str, Optional[float]] = {"in_flight": float(in_flight)}
    for key in ("oldest_started_age_s", "last_done_age_s"):
        value = load.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            out[key] = float(value)
        else:
            out[key] = None
    return out


def peer_is_working(load: Optional[Dict[str, Optional[float]]], *,
                    progress_window: float = PROGRESS_WINDOW_SECONDS) -> Optional[bool]:
    """True if B is busy and making progress, False if it is not, None if unknown."""
    if load is None:
        return None
    if not load.get("in_flight"):
        return False
    done = load.get("last_done_age_s")
    return done is not None and done <= progress_window


# What kind of failure an offloaded call was, from the exception it raised.
FAILURE_DOWN = "down"        # could not reach the server at all
FAILURE_TIMEOUT = "timeout"  # reached it, but no answer in time: busy or hung, ask it
FAILURE_ERROR = "error"      # it answered with an error, or something else went wrong


def classify_failure(exc: Optional[BaseException]) -> str:
    """Sort an offload failure into down / timeout / error.

    Matched on class names through the MRO and the ``__cause__`` chain rather than on
    ``requests`` types, so this module stays importable without ``requests`` and also
    understands the stdlib's own socket errors. A connect timeout is "down" even though
    it is also a timeout: nothing reached the server, so there is nothing to ask it about.
    """
    seen = 0
    while exc is not None and seen < 5:
        names = {cls.__name__ for cls in type(exc).__mro__}
        if "ConnectTimeout" in names:
            return FAILURE_DOWN
        if names & {"ReadTimeout", "TimeoutError", "timeout"}:
            return FAILURE_TIMEOUT
        if "ConnectionError" in names:
            return FAILURE_DOWN
        exc = exc.__cause__
        seen += 1
    return FAILURE_ERROR
