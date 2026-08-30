"""Who gets the shared model when captions and the sermon summariser both want it.

One GGUF serves two callers on the machine that holds it: live captions arriving from a
paired machine over ``/api/translate``, and sermon summary chunks over
``/api/llm/summarize``. They contend on one generation lock, and the lock is unfair — a
caption that arrives mid-chunk waits out the whole generation. A summary nobody is
waiting for can afford that wait; a caption a congregation is reading cannot, because the
client gives up after its timeout and shows the untranslated source instead.

So the summariser stands aside. The question "must it stand aside right now" is asked in
three places with three different sets of facts — the server admitting a peer's chunk, the
summariser pausing between its own chunks, the worker deciding whether to start a run at
all — and this module is that one question, so the three cannot drift apart.

Two things this module exists to get right, both learned from the deployment where the
previous attempt was inert:

* **The machine holding the model is not the machine that knows.** It is not transcribing,
  so anything it infers from its own transcription state says "idle" while it is
  translating a service's captions. Its evidence arrives over the wire, and it has to be
  recorded when it arrives — hence ``PeerActivity`` rather than a flag.
* **A live service is quiet more often than you would think.** Minutes pass during music
  and prayer with no caption offloaded, so caption traffic alone cannot answer "is a
  service running". The heartbeat can: a paired machine pings this one every ~20s for
  exactly as long as its transcription is running, silence included. That is the signal
  the deferral is built on, and it is why the quiet window is tens of seconds rather than
  a few.

``PeerActivity`` is also deliberately not the general "who has talked to me lately" map
kept elsewhere: that one is refreshed by the summarise route itself, so a summariser
reading it would see its own footprint and conclude the machine is permanently busy.
"""

import threading
from typing import Dict, Optional, Tuple

# A paired machine pings /api/translate/heartbeat every ~20s while it transcribes. The
# window has to clear that comfortably, or a service looks idle in the gap between two
# pings; it also decides how soon after a service ends summarising may begin, so it is a
# small multiple of the ping interval rather than a large one.
DEFAULT_QUIET_SECONDS = 45.0

# How long the local pump's backlog hint stays believable. Generous on purpose: the pump
# publishes once per cycle, and a cycle blocked on a slow peer can take far longer than
# the caption interval. A window shorter than the worst cycle goes stale precisely when
# captions are most backed up, which is the opposite of what it is for.
DEFAULT_STALE_SECONDS = 60.0

# Signals a paired machine gives off. Kept apart because they mean different things: a
# caption is work in flight, a heartbeat is a service being alive between captions.
KIND_CAPTION = "caption"
KIND_HEARTBEAT = "heartbeat"

# Why the summariser is waiting — the 503 body, the log line, and what the tests assert on.
WAIT_TRANSCRIBING = "a service is being transcribed here"
WAIT_PEER_SESSION = "a paired machine is running a service"
WAIT_PEER_CAPTIONS = "captions are being translated here"
WAIT_LOCAL_BACKLOG = "captions are queued here"


def _seconds_since(then: float, now: float) -> float:
    """Age of a timestamp, never negative.

    A clock that steps backwards must not make a stale hint look fresh, nor a fresh one
    look ancient; treating the future as "just now" is the safe direction for both.
    """
    try:
        age = float(now) - float(then)
    except (TypeError, ValueError):
        return float("inf")
    return max(0.0, age)


class PeerActivity:
    """When each kind of signal from a paired machine was last seen.

    Only the most recent time per kind is kept, because the question is "is anyone out
    there running a service", not "which of them". Thread-safe: the recorders are Flask
    request handlers and the reader is a background worker.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: Dict[str, float] = {}
        self._who: Dict[str, str] = {}

    def record(self, client: str, kind: str, now: float) -> None:
        """Note that ``client`` gave off ``kind`` at ``now``."""
        with self._lock:
            previous = self._last.get(kind, 0.0)
            if now >= previous:
                self._last[kind] = float(now)
                self._who[kind] = client

    def last_seen(self, kind: str) -> float:
        """When ``kind`` last arrived, or 0.0 if it never has."""
        with self._lock:
            return self._last.get(kind, 0.0)

    def last_client(self, kind: str) -> Optional[str]:
        """Which machine last gave off ``kind``, for the log line."""
        with self._lock:
            return self._who.get(kind)

    def snapshot(self, now: float) -> Dict[str, float]:
        """Age in seconds of every kind seen, for diagnostics."""
        with self._lock:
            items = list(self._last.items())
        return {kind: _seconds_since(at, now) for kind, at in items}

    def forget(self, kind: Optional[str] = None) -> None:
        """Drop what has been recorded — all of it, or one kind (unpairing, tests)."""
        with self._lock:
            if kind is None:
                self._last = {}
                self._who = {}
            else:
                self._last.pop(kind, None)
                self._who.pop(kind, None)


def local_pump_busy(pending_count: int, published_at: float, now: float,
                    stale_after: float = DEFAULT_STALE_SECONDS) -> bool:
    """Whether this machine's own translation pump says captions are queued.

    Only meaningful while that pump is running: a machine that is not transcribing never
    publishes, and an old count must not be read as a live backlog — hence the staleness
    window rather than trusting the count alone.
    """
    if _seconds_since(published_at, now) > stale_after:
        return False
    try:
        return int(pending_count) > 0
    except (TypeError, ValueError):
        return False


def summariser_wait_reason(*, now: float,
                           transcribing: bool = False,
                           last_heartbeat_at: float = 0.0,
                           last_caption_at: float = 0.0,
                           local_pending_count: int = 0,
                           local_pending_at: float = 0.0,
                           quiet_seconds: float = DEFAULT_QUIET_SECONDS,
                           stale_after: float = DEFAULT_STALE_SECONDS,
                           defer_while_live: bool = True,
                           pause_on_backlog: bool = True) -> Optional[str]:
    """Why the summariser must stand aside right now, or None if it may proceed.

    The three ``defer_while_live`` arms are the strong rule — no summarising at all while
    a service is live, here or on a machine that offloads here. ``pause_on_backlog`` keeps
    its original narrower meaning: stand aside while this machine's own pump has captions
    queued. Turning ``defer_while_live`` off and leaving ``pause_on_backlog`` on restores
    the earlier interleaving behaviour exactly.

    Waiting never loses work. Everything owed stays queued or is recomputed from the
    session database at the end of the service, so the cost of standing aside is that a
    summary arrives later — which is the trade this whole module makes deliberately.
    """
    if defer_while_live:
        if transcribing:
            return WAIT_TRANSCRIBING
        if quiet_seconds > 0:
            if _seconds_since(last_heartbeat_at, now) < quiet_seconds:
                return WAIT_PEER_SESSION
            if _seconds_since(last_caption_at, now) < quiet_seconds:
                return WAIT_PEER_CAPTIONS
    if pause_on_backlog and local_pump_busy(local_pending_count, local_pending_at,
                                            now, stale_after):
        return WAIT_LOCAL_BACKLOG
    return None


def quiet_window(config_value: object, default: float = DEFAULT_QUIET_SECONDS) -> float:
    """A configured quiet window, clamped to something that can actually work.

    The floor is above the heartbeat interval on purpose: a window under ~20s would clear
    between two pings of a service that is still running, which is the one way this
    mechanism can fail silently.
    """
    try:
        value = float(config_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(25.0, min(900.0, value))


def stale_window(config_value: object, default: float = DEFAULT_STALE_SECONDS) -> float:
    """A configured staleness window for the local backlog hint, clamped."""
    try:
        value = float(config_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(5.0, min(600.0, value))


def hold_poll_seconds(waited: float) -> Tuple[float, bool]:
    """How long to sleep before re-checking a hold, and whether to log this time.

    Backs off from one second to five so a run parked for a whole service is not spinning,
    and reports the first check and then every minute so the log says why it is waiting
    without repeating itself.
    """
    interval = 1.0 if waited < 10.0 else 5.0
    should_log = waited == 0.0 or (int(waited) % 60 == 0 and waited >= 60.0)
    return interval, should_log
