"""When a sermon summary run may start, and how long it may hold a session back.

Two rules live here, both consequences of the same decision: summarising never competes
with a live service.

**Queue nothing mid-service.** The automatic trigger fires from the phase-detector tick,
which runs throughout a service. Deferring at the point of *generation* would still write
a "pending" row for every sermon the detector closed, and a pending row is indistinguishable
from a queued one to the end-of-session catch-up, which skips what is already recorded. So
the whole scan is skipped instead, and nothing is lost: what a service owes is recomputed
at its end from the blocks and the stored rows, and the transcript in the session database
is the durable record either way.

**Count only the time a run is actually working.** The end-of-session run holds the
finished session's files back from delivery, with a deadline so a crashed summariser
cannot hold them forever. Once a run can also stand aside for a *new* service, that
deadline becomes a trap: it would expire while the run is parked, the session would be
delivered and its local copy deleted, and the run would resume and write a summary into a
database nobody will ever open. So paused time does not count against the deadline — and
because "does not count" could otherwise mean "forever", a separate wall-clock ceiling
ends it regardless. A late delivery is recoverable; a lost one is not, so the ceiling
gives up the summary rather than the session.
"""

from typing import Optional, Tuple

# Longest a finished session may be held back in total, paused time included, before it
# is delivered without its summary. Six hours covers a service that is followed by another
# on the same day; past that, something is wrong and the files matter more.
DEFAULT_PAUSE_CEILING_SECONDS = 21600.0

HOLD_TRANSCRIBING = "a service is being transcribed here"


def defer_scan(*, enabled: bool, ignore_settle: bool, transcription_running: bool,
               defer_while_live: bool) -> bool:
    """Whether the automatic scan must not look for sermons to queue yet.

    ``ignore_settle`` marks the two explicit callers — the end-of-session catch-up and the
    operator pressing Summarise — and neither is deferred here: they are the mechanism
    this deferral relies on, and an operator asking for a summary is asking for one. Only
    the tick is held back.
    """
    if ignore_settle:
        return False
    if not enabled:
        return True
    return bool(defer_while_live and transcription_running)


def hold_reason(*, transcription_running: bool, transcription_starting: bool,
                defer_while_live: bool) -> Optional[str]:
    """Why the summary worker must stand aside before taking or continuing work.

    A session that is *starting* counts: the flag flips a moment after the operator
    presses Start, and a chunk begun in that window is one the new service pays for.
    """
    if not defer_while_live:
        return None
    if transcription_running or transcription_starting:
        return HOLD_TRANSCRIBING
    return None


def pause_fields(*, now: float, paused_at: float, paused_total: float,
                 pausing: bool) -> Tuple[float, float]:
    """New ``(paused_at, paused_total)`` when entering or leaving a stand-aside.

    Entering while already paused keeps the original start, so a poll loop calling this
    every few seconds does not restart the clock; leaving while not paused is a no-op, so
    the first call after a run that never paused is harmless.
    """
    if pausing:
        if paused_at:
            return (paused_at, paused_total)
        return (float(now), paused_total)
    if not paused_at:
        return (0.0, paused_total)
    return (0.0, paused_total + max(0.0, float(now) - float(paused_at)))


def working_seconds(*, now: float, since: float, paused_at: float,
                    paused_total: float) -> float:
    """How long the run has actually been working, excluding time spent standing aside."""
    elapsed = max(0.0, float(now) - float(since))
    parked = float(paused_total)
    if paused_at:
        parked += max(0.0, float(now) - float(paused_at))
    return max(0.0, elapsed - parked)


def finalise_expired(*, now: float, since: float, paused_at: float, paused_total: float,
                     deadline: float,
                     max_total: float = DEFAULT_PAUSE_CEILING_SECONDS) -> bool:
    """Whether the hold on a finished session has run out.

    Two bounds, because they answer different questions: ``deadline`` is how long a run
    may *work* before it is presumed stuck, and ``max_total`` is how long the session may
    be held at all. A run that has spent an hour parked for a new service has not used any
    of the first and all of the second.
    """
    if max_total > 0 and (float(now) - float(since)) >= max_total:
        return True
    return working_seconds(now=now, since=since, paused_at=paused_at,
                           paused_total=paused_total) >= deadline
