"""What the live pump does with a caption that came back untranslated.

``translate_live_text`` returns the *source text* when it cannot translate — the remote
peer timed out and ``remote.fallback`` is "skip", or the LLM declined and
``llm.fallback`` is "skip". That is the right thing to put on screen: the operator needs
to see that the speaker said something. It is emphatically the wrong thing to write to
the database, and writing it was a real defect: the row stops matching
``translated_text IS NULL``, so the pump never retries it, the backfill cannot find it,
and the caption is Russian forever — in the live view, the session database, the SRT and
the HTML export. In one measured service that was 30 captions of 448.

So: display it, never persist it, retry it, and give up after a few tries. The give-up is
not a detail. Without it a peer that is merely slow costs one timeout per segment per
cycle, and the repair would stall the pump harder than the bug it fixes; the cooldown
spaces the retries and the cap ends them. When the cap is reached the row is left NULL
rather than filled with the source, because a missing caption is recoverable — the
backfill and the replay harness both key on NULL — and a poisoned one is not.
"""

import sqlite3
from typing import Dict, Optional, Set, Tuple

from stt.translation_backfill import BackfillAttempts

# Tries before a caption is left alone. Enough to ride out a chunk of contention or a
# brief peer stall, few enough that a genuinely dead peer stops costing timeouts.
DEFAULT_MAX_ATTEMPTS = 3

# Space between retries of one caption. A caption that failed on a 15s timeout will
# almost certainly fail again immediately; waiting a cycle or two costs nothing anyone is
# watching, because the source text is already on screen.
DEFAULT_COOLDOWN_SECONDS = 20.0

# How long a caption may keep waiting on a paired server that says it is busy and making
# progress (see stt.peer_load). These are live captions: a translation that arrives
# after the caption has left the screen helps nobody watching, and the backfill repairs
# the archive anyway. 30s is one more try after the first timeout, about as long as a
# caption stays on screen. A server that is *not* working gets one try, not this.
DEFAULT_MAX_WAIT_SECONDS = 30.0


def persist_decision(mt_engine: str, none_engine: str,
                     model_ready: bool) -> Tuple[bool, bool]:
    """``(display_it, persist_it)`` for one freshly translated segment.

    Three cases, and the difference between the last two is the whole point:

    * the model is still loading — the returned text is an echo of the source and the
      caption has not really been attempted yet, so neither show nor store it;
    * nothing translated it (``none_engine``) — show the source, store nothing, so the
      row stays NULL and is tried again;
    * something did — show it and store it, as always.
    """
    if not model_ready:
        return (False, False)
    if mt_engine == none_engine:
        return (True, False)
    return (True, True)


class LiveTranslationAttempts:
    """Per-session record of captions the pump could not translate.

    Wraps :class:`~stt.translation_backfill.BackfillAttempts` for the counting — the same
    give-up rule, already tested — and adds the two things the live path needs that the
    archive-repair path does not: a cooldown, so retries are spaced rather than
    per-cycle, and the source text, so the caption stays on screen while its row stays
    NULL.
    """

    def __init__(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
                 max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS) -> None:
        self._attempts = BackfillAttempts(max_attempts=max_attempts)
        self._cooldown = float(cooldown_seconds)
        self._max_wait = float(max_wait_seconds)
        self._last_try: Dict[int, float] = {}
        self._source: Dict[int, str] = {}
        self._first_failure: Dict[int, float] = {}
        self._given_up: Set[int] = set()

    def should_attempt(self, segment_id: int, now: float) -> bool:
        """Whether this caption may be sent to the model again on this cycle."""
        if self.exhausted(segment_id):
            return False
        last = self._last_try.get(segment_id)
        if last is None:
            return True
        return (now - last) >= self._cooldown

    def record_failure(self, segment_id: int, source_text: str, now: float,
                       peer_working: Optional[bool] = None) -> None:
        """Note that nothing translated this caption, keeping its text for the display.

        ``peer_working`` is what a paired translation server said about itself after
        this failure (stt.peer_load.peer_is_working):

        * False: it is down, or not getting through its work. One try is all a caption
          gets; waiting longer only spends more timeouts.
        * True: it is busy and making progress. The try is not counted, so the caption
          keeps waiting, until ``max_wait_seconds`` after its first failure.
        * None: unknown, which covers a server too old to say and every failure that
          did not involve a server at all. The plain attempt cap applies.
        """
        now = float(now)
        self._last_try[segment_id] = now
        self._source[segment_id] = source_text
        first = self._first_failure.setdefault(segment_id, now)
        if peer_working is False:
            self._given_up.add(segment_id)
        elif peer_working is True:
            if now - first >= self._max_wait:
                self._given_up.add(segment_id)
        else:
            self._attempts.record(segment_id)

    def record_success(self, segment_id: int) -> None:
        """Forget a caption that came back translated, so it costs nothing to carry."""
        self._attempts.succeeded(segment_id)
        self._last_try.pop(segment_id, None)
        self._source.pop(segment_id, None)
        self._first_failure.pop(segment_id, None)
        self._given_up.discard(segment_id)

    def exhausted(self, segment_id: int) -> bool:
        """Whether this caption has used up its retries and should be left alone."""
        return segment_id in self._given_up or self._attempts.exhausted(segment_id)

    def display_text(self, segment_id: int) -> Optional[str]:
        """The untranslated text to keep showing, or None if there is nothing pending."""
        return self._source.get(segment_id)

    def reset(self) -> None:
        """Forget everything — call when the session changes.

        Segment ids restart low in a new session database, so a carried-over count would
        be applied to an unrelated caption.
        """
        self._attempts.reset()
        self._last_try = {}
        self._source = {}
        self._first_failure = {}
        self._given_up = set()

    def size(self) -> int:
        """How many captions are being carried (for tests and diagnostics)."""
        return len(self._source)


# --- what the session database records about it --------------------------------
#
# A caption that needed a retry or a backfill used to be stored exactly like one that
# translated first time, and a caption that never translated left no trace beyond the
# log. Two columns make the path visible after the fact:
#
#   mt_attempts  how many times a translation was asked for. Written on every failed
#                try too, so a row that is still NULL says how hard it was tried.
#   mt_via       which path finally translated it: "live" (first try), "retry" (a
#                later try by the live loop) or "backfill" (repaired after scrolling
#                out of the live window, so it was never shown translated).
#
# NULL in both means the row predates the columns, or was never sent to the
# translation loop (Whisper translate writes the target language directly).

VIA_LIVE = "live"
VIA_RETRY = "retry"
VIA_BACKFILL = "backfill"

ATTEMPT_COLUMNS = (("mt_attempts", "INTEGER"), ("mt_via", "TEXT"))

# SET fragments for the UPDATE that stores a translation. SQLite evaluates every
# right-hand side against the row as it was before the update, so the CASE sees the
# failed tries already counted, not this one.
LIVE_SUCCESS_SET = (
    "mt_attempts = COALESCE(mt_attempts, 0) + 1, "
    "mt_via = CASE WHEN COALESCE(mt_attempts, 0) = 0 THEN '%s' ELSE '%s' END" % (VIA_LIVE, VIA_RETRY)
)
BACKFILL_SUCCESS_SET = "mt_attempts = COALESCE(mt_attempts, 0) + 1, mt_via = '%s'" % VIA_BACKFILL


def missing_attempt_columns(existing: Set[str]) -> Tuple[Tuple[str, str], ...]:
    """The attempt columns a session database still lacks, as (name, type)."""
    return tuple(col for col in ATTEMPT_COLUMNS if col[0] not in existing)


def record_failed_attempt(conn: sqlite3.Connection, row_id: int) -> None:
    """Count one try that came back untranslated.

    Only while the row is still untranslated: a late failure from another path must
    not add to a caption that has since been translated.
    """
    conn.execute(
        "UPDATE transcriptions SET mt_attempts = COALESCE(mt_attempts, 0) + 1"
        " WHERE id = ? AND translated_text IS NULL",
        (row_id,),
    )
