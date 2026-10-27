"""Keeping the live translation stream append-only and every shown line fixed.

Every ``translation_update`` carries the whole window of translated captions,
and a caption is left out until its translation exists. Translations do not
finish in order. Offloaded calls run concurrently and a backlog drains
newest-first, so caption N+1 is often ready before caption N. Sent as it
stands, the stream then grows *in the middle*: N+1 is shown, and a moment
later N appears above it.

A display that reveals the caption gradually cannot absorb that.
ChurchPresenter's drip feed treats the window as one continuous caption and
keeps its reveal cursor only while the text it has shown is still a prefix of
the new caption. A line inserted above that text breaks the prefix, so the
cursor jumps to the end: everything not yet revealed appears at once and the
lines shift. On 2026-10-25 that happened to 35 of 540 captions, about once
every two and a half minutes. The root page re-reveals from a character count
and shifts the same way.

Two rules follow, both about the people reading, who lose their place when text
they have read moves:

* **A caption gets one bounded chance to appear in order.** A finished caption
  waits while an earlier one is still being translated, up to ``hold_seconds``.
  Past that the earlier caption is given up on for the live display: the ones
  behind it go out, and it is never inserted afterwards, even when its
  translation lands a moment later. By then the room has moved on, and
  inserting it is the jump this exists to prevent.
* **A line on screen never changes.** A caption whose translation failed is not
  shown in the source language as a stand-in for a translation to replace
  later. It is skipped.

A skipped or given-up caption is lost only from the live display. The session
database and SRT still get the translation through the backfill, which repairs
captions the live window could not.

Every decision is reported through :meth:`EmissionOrder.drain_events` so a
service can be traced afterwards. The events carry caption ids and timings and
never caption text, which is what lets their log tag appear in the diagnostic
report.

Stdlib-only. The live loop owns one instance and calls it once per cycle.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

#: Default for ``live_translation.in_order_hold_seconds``. Replaying 2026-10-25's
#: timings: 10s lost 6 translated captions from the live display (their
#: translations took longer), and 15s lost none, with the held captions waiting
#: a median 2.0s and at most 4.0s. 20s lost none either but let one wait 9s.
DEFAULT_HOLD_SECONDS = 15.0


class EmissionOrder:
    """Per-session state that keeps the emitted caption list append-only."""

    def __init__(self) -> None:
        self._session: Optional[str] = None
        self._first_pending: Dict[int, float] = {}
        self._given_up: Dict[int, float] = {}
        self._held: Dict[int, float] = {}
        self._shown: Set[int] = set()
        self._late: Set[int] = set()
        self._events: List[str] = []

    def begin_cycle(self, session_id: Optional[str]) -> None:
        """Forget everything when the session changes: ids restart with a new database."""
        if session_id != self._session:
            self._session = session_id
            self._first_pending.clear()
            self._given_up.clear()
            self._held.clear()
            self._shown.clear()
            self._late.clear()

    def skip(self, seg_id: int, reason: str, now: float) -> None:
        """Keep ``seg_id`` off the live display for good, because ``reason``."""
        if seg_id in self._shown or seg_id in self._given_up:
            return
        self._given_up[seg_id] = now
        self._first_pending.pop(seg_id, None)
        self._events.append(f"skipped {seg_id}: {reason}")

    def drain_events(self) -> List[str]:
        """What was decided since the last call, oldest first, for the log."""
        events, self._events = self._events, []
        return events

    def order(self, ready: List[Dict[str, Any]], pending: Iterable[int],
              now: float, hold_seconds: float) -> List[Dict[str, Any]]:
        """The captions in ``ready`` (ascending id) that may be sent this cycle.

        ``pending`` are window ids still waiting for a translation. The oldest
        pending id that has not used up ``hold_seconds`` is a barrier: ready
        captions after it wait, except those already sent. A caption given up
        on is never sent. ``hold_seconds`` of 0 turns the ordering off.
        """
        pending_ids = sorted(set(pending))
        window_floor = min([int(s["id"]) for s in ready] + pending_ids, default=None)
        self._prune(window_floor)

        barrier: Optional[int] = None
        if hold_seconds > 0:
            for seg_id in pending_ids:
                if seg_id in self._given_up:
                    continue
                since = self._first_pending.setdefault(seg_id, now)
                if now - since >= hold_seconds:
                    self._given_up[seg_id] = now
                    self._events.append(
                        f"gave up on {seg_id} after {now - since:.1f}s untranslated; "
                        f"later captions released, it will not be shown live")
                    continue
                barrier = seg_id
                break

        out = []
        for seg in ready:
            seg_id = int(seg["id"])
            if seg_id in self._shown:
                out.append(seg)
                continue
            if seg_id in self._given_up:
                if seg_id not in self._late:
                    self._late.add(seg_id)
                    self._events.append(
                        f"kept {seg_id} off screen: translated "
                        f"{now - self._given_up[seg_id]:.1f}s after it was given up")
                continue
            self._first_pending.pop(seg_id, None)
            if barrier is not None and seg_id > barrier:
                if seg_id not in self._held:
                    self._held[seg_id] = now
                    self._events.append(f"held {seg_id} behind {barrier}")
                continue
            if seg_id in self._held:
                self._events.append(f"sent {seg_id} after holding {now - self._held.pop(seg_id):.1f}s")
            self._shown.add(seg_id)
            out.append(seg)
        return out

    def _prune(self, floor: Optional[int]) -> None:
        """Drop state for ids that have left the window, so memory stays bounded."""
        if floor is None:
            return
        self._first_pending = {k: v for k, v in self._first_pending.items() if k >= floor}
        self._given_up = {k: v for k, v in self._given_up.items() if k >= floor}
        self._held = {k: v for k, v in self._held.items() if k >= floor}
        self._shown = {k for k in self._shown if k >= floor}
        self._late = {k for k in self._late if k >= floor}
