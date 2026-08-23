"""Live operator marks, and how they combine with what the detector found.

The detector hears structure from audio and is right about the shape of a service but only
approximate about its edges: dwell means a block starts a minute or two after the preaching
does, and a boundary is back-dated once the next run begins. An operator in the room knows
the exact moment — and is the only one who can name a stretch before it is over.

The reverse is also true. An operator is watching a service, not a console: they will mark
the sermon starting and then miss the end entirely, or mark nothing for twenty minutes.
Anything that made the timeline *depend* on them would be worse than the detector alone.

So a mark is an **edge, not a phase**:

* where a mark exists it decides where the phase starts and what it is called — that is the
  thing the operator is better at;
* where no mark exists nothing changes, and the detector's own blocks stand;
* the *end* of a marked phase comes from the detector unless the operator marks one. The
  phase runs to the end of the run the detector is in — so an operator who marks a start
  and then spaces out still gets a phase that closes itself, at the moment the audio
  changed rather than at some arbitrary timeout.

Marks resolve into ordinary spans, which is what makes this small: a span already outranks
the detector everywhere downstream — the timeline, sermon_ranges, the summariser — so
nothing else has to learn what a mark is.

Stored as rows in ``service_phase_corrections`` with a ``start_ms`` and a NULL ``end_ms``.
Every existing consumer tests both, so a mark is invisible to code written before it
existed rather than being half-understood by it.

Stdlib-only and pure: blocks, marks and "now" go in, spans come out.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

# What an "ends now" press records. A label would be a lie — the operator is saying that
# something stopped, not what the next thing is.
END_LABEL = "__end__"


def is_mark(row: Mapping[str, Any]) -> bool:
    """Whether a corrections row is a live mark rather than a span or a relabel."""
    return (row.get("block_index") is None
            and bool(row.get("start_ms"))
            and not row.get("end_ms"))


def marks(corrections: Sequence[Mapping[str, Any]]) -> List[dict]:
    """The live marks among the corrections, oldest first, one per moment.

    A double press at the same instant is one decision made twice — the second row wins, so
    correcting a mis-picked label is pressing again rather than finding an undo.
    """
    by_time: Dict[int, dict] = {}
    for row in corrections:
        if not is_mark(row):
            continue
        at = int(row.get("start_ms") or 0)
        kept = by_time.get(at)
        if kept is None or int(row.get("id") or 0) >= int(kept.get("id") or 0):
            by_time[at] = dict(row)
    return [by_time[at] for at in sorted(by_time)]


def _run_end(blocks: Sequence[Mapping[str, Any]], at_ms: int) -> Optional[int]:
    """Where the detector's current run ends, for a phase marked as starting at *at_ms*.

    The run is the block containing the mark plus every consecutive block of the same kind
    after it: a sermon interrupted by a cough is two blocks of one kind, and ending the
    phase at the cough would be the detector overruling the operator on the very thing the
    operator is better at.

    None while that run is still the last block and still ongoing — there is no end yet, and
    inventing one would close a sermon that is still being preached.
    """
    ordered = sorted((b for b in blocks if b.get("start_ms") is not None),
                     key=lambda b: int(b.get("start_ms") or 0))
    index = None
    for i, block in enumerate(ordered):
        start, end = int(block.get("start_ms") or 0), int(block.get("end_ms") or 0)
        if start <= at_ms <= end:
            index = i
            break
        if start > at_ms:
            # The mark landed in a gap, or ahead of the detector — which is the ordinary
            # case live, because a block does not exist until dwell has passed. The run is
            # the one that starts next.
            index = i
            break
    if index is None:
        return None  # past every block: the detector has not caught up yet
    kind = ordered[index].get("kind")
    last = index
    while last + 1 < len(ordered) and ordered[last + 1].get("kind") == kind:
        last += 1
    if ordered[last].get("ongoing"):
        return None
    return int(ordered[last].get("end_ms") or 0)


def resolve(corrections: Sequence[Mapping[str, Any]],
            blocks: Sequence[Mapping[str, Any]],
            *, now_ms: int, min_seconds: int = 30) -> List[dict]:
    """The spans the operator's marks describe, in the shape the timeline already renders.

    Each start mark opens a phase, which ends at the earliest of:

    * the next mark of any kind — pressing "Songs starts now" says the sermon is over
      without anyone having to say so twice;
    * where the detector's run ends, which is what covers the operator who marked a start
      and then stopped marking;
    * now, while both of those are still open — a phase in progress is shown running to the
      present moment rather than not shown at all.

    ``min_seconds`` drops a *closed* span too short to be a phase: two presses a second
    apart are a correction of the first press, not a five-second sermon. An open span is
    never dropped for its length — it is still running, and it is at its shortest in the
    seconds after the press, which is exactly when the operator is looking for it.
    """
    found = marks(corrections)
    spans: List[dict] = []
    for i, mark in enumerate(found):
        label = (mark.get("label") or "").strip()
        if label == END_LABEL:
            continue  # an end closes the phase before it; it never opens one
        start = int(mark.get("start_ms") or 0)
        limits = [int(found[i + 1].get("start_ms") or 0)] if i + 1 < len(found) else []
        run_end = _run_end(blocks, start)
        if run_end is not None and run_end > start:
            limits.append(run_end)
        end = min(limits) if limits else max(now_ms, start)
        # Only what something actually closed is judged on length: two presses a second
        # apart are a correction of the first, but a phase pressed a moment ago is merely
        # young, and hiding it for half a minute is the timeline telling the operator their
        # press did not land.
        if limits and end - start < int(min_seconds) * 1000:
            continue
        spans.append({
            # The row's own id, so a mark takes its place in the same "newest wins" order
            # every other correction is resolved by: a span drawn afterwards supersedes a
            # mark, and a mark pressed afterwards supersedes the span.
            "id": int(mark.get("id") or 0),
            "start_ms": start,
            "end_ms": end,
            "label": label or "Phase",
            "kind": (mark.get("kind") or "S"),
            "note": mark.get("note") or "",
            "marked": True,
            # True while nothing has closed it yet: the timeline says "running" rather than
            # claiming an end the operator never gave and the detector has not reached.
            "open": not limits,
        })
    return spans


def describe(corrections: Sequence[Mapping[str, Any]], *, now_ms: int) -> str:
    """One line for the operator's row: what is marked right now, or ''.

    Said in the terms they pressed it in, because the row exists to be read at a glance in
    a dim room by someone who is also doing something else.
    """
    found = marks(corrections)
    if not found:
        return ""
    last = found[-1]
    label = (last.get("label") or "").strip()
    since = max(0, now_ms - int(last.get("start_ms") or 0)) // 60000
    if label == END_LABEL:
        return f"ended {since} min ago" if since else "ended just now"
    ago = f"{since} min ago" if since else "just now"
    return f"{label or 'Phase'} marked {ago}"
