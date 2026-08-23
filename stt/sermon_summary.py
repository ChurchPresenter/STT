"""Summary and chapters for one sermon, derived from the transcript under a phase block.

Extracted from the monolith so it can be imported and unit-tested without the import-time
side effects; stdlib-only, and every threshold, prompt and connection is passed in.

The detector in :mod:`stt.service_phase` already finds where the sermon was. This module
answers the next question — what was it about — for an operator who wants the answer while
the service is still running, a few minutes after the preaching stops.

Two constraints shape everything here:

* **The model never gets to invent a timestamp.** It proposes chapter times in mm:ss, and
  :func:`snap_chapters` resolves each one to a real ``transcriptions.ts_ms`` inside the
  block or drops it. A chapter marker is published content: one that points at a moment
  that does not exist is worse than no chapter at all, and a model asked for times will
  always produce plausible ones.
* **A sermon does not fit in one call.** A 30-minute sermon is ~7-10k tokens against a
  default ``n_ctx`` of 2048, so the work is map-reduce: short per-chunk calls whose gists
  are then reduced to one summary. That is not only a context-window workaround — it is
  what keeps live caption translation responsive, because the shared GGUF is released
  between chunks instead of being held for one multi-minute generation.

Identity is the transcript fingerprint, never the block index. Phase blocks renumber,
merge, and back-date their ``end_ms`` as a service runs (see service_phase.track_blocks),
so a summary keyed by index would silently come to describe a different stretch of the
service than the one it was written from.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from stt.llm_translate import estimate_tokens
from stt.service_phase import match_corrections

TokenCounter = Callable[[str], int]

# Phase labels the detector numbers ("Sermon 1", "Sermon 2"), so the trigger matches on
# the prefix rather than equality. _base_name in stt/phase_learn.py takes the same view.
SERMON_LABEL_PREFIX = "Sermon"


class Row(NamedTuple):
    """One finalized transcript row: the id, its epoch-ms stamp, and the text."""

    id: int
    ts_ms: int
    text: str


class Chapter(NamedTuple):
    """A chapter marker. ``ts_ms`` is always a real row's stamp, never a proposed one."""

    ts_ms: int
    title: str


class Chunk:
    """A run of consecutive rows that fits one map call, with the range it covers."""

    __slots__ = ("end_ms", "index", "rows", "start_ms")

    def __init__(self, index: int, rows: Sequence[Row]) -> None:
        self.index = index
        self.rows = list(rows)
        self.start_ms = self.rows[0].ts_ms if self.rows else 0
        self.end_ms = self.rows[-1].ts_ms if self.rows else 0

    @property
    def text(self) -> str:
        return " ".join(r.text for r in self.rows if r.text)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Chunk {self.index} rows={len(self.rows)} {self.start_ms}-{self.end_ms}>"


def _tokens(text: str, counter: Optional[TokenCounter]) -> int:
    """Token count via ``counter``, falling back to the heuristic if it raises.

    Mirrors llm_translate._count: the local GGUF path can hand us the model's own
    tokenizer, and a tokenizer that throws must degrade rather than abort a summary.
    """
    if counter is not None:
        try:
            return int(counter(text))
        except Exception:
            pass
    return estimate_tokens(text)


# --- Reading ----------------------------------------------------------------------


def read_sermon_rows(conn: "sqlite3.Connection", start_ms: int, end_ms: int) -> List[Row]:
    """Finalized, visible transcript rows inside one block, oldest first.

    Unlike phase_learn.read_phase_text this keeps the row id and stamp (chapters have to
    resolve to a real moment) and excludes denied rows. The detector deliberately keeps
    denied rows because auto-denied music is evidence a song is playing; a summary is
    published prose, so hallucination-flagged and music rows have no business in it.
    """
    try:
        cur = conn.execute(
            "SELECT id, ts_ms, text FROM transcriptions "
            "WHERE is_final = 1 AND denied = 0 AND ts_ms IS NOT NULL "
            "AND ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms, id",
            (int(start_ms), int(end_ms)))
    except sqlite3.Error:
        return []
    return [Row(int(r[0]), int(r[1]), (r[2] or "").strip()) for r in cur if (r[2] or "").strip()]


def fingerprint(rows: Sequence[Row]) -> str:
    """Stable identity for the text a summary was written from.

    Row ids are included alongside the text so that re-transcribing a stretch (same words,
    new rows) is treated as new material, and so that a correction to one caption
    invalidates the summary that quoted it.
    """
    h = hashlib.sha256()
    for r in rows:
        h.update(str(r.id).encode("utf-8"))
        h.update(b"\x1f")
        h.update(r.text.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def dominant_source_language(conn: "sqlite3.Connection", start_ms: int, end_ms: int) -> str:
    """The language most of this stretch was recognised as, lowercased, or ''.

    Only ever used to answer "would translating this actually produce another language".
    The column is per-row and populated by detection, so it is a majority question rather
    than a lookup: one misdetected caption in a Russian sermon must not make it English.
    """
    try:
        cur = conn.execute(
            "SELECT LOWER(source_language), COUNT(*) FROM transcriptions "
            "WHERE is_final = 1 AND denied = 0 AND ts_ms IS NOT NULL "
            "AND source_language IS NOT NULL AND TRIM(source_language) != '' "
            "AND ts_ms >= ? AND ts_ms <= ? GROUP BY LOWER(source_language) "
            "ORDER BY COUNT(*) DESC LIMIT 1",
            (int(start_ms), int(end_ms)))
        row = cur.fetchone()
    except sqlite3.Error:
        return ""
    return (row[0] or "").strip() if row else ""


def _base_language(code: str) -> str:
    """The base subtag of a language code: ``en-US`` and ``en_us`` are both ``en``."""
    return (code or "").strip().lower().replace("_", "-").split("-")[0]


def same_language(source: str, target: str) -> bool:
    """Whether translating *source* into *target* could only return what it was given.

    An empty source — nothing detected — is never treated as a match, because the cost of
    being wrong here is a summary that never gets translated at all.
    """
    base = _base_language(source)
    return bool(base) and base == _base_language(target)


def row_signature(conn: "sqlite3.Connection", start_ms: int, end_ms: int) -> Tuple[int, int, int]:
    """A cheap stand-in for :func:`fingerprint`: has this stretch changed since last time?

    The scan runs every detector tick for the rest of a service, and reading a finished
    sermon's rows and hashing them only to learn it is already summarised is a full-sermon
    read plus a sha over tens of kilobytes, repeated every twenty seconds inside the loop
    that also pushes captions to the UI.

    This is one indexed aggregate that transfers three integers: rows added, denied or
    re-transcribed move the count or the id, and edited text moves the character total. It
    is deliberately not an identity — an edit that keeps the length exactly is invisible to
    it — so it may only ever be used to skip work, never to decide what a summary covers.
    That decision stays with the fingerprint, which is computed from the rows themselves.
    """
    try:
        cur = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0), COALESCE(SUM(LENGTH(text)), 0) "
            "FROM transcriptions WHERE is_final = 1 AND denied = 0 AND ts_ms IS NOT NULL "
            "AND ts_ms >= ? AND ts_ms <= ?",
            (int(start_ms), int(end_ms)))
        row = cur.fetchone()
    except sqlite3.Error:
        # Unreadable reads as "changed", so the caller falls through to the real read and
        # decides there. Skipping on an error would be the one wrong answer.
        return (-1, -1, -1)
    if not row:
        return (0, 0, 0)
    return (int(row[0] or 0), int(row[1] or 0), int(row[2] or 0))


def transcript_text(rows: Sequence[Row]) -> str:
    """The sermon as one block of prose, for storage and for the review page."""
    return " ".join(r.text for r in rows if r.text)


# --- Chunking ---------------------------------------------------------------------


def chunk_rows(rows: Sequence[Row], budget_tokens: int, *,
               counter: Optional[TokenCounter] = None) -> List[Chunk]:
    """Pack rows into chunks that each fit ``budget_tokens``.

    A row is never split: it is one caption, and half a caption is neither quotable nor
    attributable to a timestamp. A single row larger than the budget therefore gets a
    chunk of its own and is allowed to exceed it — declining it would silently drop that
    stretch of the sermon, and the model truncating an over-long prompt is the better
    failure of the two.
    """
    if budget_tokens <= 0 or not rows:
        return []
    chunks: List[Chunk] = []
    current: List[Row] = []
    used = 0
    for row in rows:
        cost = _tokens(row.text, counter) + 1  # +1 for the joining space
        if current and used + cost > budget_tokens:
            chunks.append(Chunk(len(chunks), current))
            current, used = [], 0
        current.append(row)
        used += cost
    if current:
        chunks.append(Chunk(len(chunks), current))
    return chunks


# --- Time formatting --------------------------------------------------------------


def format_offset(ts_ms: int, base_ms: int) -> str:
    """``ts_ms`` as mm:ss (or h:mm:ss past an hour) relative to the sermon start."""
    secs = max(0, int(ts_ms) - int(base_ms)) // 1000
    hours, rem = divmod(secs, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


_OFFSET_RE = re.compile(r"^(?:(\d{1,2}):)?(\d{1,3}):(\d{2})$")


def parse_offset(text: str) -> Optional[int]:
    """mm:ss / h:mm:ss to milliseconds, or None if it is not a timestamp."""
    m = _OFFSET_RE.match((text or "").strip())
    if not m:
        return None
    hours = int(m.group(1) or 0)
    minutes, seconds = int(m.group(2)), int(m.group(3))
    if seconds >= 60:
        return None
    return ((hours * 3600) + (minutes * 60) + seconds) * 1000


# --- Prompts ----------------------------------------------------------------------

# Written to be language-neutral and to say "in the same language as the transcript" rather
# than naming one: the summary is for the operator who just heard the sermon, and the
# services this runs on are not in English. Naming a language here would also make the
# prompt wrong the moment the installation changes congregation.

MAP_SYSTEM = (
    "You are summarising one part of a church sermon transcript.\n"
    "Write 2-4 short sentences covering what is actually said in this part: the point being "
    "made, any scripture referenced, and any story or illustration used.\n"
    "Write in the same language as the transcript.\n"
    "Do not add anything that is not in the text. Do not add a preface, a heading, or a "
    "closing remark. Output only the summary sentences."
)


def build_map_prompt(chunk: Chunk, *, base_ms: int) -> Tuple[str, str]:
    """``(system, user)`` for one chunk.

    The time range is stated so the reduce step can place the gists on a timeline without
    the map step being asked to produce timestamps of its own.
    """
    span = f"[{format_offset(chunk.start_ms, base_ms)}-{format_offset(chunk.end_ms, base_ms)}]"
    return MAP_SYSTEM, f"Part {chunk.index + 1} {span}:\n\n{chunk.text}"


def chapter_range(minutes: int, *, min_minutes_per_chapter: int = 4,
                  max_minutes_per_chapter: int = 10, hard_max: int = 12) -> Tuple[int, int]:
    """How many chapters this sermon may have, as ``(floor, ceiling)``.

    A fixed band cannot fit every speaker. One preacher works through eight points in twenty
    minutes; another develops three across forty, and both are ordinary. Asked for the same
    3-8 either way, the model splits a movement to reach a floor it cannot honestly fill, or
    merges two to stay under a ceiling that does not belong to that sermon.

    So duration sets the bounds and the content chooses inside them: at most one chapter per
    ``min_minutes_per_chapter``, at least one per ``max_minutes_per_chapter``. The hard cap
    survives because past a dozen markers nobody scrubs the list — it is a table of contents
    again, which is the thing the cap existed to prevent.
    """
    minutes = max(0, int(minutes))
    ceiling = min(int(hard_max), max(2, minutes // max(1, int(min_minutes_per_chapter))))
    floor = max(2, minutes // max(1, int(max_minutes_per_chapter)))
    # A short sermon can push the floor above the ceiling; the ceiling is the real limit.
    floor = min(floor, ceiling)
    return floor, ceiling


def _reduce_system(floor: int, ceiling: int) -> str:
    """The reduce instructions, parameterised by the range this sermon allows.

    A range rather than a target: asked for a number, a model reaches it, and the last
    chapter or two of a sermon that had fewer movements than that are invented divisions.
    """
    span = f"{floor}" if floor == ceiling else f"between {floor} and {ceiling}"
    return (
        "You are given ordered summaries of consecutive parts of one church sermon, each "
        "with the time range it covers.\n"
        "Write in the same language as the summaries.\n"
        "Reply with exactly these two sections and nothing else:\n"
        "\n"
        "### Summary\n"
        "One paragraph of 4-6 sentences: what this sermon was about, its main point, and "
        "the scripture it worked from.\n"
        "\n"
        "### Chapters\n"
        f"{span} lines, each `m:ss Title`. Use as many as this sermon actually has — do not "
        f"split one movement in two to reach a number.\n"
        "Chapters are the sermon's major movements, not every individual point. Each title "
        "is a short phrase describing what that stretch is about.\n"
        "Use only timestamps that appear in the time ranges above. Never invent a time. "
        "The first chapter starts at the beginning of the sermon.\n"
        "Do not add any other section, preface, or closing remark."
    )


def build_reduce_prompt(gists: Sequence[Tuple[str, str]], *,
                        floor: int = 3, ceiling: int = 8) -> Tuple[str, str]:
    """``(system, user)`` for the reduce call.

    ``gists`` is ``(span_label, gist_text)`` in order — the span label carries the time
    range, which is the only timing information the model is ever shown.
    """
    body = "\n\n".join(f"{span}\n{text}" for span, text in gists if text)
    return _reduce_system(floor, ceiling), body


_TRANSLATE_SYSTEM = (
    "You are translating a summary of a church sermon into {language}.\n"
    "Reply with exactly these two sections and nothing else:\n"
    "\n"
    "### Summary\n"
    "The summary, translated into {language}.\n"
    "\n"
    "### Chapters\n"
    "The chapter titles, translated into {language}, one per line, numbered `1.` to `{n}` "
    "and in the same order. Give exactly {n} lines — one for each title, translated, and "
    "nothing else on the line.\n"
    "\n"
    "Translate only. Do not summarise further, add, explain or reorder."
)


def build_translate_prompt(summary: str, titles: Sequence[str],
                           language: str) -> Tuple[str, str]:
    """``(system, user)`` for translating a finished summary and its chapter titles.

    A separate pass rather than asking the reduce step for both languages at once. The
    timestamps are already settled and snapped to real rows by then, so translating cannot
    disturb them — whereas a model asked for two chapter lists in one reply will sooner or
    later return two that differ, and there is no honest way to decide which times belong to
    which titles.

    The titles are numbered so the reply can be matched back by position: a translated title
    against the wrong timestamp is worse than no translation at all.
    """
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    system = _TRANSLATE_SYSTEM.format(language=language, n=len(titles))
    return system, f"### Summary\n{summary}\n\n### Chapters\n{numbered}"


_NUMBERED = re.compile(r"^\s*(?:[-*•]\s*)?(\d{1,3})[.)]\s*(.+?)\s*$")


def parse_translation(raw: str, expected: int) -> Tuple[str, List[str]]:
    """``(summary, titles)`` from a translation reply, or ``("", [])`` if it does not fit.

    All or nothing on the titles. A short or reordered list would pair translations with the
    wrong timestamps, and a chapter marker that points at the wrong moment is the failure
    this whole feature is built to avoid — so a reply that does not yield exactly ``expected``
    numbered lines is discarded rather than salvaged.
    """
    sections = parse_sections(raw)
    summary = (sections.get("summary") or "").strip()
    found: Dict[int, str] = {}
    for line in (sections.get("chapters") or "").splitlines():
        m = _NUMBERED.match(line)
        if not m:
            continue
        title = m.group(2).strip().strip("*_").strip()
        if title:
            found[int(m.group(1))] = title
    titles = [found[i + 1] for i in range(expected)] if len(found) >= expected and all(
        (i + 1) in found for i in range(expected)) else []
    return summary, titles


# --- Parsing ----------------------------------------------------------------------

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$")


def parse_sections(raw: str) -> Dict[str, str]:
    """Split a markdown reply into ``{lowercased heading: body}``.

    Tolerant of the heading level and of decoration, because the model varies both between
    replies. Text before the first heading is returned under ``""`` so a reply that came
    back with no headings at all is still recoverable by the caller.
    """
    sections: Dict[str, List[str]] = {"": []}
    current = ""
    for line in (raw or "").splitlines():
        m = _HEADING_RE.match(line)
        if m:
            current = m.group(1).strip().strip("*_").lower()
            sections.setdefault(current, [])
            continue
        sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


_CHAPTER_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?\[?\s*"          # optional bullet, optional opening bracket
    r"((?:\d{1,2}:)?\d{1,3}:\d{2})"          # the timestamp
    r"\s*\]?\s*[—–:.\-]?\s*"       # optional closing bracket and separator
    r"(.+?)\s*$")


def parse_chapters(section: str) -> List[Tuple[int, str]]:
    """``(offset_ms, title)`` for each parseable line, in the order given.

    Lines without a leading timestamp are skipped rather than guessed at: a chapter with
    no time is not a chapter, and inferring one would be exactly the fabrication that
    snap_chapters exists to prevent.
    """
    out: List[Tuple[int, str]] = []
    for line in (section or "").splitlines():
        m = _CHAPTER_RE.match(line)
        if not m:
            continue
        offset = parse_offset(m.group(1))
        title = m.group(2).strip().strip("*_").strip()
        if offset is None or not title:
            continue
        out.append((offset, title))
    return out


# --- The anti-fabrication step ----------------------------------------------------


def snap_chapters(proposed: Sequence[Tuple[int, str]], rows: Sequence[Row], *,
                  start_ms: int, max_chapters: int = 8) -> List[Chapter]:
    """Resolve proposed offsets onto real transcript rows, dropping what cannot resolve.

    Every returned ``ts_ms`` is some row's own stamp. An offset that lands outside the
    sermon is dropped outright rather than clamped: past the end it is the model
    inventing material, and clamping would turn a fabrication into a plausible-looking
    marker on the last sentence.

    The first chapter is moved to the first row, because the model is told the sermon
    starts at 0:00 but the block's first row is a second or two in, and a chapter list
    that starts after the beginning leaves the opening unlabelled. That is still a real
    row, so the no-invented-timestamp rule holds.
    """
    if not rows:
        return []
    stamps = [r.ts_ms for r in rows]
    lo, hi = stamps[0], stamps[-1]

    snapped: List[Chapter] = []
    for offset, title in proposed:
        target = int(start_ms) + int(offset)
        if target < lo - 60_000 or target > hi:
            continue  # outside the sermon; a minute of slack for the block's own back-dating
        nearest = min(stamps, key=lambda s: abs(s - target))
        snapped.append(Chapter(nearest, title))

    snapped.sort(key=lambda c: c.ts_ms)

    # One marker per moment, and strictly increasing: two chapters snapping to the same row
    # would render as a duplicate, and the first title is the one the model put earlier.
    deduped: List[Chapter] = []
    for chapter in snapped:
        if deduped and chapter.ts_ms <= deduped[-1].ts_ms:
            continue
        deduped.append(chapter)

    if deduped:
        deduped[0] = Chapter(lo, deduped[0].title)
    return deduped[:max(1, int(max_chapters))]


# --- The trigger predicate --------------------------------------------------------


def ready_sermons(blocks: Sequence[dict], *, now_ms: int, settle_seconds: int = 180,
                  min_minutes: int = 8, include_ongoing: bool = False,
                  label_prefix: str = SERMON_LABEL_PREFIX) -> List[dict]:
    """Sermon blocks that have finished and stayed finished long enough to summarise.

    The settle window is the whole point. track_blocks back-dates a closing block's
    ``end_ms`` to where the following run began, so a block that just closed is still
    moving; summarising immediately would write a summary of a range that then changes
    and be invalidated on the next tick. Waiting past the window costs a few minutes of a
    service that has moved on to closing songs anyway.

    ``include_ongoing`` is for the operator pressing the button. The detector always calls
    the last block ongoing, so a sermon that is still the final block — the usual case at
    the end of a service — is otherwise unreachable by any automatic rule. The operator can
    see the preaching has stopped; the detector cannot yet.
    """
    out = []
    for block in blocks:
        if block.get("ongoing") and not include_ongoing:
            continue
        label = (block.get("label") or "")
        if not label.startswith(label_prefix):
            continue
        if int(block.get("minutes") or 0) < int(min_minutes):
            continue
        end_ms = int(block.get("end_ms") or 0)
        if not end_ms:
            continue
        # An ongoing block's end is simply "now", so there is nothing to settle.
        if not block.get("ongoing") and now_ms - end_ms < int(settle_seconds) * 1000:
            continue
        out.append(block)
    return out


def explain_no_sermons(blocks: Sequence[dict], *, min_minutes: int = 8,
                       label_prefix: str = SERMON_LABEL_PREFIX) -> str:
    """Why nothing was queued, in the operator's terms.

    "No sermon block to summarise" covers four different situations, and the one that
    actually happens most on an archived service — the detector never ran over it, so there
    are no blocks at all — is the one where the message sends the operator looking in
    entirely the wrong place. Each case names its own fix.
    """
    if not blocks:
        return ("this service has no detected phases yet — use 'Re-run & save' first, "
                "then summarise")
    sermons = [b for b in blocks if (b.get("label") or "").startswith(label_prefix)]
    if not sermons:
        labels = sorted({(b.get("label") or "unnamed") for b in blocks})
        return (f"no block in this service is labelled '{label_prefix}' "
                f"(found: {', '.join(labels)}) — relabel one, or re-run the detector")
    longest = max(int(b.get("minutes") or 0) for b in sermons)
    if longest < int(min_minutes):
        return (f"the longest sermon block is {longest} min, under the {min_minutes} min "
                f"minimum — lower sermon_summary.min_minutes to include it")
    return "every sermon in this service already has a summary"


def sermon_ranges(blocks: Sequence[dict], corrections: Sequence[dict] = (), *,
                  min_minutes: int = 8,
                  label_prefix: str = SERMON_LABEL_PREFIX) -> List[dict]:
    """The stretches to summarise, with the operator's corrections applied.

    The detector decides where a sermon is from audio alone, and it is right about the
    structure but only approximate about the edges — dwell means a block starts a minute or
    two after the preaching does, and a boundary is back-dated when the next run begins. For
    a caption timeline that is fine. For a summary it is not: the first minutes of the
    introduction, or a stretch of the song before it, change what the model is reading.

    So an operator who has corrected a phase on /service-phase is correcting what gets
    summarised too. Two kinds of correction matter here and they compose:

    * a correction against a block relabels it — which is how a stretch the detector called
      "Speaking" becomes a sermon, and how one it wrongly called a sermon stops being one;
    * a correction carrying its own ``start_ms``/``end_ms`` (the grouping control) states the
      range outright, and replaces every detector block it overlaps.

    Returned in the shape of blocks, so callers cannot tell the difference.
    """
    # Matched by service_phase.match_corrections rather than by index here: a correction
    # carrying its own span follows the phase it named across a re-run that renumbered the
    # blocks, which is the whole reason the span is recorded.
    by_index = match_corrections(blocks, corrections)
    # Newest wins where drawn spans overlap. A grouping correction is always an insert, so
    # an operator adjusting the same boundary twice leaves both on record — and taking both
    # would summarise one sermon twice, from two ranges they have already superseded. The
    # older rows stay in the table, because corrections are the operator's record and the
    # learner reads them; they simply stop defining a range.
    spans: List[dict] = []
    for c in sorted((c for c in corrections
                     if c.get("block_index") is None and c.get("start_ms") and c.get("end_ms")),
                    key=lambda c: c.get("id") or 0, reverse=True):
        if any(int(c["start_ms"]) < int(k["end_ms"]) and int(c["end_ms"]) > int(k["start_ms"])
               for k in spans):
            continue
        spans.append(c)

    out: List[dict] = []
    for i, block in enumerate(blocks):
        fix = by_index.get(i)
        label = (fix.get("label") if fix and fix.get("label") else block.get("label")) or ""
        start, end = int(block.get("start_ms") or 0), int(block.get("end_ms") or 0)
        # A span the operator drew wins over whatever the detector put under it.
        if any(int(c["start_ms"]) < end and int(c["end_ms"]) > start for c in spans):
            continue
        out.append({**block, "label": label, "source": "correction" if fix else "detector"})

    for c in spans:
        label = c.get("label") or ""
        start, end = int(c["start_ms"]), int(c["end_ms"])
        out.append({
            "index": None, "kind": c.get("kind") or "S", "label": label,
            "start_ms": start, "end_ms": end,
            "minutes": max(1, round((end - start) / 60000.0)),
            "ongoing": False, "confidence": 1.0, "source": "correction",
        })

    out.sort(key=lambda b: int(b.get("start_ms") or 0))
    return [b for b in out
            if (b.get("label") or "").startswith(label_prefix)
            and int(b.get("minutes") or 0) >= int(min_minutes)]


def unfinished(blocks: Sequence[dict], stored: Sequence[dict], *,
               min_minutes: int = 8,
               label_prefix: str = SERMON_LABEL_PREFIX) -> List[dict]:
    """Sermon blocks with no usable summary yet, for the end-of-session catch-up.

    A row left ``pending`` or ``running`` counts as unfinished: the process that was going to
    do the work is the one that just stopped, so nothing is coming to finish it. So does
    ``error`` — the usual cause is a model that was unreachable at the time, and the end of a
    service is exactly when that has changed.

    Matched on the block's own range rather than on a fingerprint, because the caller is
    asking "does this sermon have a summary", not "is this specific text summarised".
    """
    done = {(int(r.get("start_ms") or 0), (r.get("label") or ""))
            for r in stored if r.get("status") == STATUS_DONE}
    out = []
    for block in blocks:
        label = (block.get("label") or "")
        if not label.startswith(label_prefix):
            continue
        if int(block.get("minutes") or 0) < int(min_minutes):
            continue
        if (int(block.get("start_ms") or 0), label) in done:
            continue
        out.append(block)
    return out


def progress_text(done: int, total: int, *, waiting: bool = False,
                  reducing: bool = False, translating: bool = False) -> str:
    """What a run in flight should say about itself.

    A sermon takes minutes, and "summarising" alone for all of them is indistinguishable
    from stuck. The waiting case matters most: during a service the machine holding the model
    defers to its captions, so a part can sit for minutes having asked and been told to come
    back — which is correct behaviour and looks identical to a hang unless it says so.
    """
    if translating:
        return "translating the summary"
    if reducing:
        return "writing the summary"
    if total <= 0:
        return "starting"
    place = f"part {min(int(done) + 1, int(total))} of {int(total)}"
    return f"{place} — waiting for the paired machine" if waiting else place


# --- Persistence ------------------------------------------------------------------
#
# One table, in the session's own database beside the transcript it was written from. A
# sermon transcript is verbatim congregation speech, so it lives in exactly one place and
# is deleted when the session is: an aggregate store would outlive the recording it came
# from. Cross-service review reads the archive instead, one database at a time.

_DDL = (
    """CREATE TABLE IF NOT EXISTS sermon_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, start_ms INTEGER, end_ms INTEGER,
        fingerprint TEXT UNIQUE, transcript TEXT, summary TEXT, chapters_json TEXT,
        model TEXT, status TEXT, error TEXT, generated_at TEXT)""",
)

# Columns added after the table first shipped. CREATE TABLE IF NOT EXISTS is a no-op on an
# existing table, so a session summarised under an earlier build keeps the older shape and
# the write would fail on the missing column.
_ADDED_COLUMNS = (("sermon_summaries", "progress", "TEXT"),
                  ("sermon_summaries", "summary_translated", "TEXT"))

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"


def ensure_tables(conn: "sqlite3.Connection") -> None:
    """Create the table if absent and add any later columns. Safe to call on every run."""
    for stmt in _DDL:
        conn.execute(stmt)
    for table, column, decl in _ADDED_COLUMNS:
        try:
            have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        except sqlite3.Error:
            pass  # a progress note is never worth failing a summary over
    conn.commit()


def set_progress(conn: "sqlite3.Connection", fingerprint: str, text: str) -> int:
    """Note how far a run has got, touching nothing else.

    An UPDATE rather than a save_summary upsert, for the same reason mark_error is: the
    caller knows the fingerprint and little else, and writing a row from those defaults would
    replace a real sermon's label and range with placeholders.
    """
    try:
        cur = conn.execute("UPDATE sermon_summaries SET progress = ? WHERE fingerprint = ?",
                           (text, fingerprint))
    except sqlite3.Error:
        return 0
    conn.commit()
    return int(cur.rowcount or 0)


def save_summary(conn: "sqlite3.Connection", *, fingerprint: str, label: str,
                 start_ms: int, end_ms: int, status: str,
                 transcript: str = "", summary: str = "",
                 chapters: Sequence[Chapter] = (), model: str = "",
                 error: str = "", generated_at: str = "",
                 summary_translated: str = "",
                 titles_translated: Sequence[str] = ()) -> int:
    """Upsert one sermon's summary, keyed by fingerprint. Returns the row id.

    Keyed on the fingerprint rather than the block index so that a re-derived block whose
    boundaries moved writes a new row, while an unchanged one is updated in place through
    its pending -> running -> done progression.
    """
    ensure_tables(conn)
    # The translation is stored on the chapter, not in a list beside it: a parallel array
    # can fall out of step with the timestamps, and this pairing cannot.
    chapters_json = json.dumps(
        [{"ts_ms": c.ts_ms, "title": c.title,
          "title_translated": titles_translated[i] if i < len(titles_translated) else ""}
         for i, c in enumerate(chapters)], ensure_ascii=False)
    conn.execute(
        "INSERT INTO sermon_summaries (fingerprint, label, start_ms, end_ms, transcript, "
        "summary, chapters_json, model, status, error, generated_at, summary_translated) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(fingerprint) DO UPDATE SET label=excluded.label, "
        "start_ms=excluded.start_ms, end_ms=excluded.end_ms, "
        "transcript=excluded.transcript, summary=excluded.summary, "
        "chapters_json=excluded.chapters_json, model=excluded.model, "
        "status=excluded.status, error=excluded.error, generated_at=excluded.generated_at, "
        "summary_translated=excluded.summary_translated, progress=''",
        (fingerprint, label, int(start_ms), int(end_ms), transcript, summary,
         chapters_json, model, status, error, generated_at, summary_translated))
    conn.commit()
    row = conn.execute("SELECT id FROM sermon_summaries WHERE fingerprint = ?",
                       (fingerprint,)).fetchone()
    return int(row[0]) if row else 0


def _to_dict(row: Dict[str, Any]) -> dict:
    """One stored row as the API shape, reading by name with defaults.

    Deliberately not positional. The archive is opened read-only, so a reader meets whatever
    columns that database happens to have and cannot migrate it — and a summary written before
    a column existed is the normal case, not an edge one. Indexing by offset made every added
    column a way to fail the whole read: naming summary_translated in the SELECT made every
    session summarised before it report as having no summaries at all.
    """
    try:
        chapters = json.loads(str(row.get("chapters_json") or "[]"))
    except (ValueError, TypeError):
        chapters = []
    # Chapters gained their translation the same way, so they get the same treatment.
    chapters = [{"ts_ms": int(c.get("ts_ms") or 0), "title": c.get("title") or "",
                 "title_translated": c.get("title_translated") or ""}
                for c in chapters if isinstance(c, dict)]
    return {
        "id": int(row.get("id") or 0),
        "fingerprint": row.get("fingerprint") or "",
        "label": row.get("label") or "",
        "start_ms": int(row.get("start_ms") or 0),
        "end_ms": int(row.get("end_ms") or 0),
        "transcript": row.get("transcript") or "",
        "summary": row.get("summary") or "",
        "chapters": chapters,
        "model": row.get("model") or "",
        "status": row.get("status") or "",
        "error": row.get("error") or "",
        "generated_at": row.get("generated_at") or "",
        "progress": row.get("progress") or "",
        "summary_translated": row.get("summary_translated") or "",
    }


def _rows(cur: "sqlite3.Cursor") -> List[Dict[str, Any]]:
    """Fetched rows as dicts keyed by the column names this database actually has."""
    names = [d[0] for d in (cur.description or ())]
    return [dict(zip(names, r)) for r in cur.fetchall()]


# SELECT * rather than a column list, so a column added later cannot break reading a session
# written earlier, nor a session written later break an older reader. Ordering is stated
# explicitly because SELECT * says nothing about it.
_SELECT = "SELECT * FROM sermon_summaries"


def load_summaries(conn: "sqlite3.Connection") -> List[dict]:
    """Every stored summary for this session, in service order."""
    try:
        return [_to_dict(r) for r in _rows(conn.execute(_SELECT + " ORDER BY start_ms, id"))]
    except sqlite3.Error as e:
        # Never break the page over one session — but say why, because swallowing this
        # silently is what turned a missing column into "no sermon has been summarised".
        print(f"[SERMON] could not read summaries ({type(e).__name__}: {e})")
        return []


def load_summary(conn: "sqlite3.Connection", fingerprint: str) -> Optional[dict]:
    """One stored summary by fingerprint, or None."""
    try:
        rows = _rows(conn.execute(_SELECT + " WHERE fingerprint = ?", (fingerprint,)))
    except sqlite3.Error as e:
        print(f"[SERMON] could not read summary ({type(e).__name__}: {e})")
        return None
    return _to_dict(rows[0]) if rows else None


def delete_summary(conn: "sqlite3.Connection", fingerprint: str) -> int:
    """Drop one summary so it can be regenerated. Returns rows removed."""
    try:
        cur = conn.execute("DELETE FROM sermon_summaries WHERE fingerprint = ?", (fingerprint,))
    except sqlite3.Error:
        return 0
    conn.commit()
    return int(cur.rowcount or 0)


def mark_error(conn: "sqlite3.Connection", fingerprint: str, error: str) -> int:
    """Record that a summary failed, touching only the status and the message.

    Deliberately an UPDATE and not a save_summary upsert: the failure handler often knows
    nothing but the fingerprint, and writing a row from those defaults would replace a real
    sermon's label and time range with placeholders — losing the very thing that says which
    sermon failed. Returns rows updated; 0 means there was nothing to mark.
    """
    try:
        cur = conn.execute(
            "UPDATE sermon_summaries SET status = ?, error = ? WHERE fingerprint = ?",
            (STATUS_ERROR, error, fingerprint))
    except sqlite3.Error:
        return 0
    conn.commit()
    return int(cur.rowcount or 0)


def supersede(conn: "sqlite3.Connection", *, label: str, start_ms: int, end_ms: int,
              keep: str) -> int:
    """Drop earlier summaries of the same sermon, keeping fingerprint ``keep``.

    Matched by *overlap*, not by an identical start. A summary the operator has superseded by
    correcting the boundary describes a range that no longer exists, and leaving it beside
    the new one gives the same sermon two summaries that disagree — the reader has no way to
    tell which is the current one. Overlap is the test because that is what "the same
    sermon" means once an operator can move the edges.

    Same reason a partial summary made on request while the preaching was still going is
    replaced by the complete one when the block closes.
    """
    try:
        cur = conn.execute(
            "DELETE FROM sermon_summaries WHERE label = ? AND fingerprint != ? "
            "AND start_ms < ? AND end_ms > ?",
            (label, keep, int(end_ms), int(start_ms)))
    except sqlite3.Error:
        return 0
    conn.commit()
    return int(cur.rowcount or 0)


def has_summaries(conn: "sqlite3.Connection") -> bool:
    """Whether this session holds any summary — the archive listing's filter."""
    try:
        row = conn.execute("SELECT 1 FROM sermon_summaries LIMIT 1").fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def render_markdown(entry: dict) -> str:
    """One summary as markdown, for the review page's copy-out.

    Chapter times are rendered relative to the sermon's own start, which is what a reader
    scrubbing a recording of the sermon needs; the absolute stamps stay in the database.
    """
    base = int(entry.get("start_ms") or 0)
    lines = [f"# {entry.get('label') or 'Sermon'}", ""]
    if entry.get("summary"):
        lines += [entry["summary"], ""]
    if entry.get("summary_translated"):
        lines += [entry["summary_translated"], ""]
    chapters = entry.get("chapters") or []
    if chapters:
        lines.append("## Chapters")
        for c in chapters:
            at = format_offset(int(c.get("ts_ms") or 0), base)
            lines.append(f"- {at} {c.get('title') or ''}")
            # Indented under its own timestamp rather than as a second list: one chapter,
            # named twice, is not two chapters.
            if c.get("title_translated"):
                lines.append(f"  - {at} {c['title_translated']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
