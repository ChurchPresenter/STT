"""Where a service currently is: songs, speaking, or quiet — and a guess at which part.

Extracted from speech_to_text.py so it can be imported and unit-tested without the
monolith's import-time side effects. Stdlib-only; every config section and phrase list is
passed in explicitly, so nothing here knows about the live config or the database.

The structure comes from audio, not from words. Every finalized row already carries a PANNs
`speech_type` and `music_prob`, and binned per minute those reproduce a service's shape
directly: long Music runs are songs, long Speaking runs are sermons. Measured over ten real
services (94-168 minutes each), that skeleton is reliable enough to track live.

Words only annotate and tentatively *name* the blocks they fall in. They are deliberately
not allowed to move a boundary, because the cues are much weaker than they look:

* "Аминь." is a standalone caption 99% of the time, but only 33% of its 114 occurrences in
  the archive fall within a minute of an audio boundary — against a 20% random baseline.
  It ends prayers, poems and songs *inside* a block far more often than it ends the block.
* Communion separates by keyword *volume*, not position: the one communion service scored
  34 hits against <=3 for every other Sunday, while a sermon merely discussing communion
  scored 8 spread across 84 minutes.

So a name is always returned beside the structural kind and never in place of it. When an
operator reviews a session, a misnamed block and a mis-split block have to be
distinguishable, or the corrections cannot say which half of this is wrong.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

# Structural classes. Deliberately three, matching the PANNs vocabulary the pipeline
# already produces (stt/segments.py:panns_label_from_prob) rather than inventing a
# parallel one.
MUSIC = "M"
SPEECH = "S"
QUIET = "_"

# A block whose kind is one of these is never named; it is the gap between the parts.
_UNNAMED_KINDS = (QUIET,)


class Bin:
    """One time bucket of finalized rows, with everything a detector needs.

    A plain object rather than a tuple because it is persisted per-minute as the
    finetuning record: a future detector has to be replayable over past services without
    re-deriving anything from audio, so the fields are named and stable.
    """

    __slots__ = ("cues", "end_ms", "index", "music", "quiet", "speech", "start_ms", "words")

    def __init__(self, index: int, start_ms: int, end_ms: int) -> None:
        self.index = index
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.music = 0
        self.speech = 0
        self.quiet = 0
        self.words = 0
        self.cues: Dict[str, int] = {}

    @property
    def rows(self) -> int:
        return self.music + self.speech + self.quiet

    def to_dict(self) -> dict:
        return {
            "index": self.index, "start_ms": self.start_ms, "end_ms": self.end_ms,
            "music": self.music, "speech": self.speech, "quiet": self.quiet,
            "words": self.words, "cues": dict(self.cues),
        }


class Block:
    """A contiguous run of one structural kind, with a tentative name."""

    __slots__ = (
        "confidence",
        "cues",
        "end_bin",
        "end_ms",
        "index",
        "kind",
        "label",
        "ongoing",
        "start_bin",
        "start_ms",
    )

    def __init__(self, index: int, kind: str, start_bin: int, end_bin: int,
                 start_ms: int, end_ms: int) -> None:
        self.index = index
        self.kind = kind
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.label: Optional[str] = None
        self.confidence: float = 0.0
        self.cues: Dict[str, int] = {}
        self.ongoing = False

    @property
    def minutes(self) -> int:
        return self.end_bin - self.start_bin + 1

    def to_dict(self) -> dict:
        return {
            "index": self.index, "kind": self.kind,
            "start_bin": self.start_bin, "end_bin": self.end_bin,
            "start_ms": self.start_ms, "end_ms": self.end_ms,
            "minutes": self.minutes, "label": self.label,
            "confidence": round(self.confidence, 2), "cues": dict(self.cues),
            "ongoing": self.ongoing,
        }


def compile_cues(phrases: Dict[str, Sequence[str]]) -> Dict[str, "re.Pattern[str]"]:
    """Compile a {cue name: [phrase, ...]} config map into whole-word patterns.

    Phrases are matched case-insensitively at word boundaries so "хлеб" cannot fire on
    "хлебом" mid-word by accident — and are taken from config rather than hardcoded,
    because the cues are language- and congregation-specific.

    Keys beginning with "_" are skipped: config.default.json documents every block with
    sibling ``_comment`` keys, and iterating one of those as a phrase list would walk the
    characters of the comment string and compile each letter as a cue.
    """
    out: Dict[str, "re.Pattern[str]"] = {}
    for name, items in (phrases or {}).items():
        if name.startswith("_") or isinstance(items, str):
            continue
        parts = [p.strip() for p in (items or []) if p and p.strip()]
        if not parts:
            continue
        try:
            out[name] = re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)
        except re.error:
            continue  # a bad phrase list must not take the detector down
    return out


def bin_rows(rows: Sequence[Tuple], bin_seconds: int = 60, *,
             music_prob_threshold: float = 0.5,
             cues: Optional[Dict[str, "re.Pattern[str]"]] = None) -> List[Bin]:
    """Bucket finalized rows into fixed-width bins.

    ``rows`` are ``(ts_ms, speech_type, music_prob, text)`` tuples ordered by ts_ms — plain
    tuples so this module never touches SQL. Bins are contiguous from the first row, so a
    silent stretch is an empty bin rather than a missing one; the tracker needs the gap to
    exist in order to see it.
    """
    usable = [r for r in rows if r and r[0] is not None]
    if not usable:
        return []
    width = max(1, int(bin_seconds)) * 1000
    t0 = int(usable[0][0])
    span = int(usable[-1][0]) - t0
    count = span // width + 1
    bins = [Bin(i, t0 + i * width, t0 + (i + 1) * width) for i in range(count)]

    for ts_ms, speech_type, music_prob, text in usable:
        b = bins[min(count - 1, (int(ts_ms) - t0) // width)]
        if speech_type == "Music" or (music_prob or 0.0) > music_prob_threshold:
            b.music += 1
        elif speech_type == "Speaking":
            b.speech += 1
        else:
            b.quiet += 1
        body = text or ""
        b.words += len(body.split())
        for name, pattern in (cues or {}).items():
            hits = len(pattern.findall(body))
            if hits:
                b.cues[name] = b.cues.get(name, 0) + hits
    return bins


def classify_bin(b: Bin, *, dominance: float = 0.35) -> str:
    """The structural class of one bin.

    ``dominance`` is a share of the bin's rows rather than a majority: a minute of preaching
    still contains pauses tagged Quiet, and a minute of singing contains announcements, so
    requiring an outright majority would classify most of a service as nothing at all.
    """
    total = b.rows
    if total == 0:
        return QUIET
    floor = max(1, dominance * total)
    if b.music >= floor:
        return MUSIC
    if b.speech >= floor:
        return SPEECH
    return QUIET


def track_blocks(classes: Sequence[str], bins: Sequence[Bin], *,
                 enter_minutes: int = 2, exit_minutes: int = 3) -> List[Block]:
    """Group per-bin classes into blocks, causally.

    The state only changes after ``enter_minutes`` consecutive bins of a new class
    (``exit_minutes`` for quiet, which is looser because a lull inside a sermon is common
    and must not end it). Nothing here reads ahead of the bin being processed, so the answer
    produced live at minute N is the same answer a replay produces at minute N — which is
    what makes a recorded session a fair test of the live detector.

    Dwell is the whole quality knob. Measured against the offline segmenter over ten real
    services, 2/3 recovers 89% of major blocks at a 1-minute median lag with about one
    spurious flip per service; 3/4 gives 81% at 2 minutes with a quarter of the flips.
    """
    if not classes:
        return []
    enter = max(1, int(enter_minutes))
    leave = max(1, int(exit_minutes))

    blocks: List[Block] = []
    state = classes[0]
    state_start = 0
    run_class: Optional[str] = None
    run_len = 0

    for i, c in enumerate(classes):
        if c == run_class:
            run_len += 1
        else:
            run_class, run_len = c, 1
        needed = leave if c == QUIET else enter
        if c != state and run_len >= needed:
            # The new block began where its run began, not where we became sure of it.
            switch_at = i - run_len + 1
            if switch_at > state_start:
                blocks.append(Block(len(blocks), state, state_start, switch_at - 1,
                                    bins[state_start].start_ms, bins[switch_at - 1].end_ms))
                state_start = switch_at
            state = c

    last = Block(len(blocks), state, state_start, len(classes) - 1,
                 bins[state_start].start_ms, bins[len(classes) - 1].end_ms)
    last.ongoing = True
    blocks.append(last)
    return blocks


def _sum_cues(bins: Sequence[Bin], block: Block) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for b in bins[block.start_bin:block.end_bin + 1]:
        for name, n in b.cues.items():
            out[name] = out.get(name, 0) + n
    return out


def label_blocks(blocks: List[Block], bins: Sequence[Bin], cfg: Optional[dict] = None, *,
                 first_sunday: bool = False) -> List[Block]:
    """Attach a tentative name, a confidence, and the cues behind it to each block.

    Names are guesses and are reported as such. The rules are only those the archive
    actually supports — position, duration, and cue volume — and no expected running order
    is assumed anywhere: the same ten services contain both a Sunday morning that alternates
    two or three sermons with songs and a Wednesday that is songs, a short talk, then one
    45-minute sermon. A template would be wrong about half of them.
    """
    cfg = cfg or {}
    sermon_min = int(cfg.get("sermon_min_minutes", 8))
    songs_min = int(cfg.get("songs_min_minutes", 3))
    communion_min = int(cfg.get("communion_min_hits", 12))
    closing_max = int(cfg.get("closing_max_minutes", 6))

    named = [b for b in blocks if b.kind not in _UNNAMED_KINDS]
    sermon_n = 0
    songs_n = 0
    seen_sermon = False

    for b in blocks:
        b.cues = _sum_cues(bins, b)
        if b.kind in _UNNAMED_KINDS:
            b.label, b.confidence = None, 0.0
            continue

        if b.kind == MUSIC:
            if b.minutes >= songs_min:
                songs_n += 1
                b.label, b.confidence = f"Songs {songs_n}", 0.7
            else:
                b.label, b.confidence = "Music", 0.4
            continue

        # Speaking. Communion outranks sermon: it replaces a sermon slot rather than
        # sitting beside one, so a block that clears the cue threshold is not also counted
        # as the next sermon.
        communion_hits = b.cues.get("communion", 0)
        if communion_hits >= communion_min:
            b.label = "Communion"
            b.confidence = 0.8 if first_sunday else 0.6
            continue

        is_last_named = named and b is named[-1]
        if not seen_sermon and b.minutes < sermon_min:
            b.label, b.confidence = "Opening", 0.5
        elif b.minutes >= sermon_min:
            sermon_n += 1
            seen_sermon = True
            b.label, b.confidence = f"Sermon {sermon_n}", 0.7
        elif is_last_named and b.minutes <= closing_max:
            b.label, b.confidence = "Closing", 0.5
        else:
            b.label, b.confidence = "Speaking", 0.3

        # An ongoing block may still grow into a sermon; say so rather than commit.
        if b.ongoing and b.label in ("Speaking", "Opening"):
            b.confidence = min(b.confidence, 0.3)
    return blocks


def analyze(rows: Sequence[Tuple], cfg: Optional[dict] = None, *,
            first_sunday: bool = False) -> dict:
    """Full pass: rows in, current phase and block timeline out.

    Re-run from scratch on every tick rather than updated incrementally. A three-hour
    service is ~180 bins, so the cost is nothing, and it means a mid-service restart loses
    no state and cannot drift from what a replay would produce.
    """
    cfg = cfg or {}
    cues = compile_cues(cfg.get("cue_phrases", {}))
    bins = bin_rows(
        rows,
        bin_seconds=int(cfg.get("bin_seconds", 60)),
        music_prob_threshold=float(cfg.get("music_prob_threshold", 0.5)),
        cues=cues,
    )
    if not bins:
        return {"current": None, "blocks": [], "bins": [], "classes": ""}

    classes = [classify_bin(b, dominance=float(cfg.get("dominance", 0.35))) for b in bins]
    blocks = track_blocks(
        classes, bins,
        enter_minutes=int(cfg.get("enter_minutes", 2)),
        exit_minutes=int(cfg.get("exit_minutes", 3)),
    )
    label_blocks(blocks, bins, cfg, first_sunday=first_sunday)
    current = blocks[-1] if blocks else None
    return {
        "current": current.to_dict() if current else None,
        "blocks": [b.to_dict() for b in blocks],
        "bins": [b.to_dict() for b in bins],
        "classes": "".join(classes),
    }


# --- Persistence ------------------------------------------------------------------
#
# Three tables, and the split between them is the point of the feature. The detector
# rewrites its own output wholesale on every tick, so operator corrections have to live
# somewhere it never touches — otherwise the next tick silently destroys the only real
# labels anyone has. Bins are stored too, because a future detector must be replayable
# over past services without re-deriving anything from audio.

_DDL = (
    """CREATE TABLE IF NOT EXISTS service_phase_bins (
        bin_index INTEGER PRIMARY KEY, start_ms INTEGER, end_ms INTEGER,
        music INTEGER, speech INTEGER, quiet INTEGER, words INTEGER, cues_json TEXT)""",
    """CREATE TABLE IF NOT EXISTS service_phase_blocks (
        block_index INTEGER PRIMARY KEY, kind TEXT, start_bin INTEGER, end_bin INTEGER,
        start_ms INTEGER, end_ms INTEGER, minutes INTEGER, label TEXT,
        confidence REAL, cues_json TEXT, ongoing INTEGER)""",
    """CREATE TABLE IF NOT EXISTS service_phase_corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, block_index INTEGER, start_ms INTEGER,
        end_ms INTEGER, kind TEXT, label TEXT, note TEXT, corrected_at TEXT)""",
)


def ensure_tables(conn: "sqlite3.Connection") -> None:
    """Create the three tables if absent. Safe to call on every tick."""
    for stmt in _DDL:
        conn.execute(stmt)
    conn.commit()


def read_rows(conn: "sqlite3.Connection") -> List[Tuple]:
    """The finalized rows the detector runs on, oldest first.

    Denied rows are kept deliberately: music segments are auto-denied from the transcript
    (``denied_reason`` like ``music:0.5``) and excluding them would erase exactly the
    evidence that a song is playing.
    """
    return list(conn.execute(
        "SELECT ts_ms, speech_type, music_prob, text FROM transcriptions "
        "WHERE is_final = 1 AND ts_ms IS NOT NULL ORDER BY ts_ms"))


def save_analysis(conn: "sqlite3.Connection", analysis: dict) -> None:
    """Replace the stored detector output with this tick's.

    DELETE-then-INSERT rather than an upsert: blocks merge and renumber as a service runs,
    so a stale row from a previous tick would otherwise survive as a phantom block. Only
    the detector's own tables are touched — corrections are never in scope here.
    """
    ensure_tables(conn)
    conn.execute("DELETE FROM service_phase_bins")
    conn.execute("DELETE FROM service_phase_blocks")
    conn.executemany(
        "INSERT INTO service_phase_bins (bin_index, start_ms, end_ms, music, speech, quiet, "
        "words, cues_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(b["index"], b["start_ms"], b["end_ms"], b["music"], b["speech"], b["quiet"],
          b["words"], json.dumps(b["cues"], ensure_ascii=False)) for b in analysis.get("bins", [])])
    conn.executemany(
        "INSERT INTO service_phase_blocks (block_index, kind, start_bin, end_bin, start_ms, "
        "end_ms, minutes, label, confidence, cues_json, ongoing) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(b["index"], b["kind"], b["start_bin"], b["end_bin"], b["start_ms"], b["end_ms"],
          b["minutes"], b["label"], b["confidence"],
          json.dumps(b["cues"], ensure_ascii=False), 1 if b["ongoing"] else 0)
         for b in analysis.get("blocks", [])])
    conn.commit()


def save_correction(conn: "sqlite3.Connection", block_index: Optional[int], *,
                    kind: Optional[str], label: Optional[str],
                    start_ms: Optional[int] = None, end_ms: Optional[int] = None,
                    note: str = "", corrected_at: str = "") -> int:
    """Record an operator's correction, replacing any earlier one for the same block.

    Returns the row id. One correction per block_index so a reviewer can change their mind
    without accumulating contradictory truth; a correction with no block_index (a boundary
    the detector missed entirely) is always an insert.
    """
    ensure_tables(conn)
    if block_index is not None:
        conn.execute("DELETE FROM service_phase_corrections WHERE block_index = ?", (block_index,))
    cur = conn.execute(
        "INSERT INTO service_phase_corrections (block_index, start_ms, end_ms, kind, label, "
        "note, corrected_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (block_index, start_ms, end_ms, kind, label, note, corrected_at))
    conn.commit()
    return int(cur.lastrowid or 0)


def load_corrections(conn: "sqlite3.Connection") -> List[dict]:
    """Operator corrections, oldest first. Missing table = none recorded yet."""
    try:
        rows = conn.execute(
            "SELECT id, block_index, start_ms, end_ms, kind, label, note, corrected_at "
            "FROM service_phase_corrections ORDER BY id").fetchall()
    except sqlite3.Error:
        return []
    keys = ("id", "block_index", "start_ms", "end_ms", "kind", "label", "note", "corrected_at")
    return [dict(zip(keys, r)) for r in rows]


def load_analysis(conn: "sqlite3.Connection") -> dict:
    """The stored detector output, for reviewing a session that is no longer running."""
    try:
        bins = conn.execute(
            "SELECT bin_index, start_ms, end_ms, music, speech, quiet, words, cues_json "
            "FROM service_phase_bins ORDER BY bin_index").fetchall()
        blocks = conn.execute(
            "SELECT block_index, kind, start_bin, end_bin, start_ms, end_ms, minutes, label, "
            "confidence, cues_json, ongoing FROM service_phase_blocks ORDER BY block_index").fetchall()
    except sqlite3.Error:
        return {"current": None, "blocks": [], "bins": [], "classes": ""}

    def _cues(raw: Optional[str]) -> dict:
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            return {}

    out_bins = [{"index": r[0], "start_ms": r[1], "end_ms": r[2], "music": r[3], "speech": r[4],
                 "quiet": r[5], "words": r[6], "cues": _cues(r[7])} for r in bins]
    out_blocks = [{"index": r[0], "kind": r[1], "start_bin": r[2], "end_bin": r[3],
                   "start_ms": r[4], "end_ms": r[5], "minutes": r[6], "label": r[7],
                   "confidence": r[8], "cues": _cues(r[9]), "ongoing": bool(r[10])}
                  for r in blocks]
    classes = "".join(
        next((b["kind"] for b in out_blocks if b["start_bin"] <= i <= b["end_bin"]), QUIET)
        for i in range(len(out_bins)))
    return {"current": out_blocks[-1] if out_blocks else None,
            "blocks": out_blocks, "bins": out_bins, "classes": classes}
