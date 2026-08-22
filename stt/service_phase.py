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

from stt.phase_rules import Span, apply_rules

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
        "unusual",
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
        self.unusual: List[str] = []

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
            "ongoing": self.ongoing, "unusual": list(self.unusual),
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


# PANNs' own top label, consulted when the derived speech_type says Quiet.
#
# speech_type is not the classifier's answer — it is a derivation of it:
# segments.panns_label_from_prob calls a row Music only above music_prob_threshold, and
# otherwise Quiet or Speaking purely on whether audio_db clears quiet_db_threshold. On a
# quiet feed that gate silences almost everything. Measured on one live service: 27 rows
# tagged "Speech" and 14 tagged "Music" were all recorded as Quiet, because the input sat at
# about -40 dB — the threshold itself — while the smoothed music probability averaged 0.22
# through the singing.
#
# The tag is the signal that survived. Matched loosely because PANNs emits a vocabulary, not
# a class: "Music", "Singing", "Choir", "Musical instrument" are all the congregation
# singing, and "Speech", "Narration, monologue" are all the preacher.
_MUSIC_TAGS = ("music", "sing", "choir", "instrument", "guitar", "piano", "organ", "drum")
_SPEECH_TAGS = ("speech", "narration", "monologue", "conversation", "speaking")


def tag_class(audio_tag: Optional[str]) -> Optional[str]:
    """MUSIC, SPEECH, or None for a PANNs tag that says neither."""
    tag = (audio_tag or "").strip().lower()
    if not tag:
        return None
    if any(k in tag for k in _MUSIC_TAGS):
        return MUSIC
    if any(k in tag for k in _SPEECH_TAGS):
        return SPEECH
    return None


def bin_rows(rows: Sequence[Tuple], bin_seconds: int = 60, *,
             music_prob_threshold: float = 0.5,
             use_audio_tag: bool = True,
             cues: Optional[Dict[str, "re.Pattern[str]"]] = None,
             cues_translated: Optional[Dict[str, "re.Pattern[str]"]] = None) -> List[Bin]:
    """Bucket finalized rows into fixed-width bins.

    ``rows`` are ``(ts_ms, speech_type, music_prob, text[, translated_text])`` tuples
    ordered by ts_ms — plain tuples so this module never touches SQL. Bins are contiguous
    from the first row, so a silent stretch is an empty bin rather than a missing one; the
    tracker needs the gap to exist in order to see it.

    Cues are counted against the transcript and the translation together, because either
    can miss what the other catches. The transcript can spell a word the phrase list does
    not anticipate, while the translation normalises those variants; the translation in
    turn paraphrases and sometimes drops a term entirely. Measured over the archive, taking
    both raised communion hits 17% and opening-phrase hits 28% over the transcript alone —
    and that is with only 53% of rows carrying a translation at all.

    The per-row count is the *larger* of the two, never the sum: a caption whose transcript
    and translation both say the word is still one utterance, and summing would inflate
    translated rows relative to untranslated ones, which is exactly the comparison the
    communion threshold rests on.
    """
    usable = [r for r in rows if r and r[0] is not None]
    if not usable:
        return []
    width = max(1, int(bin_seconds)) * 1000
    t0 = int(usable[0][0])
    span = int(usable[-1][0]) - t0
    count = span // width + 1
    bins = [Bin(i, t0 + i * width, t0 + (i + 1) * width) for i in range(count)]

    for row in usable:
        ts_ms, speech_type, music_prob, text = row[0], row[1], row[2], row[3]
        translated = row[4] if len(row) > 4 else None
        audio_tag = row[5] if len(row) > 5 else None
        b = bins[min(count - 1, (int(ts_ms) - t0) // width)]
        if speech_type == "Music" or (music_prob or 0.0) > music_prob_threshold:
            b.music += 1
        elif speech_type == "Speaking":
            b.speech += 1
        else:
            # Quiet is the label this pipeline reaches for when it is unsure, so it is the
            # one worth a second opinion. Deliberately only here: where speech_type is
            # confident it stays authoritative, and the dwell settings were tuned against it.
            fallback = tag_class(audio_tag) if use_audio_tag else None
            if fallback == MUSIC:
                b.music += 1
            elif fallback == SPEECH:
                b.speech += 1
            else:
                b.quiet += 1
        body = text or ""
        b.words += len(body.split())
        for name in set(cues or {}) | set(cues_translated or {}):
            src = (cues or {}).get(name)
            tgt = (cues_translated or {}).get(name)
            hits = max(
                len(src.findall(body)) if src is not None else 0,
                len(tgt.findall(translated or "")) if tgt is not None else 0,
            )
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


def flag_unusual(blocks: Sequence[Block], cfg: Optional[dict] = None) -> List[str]:
    """Mark blocks whose duration is outside anything the archive contains, and describe
    the service as a whole. Returns the service-level notes.

    Duration is genuinely informative — a song set does not run for an hour — but it is
    used to *flag*, never to relabel. On a music-heavy service or a Christmas play the long
    block is real, and a rule that "corrected" it would mislabel precisely the service most
    worth labelling. So the audio still decides what a block *is*; an implausible duration
    only lowers confidence and asks the operator to look.

    Every message quotes the threshold it actually applied, rather than a figure measured
    somewhere else. The shipped defaults were taken from one congregation's archive, and a
    message that says "nothing exceeds 27" while comparing against a tuned 45 is worse than
    no message: it reads as evidence when it is a leftover. What is typical here is whatever
    this installation has configured, and the learner moves it.
    """
    cfg = cfg or {}
    music_max = int(cfg.get("typical_music_max_minutes", 30))
    speech_max = int(cfg.get("typical_speaking_max_minutes", 60))
    share_lo = float(cfg.get("typical_music_share_min", 0.20))
    share_hi = float(cfg.get("typical_music_share_max", 0.70))

    for b in blocks:
        b.unusual = []
        # An ongoing block is judged too. Exceeding a maximum is monotone — once a block
        # has run longer than anything in the archive, finishing later cannot make that
        # untrue — and a sermon that has already overrun is exactly what an operator
        # wants flagged while it is still happening rather than afterwards.
        if b.kind == MUSIC and b.minutes > music_max:
            b.unusual.append(f"music runs {b.minutes} min; longer than the {music_max} min "
                             f"this installation treats as usual")
        elif b.kind == SPEECH and b.minutes > speech_max:
            b.unusual.append(f"speaking runs {b.minutes} min; longer than the {speech_max} min "
                             f"this installation treats as usual")
        if b.unusual:
            # Structure is still trusted; only the name becomes a weaker claim.
            b.confidence = min(b.confidence, 0.3)

    notes: List[str] = []
    music = sum(b.minutes for b in blocks if b.kind == MUSIC)
    speech = sum(b.minutes for b in blocks if b.kind == SPEECH)
    if music + speech >= 30:  # too short to characterise before that
        share = music / float(music + speech)
        band = f"configured band is {share_lo:.0%}-{share_hi:.0%}"
        if share > share_hi:
            notes.append(f"More musical than usual ({share:.0%} of the service; {band}) "
                         f"— a music service or a play would look like this.")
        elif share < share_lo:
            notes.append(f"Less musical than usual ({share:.0%} of the service; {band}).")
    return notes


def sum_cues(bins: Sequence[Bin], block: Block) -> Dict[str, int]:
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
        b.cues = sum_cues(bins, b)
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

    return number_songs_from_opening(blocks, songs_min=songs_min)


def number_songs_from_opening(blocks: List[Block], *, songs_min: int) -> List[Block]:
    """Restart the Songs count at the opening, leaving earlier music unnumbered.

    Music before the opening is the band rehearsing to an empty room, not the first song of
    the service: numbering it put Songs 1 half an hour before anyone arrived and left the
    real first song called Songs 3. Rehearsal keeps the plain "Music" name the detector
    already gives short music, so no new category enters the vocabulary.

    A no-op when nothing was labelled Opening — a session with no detected opening keeps
    counting from the first music block, which is the best available guess.
    """
    opening_at = next((b.index for b in blocks if b.label == "Opening"), None)
    if opening_at is None:
        return blocks

    songs_n = 0
    for b in blocks:
        if b.kind != MUSIC or b.label is None:
            continue
        if b.index < opening_at:
            b.label, b.confidence = "Music", 0.4
        elif b.minutes >= songs_min:
            songs_n += 1
            b.label, b.confidence = f"Songs {songs_n}", 0.7
    return blocks


def compile_fragment_cues(groups: Optional[Dict[str, Dict[str, Sequence[str]]]]
                          ) -> Dict[str, "re.Pattern[str]"]:
    """Compile grouped fragments into ordinary cues named ``group:fragment``.

    A liturgical formula is several distinct lines, and how many *different* ones appear is
    the signal — a sermon may say "the cup" twenty times without ever reading the words of
    institution. Flattening each fragment into its own cue name means the existing per-bin
    counting, the summing onto blocks and the stored cues_json all carry it with no schema
    change; a rule then asks for a count of distinct ``group:`` keys.
    """
    out: Dict[str, "re.Pattern[str]"] = {}
    for group, fragments in (groups or {}).items():
        if group.startswith("_") or not isinstance(fragments, dict):
            continue
        for fragment, phrases in fragments.items():
            if fragment.startswith("_"):
                continue
            compiled = compile_cues({fragment: phrases})
            if fragment in compiled:
                out["%s:%s" % (group, fragment)] = compiled[fragment]
    return out


def analyze(rows: Sequence[Tuple], cfg: Optional[dict] = None, *,
            first_sunday: bool = False, rules: Optional[Sequence] = None) -> dict:
    """Full pass: rows in, current phase and block timeline out.

    Re-run from scratch on every tick rather than updated incrementally. A three-hour
    service is ~180 bins, so the cost is nothing, and it means a mid-service restart loses
    no state and cannot drift from what a replay would produce.

    ``rules`` are parsed phase rules (stt/phase_rules). When given they name the blocks and
    may return spans — a phase covering several blocks, which is what communion looks like.
    Without them the built-in labeller runs, so a caller that has no rule file still works.
    """
    cfg = cfg or {}
    cues = dict(compile_cues(cfg.get("cue_phrases", {})))
    cues.update(compile_fragment_cues(cfg.get("cue_fragments")))
    cues_translated = dict(compile_cues(cfg.get("cue_phrases_translated", {})))
    cues_translated.update(compile_fragment_cues(cfg.get("cue_fragments_translated")))
    bins = bin_rows(
        rows,
        bin_seconds=int(cfg.get("bin_seconds", 60)),
        music_prob_threshold=float(cfg.get("music_prob_threshold", 0.5)),
        use_audio_tag=bool(cfg.get("use_audio_tag", True)),
        cues=cues,
        cues_translated=cues_translated,
    )
    if not bins:
        return {"current": None, "blocks": [], "bins": [], "classes": "", "notes": [],
                "spans": []}

    classes = [classify_bin(b, dominance=float(cfg.get("dominance", 0.35))) for b in bins]
    blocks = track_blocks(
        classes, bins,
        enter_minutes=int(cfg.get("enter_minutes", 2)),
        exit_minutes=int(cfg.get("exit_minutes", 3)),
    )
    spans: List[Span] = []
    if rules:
        for b in blocks:
            b.cues = sum_cues(bins, b)
        spans = apply_rules(blocks, rules, first_sunday=first_sunday)
    else:
        label_blocks(blocks, bins, cfg, first_sunday=first_sunday)
    notes = flag_unusual(blocks, cfg)
    current = blocks[-1] if blocks else None
    return {
        "current": current.to_dict() if current else None,
        "blocks": [b.to_dict() for b in blocks],
        "bins": [b.to_dict() for b in bins],
        "classes": "".join(classes),
        "notes": notes,
        "spans": [_span_dict(s, blocks) for s in spans],
    }


def _span_dict(span: Span, blocks: Sequence[Block]) -> dict:
    """A span in the same shape the page already renders an operator's group in."""
    first, last = blocks[span.start_index], blocks[span.end_index]
    return {
        "label": span.label, "confidence": round(span.confidence, 2),
        "start_index": span.start_index, "end_index": span.end_index,
        "start_ms": first.start_ms, "end_ms": last.end_ms,
        "minutes": last.end_bin - first.start_bin + 1,
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
        confidence REAL, cues_json TEXT, ongoing INTEGER, unusual_json TEXT)""",
    """CREATE TABLE IF NOT EXISTS service_phase_spans (
        span_index INTEGER PRIMARY KEY, label TEXT, confidence REAL,
        start_index INTEGER, end_index INTEGER, start_ms INTEGER, end_ms INTEGER,
        minutes INTEGER)""",
    """CREATE TABLE IF NOT EXISTS service_phase_corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, block_index INTEGER, start_ms INTEGER,
        end_ms INTEGER, kind TEXT, label TEXT, note TEXT, corrected_at TEXT)""",
)


# Columns added after the tables first shipped. CREATE TABLE IF NOT EXISTS is a no-op on
# an existing table, so a session started under an earlier build keeps the older shape and
# the INSERT would fail on the missing column.
_ADDED_COLUMNS = (("service_phase_blocks", "unusual_json", "TEXT"),)


def ensure_tables(conn: "sqlite3.Connection") -> None:
    """Create the three tables if absent and add any later columns. Safe on every tick."""
    for stmt in _DDL:
        conn.execute(stmt)
    for table, column, decl in _ADDED_COLUMNS:
        try:
            have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        except sqlite3.Error:
            pass  # a derived table is rebuilt each tick; never block a session over it
    conn.commit()


def read_rows(conn: "sqlite3.Connection") -> List[Tuple]:
    """The finalized rows the detector runs on, oldest first.

    Both the transcript and its translation are read: either can miss a cue the other
    catches (see bin_rows). Denied rows are kept deliberately: music segments are auto-denied from the transcript
    (``denied_reason`` like ``music:0.5``) and excluding them would erase exactly the
    evidence that a song is playing.
    """
    try:
        return list(conn.execute(
            "SELECT ts_ms, speech_type, music_prob, text, translated_text, audio_tag "
            "FROM transcriptions "
            "WHERE is_final = 1 AND ts_ms IS NOT NULL ORDER BY ts_ms"))
    except sqlite3.Error:
        # A session recorded before translation existed has no such column; the cue
        # union simply degrades to the transcript alone rather than the read failing.
        return list(conn.execute(
            "SELECT ts_ms, speech_type, music_prob, text FROM transcriptions "
            "WHERE is_final = 1 AND ts_ms IS NOT NULL ORDER BY ts_ms"))


def save_analysis(conn: "sqlite3.Connection", analysis: dict) -> Dict[str, int]:
    """Persist this tick's detector output, writing only what actually changed.

    Returns ``{"bins": n, "blocks": n}`` — the rows written, so a caller (or a test) can
    see that an idle tick is genuinely free.

    The detector re-derives the whole session every tick, but almost none of it moves: a
    minute-wide bin is only mutable while it is the current minute, so across a real
    167-minute service exactly one bin changes per minute (max two), and the block list
    changes far more rarely still. Rewriting both tables wholesale on a 20-second tick cost
    ~46,000 row-writes per service to record ~500 real changes, and grew the WAL by
    hundreds of KB per minute — on the same disk the session recording is being written to.

    So bins are diffed against what is stored and upserted individually. The comparison
    read is deliberate: reads do not grow the WAL, and one SELECT of a few hundred narrow
    rows is far cheaper than the writes it avoids. It also keeps this stateless — a
    mid-service restart re-derives and re-diffs correctly with nothing held in memory.

    Blocks keep DELETE-then-INSERT, because they merge and renumber as a service runs and a
    stale row would survive as a phantom block — but only when they have actually changed,
    and there are only a couple of dozen of them.

    Only the detector's own tables are touched; corrections are never in scope here.
    """
    ensure_tables(conn)
    written = {"bins": 0, "blocks": 0, "spans": 0}

    bins = analysis.get("bins", [])
    # cues_json is part of the comparison, not just the payload. Translation lands via a
    # later async UPDATE on a row the bin already counted, so a cue can appear in a settled
    # bin with every other field unchanged — and an operator correcting a word does the
    # same. Diffing on counts alone silently loses both.
    stored_bins = {
        r[0]: (r[1], r[2], r[3], r[4], r[5], r[6], r[7])
        for r in conn.execute("SELECT bin_index, start_ms, end_ms, music, speech, quiet, "
                              "words, cues_json FROM service_phase_bins")
    }
    changed_bins = []
    for b in bins:
        row = (b["start_ms"], b["end_ms"], b["music"], b["speech"], b["quiet"], b["words"],
               json.dumps(b["cues"], ensure_ascii=False))
        if stored_bins.get(b["index"]) != row:
            changed_bins.append((b["index"], *row))
    if changed_bins:
        conn.executemany(
            "INSERT OR REPLACE INTO service_phase_bins (bin_index, start_ms, end_ms, music, "
            "speech, quiet, words, cues_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", changed_bins)
        written["bins"] = len(changed_bins)
    # Bins only ever grow during a session, but a reused database would leave a tail.
    if stored_bins and max(stored_bins) >= len(bins):
        conn.execute("DELETE FROM service_phase_bins WHERE bin_index >= ?", (len(bins),))
        written["bins"] += 1

    block_rows = [
        (b["index"], b["kind"], b["start_bin"], b["end_bin"], b["start_ms"], b["end_ms"],
         b["minutes"], b["label"], b["confidence"],
         json.dumps(b["cues"], ensure_ascii=False), 1 if b["ongoing"] else 0,
         json.dumps(b.get("unusual", []), ensure_ascii=False))
        for b in analysis.get("blocks", [])
    ]
    # Blocks get the same treatment, and need it: the last block is ongoing, so its end and
    # minutes grow on every single tick. Rewriting the table for that one row meant the
    # block table alone outwrote the bins three to one.
    stored_blocks = {
        r[0]: tuple(r) for r in conn.execute(
            "SELECT block_index, kind, start_bin, end_bin, start_ms, end_ms, minutes, label, "
            "confidence, cues_json, ongoing, unusual_json FROM service_phase_blocks")
    }
    changed_blocks = [r for r in block_rows if stored_blocks.get(r[0]) != r]
    if changed_blocks:
        conn.executemany(
            "INSERT OR REPLACE INTO service_phase_blocks (block_index, kind, start_bin, "
            "end_bin, start_ms, end_ms, minutes, label, confidence, cues_json, ongoing, "
            "unusual_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", changed_blocks)
        written["blocks"] = len(changed_blocks)
    # Blocks merge as a service runs, so the list can get *shorter*. A leftover row would
    # survive as a phantom block on the timeline.
    if stored_blocks and max(stored_blocks) >= len(block_rows):
        conn.execute("DELETE FROM service_phase_blocks WHERE block_index >= ?", (len(block_rows),))
        written["blocks"] += 1

    span_rows = [
        (i, s["label"], s["confidence"], s["start_index"], s["end_index"],
         s["start_ms"], s["end_ms"], s["minutes"])
        for i, s in enumerate(analysis.get("spans", []))
    ]
    # A span is a claim about a range of blocks, so it moves whenever the blocks under it
    # do. There are a handful per service, and they renumber together, so the diffing the
    # bins need would cost more than it saves.
    stored_spans = [tuple(r) for r in conn.execute(
        "SELECT span_index, label, confidence, start_index, end_index, start_ms, end_ms, "
        "minutes FROM service_phase_spans ORDER BY span_index")]
    if stored_spans != span_rows:
        conn.execute("DELETE FROM service_phase_spans")
        if span_rows:
            conn.executemany(
                "INSERT INTO service_phase_spans (span_index, label, confidence, start_index, "
                "end_index, start_ms, end_ms, minutes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                span_rows)
        written["spans"] = len(span_rows)

    if written["bins"] or written["blocks"] or written.get("spans"):
        conn.commit()
    return written


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


def save_group_correction(conn: "sqlite3.Connection", start_ms: int, end_ms: int, *,
                          kind: Optional[str], label: Optional[str],
                          note: str = "", corrected_at: str = "") -> int:
    """Record one phase spanning several detected blocks, replacing any group on that span.

    The detector breaks a block wherever the audio kind changes, so a worship set with a
    word between the songs arrives as three blocks and one sermon interrupted by a cough
    arrives as two. This is how an operator says they were one thing. The row carries no
    block_index — it deliberately does not belong to any single block — and the span is
    what identifies it, so re-grouping the same range corrects it instead of stacking a
    second claim on top.
    """
    ensure_tables(conn)
    conn.execute("DELETE FROM service_phase_corrections WHERE block_index IS NULL "
                 "AND start_ms = ? AND end_ms = ?", (start_ms, end_ms))
    cur = conn.execute(
        "INSERT INTO service_phase_corrections (block_index, start_ms, end_ms, kind, label, "
        "note, corrected_at) VALUES (NULL, ?, ?, ?, ?, ?, ?)",
        (start_ms, end_ms, kind, label, note, corrected_at))
    conn.commit()
    return int(cur.lastrowid or 0)


def delete_correction(conn: "sqlite3.Connection", block_index: int) -> int:
    """Drop the correction for one block, returning how many rows went.

    The counterpart to save_correction: a reviewer who mislabels a block should be able to
    withdraw the claim entirely, not just overwrite it with another guess. Corrections with
    no block_index are groups spanning several blocks; delete_correction_by_id reaches those.
    """
    ensure_tables(conn)
    cur = conn.execute("DELETE FROM service_phase_corrections WHERE block_index = ?", (block_index,))
    conn.commit()
    return int(cur.rowcount or 0)


def delete_correction_by_id(conn: "sqlite3.Connection", row_id: int) -> int:
    """Drop one correction by row id, returning how many rows went.

    A group has no block_index to name it by, so undoing one needs the id the page already
    holds from load_corrections. Ungrouping is exactly this: remove the span and the blocks
    underneath it reappear as the detector found them.
    """
    ensure_tables(conn)
    cur = conn.execute("DELETE FROM service_phase_corrections WHERE id = ?", (row_id,))
    conn.commit()
    return int(cur.rowcount or 0)


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
            "confidence, cues_json, ongoing, unusual_json "
            "FROM service_phase_blocks ORDER BY block_index").fetchall()
    except sqlite3.Error:
        return {"current": None, "blocks": [], "bins": [], "classes": "", "notes": [],
                "spans": []}
    try:
        spans = conn.execute(
            "SELECT label, confidence, start_index, end_index, start_ms, end_ms, minutes "
            "FROM service_phase_spans ORDER BY span_index").fetchall()
    except sqlite3.Error:
        spans = []   # a session recorded before spans existed simply has none

    def _cues(raw: Optional[str]) -> dict:
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            return {}

    def _list(raw: Optional[str]) -> list:
        try:
            return json.loads(raw) if raw else []
        except ValueError:
            return []

    out_bins = [{"index": r[0], "start_ms": r[1], "end_ms": r[2], "music": r[3], "speech": r[4],
                 "quiet": r[5], "words": r[6], "cues": _cues(r[7])} for r in bins]
    out_blocks = [{"index": r[0], "kind": r[1], "start_bin": r[2], "end_bin": r[3],
                   "start_ms": r[4], "end_ms": r[5], "minutes": r[6], "label": r[7],
                   "confidence": r[8], "cues": _cues(r[9]), "ongoing": bool(r[10]),
                   "unusual": _list(r[11] if len(r) > 11 else None)}
                  for r in blocks]
    classes = "".join(
        next((b["kind"] for b in out_blocks if b["start_bin"] <= i <= b["end_bin"]), QUIET)
        for i in range(len(out_bins)))
    out_spans = [{"label": r[0], "confidence": r[1], "start_index": r[2], "end_index": r[3],
                  "start_ms": r[4], "end_ms": r[5], "minutes": r[6]} for r in spans]
    return {"current": out_blocks[-1] if out_blocks else None,
            "blocks": out_blocks, "bins": out_bins, "classes": classes, "spans": out_spans}
