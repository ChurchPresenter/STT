"""Replay a past service through the phase detector and score the labels it produces.

Every claim about phase accuracy in this project so far has been a sentence. The config
comments quote figures — "89% of major blocks at a 1-minute median lag" — that nothing in
the tree can reproduce, and the detector was tuned against an offline segmenter that was
never committed. So a change to a threshold is argued the only way it can be: by looking at
one service and deciding it reads better. That is how a change that fixes one service and
breaks two gets shipped.

A session database is a far better witness. It holds the per-minute bins the detector saw,
the blocks it emitted, and — the part that matters — the labels a human corrected it to. Re-
running a candidate setting over one turns "this ought to help" into a count.

Three things shape the design:

* **The comparison is the output.** A single run's agreement figure means little on its own;
  what matters is which minutes changed against the run before, and in which direction. So
  :func:`compare` reports fixed and broken separately, and :func:`shipped_run` makes the
  baseline free — the blocks a service already stored are a run, no detector required.
* **Truth is what a human said**, and it is read through the same code the server uses:
  :func:`stt.phase_learn.read_corrected_phases` for corrections and groups, and
  :func:`stt.phase_marks.resolve` for live marks. A harness carrying its own copy of those
  rules measures its copy.
* **Labels are allowed to settle, so churn has to be measurable.** A rule that ranks blocks
  against the whole service can change an earlier block's label when a later one arrives.
  :func:`progressive` counts exactly that, over one-minute prefixes, so the cost of such a
  rule is a number rather than an argument.

Stdlib-only and model-free: binning and labelling come from :mod:`stt.service_phase`, which
is already pure. No audio, no GPU, no network.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from stt.phase_learn import read_corrected_phases
from stt.phase_marks import resolve as resolve_marks
from stt.service_phase import analyze_bins, bins_from_stored, bins_from_transcript, read_rows

# Everything is measured in whole minutes of service time, because that is the unit the
# detector decides in: a bin is a minute, and a boundary can only ever land on a bin edge.
DEFAULT_BIN_SECONDS = 60

SOURCE_BINS = "bins"
SOURCE_ROWS = "rows"


@dataclass(frozen=True)
class TruthSpan:
    """A stretch of a service a human put a name to."""

    start_ms: int
    end_ms: int
    label: str
    source: str          # "correction" | "mark"

    @property
    def base(self) -> str:
        return base_label(self.label)


@dataclass(frozen=True)
class Recording:
    """One archived service, and everything needed to re-label it."""

    path: str
    session: str
    stored_bins: List[dict] = field(default_factory=list)
    rows: List[Tuple] = field(default_factory=list)
    stored_blocks: List[dict] = field(default_factory=list)
    truth: List[TruthSpan] = field(default_factory=list)
    profile: Optional[str] = None


@dataclass(frozen=True)
class Run:
    """What a detector said about a service."""

    label: str
    blocks: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class LabelScore:
    label: str
    truth_minutes: int
    run_minutes: int
    overlap_minutes: int

    @property
    def precision(self) -> float:
        return _ratio(self.overlap_minutes, self.run_minutes)

    @property
    def recall(self) -> float:
        return _ratio(self.overlap_minutes, self.truth_minutes)


@dataclass(frozen=True)
class Score:
    """How well one run matched what a human said, in minutes."""

    label: str
    judged_minutes: int
    agreed_minutes: int
    per_label: Dict[str, LabelScore] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    truth_counts: Dict[str, int] = field(default_factory=dict)
    spurious: List[dict] = field(default_factory=list)
    missed: List[dict] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        return _ratio(self.agreed_minutes, self.judged_minutes)


@dataclass(frozen=True)
class Comparison:
    before: Score
    after: Score
    fixed: List[dict] = field(default_factory=list)
    broken: List[dict] = field(default_factory=list)

    @property
    def agreement_delta(self) -> float:
        return round(self.after.agreement - self.before.agreement, 4)


@dataclass(frozen=True)
class LabelChange:
    """A settled block whose label differed between one prefix of a service and the next."""

    at_minute: int
    start_ms: int
    was: Optional[str]
    now: Optional[str]


def _ratio(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def base_label(label: Optional[str]) -> str:
    """A phase name without its ordinal: "Sermon 2" -> "Sermon".

    Scoring compares kinds of phase, not the detector's numbering. Whether the second
    sermon is called 2 or 3 is a renumbering artefact; whether that stretch is preaching at
    all is the question being asked.
    """
    text = str(label or "").strip()
    parts = text.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].strip()
    return text


def _read_only_uri(db_path: str) -> str:
    """A sqlite URI that cannot write, journal, or create the file.

    ``immutable=1`` rather than ``mode=ro``, for the reason stt/translation_replay gives:
    a session database may be opened while its WAL still holds committed frames, and
    mode=ro would create a -shm sidecar next to a file this tool has no business touching.
    """
    return "file:%s?immutable=1" % urllib.parse.quote(db_path)


def _json_or(raw: Any, fallback: Any) -> Any:
    try:
        return json.loads(raw) if raw else fallback
    except (ValueError, TypeError):
        return fallback


def read_truth(conn: "sqlite3.Connection", session: str, blocks: Sequence[dict], *,
               now_ms: int = 0) -> List[TruthSpan]:
    """Every stretch of this service a human named, corrections and marks alike.

    Marks are resolved through :func:`stt.phase_marks.resolve` — the one implementation of
    what a mark means — and only closed spans count: an open one is a phase that was still
    running, which says nothing about where it ended. Where a mark and a correction overlap,
    the correction wins: it was made afterwards, with the timeline in view.
    """
    spans: List[TruthSpan] = []
    for phase in read_corrected_phases(conn, session):
        if phase.end_ms > phase.start_ms:
            spans.append(TruthSpan(int(phase.start_ms), int(phase.end_ms),
                                   str(phase.label), "correction"))
    corrections = _read_corrections(conn)
    if corrections:
        anchor = now_ms or max((int(b.get("end_ms") or 0) for b in blocks), default=0)
        for span in resolve_marks(corrections, blocks, now_ms=anchor):
            if span.get("open"):
                continue
            start, end = int(span.get("start_ms") or 0), int(span.get("end_ms") or 0)
            if end <= start:
                continue
            if any(s.start_ms < end and start < s.end_ms for s in spans):
                continue
            spans.append(TruthSpan(start, end, str(span.get("label") or ""), "mark"))
    spans.sort(key=lambda s: s.start_ms)
    return spans


def _read_corrections(conn: "sqlite3.Connection") -> List[dict]:
    try:
        rows = conn.execute(
            "SELECT id, block_index, start_ms, end_ms, kind, label, note, corrected_at "
            "FROM service_phase_corrections ORDER BY id").fetchall()
    except sqlite3.Error:
        return []
    keys = ("id", "block_index", "start_ms", "end_ms", "kind", "label", "note",
            "corrected_at")
    return [dict(zip(keys, r)) for r in rows]


def load_recording(db_path: str, *, session: str = "") -> Recording:
    """Read one archived service. The database is opened read-only and left untouched."""
    conn = sqlite3.connect(_read_only_uri(db_path), uri=True)
    try:
        name = session or db_path.rsplit("/", 1)[-1]
        stored_bins = _load_bins(conn)
        stored_blocks = _load_blocks(conn)
        try:
            rows = read_rows(conn)
        except sqlite3.Error:
            rows = []
        truth = read_truth(conn, name, stored_blocks)
        return Recording(path=db_path, session=name, stored_bins=stored_bins,
                         rows=list(rows), stored_blocks=stored_blocks, truth=truth,
                         profile=_stored_profile(conn))
    finally:
        conn.close()


def _load_bins(conn: "sqlite3.Connection") -> List[dict]:
    try:
        rows = conn.execute(
            "SELECT bin_index, start_ms, end_ms, music, speech, quiet, words, cues_json "
            "FROM service_phase_bins ORDER BY bin_index").fetchall()
    except sqlite3.Error:
        return []
    keys = ("index", "start_ms", "end_ms", "music", "speech", "quiet", "words", "cues_json")
    return [dict(zip(keys, r)) for r in rows]


def _load_blocks(conn: "sqlite3.Connection") -> List[dict]:
    try:
        rows = conn.execute(
            "SELECT block_index, kind, start_bin, end_bin, start_ms, end_ms, minutes, label, "
            "confidence, cues_json, ongoing, unusual_json "
            "FROM service_phase_blocks ORDER BY block_index").fetchall()
    except sqlite3.Error:
        return []
    out = []
    for r in rows:
        out.append({
            "index": r[0], "kind": r[1], "start_bin": r[2], "end_bin": r[3],
            "start_ms": r[4], "end_ms": r[5], "minutes": r[6], "label": r[7],
            "confidence": r[8], "cues": _json_or(r[9], {}), "ongoing": bool(r[10]),
            "unusual": _json_or(r[11], []),
        })
    return out


def _stored_profile(conn: "sqlite3.Connection") -> Optional[str]:
    """Which service profile produced this recording, if it recorded one."""
    try:
        row = conn.execute(
            "SELECT value FROM session_meta WHERE key = ?",
            ("service_phase.profile",)).fetchone()
    except sqlite3.Error:
        return None
    return str(row[0]) if row and row[0] else None


def shipped_run(recording: Recording) -> Run:
    """What the service actually produced — the free baseline, no detector required."""
    return Run(label="shipped", blocks=list(recording.stored_blocks))


def replay(recording: Recording, cfg: Mapping[str, Any],
           rules: Optional[Sequence[Any]] = None, *,
           source: str = SOURCE_BINS, label: str = "replay",
           first_sunday: bool = False) -> Run:
    """Re-label a service under candidate settings.

    ``source`` decides where the bins come from, and the choice is not cosmetic. Stored bins
    carry the cue counts computed with the phrase lists in force when the service was
    recorded, so a candidate that changes ``cue_phrases`` or ``cue_fragments`` must replay
    from ``rows`` or it will measure the old phrases and report no difference.
    """
    if source == SOURCE_ROWS:
        bins = bins_from_transcript(recording.rows, dict(cfg))
    else:
        bins = bins_from_stored(recording.stored_bins)
    # A recording is a finished service, so nothing in it is still growing.
    result = analyze_bins(bins, dict(cfg), first_sunday=first_sunday, rules=rules, live=False)
    return Run(label=label, blocks=list(result.get("blocks") or []),
               notes=list(result.get("notes") or []))


def _minutes(start_ms: int, end_ms: int, bin_seconds: int) -> int:
    step = max(1, bin_seconds) * 1000
    return max(0, (int(end_ms) - int(start_ms)) // step)


def _overlap_minutes(a: Tuple[int, int], b: Tuple[int, int], bin_seconds: int) -> int:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return _minutes(start, end, bin_seconds) if end > start else 0


def score(run: Run, truth: Sequence[TruthSpan], *,
          bin_seconds: int = DEFAULT_BIN_SECONDS, label: str = "") -> Score:
    """How much of what a human named this run got right, counted in minutes.

    Only the stretches a human named are judged. A service with one corrected block is
    scored on that block alone — which is honest about how much evidence there is, and is
    why :attr:`Score.judged_minutes` travels beside the agreement figure. Scoring the
    unnamed remainder against the detector's own output would be scoring it against itself.
    """
    judged = 0
    agreed = 0
    per: Dict[str, LabelScore] = {}
    truth_minutes: Dict[str, int] = {}
    run_minutes: Dict[str, int] = {}
    overlap: Dict[str, int] = {}

    for span in truth:
        minutes = _minutes(span.start_ms, span.end_ms, bin_seconds)
        judged += minutes
        truth_minutes[span.base] = truth_minutes.get(span.base, 0) + minutes
        for block in run.blocks:
            got = base_label(block.get("label"))
            shared = _overlap_minutes((span.start_ms, span.end_ms),
                                      (int(block.get("start_ms") or 0),
                                       int(block.get("end_ms") or 0)), bin_seconds)
            if not shared:
                continue
            run_minutes[got] = run_minutes.get(got, 0) + shared
            if got == span.base:
                agreed += shared
                overlap[got] = overlap.get(got, 0) + shared

    for name in set(truth_minutes) | set(run_minutes):
        per[name] = LabelScore(name, truth_minutes.get(name, 0),
                               run_minutes.get(name, 0), overlap.get(name, 0))

    spurious = []
    for block in run.blocks:
        name = base_label(block.get("label"))
        if not name:
            continue
        window = (int(block.get("start_ms") or 0), int(block.get("end_ms") or 0))
        if any(s.base == name and _overlap_minutes(window, (s.start_ms, s.end_ms),
                                                   bin_seconds) for s in truth):
            continue
        # Only a stretch a human named can be called wrong; elsewhere there is no evidence.
        if any(_overlap_minutes(window, (s.start_ms, s.end_ms), bin_seconds) for s in truth):
            spurious.append({"label": block.get("label"), "start_ms": window[0],
                             "end_ms": window[1],
                             "minutes": _minutes(window[0], window[1], bin_seconds)})

    missed = []
    for span in truth:
        if not any(base_label(b.get("label")) == span.base
                   and _overlap_minutes((span.start_ms, span.end_ms),
                                        (int(b.get("start_ms") or 0),
                                         int(b.get("end_ms") or 0)), bin_seconds)
                   for b in run.blocks):
            missed.append({"label": span.label, "start_ms": span.start_ms,
                           "end_ms": span.end_ms, "source": span.source})

    counts: Dict[str, int] = {}
    for block in run.blocks:
        name = base_label(block.get("label"))
        if name:
            counts[name] = counts.get(name, 0) + 1
    truth_counts: Dict[str, int] = {}
    for span in truth:
        truth_counts[span.base] = truth_counts.get(span.base, 0) + 1

    return Score(label=label or run.label, judged_minutes=judged, agreed_minutes=agreed,
                 per_label=per, counts=counts, truth_counts=truth_counts,
                 spurious=spurious, missed=missed)


def compare(before: Score, after: Score) -> Comparison:
    """What changed between two runs, in both directions.

    Named the way the translation harness names it, and for the same reason: a candidate
    that fixes four things and breaks six looks like an improvement until the breakages are
    counted separately.
    """
    fixed = [item for item in before.spurious
             if not _same_span(item, after.spurious)]
    broken = [item for item in after.spurious
              if not _same_span(item, before.spurious)]
    for item in before.missed:
        if not _same_span(item, after.missed):
            fixed.append(dict(item, kind="missed"))
    for item in after.missed:
        if not _same_span(item, before.missed):
            broken.append(dict(item, kind="missed"))
    return Comparison(before=before, after=after, fixed=fixed, broken=broken)


def _same_span(item: Mapping[str, Any], others: Sequence[Mapping[str, Any]]) -> bool:
    return any(o.get("start_ms") == item.get("start_ms")
               and o.get("label") == item.get("label") for o in others)


def progressive(recording: Recording, cfg: Mapping[str, Any],
                rules: Optional[Sequence[Any]] = None, *,
                step_minutes: int = 1, first_sunday: bool = False) -> List[LabelChange]:
    """Every time a settled block's label changed as the service went on.

    The instrument for "labels may settle". Re-runs the detector over successive prefixes
    and records each closed block whose name differs from what the previous prefix called
    it. An ongoing block is excluded: it is still growing, and its name is expected to move.

    A count of zero means the detector never revised itself. A rule that ranks blocks
    against the whole service will not score zero, and the point of this function is that
    the cost is then a number to weigh rather than an argument to have.
    """
    changes: List[LabelChange] = []
    bins = bins_from_stored(recording.stored_bins)
    if not bins:
        return changes
    previous: Dict[int, Optional[str]] = {}
    step = max(1, int(step_minutes))
    for end in range(step, len(bins) + 1, step):
        result = analyze_bins(bins[:end], dict(cfg), first_sunday=first_sunday, rules=rules)
        for block in result.get("blocks") or []:
            if block.get("ongoing"):
                continue
            start_ms = int(block.get("start_ms") or 0)
            now = block.get("label")
            if start_ms in previous and previous[start_ms] != now:
                changes.append(LabelChange(at_minute=end, start_ms=start_ms,
                                           was=previous[start_ms], now=now))
            previous[start_ms] = now
    return changes


def render_score(score_: Score) -> str:
    """One run, as a few lines a commit message can carry."""
    lines = ["%s: %.1f%% of %d judged minutes (%d agreed)" % (
        score_.label, score_.agreement * 100, score_.judged_minutes,
        score_.agreed_minutes)]
    for name in sorted(score_.per_label):
        per = score_.per_label[name]
        lines.append("  %-12s truth %3dm  run %3dm  overlap %3dm  p=%.2f r=%.2f" % (
            name, per.truth_minutes, per.run_minutes, per.overlap_minutes,
            per.precision, per.recall))
    if score_.counts:
        lines.append("  counts: " + ", ".join(
            "%s x%d" % (k, v) for k, v in sorted(score_.counts.items())))
    for extra in score_.spurious:
        lines.append("  spurious: %s (%d min)" % (extra.get("label"),
                                                  int(extra.get("minutes") or 0)))
    for gap in score_.missed:
        lines.append("  missed: %s (%s)" % (gap.get("label"), gap.get("source")))
    return "\n".join(lines)


def render_comparison(comparison: Comparison) -> str:
    lines = [render_score(comparison.before), render_score(comparison.after),
             "agreement %+0.1f points" % (comparison.agreement_delta * 100)]
    for item in comparison.fixed:
        lines.append("  fixed:  %s" % (item.get("label"),))
    for item in comparison.broken:
        lines.append("  broken: %s" % (item.get("label"),))
    if not comparison.fixed and not comparison.broken:
        lines.append("  nothing changed")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay archived services through the phase detector and score them.")
    parser.add_argument("databases", nargs="+", help="session database paths")
    parser.add_argument("--config", help="JSON file of service_phase settings to replay with")
    parser.add_argument("--rules", help="phase rule file to replay with")
    parser.add_argument("--source", choices=(SOURCE_BINS, SOURCE_ROWS), default=SOURCE_BINS)
    parser.add_argument("--progressive", action="store_true",
                        help="also report how often a settled label changed")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            cfg = json.load(handle)
    rules = None
    if args.rules:
        from stt.phase_rules import load_rules
        rules = load_rules(args.rules, args.rules)

    for path in args.databases:
        recording = load_recording(path)
        if not recording.truth:
            print("%s: nothing corrected; no ground truth to score against" % recording.session)
            continue
        before = score(shipped_run(recording), recording.truth, label="shipped")
        after = score(replay(recording, cfg, rules, source=args.source),
                      recording.truth, label="candidate")
        print("== %s%s" % (recording.session,
                           " [%s]" % recording.profile if recording.profile else ""))
        print(render_comparison(compare(before, after)))
        if args.progressive:
            churn = progressive(recording, cfg, rules)
            print("  settled labels changed %d time(s)" % len(churn))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
