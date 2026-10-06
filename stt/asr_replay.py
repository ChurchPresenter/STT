"""Replay a past service through the sentence filters and count what they would keep.

Whether the hallucination and music filters throw away speech was, until now, found by
reading a session database by hand after somebody noticed words were missing. A threshold or a
rule change was judged by one service and a feeling. A session database already holds every
row the pipeline produced, the audio type and tag each was finalised with, and what the
filters did to it, so a candidate rule can be run over it and the answer is a count.

The comparison is the output, as in :mod:`stt.phase_replay`: one run's totals say little, but
which rows changed against the run before, and in which direction, says whether a change helped.
:func:`shipped_run` makes the baseline free, because what a session stored is a run.

Two things this deliberately is not:

* **Not a re-transcription.** Rows are replayed through :func:`stt.segment_disposition.decide`,
  the function production calls, on the text Whisper emitted. The model never runs.
* **Not a printer of captions.** A caption is verbatim congregation speech. Reports name row
  ids and counts only, so a result can be shared and grepped back to nobody.

The audio label of a row is the one it was stored with, except that a row the detector tagged
Speech but typed Music is re-labelled Speaking when ``speech_overrides_music`` is on (the rule
:func:`stt.segments.effective_music_prob` applies live). The detector's smoothing history is not
stored, so this is the rule's effect, not a bit-exact rerun of the smoother.

Stdlib-only and model-free.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence

from stt.segment_disposition import decide
from stt.text_utils import DEFAULT_WHISPER_HALLUCINATIONS

# Reasons a row is denied by something other than the sentence filters replayed here. They
# are carried through unchanged: replaying them would need state (recent sentences, the word
# floor) this harness does not have.
CARRIED_REASONS = frozenset({"dup", "short", "cjk_shadow"})

OUTCOME_KEPT = "kept"


@dataclass(frozen=True)
class Row:
    """One finalised row of a session, as stored."""

    id: int
    text: str
    original_text: str
    denied: bool
    reason: Optional[str]
    speech_type: Optional[str]
    audio_tag: Optional[str]


@dataclass(frozen=True)
class Settings:
    """What the replay runs with."""

    phrases: Sequence[str] = tuple(DEFAULT_WHISPER_HALLUCINATIONS)
    speech_overrides_music: bool = True
    transcribe_music: bool = False
    music_reason: str = "music"


@dataclass(frozen=True)
class Run:
    """The outcome per row id: ``None`` when kept, else the reason it was denied."""

    label: str
    outcomes: Mapping[int, Optional[str]]


@dataclass(frozen=True)
class Comparison:
    before: str
    after: str
    rescued: List[int] = field(default_factory=list)       # denied before, kept after
    newly_denied: List[int] = field(default_factory=list)  # kept before, denied after
    reason_changed: List[int] = field(default_factory=list)
    unchanged: int = 0


@dataclass(frozen=True)
class Score:
    """A run judged against ids a human has marked."""

    label: str
    speech_lost: List[int]    # marked speech the run denied
    junk_kept: List[int]      # marked junk the run kept
    speech_total: int
    junk_total: int

    @property
    def speech_recall(self) -> float:
        return 1.0 - (len(self.speech_lost) / self.speech_total) if self.speech_total else 1.0

    @property
    def junk_rejected(self) -> float:
        return 1.0 - (len(self.junk_kept) / self.junk_total) if self.junk_total else 1.0


def _read_only_uri(db_path: str) -> str:
    # immutable=1, as stt/phase_replay: a live session's WAL must not gain a -shm sidecar here.
    return "file:%s?immutable=1" % urllib.parse.quote(db_path)


def load_rows(db_path: str) -> List[Row]:
    """The finalised rows of a session database, oldest first."""
    conn = sqlite3.connect(_read_only_uri(db_path), uri=True)
    try:
        cur = conn.execute(
            "SELECT id, text, original_text, denied, denied_reason, speech_type, audio_tag "
            "FROM transcriptions WHERE is_final = 1 ORDER BY id")
        return [
            Row(id=r[0], text=r[1] or "", original_text=r[2] or r[1] or "", denied=bool(r[3]),
                reason=r[4], speech_type=r[5], audio_tag=r[6])
            for r in cur.fetchall()
        ]
    finally:
        conn.close()


def shipped_run(rows: Sequence[Row], label: str = "shipped") -> Run:
    """What the session actually did: its stored decisions are a run."""
    return Run(label, {r.id: (r.reason or "denied") if r.denied else None for r in rows})


def music_label(row: Row, settings: Settings) -> Optional[str]:
    """The audio type to replay a row with."""
    if settings.speech_overrides_music and row.speech_type == "Music" and row.audio_tag == "Speech":
        return "Speaking"
    return row.speech_type


def replay(rows: Sequence[Row], settings: Settings, label: str = "candidate") -> Run:
    """Run every row through the production sentence decision with ``settings``."""
    outcomes: Dict[int, Optional[str]] = {}
    for row in rows:
        if row.denied and row.reason in CARRIED_REASONS:
            outcomes[row.id] = row.reason
            continue
        verdict = decide(
            row.original_text,
            phrases=settings.phrases,
            music_label=music_label(row, settings),
            transcribe_music=settings.transcribe_music,
            music_reason=settings.music_reason,
        )
        outcomes[row.id] = verdict.reason if verdict.denied else None
    return Run(label, outcomes)


def compare(before: Run, after: Run) -> Comparison:
    """Which rows a change rescued, which it newly denied, and which it re-labelled."""
    rescued: List[int] = []
    newly_denied: List[int] = []
    reason_changed: List[int] = []
    unchanged = 0
    for row_id, was in before.outcomes.items():
        now = after.outcomes.get(row_id, was)
        if was == now:
            unchanged += 1
        elif was is not None and now is None:
            rescued.append(row_id)
        elif was is None and now is not None:
            newly_denied.append(row_id)
        else:
            reason_changed.append(row_id)
    return Comparison(before.label, after.label, rescued, newly_denied, reason_changed, unchanged)


def score(run: Run, speech_ids: FrozenSet[int], junk_ids: FrozenSet[int]) -> Score:
    """Judge a run against ids a human marked as real speech or as junk."""
    lost = sorted(i for i in speech_ids if run.outcomes.get(i) is not None)
    kept = sorted(i for i in junk_ids if i in run.outcomes and run.outcomes[i] is None)
    return Score(run.label, lost, kept, len(speech_ids), len(junk_ids))


def reason_counts(run: Run) -> Dict[str, int]:
    """Rows per outcome, with kept rows under ``kept``."""
    counts: Counter = Counter(OUTCOME_KEPT if v is None else v for v in run.outcomes.values())
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def render_comparison(cmp: Comparison, before: Run, after: Run) -> str:
    """Counts and row ids; never caption text."""
    def ids(values: Sequence[int]) -> str:
        shown = ", ".join(str(v) for v in values[:40])
        return shown + (" ..." if len(values) > 40 else "")
    lines = [
        "  %s: %s" % (cmp.before, reason_counts(before)),
        "  %s: %s" % (cmp.after, reason_counts(after)),
        "  rescued %d (denied -> kept): %s" % (len(cmp.rescued), ids(cmp.rescued)),
        "  newly denied %d (kept -> denied): %s" % (len(cmp.newly_denied), ids(cmp.newly_denied)),
        "  reason changed %d, unchanged %d" % (len(cmp.reason_changed), cmp.unchanged),
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay archived sessions through the sentence filters and compare with what shipped.")
    parser.add_argument("databases", nargs="+", help="session database paths")
    parser.add_argument("--config", help="JSON with hallucination phrases and speech_type_detection to replay with")
    parser.add_argument("--no-speech-override", action="store_true",
                        help="replay without the Speech-tag-beats-Music rule")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            cfg = json.load(handle)
    halluc = cfg.get("hallucination_filter", {})
    sdt = cfg.get("speech_type_detection", {})
    settings = Settings(
        phrases=tuple(halluc.get("phrases", DEFAULT_WHISPER_HALLUCINATIONS)) if halluc.get("enabled", True) else (),
        speech_overrides_music=not args.no_speech_override and sdt.get("speech_overrides_music", True),
        transcribe_music=bool(sdt.get("transcribe_detected_music", False)),
        music_reason="music:%g" % sdt.get("music_prob_threshold", 0.5),
    )
    for path in args.databases:
        rows = load_rows(path)
        shipped = shipped_run(rows)
        candidate = replay(rows, settings)
        print("== %s (%d final rows)" % (path, len(rows)))
        print(render_comparison(compare(shipped, candidate), shipped, candidate))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
