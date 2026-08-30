"""Proposing detector settings from the services an operator has actually corrected.

The shipped thresholds are one congregation's measurements. This reads the corrections
accumulated across archived sessions and says what *this* installation's numbers look like,
so nobody has to inherit someone else's Sunday.

Three deliberate limits:

* Only corrections count. The detector's own output is not evidence — the shipped figures
  were derived that way, and learning from it would teach the detector to agree with itself.
* Nothing is applied. Every function here returns a proposal with the evidence behind it;
  writing is the caller's decision and the operator's click. A single mistaken correction
  should not silently move a threshold between one service and the next.
* Below a minimum sample count a knob stays on its baseline and says so. A proposal from
  two examples is not a measurement, and presenting it as one is worse than staying quiet.

Stdlib only, and every path is passed in: this module never reads the live config, never
finds the archive itself, and never opens a database it was not handed.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# How many corrected examples a knob needs before it is worth proposing anything. Four is
# not statistics; it is the point at which a number stops being one operator's afternoon.
MIN_SAMPLES = 4

# Words too common to be evidence of anything. Kept tiny and structural on purpose — a real
# stopword list is language-specific, and this module must not assume a language.
_TOKEN = re.compile(r"\w+", re.UNICODE)


class Proposal:
    """One suggested setting, and everything needed to judge it."""

    __slots__ = ("baseline", "current", "evidence", "key", "samples", "suggested",
                 "target")

    def __init__(self, key: str, *, baseline: Any, current: Any, suggested: Any,
                 samples: int, evidence: str, target: str = "profile") -> None:
        self.key = key
        self.baseline = baseline
        self.current = current
        self.suggested = suggested
        self.samples = samples
        self.evidence = evidence
        # Where applying this actually writes. Worth carrying, because the two destinations
        # are not interchangeable: a threshold belongs in the rule the detector reads, and
        # writing it to the config block instead is how sermon_min_minutes came to be
        # proposed, applied, and then ignored by everything.
        self.target = target

    @property
    def actionable(self) -> bool:
        """Worth showing an Apply button for: enough evidence, and not already the value."""
        return self.samples >= MIN_SAMPLES and self.suggested != self.current

    def to_dict(self) -> dict:
        return {"key": self.key, "baseline": self.baseline, "current": self.current,
                "suggested": self.suggested, "samples": self.samples,
                "evidence": self.evidence, "actionable": self.actionable,
                "target": self.target}


class CorrectedPhase:
    """One stretch of one service that a human put a name to."""

    __slots__ = ("end_ms", "kind", "label", "minutes", "session", "start_ms", "text")

    def __init__(self, session: str, label: str, kind: str,
                 start_ms: int, end_ms: int, minutes: int, text: str = "") -> None:
        self.session = session
        self.label = label
        self.kind = kind
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.minutes = minutes
        self.text = text


def read_corrected_phases(conn: "sqlite3.Connection", session: str) -> List[CorrectedPhase]:
    """The named stretches of one session: per-block corrections and grouped spans alike.

    A correction carries a name but not always a shape — the page sends only the block index
    for a single block — so the block table supplies the times. A group carries its own span
    and no index, which is exactly why both paths exist.
    """
    try:
        corrections = conn.execute(
            "SELECT block_index, start_ms, end_ms, kind, label FROM service_phase_corrections "
            "WHERE label IS NOT NULL AND label != ''").fetchall()
    except sqlite3.Error:
        return []
    blocks: Dict[int, Tuple] = {}
    try:
        blocks = {r[0]: r for r in conn.execute(
            "SELECT block_index, kind, start_ms, end_ms, minutes FROM service_phase_blocks")}
    except sqlite3.Error:
        pass

    out: List[CorrectedPhase] = []
    for block_index, start_ms, end_ms, kind, label in corrections:
        # A stored span first. Looking the index up in service_phase_blocks reads that
        # block's *current* times, and blocks renumber whenever the detector's output changes
        # shape — so a drifted index would teach the learner a phase of the wrong length
        # under the operator's name. The span is what the operator actually pointed at.
        if start_ms and end_ms and end_ms > start_ms:
            out.append(CorrectedPhase(session, label, kind or "", start_ms, end_ms,
                                      max(1, round((end_ms - start_ms) / 60000.0))))
        elif block_index is not None and block_index in blocks:
            _, bkind, bstart, bend, bminutes = blocks[block_index]
            out.append(CorrectedPhase(session, label, kind or bkind or "",
                                      bstart, bend, bminutes or 0))
    return out


def read_phase_text(conn: "sqlite3.Connection", start_ms: int, end_ms: int) -> str:
    """The transcript under one corrected stretch, for mining the words that mark it."""
    try:
        rows = conn.execute(
            "SELECT text FROM transcriptions WHERE is_final = 1 AND ts_ms >= ? AND ts_ms <= ?",
            (start_ms, end_ms)).fetchall()
    except sqlite3.Error:
        return ""
    return " ".join(r[0] or "" for r in rows)


def collect(sessions: Iterable[Tuple[str, "sqlite3.Connection"]], *,
            with_text: bool = False) -> List[CorrectedPhase]:
    """Every corrected stretch across the sessions handed in, oldest session first."""
    out: List[CorrectedPhase] = []
    for name, conn in sessions:
        for phase in read_corrected_phases(conn, name):
            if with_text:
                phase.text = read_phase_text(conn, phase.start_ms, phase.end_ms)
            out.append(phase)
    return out


def _base_name(label: str) -> str:
    """'Sermon 2' and 'Sermon' are the same phase for the purpose of measuring it."""
    return re.sub(r"\s+\d+$", "", (label or "").strip())


def _percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. No numpy — this module is stdlib-only by contract."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[idx]


def propose_durations(phases: Sequence[CorrectedPhase], cfg: Dict[str, Any],
                      baseline: Dict[str, Any]) -> List[Proposal]:
    """Length thresholds, from how long the named phases actually ran here.

    A sermon threshold is really the question "how short can an address be here before it is
    something else", so the proposal is the shortest corrected sermon rather than a middle:
    a threshold above the shortest real one would have misnamed it.
    """
    by_name: Dict[str, List[int]] = {}
    for p in phases:
        by_name.setdefault(_base_name(p.label), []).append(p.minutes)

    out: List[Proposal] = []
    sermons = by_name.get("Sermon", [])
    out.append(Proposal(
        "sermon_min_minutes",
        baseline=baseline.get("sermon_min_minutes"),
        current=cfg.get("sermon_min_minutes", baseline.get("sermon_min_minutes")),
        suggested=int(_percentile(sermons, 0.1)) if len(sermons) >= MIN_SAMPLES
        else cfg.get("sermon_min_minutes", baseline.get("sermon_min_minutes")),
        samples=len(sermons),
        evidence=_lengths("sermons", sermons)))

    songs = by_name.get("Songs", [])
    out.append(Proposal(
        "songs_min_minutes",
        baseline=baseline.get("songs_min_minutes"),
        current=cfg.get("songs_min_minutes", baseline.get("songs_min_minutes")),
        suggested=max(1, int(_percentile(songs, 0.1))) if len(songs) >= MIN_SAMPLES
        else cfg.get("songs_min_minutes", baseline.get("songs_min_minutes")),
        samples=len(songs),
        evidence=_lengths("song sets", songs)))

    music = [p.minutes for p in phases if p.kind == "M"]
    out.append(Proposal(
        "typical_music_max_minutes",
        baseline=baseline.get("typical_music_max_minutes"),
        current=cfg.get("typical_music_max_minutes", baseline.get("typical_music_max_minutes")),
        suggested=int(_percentile(music, 1.0)) if len(music) >= MIN_SAMPLES
        else cfg.get("typical_music_max_minutes", baseline.get("typical_music_max_minutes")),
        samples=len(music),
        evidence=_lengths("music blocks", music)))

    speech = [p.minutes for p in phases if p.kind == "S"]
    out.append(Proposal(
        "typical_speaking_max_minutes",
        baseline=baseline.get("typical_speaking_max_minutes"),
        current=cfg.get("typical_speaking_max_minutes",
                        baseline.get("typical_speaking_max_minutes")),
        suggested=int(_percentile(speech, 1.0)) if len(speech) >= MIN_SAMPLES
        else cfg.get("typical_speaking_max_minutes",
                     baseline.get("typical_speaking_max_minutes")),
        samples=len(speech),
        evidence=_lengths("speaking blocks", speech)))
    return out


def _lengths(what: str, values: Sequence[int]) -> str:
    if not values:
        return "no corrected %s yet" % what
    ordered = sorted(values)
    shown = ", ".join(str(v) for v in ordered[:8])
    more = "" if len(ordered) <= 8 else ", …"
    return "%d corrected %s: %s%s min" % (len(ordered), what, shown, more)


def propose_fragments(phases: Sequence[CorrectedPhase], group: str, label: str, *,
                      known: Optional[Dict[str, Sequence[str]]] = None,
                      min_share: float = 0.75, max_new: int = 6) -> List[Proposal]:
    """Phrases that show up in every corrected instance of one phase and nowhere else.

    This is the part that turns "the recogniser's own wording differs from the printed form" into a rule
    without anyone typing it. A candidate has to appear in most of the corrected examples of
    the phase and in none of the other phases — a word common to both is a word about the
    service, not about this part of it.
    """
    wanted = [p for p in phases if _base_name(p.label) == label and p.text]
    others = [p for p in phases if _base_name(p.label) != label and p.text]
    if len(wanted) < MIN_SAMPLES:
        return [Proposal("cue_fragments.%s" % group, baseline=None, current=None,
                         suggested=[], samples=len(wanted), target="config",
                         evidence="needs %d corrected %s, has %d"
                                  % (MIN_SAMPLES, label, len(wanted)))]

    have = {w.lower() for phrases in (known or {}).values() for w in phrases}
    counts: Dict[str, int] = {}
    for p in wanted:
        for phrase in set(_phrases(p.text)):
            counts[phrase] = counts.get(phrase, 0) + 1
    elsewhere: Dict[str, int] = {}
    for p in others:
        for phrase in set(_phrases(p.text)):
            elsewhere[phrase] = elsewhere.get(phrase, 0) + 1

    need = max(1, round(min_share * len(wanted)))
    candidates = [
        (phrase, n) for phrase, n in counts.items()
        if n >= need and not elsewhere.get(phrase) and phrase not in have
    ]
    candidates.sort(key=lambda x: (-x[1], x[0]))
    picked = [phrase for phrase, _ in candidates[:max_new]]
    return [Proposal(
        "cue_fragments.%s" % group, baseline=None, current=sorted(have)[:6],
        suggested=picked, samples=len(wanted), target="config",
        evidence="phrases in %d of %d corrected %s and in none of the %d other phases"
                 % (need, len(wanted), label, len(others)))]


def _phrases(text: str, n: int = 3) -> List[str]:
    """Word n-grams, lowercased. Three words is long enough to be a quotation."""
    words = [w.lower() for w in _TOKEN.findall(text or "")]
    return [" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]


class ServiceShape:
    """What one corrected service turned out to look like."""

    __slots__ = ("counts", "end_minutes", "profile", "session")

    def __init__(self, session: str, counts: Dict[str, int], end_minutes: int,
                 profile: Optional[str] = None) -> None:
        self.session = session
        self.counts = counts
        self.end_minutes = end_minutes
        self.profile = profile


def read_marked_phases(conn: "sqlite3.Connection", session: str, *,
                       resolve: Any, now_ms: int = 0) -> List[CorrectedPhase]:
    """The phases an operator marked live, as evidence.

    A mark is the strongest statement about a boundary anyone makes — a person in the room
    saying "the sermon starts now" — and it was being discarded: a mark carries no end and
    no block index, so both branches of :func:`read_corrected_phases` reject it.

    ``resolve`` is passed in rather than imported so this module keeps its one job and its
    stdlib-only imports; the caller hands it :func:`stt.phase_marks.resolve`, which is the
    single implementation of where a marked phase ends. Only closed spans count: an open one
    was still running, and says nothing about how long that phase was.
    """
    try:
        rows = conn.execute(
            "SELECT id, block_index, start_ms, end_ms, kind, label, note, corrected_at "
            "FROM service_phase_corrections ORDER BY id").fetchall()
        blocks = conn.execute(
            "SELECT block_index, kind, start_ms, end_ms, minutes, label, ongoing "
            "FROM service_phase_blocks ORDER BY block_index").fetchall()
    except sqlite3.Error:
        return []
    keys = ("id", "block_index", "start_ms", "end_ms", "kind", "label", "note",
            "corrected_at")
    corrections = [dict(zip(keys, r)) for r in rows]
    block_dicts = [{"block_index": b[0], "kind": b[1], "start_ms": b[2], "end_ms": b[3],
                    "minutes": b[4], "label": b[5], "ongoing": bool(b[6])} for b in blocks]
    anchor_ms = now_ms or max((int(b["end_ms"] or 0) for b in block_dicts), default=0)
    out: List[CorrectedPhase] = []
    for span in resolve(corrections, block_dicts, now_ms=anchor_ms):
        if span.get("open"):
            continue
        start, end = int(span.get("start_ms") or 0), int(span.get("end_ms") or 0)
        if end <= start:
            continue
        out.append(CorrectedPhase(session, str(span.get("label") or ""),
                                  str(span.get("kind") or ""), start, end,
                                  max(1, round((end - start) / 60000.0))))
    return out


def group_by_service(phases: Sequence[CorrectedPhase],
                     profiles: Optional[Mapping[str, str]] = None) -> List[ServiceShape]:
    """One entry per corrected service: what it contained and how long it ran."""
    by_session: Dict[str, List[CorrectedPhase]] = {}
    for phase in phases:
        by_session.setdefault(phase.session, []).append(phase)
    shapes: List[ServiceShape] = []
    for session, found in sorted(by_session.items()):
        counts: Dict[str, int] = {}
        for phase in found:
            name = _base_name(phase.label)
            counts[name] = counts.get(name, 0) + 1
        start = min(p.start_ms for p in found)
        end = max(p.end_ms for p in found)
        shapes.append(ServiceShape(session, counts,
                                   max(1, round((end - start) / 60000.0)),
                                   (profiles or {}).get(session)))
    return shapes


def _mode(values: Sequence[int]) -> Optional[int]:
    """The commonest value, ties going to the smaller.

    A mode rather than a maximum, deliberately. One Christmas service with three sermons
    must not raise a church's cap for every ordinary Sunday afterwards; the usual shape is
    what a limit should describe.
    """
    if not values:
        return None
    counts: Dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def propose_counts(shapes: Sequence[ServiceShape], limits: Mapping[str, Any], *,
                   label: str = "Sermon") -> List[Proposal]:
    """How many of a phase this installation's services usually contain."""
    seen = [s.counts.get(label, 0) for s in shapes if s.counts.get(label)]
    key = "max_%ss" % label.lower()
    current = ((limits or {}).get(label) or {}).get("max")
    if len(seen) < MIN_SAMPLES:
        return [Proposal(key, baseline=None, current=current, suggested=current,
                         samples=len(seen),
                         evidence="needs %d corrected services, has %d"
                                  % (MIN_SAMPLES, len(seen)))]
    return [Proposal(key, baseline=None, current=current, suggested=_mode(seen),
                     samples=len(seen),
                     evidence="%d corrected services: %s"
                              % (len(seen), ", ".join(str(n) for n in seen)))]


def propose_service_length(shapes: Sequence[ServiceShape],
                           service: Mapping[str, Any]) -> List[Proposal]:
    """How long a service of this kind runs, for a rule about when one can still start."""
    lengths = [s.end_minutes for s in shapes if s.end_minutes]
    current = (service or {}).get("service_length_minutes")
    if len(lengths) < MIN_SAMPLES:
        return [Proposal("service_length_minutes", baseline=None, current=current,
                         suggested=current, samples=len(lengths),
                         evidence="needs %d corrected services, has %d"
                                  % (MIN_SAMPLES, len(lengths)))]
    return [Proposal("service_length_minutes", baseline=None, current=current,
                     suggested=_percentile(lengths, 0.9), samples=len(lengths),
                     evidence="90th percentile of %d services (%d-%d min)"
                              % (len(lengths), min(lengths), max(lengths)))]


def propose_all(phases: Sequence[CorrectedPhase], cfg: Dict[str, Any],
                baseline: Dict[str, Any], *,
                shapes: Optional[Sequence[ServiceShape]] = None,
                profile_name: Optional[str] = None) -> Dict[str, Any]:
    """Everything this installation's corrections support, with the evidence attached."""
    proposals = propose_durations(phases, cfg, baseline)
    proposals += propose_fragments(phases, "communion_verse", "Communion",
                                   known=(cfg.get("cue_fragments") or {}).get("communion_verse"))
    shapes = list(shapes or group_by_service(phases))
    proposals += propose_counts(shapes, cfg.get("limits") or {})
    proposals += propose_service_length(shapes, cfg)
    return {
        "min_samples": MIN_SAMPLES,
        "sessions": len({p.session for p in phases}),
        "corrections": len(phases),
        "services": len(shapes),
        "profile": profile_name,
        "proposals": [p.to_dict() for p in proposals],
        "actionable": sum(1 for p in proposals if p.actionable),
    }


def apply_proposals(cfg: Dict[str, Any], proposals: Sequence[Dict[str, Any]],
                    keys: Sequence[str]) -> Dict[str, Any]:
    """A copy of the service_phase config with the named proposals taken.

    Returns a new dict rather than mutating: the caller holds the live config, and a
    half-applied settings change is worse than none. Only proposals the operator named are
    taken, and only if they were actionable when they were shown.
    """
    out = json.loads(json.dumps(cfg or {}))
    for proposal in proposals:
        key = proposal.get("key") or ""
        if key not in keys or not proposal.get("actionable"):
            continue
        if key.startswith("cue_fragments."):
            group = key.split(".", 1)[1]
            fragments = out.setdefault("cue_fragments", {}).setdefault(group, {})
            for i, phrase in enumerate(proposal.get("suggested") or []):
                fragments["learned_%d" % (len(fragments) + i)] = [re.escape(phrase)]
        else:
            out[key] = proposal.get("suggested")
    return out
