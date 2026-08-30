"""Naming a service's blocks from a rule file rather than from Python.

The detector's *structure* — songs, speaking, quiet — comes from audio and is the reliable
part. The *names* on top of it are conventions, and conventions differ: one congregation
opens with announcements and closes with communion on the first Sunday, another runs three
sermons with song sets between, another has no communion at all. Encoding one church's
running order in `label_blocks` made every install inherit it.

So the names live in a rule file. `config/service_phases.default.json` ships a deliberately
plain baseline; `config/service_phases.json` is the operator's copy and is what the learner
edits. Nothing here knows about the live config, the database, or any particular church —
rules arrive as a dict, blocks and bins arrive as objects, and a name comes back.

Rules are tried in order and the first match wins, which is the only ordering guarantee a
rule author has to hold in their head. A rule that fails to parse is dropped rather than
raised: a typo in a phrase list must cost its own rule, not the whole detector.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

MUSIC = "M"
SPEECH = "S"
QUIET = "_"


class Rule:
    """One named phase and the conditions under which a block earns that name."""

    __slots__ = ("confidence", "confidence_first_sunday", "match", "name", "not_when",
                 "number", "number_from", "ongoing_confidence_max", "span")

    def __init__(self, name: str, *, match: Dict[str, Any],
                 confidence: float = 0.5, number: bool = False,
                 number_from: Optional[str] = None,
                 not_when: Optional[Dict[str, Any]] = None,
                 span: Optional[Dict[str, Any]] = None,
                 confidence_first_sunday: Optional[float] = None,
                 ongoing_confidence_max: Optional[float] = None) -> None:
        self.name = name
        self.match = match
        self.confidence = confidence
        self.number = number
        self.number_from = number_from
        self.not_when = not_when
        self.span = span
        # Communion's usual slot is the first Sunday, so the same evidence is worth more
        # then. A rule that does not care simply omits it.
        self.confidence_first_sunday = confidence_first_sunday
        # A block still running may yet grow into something else; say so rather than commit.
        self.ongoing_confidence_max = ongoing_confidence_max


class Span:
    """A phase covering several blocks, produced by a rule with a ``span`` clause."""

    __slots__ = ("confidence", "end_index", "label", "start_index")

    def __init__(self, label: str, start_index: int, end_index: int, confidence: float) -> None:
        self.label = label
        self.start_index = start_index
        self.end_index = end_index
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {"label": self.label, "start_index": self.start_index,
                "end_index": self.end_index, "confidence": round(self.confidence, 2)}


def load_rules(user_path: str, default_path: str) -> List[Rule]:
    """The operator's rule file if it has one, otherwise the shipped baseline.

    Deliberately not a key-by-key merge. Rules are an ordered list where position decides
    which one wins, so splicing shipped rules into an operator's list would silently change
    the answer on an upgrade — the one thing a file you are asked to tune must never do.
    An operator who wants a new shipped rule copies it in, and the learner writes whole
    files for the same reason.
    """
    for path in (user_path, default_path):
        if not path:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            continue
        rules = parse_rules(raw)
        if rules:
            return rules
    return []


def parse_rules(raw: Optional[Dict[str, Any]]) -> List[Rule]:
    """Turn the rule file's ``phases`` list into Rule objects, skipping unusable entries.

    Silently dropping a malformed rule matches how compile_cues treats a bad phrase list:
    the operator edits these files by hand and through the learner, and one bad entry must
    not leave a service with no names at all.
    """
    out: List[Rule] = []
    for item in ((raw or {}).get("phases") or []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        match = item.get("match")
        if not isinstance(name, str) or not name.strip() or not isinstance(match, dict):
            continue
        span = item.get("span") if isinstance(item.get("span"), dict) else None
        not_when = item.get("not_when") if isinstance(item.get("not_when"), dict) else None
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        out.append(Rule(
            name.strip(), match=match, confidence=confidence,
            number=bool(item.get("number", False)),
            number_from=item.get("number_from") if isinstance(item.get("number_from"), str) else None,
            not_when=not_when, span=span,
            confidence_first_sunday=_opt_num(item.get("confidence_first_sunday")),
            ongoing_confidence_max=_opt_num(item.get("ongoing_confidence_max")),
        ))
    return out


def _opt_num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _distinct_fragments(cues: Dict[str, int], group: str) -> int:
    """How many different fragments of one cue group are present.

    Fragments are flattened into cue names as ``group:fragment`` when the phrase list is
    compiled, so the existing per-bin counting and persistence carry them with no schema
    change. Distinctness is what matters here: two different lines of a liturgical formula
    is evidence that it was read, whereas one word twenty times is a topic.
    """
    prefix = group + ":"
    return sum(1 for name, n in cues.items() if n and name.startswith(prefix))


def _matches(block: Any, cond: Dict[str, Any], *, is_last_named: bool = False) -> bool:
    """Does one block satisfy one condition clause?"""
    if cond.get("is_last_named") is True and not is_last_named:
        return False
    kind = cond.get("kind")
    if isinstance(kind, str) and block.kind != kind:
        return False
    kinds = cond.get("kinds")
    if isinstance(kinds, list) and kinds and block.kind not in kinds:
        return False
    if "min_minutes" in cond and block.minutes < _num(cond["min_minutes"], 0):
        return False
    if "max_minutes" in cond and block.minutes > _num(cond["max_minutes"], 10 ** 6):
        return False
    for group, need in (cond.get("cues") or {}).items():
        if block.cues.get(group, 0) < _num(need, 0):
            return False
    for group, need in (cond.get("cue_fragments") or {}).items():
        if _distinct_fragments(block.cues, group) < _num(need, 0):
            return False
    return True


def _num(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _span_end(blocks: Sequence[Any], start: int, span: Dict[str, Any]) -> int:
    """How far a spanning phase reaches past the block that opened it.

    Communion is the case this exists for: the reading opens it, then distribution runs as
    quiet minutes with short music over them, and it ends when the service returns to
    something substantial. Everything about "short" and "substantial" is in the rule file.
    """
    inside = span.get("extend_while_kinds")
    inside = inside if isinstance(inside, list) else []
    max_inner = _num(span.get("max_inner_block_minutes"), 10 ** 6)
    ends_on = span.get("ends_on_kind_minutes") or {}

    end = start
    for i in range(start + 1, len(blocks)):
        b = blocks[i]
        if b.kind not in inside:
            break
        limit = ends_on.get(b.kind)
        if limit is not None and b.minutes >= _num(limit, 10 ** 6):
            break
        if b.minutes > max_inner:
            break
        end = i
    return end


def apply_rules(blocks: Sequence[Any], rules: Sequence[Rule], *,
                first_sunday: bool = False,
                barred: Optional[Mapping[int, Set[str]]] = None) -> List[Span]:
    """Name every block from the rules, returning any multi-block spans that were formed.

    Blocks are labelled in place — the caller already owns them and the persistence layer
    reads `label`/`confidence` off each one. Spans are returned separately because they are
    a claim *about* several blocks rather than a property of any one of them, and the page
    renders them the same way it renders an operator's group.

    Ordinals (`Sermon 1`, `Songs 2`) are assigned in one forward pass, so the whole numbering
    re-derives on every run and never drifts from what is on screen.

    ``barred`` names rules that may not claim particular blocks, which is how a block the
    service already has enough of is handed back to the rules rather than to a fixed
    fallback. Without it a demoted sermon becomes whatever generic name was chosen for it in
    advance; with it, the fourteen minutes at the end of a service become the Closing they
    match, because the rules get to answer the question a second time.
    """
    counters: Dict[str, int] = {}
    seen: Dict[str, bool] = {}
    spans: List[Span] = []
    claimed: Dict[int, bool] = {}
    # "The last block that could carry a name" — a closing talk is recognised by sitting at
    # the end of the service, and quiet blocks after it do not count as the service going on.
    nameable = [i for i, b in enumerate(blocks) if b.kind != QUIET]
    last_named = nameable[-1] if nameable else -1

    for i, b in enumerate(blocks):
        if claimed.get(i):
            continue
        b.label, b.confidence = None, 0.0
        blocked_here = (barred or {}).get(i) or ()
        for rule in rules:
            if rule.name in blocked_here:
                continue
            if not _matches(b, rule.match, is_last_named=(i == last_named)):
                continue
            if rule.not_when is not None and _matches(b, rule.not_when,
                                                      is_last_named=(i == last_named)):
                continue
            if _blocked_by_order(rule, seen):
                continue

            b.confidence = rule.confidence
            if first_sunday and rule.confidence_first_sunday is not None:
                b.confidence = rule.confidence_first_sunday
            if rule.ongoing_confidence_max is not None and b.ongoing:
                b.confidence = min(b.confidence, rule.ongoing_confidence_max)
            if rule.number:
                counters[rule.name] = counters.get(rule.name, 0) + 1
                b.label = "%s %d" % (rule.name, counters[rule.name])
            else:
                b.label = rule.name
            seen[rule.name] = True

            if rule.span is not None:
                end = _span_end(blocks, i, rule.span)
                if end > i:
                    for j in range(i + 1, end + 1):
                        claimed[j] = True
                        blocks[j].label, blocks[j].confidence = None, 0.0
                    spans.append(Span(b.label, i, end, rule.confidence))
            break

    _renumber(blocks, rules, spans)
    return spans


def _blocked_by_order(rule: Rule, seen: Dict[str, bool]) -> bool:
    """Positional conditions: 'only before the first Sermon', 'only after one'."""
    before = rule.match.get("before_first")
    if isinstance(before, str) and seen.get(before):
        return True
    after = rule.match.get("after_first")
    if isinstance(after, str) and not seen.get(after):
        return True
    return False


def _renumber(blocks: Sequence[Any], rules: Sequence[Rule], spans: Sequence[Span]) -> None:
    """Restart a numbered rule's count at the phase named in ``number_from``.

    Songs are the reason: music before the service opens is the band rehearsing to an empty
    room, and numbering it put Songs 1 half an hour before anyone arrived. Which phase the
    count starts at is a rule-file decision, not a fact about music.
    """
    anchors = {r.name: r.number_from for r in rules if r.number and r.number_from}
    if not anchors:
        return
    inside = {j for s in spans for j in range(s.start_index, s.end_index + 1)}
    for name, anchor in anchors.items():
        at = _index_of(blocks, anchor, spans)
        if at is None:
            continue
        n = 0
        for i, b in enumerate(blocks):
            if not b.label or not b.label.startswith(name + " ") or i in inside:
                continue
            if i < at:
                b.label, b.confidence = _unnumbered(rules, name), 0.4
            else:
                n += 1
                b.label = "%s %d" % (name, n)


def _index_of(blocks: Sequence[Any], label: str, spans: Sequence[Span]) -> Optional[int]:
    for s in spans:
        if s.label == label:
            return s.start_index
    for i, b in enumerate(blocks):
        if b.label == label:
            return i
    return None


def _unnumbered(rules: Sequence[Rule], name: str) -> Optional[str]:
    """What a block of this kind is called when it does not earn a number."""
    for r in rules:
        if not r.number and r.match.get("kind") == _kind_of(rules, name):
            return r.name
    return None


def _kind_of(rules: Sequence[Rule], name: str) -> Optional[str]:
    for r in rules:
        if r.name == name:
            kind = r.match.get("kind")
            return kind if isinstance(kind, str) else None
    return None
