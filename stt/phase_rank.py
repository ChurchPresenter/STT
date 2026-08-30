"""How many of a phase a service may contain, and which candidates keep the name.

The detector names a block from that block alone: a run of speech lasting longer than the
sermon threshold is a sermon, however many of them a service turns out to have. On a real
service that produced three sermons where there were two — the third was the fourteen
minutes after the closing songs, which the operator relabelled by hand.

No per-block rule can fix that, because nothing is wrong with the block in isolation: it is
speech, and it is long enough. What is wrong is that a *third* one exists. So this asks the
question the rule vocabulary cannot: given every block in the service, which of them are
the sermons? The answer is the longest ones, because a sermon here is the long speaking of
a service, and the false positive is reliably the short one — which is also why this ranks
rather than takes the first N or the last N. A positional rule works until the spurious
block moves.

Two consequences worth stating plainly:

* **A cap is a fact about one congregation**, so nothing here ships a number. The limits
  come from that church's own profile, and with no limits configured every function is a
  no-op — an install that has said nothing about its services is left exactly as it was.
* **This is not causal.** An earlier block's label can change when a later, longer one
  arrives. Boundaries are untouched, and labels already look forward elsewhere (a Closing
  stops being a Closing when something follows it), but the churn is real and is measured
  by :func:`stt.phase_replay.progressive` rather than argued about. ``closed_only`` keeps a
  still-running block out of it, since it may yet grow into the sermon it claims to be.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

# What a demoted block is told it is, when the rule list offers nothing better.
FALLBACK_CONFIDENCE = 0.3

REASON_OVER_LIMIT = "over the limit for this service"


class Demotion:
    """One block that lost a name because the service already had enough of them."""

    __slots__ = ("block_index", "minutes", "now", "reason", "start_ms", "was")

    def __init__(self, block_index: int, start_ms: int, minutes: int,
                 was: str, now: Optional[str], reason: str) -> None:
        self.block_index = block_index
        self.start_ms = start_ms
        self.minutes = minutes
        self.was = was
        self.now = now
        self.reason = reason

    def to_dict(self) -> dict:
        return {"block_index": self.block_index, "start_ms": self.start_ms,
                "minutes": self.minutes, "was": self.was, "now": self.now,
                "reason": self.reason}

    def note(self) -> str:
        """What the operator is told, which is also the prompt for a correction."""
        return ('"%s" at minute %d looks like a %s, but this service already has as many '
                "as the profile expects — calling it %s instead" % (
                    self.was, self.minutes, base_name(self.was), self.now or "unnamed"))


def base_name(label: Optional[str]) -> str:
    """A phase name without its ordinal: "Sermon 2" -> "Sermon"."""
    text = str(label or "").strip()
    parts = text.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].strip()
    return text


def fallback_for(rules: Sequence[Any], name: str, kind: str) -> Optional[str]:
    """What a demoted block should be called instead, taken from the rule list.

    Never a constant. The rules are the operator's vocabulary — a church that renamed
    "Speaking" gets its own word back — so the fallback is the first unnumbered rule of the
    same kind that is not the rule being capped, which is the same idea ``_renumber`` uses
    when it turns a surplus "Songs 3" back into plain "Music".
    """
    for rule in rules:
        rule_name = getattr(rule, "name", None)
        if not rule_name or rule_name == name or getattr(rule, "number", False):
            continue
        match = getattr(rule, "match", None) or {}
        kinds = match.get("kinds") or ([match["kind"]] if match.get("kind") else [])
        if kinds and kind not in kinds:
            continue
        if any(key not in ("kind", "kinds") for key in match):
            continue  # a conditional rule is not a safe catch-all
        return str(rule_name)
    return None


def rank_and_limit(blocks: Sequence[Any], *, name: str, max_count: int,
                   fallback_label: Optional[str],
                   fallback_confidence: float = FALLBACK_CONFIDENCE,
                   closed_only: bool = True) -> List[Demotion]:
    """Keep the ``max_count`` longest blocks called ``name``; rename the rest in place.

    Ranked by duration, ties going to whichever came first — so a service whose two
    candidates are the same length keeps the earlier one, and the answer does not flap
    between ticks over a tie. Survivors are renumbered in *time* order, because the numbers
    are how an operator refers to them out loud.
    """
    if max_count < 0:
        return []
    candidates = [b for b in blocks if base_name(getattr(b, "label", None)) == name]
    if closed_only:
        # A block still being spoken is invisible to the cap: not demoted, because it may
        # still grow into the sermon it claims to be, and not counted either, because
        # letting it spend a place would demote a finished block that is already longer.
        # A service may therefore be briefly over its cap, and settle when the block ends.
        rankable = [b for b in candidates if not getattr(b, "ongoing", False)]
    else:
        rankable = list(candidates)
    if len(rankable) <= max_count:
        return []

    ranked = sorted(rankable, key=lambda b: (-int(getattr(b, "minutes", 0)),
                                             int(getattr(b, "start_ms", 0))))
    demoted = ranked[max_count:]

    out: List[Demotion] = []
    for block in demoted:
        was = str(getattr(block, "label", "") or "")
        out.append(Demotion(int(getattr(block, "index", 0)),
                            int(getattr(block, "start_ms", 0)),
                            int(getattr(block, "minutes", 0)),
                            was, fallback_label, REASON_OVER_LIMIT))
        block.label = fallback_label
        block.confidence = fallback_confidence
    if out:
        renumber(blocks, name)
    return out


def renumber(blocks: Sequence[Any], name: str) -> None:
    """Renumber the survivors of ``name`` in time order, or drop the number if only one."""
    survivors = [b for b in blocks if base_name(getattr(b, "label", None)) == name]
    survivors.sort(key=lambda b: int(getattr(b, "start_ms", 0)))
    for i, block in enumerate(survivors, start=1):
        block.label = "%s %d" % (name, i)


def apply_limits(blocks: Sequence[Any], rules: Sequence[Any],
                 limits: Optional[Mapping[str, Mapping[str, Any]]], *,
                 closed_only: bool = True) -> List[Demotion]:
    """Apply every configured cap to a labelled timeline.

    With no limits — the shipped state, and every install that has not described its
    services — this does nothing at all.
    """
    if not limits:
        return []
    out: List[Demotion] = []
    for name in sorted(limits):
        setting = limits.get(name) or {}
        try:
            max_count = int(setting.get("max", -1))
        except (TypeError, ValueError):
            continue
        kind = _kind_of(blocks, name)
        out.extend(rank_and_limit(
            blocks, name=name, max_count=max_count,
            fallback_label=setting.get("fallback") or fallback_for(rules, name, kind),
            closed_only=bool(setting.get("closed_only", closed_only))))
    return out


def _kind_of(blocks: Sequence[Any], name: str) -> str:
    for block in blocks:
        if base_name(getattr(block, "label", None)) == name:
            return str(getattr(block, "kind", "") or "")
    return ""


def limit_notes(demotions: Sequence[Demotion]) -> List[str]:
    """The service-level notes for what a cap changed.

    Said out loud on purpose. A detector that quietly returns a different answer teaches an
    operator nothing; one that says which block it demoted and why is asking to be corrected
    if it got it wrong, and that correction is what the learner reads back.
    """
    return [d.note() for d in demotions]


def parse_limits(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Read a profile's ``limits`` block, ignoring anything malformed.

    A hand-edited file is the normal case here, so a typo must cost the setting it is in
    and nothing else — never the service.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for name, setting in raw.items():
        if not isinstance(name, str) or name.startswith("_"):
            continue
        if not isinstance(setting, dict):
            continue
        try:
            max_count = int(setting.get("max", -1))
        except (TypeError, ValueError):
            continue
        if max_count < 0:
            continue
        entry: Dict[str, Any] = {"max": max_count}
        if isinstance(setting.get("fallback"), str) and setting["fallback"].strip():
            entry["fallback"] = setting["fallback"].strip()
        if "closed_only" in setting:
            entry["closed_only"] = bool(setting["closed_only"])
        out[name] = entry
    return out
