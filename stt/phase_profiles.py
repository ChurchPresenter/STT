"""One church's own services, described in that church's own files.

A Sunday morning service and a Wednesday evening one are not the same shape. The morning
runs two hours and ends with a long closing; the midweek meeting runs eighty minutes and
ends with a short one. Until now the detector could not tell them apart: weekday and time
of day are invisible to it, and one set of thresholds had to serve both.

The obvious fix is to put the right numbers in the app, and it is the wrong one. "Two
sermons", "a service runs about two hours", "the closing can be fifteen minutes" are facts
about one congregation, and shipping them would make every other install wrong in a way its
operator could not see. So the app ships the *ability* to say those things and no instance
of them: with no profile on disk, every function here returns what the detector already did.

A profile is a file — ``service_phases.<name>.json`` beside the operator's existing rule
file in their own config directory, which is outside the repository, gitignored, and never
overwritten by an update. Three things live in it:

* ``phases``  — the rules, exactly as the single rule file already holds them;
* ``service`` — the numbers the rules and the detector read, overriding the config block
  for this kind of service;
* ``limits``  — how many of a phase this kind of service has (see stt/phase_rank).

Which profile answers is decided by when the service started, so nobody has to remember to
pick one; an operator can still override it for an unusual week. Different churches need
nothing from this module at all — they are different installs with different config
directories, which is a separation that already exists and needs no code.
"""

import datetime
import json
import os
import re
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from stt.phase_rules import Rule, parse_rules

DEFAULT_PROFILE = "default"

# The file the single-profile install already has, and the default profile's name on disk.
BASE_FILENAME = "service_phases.json"

_SLUG_OK = re.compile(r"[^a-z0-9-]+")
_TIME = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")

MAX_SLUG = 40


class Profile:
    """The rules and numbers one kind of service is judged by."""

    __slots__ = ("limits", "name", "raw", "rules", "service", "source")

    def __init__(self, name: str, *, rules: Optional[List[Rule]] = None,
                 service: Optional[Dict[str, Any]] = None,
                 limits: Optional[Dict[str, Any]] = None,
                 source: str = "", raw: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.rules = rules or []
        self.service = service or {}
        self.limits = limits or {}
        # Which file actually answered. An operator whose profile is missing gets the
        # shipped rules instead, and without this nothing on the page could say so.
        self.source = source
        self.raw = raw or {}

    def to_dict(self) -> dict:
        return {"name": self.name, "source": self.source,
                "service": dict(self.service), "limits": dict(self.limits),
                "rules": len(self.rules)}


def slugify_profile(name: str) -> str:
    """A profile name reduced to something safe to put in a filename.

    Refuses rather than repairs anything that would leave the config directory: the name
    reaches this from an HTTP parameter, and a profile called ``../../etc/passwd`` must not
    become a path.
    """
    text = str(name or "").strip().lower().replace(" ", "-").replace("_", "-")
    text = _SLUG_OK.sub("", text).strip("-")
    return text[:MAX_SLUG]


def profile_filename(name: str) -> str:
    """The file a profile lives in. The default profile keeps the existing filename."""
    slug = slugify_profile(name)
    if not slug or slug == DEFAULT_PROFILE:
        return BASE_FILENAME
    return "service_phases.%s.json" % slug


def profile_path(config_dir: str, name: str) -> str:
    """The absolute path of a profile's file, always inside ``config_dir``."""
    return os.path.join(config_dir, profile_filename(name))


def list_profiles(config_dir: str) -> List[str]:
    """Every profile this installation has written, newest name order not implied."""
    found: List[str] = []
    try:
        entries = sorted(os.listdir(config_dir))
    except OSError:
        return found
    for entry in entries:
        if entry == BASE_FILENAME:
            found.append(DEFAULT_PROFILE)
        elif entry.startswith("service_phases.") and entry.endswith(".json"):
            middle = entry[len("service_phases."):-len(".json")]
            if middle and middle != "default":
                found.append(middle)
    return found


def _minutes_of_day(text: str) -> Optional[int]:
    match = _TIME.match(str(text or "").strip())
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _entry_matches(entry: Mapping[str, Any], when: "datetime.datetime") -> bool:
    weekdays = entry.get("weekdays")
    if weekdays:
        try:
            wanted = {int(d) for d in weekdays}
        except (TypeError, ValueError):
            return False
        if when.weekday() not in wanted:
            return False
    start = _minutes_of_day(entry.get("from", "")) if entry.get("from") else None
    end = _minutes_of_day(entry.get("to", "")) if entry.get("to") else None
    if entry.get("from") and start is None:
        return False        # a malformed window matches nothing rather than everything
    if entry.get("to") and end is None:
        return False
    minute = when.hour * 60 + when.minute
    if start is not None and minute < start:
        return False
    if end is not None and minute > end:
        return False
    return True


def select_profile(schedule: Sequence[Mapping[str, Any]], when: "datetime.datetime", *,
                   override: str = "", default: str = DEFAULT_PROFILE) -> str:
    """Which profile a service starting at ``when`` should be judged by.

    First match wins, so the schedule reads top to bottom like a run sheet. An override —
    the operator saying this week is not the usual — beats the schedule outright, because
    the one time a person disagrees with the calendar is the time they are right.
    """
    picked = slugify_profile(override)
    if picked:
        return picked
    for entry in schedule or ():
        if not isinstance(entry, Mapping):
            continue
        name = slugify_profile(entry.get("profile", ""))
        if not name:
            continue
        try:
            if _entry_matches(entry, when):
                return name
        except (TypeError, ValueError):
            continue    # one malformed entry costs itself, never the service
    return slugify_profile(default) or DEFAULT_PROFILE


def parse_profile(raw: Mapping[str, Any], name: str, *, source: str = "") -> Profile:
    """One profile file's contents, with anything malformed left out."""
    service = raw.get("service")
    limits = raw.get("limits")
    return Profile(
        name=name,
        rules=parse_rules(dict(raw)),
        service={k: v for k, v in dict(service or {}).items() if not k.startswith("_")},
        limits={k: v for k, v in dict(limits or {}).items() if not k.startswith("_")},
        source=source,
        raw=dict(raw),
    )


def load_profile(config_dir: str, name: str, template_path: str) -> Profile:
    """The named profile, falling back to the base file and then the shipped template.

    Whole-file, like :func:`stt.phase_rules.load_rules` and for the same reason: rules are
    an ordered list where position decides the winner, so splicing shipped rules into an
    operator's list would silently change the answer on an upgrade. The fallback chain is
    what makes a profile safe to try — a name with no file behind it behaves exactly as the
    install did before anyone typed it.
    """
    candidates = [profile_path(config_dir, name)]
    base = os.path.join(config_dir, BASE_FILENAME)
    if base not in candidates:
        candidates.append(base)
    candidates.append(template_path)
    for path in candidates:
        if not path:
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError):
            continue
        profile = parse_profile(raw, slugify_profile(name) or DEFAULT_PROFILE, source=path)
        if profile.rules:
            return profile
    return Profile(slugify_profile(name) or DEFAULT_PROFILE)


def merge_config(base: Mapping[str, Any], profile: Optional[Profile]) -> Dict[str, Any]:
    """The detector settings for one service: the machine's, with the profile's on top.

    The split is deliberate. Bin width, dominance and the dwell settings describe *this
    machine's audio* and stay per-install; how long a sermon runs and how many there are
    describe *this service* and come from the profile. A profile that says nothing changes
    nothing.
    """
    merged = dict(base or {})
    if profile is None:
        return merged
    merged.update(profile.service)
    if profile.limits:
        merged["limits"] = dict(profile.limits)
    return merged


def seed_missing(config_dir: str, name: str, template_path: str) -> Optional[str]:
    """Create a profile from the shipped template, only if it does not exist yet.

    The only write in this module, and it never overwrites. An operator's tuning must not be
    silently replaced by defaults — the rule the word-highlighting config learned the hard
    way — so an existing file is left alone and the caller is told nothing was written.
    """
    path = profile_path(config_dir, name)
    if os.path.exists(path):
        return None
    try:
        os.makedirs(config_dir, exist_ok=True)
        shutil.copyfile(template_path, path)
    except OSError:
        return None
    return path


# Where a learned number belongs. The keys on the left are what stt/phase_learn proposes;
# the right says which rule's threshold they actually are, because the detector reads the
# rule file and not the config block — which is why proposing them into the config block
# was a dead end.
RULE_THRESHOLDS: Dict[str, Tuple[str, str]] = {
    "sermon_min_minutes": ("Sermon", "min_minutes"),
    "songs_min_minutes": ("Songs", "min_minutes"),
    "closing_max_minutes": ("Closing", "max_minutes"),
}


def apply_to_profile(raw: Mapping[str, Any], proposals: Mapping[str, Any],
                     keys: Sequence[str]) -> Dict[str, Any]:
    """A profile with the accepted proposals written into it, as a new dict.

    Pure, like :func:`stt.phase_learn.apply_proposals`, and it writes each number where the
    detector will actually read it: a threshold into the rule that owns it, a cap into
    ``limits``, anything else into ``service``. A proposal the operator did not accept, or
    that the evidence does not support, is ignored.
    """
    out = json.loads(json.dumps(dict(raw or {})))
    for key in keys:
        proposal = proposals.get(key)
        if not isinstance(proposal, Mapping) or not proposal.get("actionable"):
            continue
        value = proposal.get("suggested")
        if key in RULE_THRESHOLDS:
            rule_name, field = RULE_THRESHOLDS[key]
            for phase in out.get("phases") or []:
                if isinstance(phase, dict) and phase.get("name") == rule_name:
                    phase.setdefault("match", {})[field] = value
                    break
        elif key.startswith("max_") and key.endswith("s"):
            name = key[len("max_"):-1].title()
            out.setdefault("limits", {}).setdefault(name, {})["max"] = value
        else:
            out.setdefault("service", {})[key] = value
    return out
