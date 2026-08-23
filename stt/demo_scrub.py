"""Turn a recorded service into one that can leave the building.

A session database is verbatim congregation speech. It names people who were in the
room, and it is the reason recordings are kept out of this repository at all. This
module reduces the risk of showing one to a stranger; it does not remove it, and it
cannot certify anything. Nothing it produces should go anywhere until a human has
read the review file it writes, end to end.

The safer option is ``stt/demo_synth.py``, which invents a service instead. Prefer it
for anything published. This exists for the case where a live demo genuinely needs to
show real captions to someone in the room.

What is preserved matters as much as what is removed: timings, word-level stamps,
music probabilities, confidences and phase blocks all stay, because those are what
make the replay look like a service rather than a text dump. Only the words change.

Stdlib-only.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

from stt import demo_redact

# Stand-ins drawn on in order, so the same original always becomes the same
# replacement within a recording and the transcript stays internally coherent.
SUBSTITUTE_NAMES: Tuple[str, ...] = (
    "Alex", "Sam", "Jordan", "Casey", "Riley", "Morgan", "Avery", "Quinn",
    "Rowan", "Sasha", "Robin", "Emery", "Finley", "Harper", "Kai", "Noor",
)

# Text columns rewritten. words_json is handled separately because its tokens must
# stay aligned with the text they came from.
TEXT_COLUMNS: Tuple[str, ...] = ("text", "original_text", "translated_text")

# Cleared outright: who made a correction is a name, and a session's provenance
# names the machine it ran on.
CLEARED_COLUMNS: Tuple[str, ...] = ("corrected_by",)

_DIGIT_RUN = re.compile(r"\b\d[\d\-\s()]{5,}\d\b")
# A run of capitalised words, which is where an unlisted name hides. Latin and
# Cyrillic both, since services here are transcribed in Russian.
_CAPITALISED_RUN = re.compile(
    r"\b[A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)+\b")
# Russian patronymics, which are always a personal name and never anything else.
_PATRONYMIC = re.compile(r"\b[А-ЯЁ][а-яё]+(?:ович|овна|евич|евна|ична|инична)\b")
# Vocative openings: "brother X", "sister X" — how a congregation names someone.
_VOCATIVE = re.compile(r"\b(?:брат|сестра|брата|сестру|братья|сёстры|"
                       r"brother|sister)\s+([A-ZА-ЯЁ][a-zа-яё]+)")


class ScrubRule:
    """One replacement, and what to call it in the report."""

    __slots__ = ("label", "pattern", "replacement")

    def __init__(self, pattern: Pattern[str], replacement: str, label: str) -> None:
        self.pattern = pattern
        self.replacement = replacement
        self.label = label


class ScrubReport:
    """What the scrub did, and what it is not sure about."""

    def __init__(self) -> None:
        self.rows_in = 0
        self.rows_out = 0
        self.rows_dropped = 0
        self.words_json_dropped = 0
        self.replacements: Dict[str, int] = {}
        self.residual_flags: List[Tuple[int, str, str]] = []
        self.changes: List[Tuple[int, str, str]] = []

    def count(self, label: str, times: int = 1) -> None:
        if times:
            self.replacements[label] = self.replacements.get(label, 0) + times


def build_rules(names: Sequence[str],
                extra: Sequence[Tuple[str, str]] = ()) -> List[ScrubRule]:
    """A replacement per name, plus whatever else the operator wants rewritten.

    Whole-word and case-insensitive, but longest-first: replacing "Ann" before
    "Anna" would leave "a" behind in the middle of the longer name.
    """
    rules: List[ScrubRule] = []
    for index, name in enumerate(sorted({n.strip() for n in names if n.strip()},
                                        key=len, reverse=True)):
        rules.append(ScrubRule(
            pattern=re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE),
            replacement=SUBSTITUTE_NAMES[index % len(SUBSTITUTE_NAMES)],
            label=f"name:{name}",
        ))
    for pattern, replacement in extra:
        rules.append(ScrubRule(re.compile(pattern, re.IGNORECASE), replacement,
                               label=f"custom:{pattern}"))
    # Anything that looks like a phone number or an account reference.
    rules.append(ScrubRule(_DIGIT_RUN, "[number]", label="digits"))
    return rules


def scrub_text(text: Optional[str],
               rules: Sequence[ScrubRule]) -> Tuple[Optional[str], Dict[str, int]]:
    """The line with every rule applied, and how many times each one fired."""
    if not text:
        return text, {}
    counts: Dict[str, int] = {}
    result = text
    for rule in rules:
        result, hits = rule.pattern.subn(rule.replacement, result)
        if hits:
            counts[rule.label] = counts.get(rule.label, 0) + hits
    return result, counts


def scrub_words_json(words_json: Optional[str], original: str,
                     scrubbed: str) -> Tuple[Optional[str], bool]:
    """Word timings that still line up with the rewritten line.

    Returns ``(value, dropped)``. When the rewrite changes how many words there are,
    the timings can no longer be trusted and are dropped rather than guessed —
    a mis-stamped word makes the live preview reveal at the wrong moment, which looks
    like a bug rather than like a redaction.
    """
    if not words_json:
        return words_json, False
    if original == scrubbed:
        return words_json, False

    import json

    try:
        words = json.loads(words_json)
    except (ValueError, TypeError):
        return None, True
    if not isinstance(words, list):
        return None, True

    original_tokens = original.split()
    scrubbed_tokens = scrubbed.split()
    if len(original_tokens) != len(scrubbed_tokens) or len(words) != len(original_tokens):
        return None, True

    # Same token count: each word keeps its own start and end, only its text changes.
    rebuilt = []
    for word, token in zip(words, scrubbed_tokens):
        if not isinstance(word, dict):
            return None, True
        replacement = dict(word)
        leading = " " if str(word.get("w", "")).startswith(" ") else ""
        replacement["w"] = leading + token
        rebuilt.append(replacement)
    return json.dumps(rebuilt, ensure_ascii=False), False


def flag_residuals(text: Optional[str]) -> List[str]:
    """What still looks like it might identify someone.

    Deliberately noisy — a false positive costs a moment's reading, and a false
    negative is the whole problem.
    """
    if not text:
        return []
    flags: List[str] = []
    for match in _PATRONYMIC.finditer(text):
        flags.append(f"patronymic {match.group(0)!r}")
    for match in _VOCATIVE.finditer(text):
        flags.append(f"named directly: {match.group(0)!r}")
    for match in _CAPITALISED_RUN.finditer(text):
        flags.append(f"capitalised run {match.group(0)!r}")
    flags.extend(demo_redact.residual_flags(text))
    return flags


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def scrub_session(src: str, dst: str, rules: Sequence[ScrubRule], *,
                  window: Optional[Tuple[int, int]] = None,
                  require_translation: bool = True) -> ScrubReport:
    """Write a scrubbed copy of ``src`` to ``dst``. ``src`` is never opened for write."""
    report = ScrubReport()
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(src, dst)

    conn = sqlite3.connect(dst)
    conn.row_factory = sqlite3.Row
    try:
        columns = _table_columns(conn, "transcriptions")
        rows = conn.execute("SELECT * FROM transcriptions ORDER BY id").fetchall()
        report.rows_in = len(rows)

        # Anchored on the first caption that would actually be replayed, not the first
        # row of any kind. A recording routinely opens with a long stretch of partials
        # and hallucinations from before the service began; anchoring on those made a
        # 30-minute excerpt contain no service at all.
        replayable = [{name: row[name] for name in columns} for row in rows
                      if _is_replayable(row)]
        first_ms = int(replayable[0]["ts_ms"]) if replayable else None
        report.rows_dropped = len(rows) - len(replayable)

        keep: List[Dict[str, Any]] = []
        for values in replayable:
            if not _in_window(values, first_ms, window):
                report.rows_dropped += 1
                continue
            if (require_translation
                    and not str(values.get("translated_text") or "").strip()):
                report.rows_dropped += 1
                continue
            keep.append(_scrub_row(values, rules, report))

        conn.execute("DELETE FROM transcriptions")
        if keep:
            insert_columns = list(keep[0].keys())
            conn.executemany(
                f"INSERT INTO transcriptions ({', '.join(insert_columns)}) "
                f"VALUES ({', '.join('?' for _ in insert_columns)})",
                [tuple(item[name] for name in insert_columns) for item in keep])
        report.rows_out = len(keep)

        _scrub_session_meta(conn, report)
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    return report


def _is_replayable(row: Any) -> bool:
    """Whether the demo would ever show this row.

    The same filter :func:`stt.demo_playback.load_schedule` applies. Rows it would
    skip — superseded partials, hidden hallucinations, the blank marker a session
    opens with — are dropped rather than scrubbed: they are never displayed, and
    carrying them would only bulk out a file that has to ship.
    """
    return bool(row["is_final"]
                and not (row["denied"] or 0)
                and row["ts_ms"] is not None
                and str(row["text"] or "").strip())


def _in_window(values: Dict[str, Any], first_ms: Optional[int],
               window: Optional[Tuple[int, int]]) -> bool:
    if window is None or first_ms is None or values.get("ts_ms") is None:
        return True
    offset = int(values["ts_ms"]) - first_ms
    return window[0] <= offset <= window[1]


def _scrub_row(values: Dict[str, Any], rules: Sequence[ScrubRule],
               report: ScrubReport) -> Dict[str, Any]:
    row_id = int(values.get("id") or 0)
    original_text = str(values.get("text") or "")

    for column in TEXT_COLUMNS:
        if column not in values:
            continue
        before = values[column]
        after, counts = scrub_text(before, rules)
        for label, hits in counts.items():
            report.count(label, hits)
        if after != before:
            report.changes.append((row_id, str(before), str(after)))
        values[column] = after

    if "words_json" in values:
        rebuilt, dropped = scrub_words_json(
            values.get("words_json"), original_text, str(values.get("text") or ""))
        values["words_json"] = rebuilt
        if dropped:
            report.words_json_dropped += 1

    for column in CLEARED_COLUMNS:
        if column in values:
            values[column] = None

    for flag in flag_residuals(values.get("text")):
        report.residual_flags.append((row_id, "text", flag))

    return values


# session_meta keys that name the machine or the operator. Short names that the
# general deny-list in demo_redact does not match ("host" is not "hostname"), and a
# session's provenance is exactly where they appear.
_META_MACHINE_KEYS: Tuple[str, ...] = (
    "host", "machine", "node", "device", "user", "peer", "operator", "serial",
)


def _scrub_session_meta(conn: sqlite3.Connection, report: ScrubReport) -> None:
    """Replace the provenance that names the machine the service ran on."""
    try:
        rows = conn.execute("SELECT key, value FROM session_meta").fetchall()
    except sqlite3.Error:
        return
    for row in rows:
        key, value = str(row["key"]), row["value"]
        if not isinstance(value, str):
            continue
        if any(part in key.lower() for part in _META_MACHINE_KEYS):
            cleaned = demo_redact.PLACEHOLDER_HOST
        else:
            cleaned = demo_redact.redact({key: value})[key]
        if cleaned != value:
            conn.execute("UPDATE session_meta SET value = ? WHERE key = ?", (cleaned, key))
            report.count("session_meta")


def write_review(path: str, report: ScrubReport) -> str:
    """Write the file a human reads before deciding this recording may travel."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Scrub review\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"captions in:      {report.rows_in}\n")
        handle.write(f"captions kept:    {report.rows_out}\n")
        handle.write(f"captions dropped: {report.rows_dropped}\n")
        handle.write(f"word timings dropped: {report.words_json_dropped}\n\n")

        handle.write("replacements\n" + "-" * 60 + "\n")
        for label, count in sorted(report.replacements.items()):
            handle.write(f"  {label}: {count}\n")
        if not report.replacements:
            handle.write("  (none)\n")

        handle.write(f"\nresidual flags ({len(report.residual_flags)})\n" + "-" * 60 + "\n")
        handle.write("Each of these may be a name the scrub did not know about.\n\n")
        for row_id, column, flag in report.residual_flags:
            handle.write(f"  row {row_id} [{column}] {flag}\n")
        if not report.residual_flags:
            handle.write("  (none)\n")

        handle.write(f"\nchanged captions ({len(report.changes)})\n" + "-" * 60 + "\n")
        for row_id, before, after in report.changes:
            handle.write(f"  row {row_id}\n    - {before}\n    + {after}\n")

        handle.write("\n" + "=" * 60 + "\n")
        handle.write("This report says what the scrubber knew to change. It cannot say\n"
                     "what it did not know about. Read the captions themselves before\n"
                     "this recording goes anywhere.\n")
    return path
