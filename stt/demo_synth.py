"""Generate a service to demonstrate, without recording anyone.

A demo needs a service to replay, and the obvious source — a real one — is
congregation speech: verbatim, often naming people who were in the room. This builds
one instead. The shape is real (phase structure, caption lengths, word timings, the
confidence spread of an ASR model, music during singing) because that is what the UI
and the phase detector read; the words are written, so there is nobody to protect.

Deterministic for a given seed, so a demo build is reproducible and a test can assert
on its output.

Stdlib-only.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The schema a session database has after every migration in the monolith has run.
# Kept in one place so create() and the tests agree on it.
SCHEMA = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, text TEXT, start_time REAL, end_time REAL, confidence REAL,
    original_text TEXT, corrected_by TEXT, needs_review INTEGER,
    translated_text TEXT, translation_language TEXT, speech_type TEXT,
    audio_tag TEXT, music_prob REAL, denied INTEGER, ts_ms INTEGER,
    words_json TEXT, is_final INTEGER DEFAULT 1, partial_seq INTEGER,
    source_language TEXT, segment_id TEXT, words_source TEXT, session_id TEXT,
    denied_reason TEXT, marked INTEGER, translation_ts_ms INTEGER,
    asr_model TEXT, mt_engine TEXT, mt_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_ts_ms ON transcriptions(ts_ms);
CREATE TABLE IF NOT EXISTS session_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS service_phase_bins (
    bin_index INTEGER PRIMARY KEY, start_ms INTEGER, end_ms INTEGER,
    words INTEGER, speech INTEGER, music INTEGER, quiet INTEGER, cues TEXT
);
CREATE TABLE IF NOT EXISTS service_phase_blocks (
    block_index INTEGER PRIMARY KEY, kind TEXT, label TEXT,
    start_ms INTEGER, end_ms INTEGER, start_bin INTEGER, end_bin INTEGER,
    confidence REAL, cues TEXT
);
CREATE TABLE IF NOT EXISTS service_phase_spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT, start_ms INTEGER, end_ms INTEGER, kind TEXT
);
CREATE TABLE IF NOT EXISTS service_phase_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, start_ms INTEGER, end_ms INTEGER,
    label TEXT, created_ms INTEGER
);
CREATE TABLE IF NOT EXISTS sermon_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT, start_ms INTEGER, end_ms INTEGER,
    summary TEXT, created_ms INTEGER
);
"""

ASR_MODEL = "large-v3"
MT_ENGINE = "nllb"
MT_MODEL = "facebook/nllb-200-distilled-1.3B"
SOURCE_LANGUAGE = "ru"
TARGET_LANGUAGE = "en"

# (source, translation) pairs, written for this purpose. Grouped by the part of the
# service they belong to so the phase detector sees a plausible arc.
WELCOME: Tuple[Tuple[str, str], ...] = (
    ("Мир вам, дорогие друзья.", "Peace be with you, dear friends."),
    ("Добро пожаловать на наше сегодняшнее собрание.",
     "Welcome to our gathering today."),
    ("Мы рады, что вы пришли разделить это время с нами.",
     "We are glad you came to share this time with us."),
    ("Давайте начнём с благодарности за прошедшую неделю.",
     "Let us begin with thanks for the week that has passed."),
    ("Пусть это время будет временем покоя и внимания.",
     "May this time be one of quiet and attention."),
)

SINGING: Tuple[Tuple[str, str], ...] = (
    ("Хвалите имя Господне, хвалите, рабы Господни.",
     "Praise the name of the Lord, praise him, servants of the Lord."),
    ("Славьте Его, ибо Он благ, ибо вовек милость Его.",
     "Give thanks to him, for he is good, for his mercy endures forever."),
    ("Воспойте новую песнь, пойте всей землёй.",
     "Sing a new song, sing all the earth."),
    ("Он вывел нас на простор и укрепил наши руки.",
     "He brought us into open space and strengthened our hands."),
)

PRAYER: Tuple[Tuple[str, str], ...] = (
    ("Склоним головы и обратимся в молитве.", "Let us bow our heads and turn to prayer."),
    ("Господи, благодарим Тебя за этот день и за это собрание.",
     "Lord, we thank you for this day and for this gathering."),
    ("Мы просим мудрости для тех, кто принимает трудные решения.",
     "We ask for wisdom for those making difficult decisions."),
    ("Утешь тех, кто сегодня переживает потерю.",
     "Comfort those who are grieving a loss today."),
    ("Мы вверяем Тебе неделю, которая перед нами. Аминь.",
     "We entrust to you the week ahead of us. Amen."),
)

READING: Tuple[Tuple[str, str], ...] = (
    ("Откроем сегодня отрывок о прощении.",
     "Let us open today to the passage about forgiveness."),
    ("И сказал Он им: прощайте, и прощены будете.",
     "And he said to them: forgive, and you will be forgiven."),
    ("Ибо какою мерою мерите, такою же отмерится и вам.",
     "For with the measure you use, it will be measured to you."),
)

SERMON: Tuple[Tuple[str, str], ...] = (
    ("Сегодня я хочу говорить о цене прощения.",
     "Today I want to speak about the cost of forgiveness."),
    ("Мы часто думаем, что простить — значит забыть. Но это не так.",
     "We often think that to forgive means to forget. But that is not so."),
    ("Прощение помнит и всё же отпускает. В этом его тяжесть.",
     "Forgiveness remembers and still lets go. That is what makes it heavy."),
    ("Оно не отменяет случившегося и не делает вид, что раны нет.",
     "It does not undo what happened, nor pretend the wound is absent."),
    ("Оно решает не требовать платы, которую вправе было бы потребовать.",
     "It decides not to demand a payment it would have every right to demand."),
    ("Именно поэтому прощение всегда стоит кому-то дорого.",
     "That is precisely why forgiveness always costs someone dearly."),
    ("Подумайте о человеке, которого вам труднее всего простить.",
     "Think of the person you find hardest to forgive."),
    ("Возможно, первым шагом будет не чувство, а решение.",
     "Perhaps the first step is not a feeling, but a decision."),
    ("Мы не начинаем с того, что нам легко. Мы начинаем с того, что верно.",
     "We do not begin with what is easy. We begin with what is right."),
    ("И обнаруживаем, что отпустив, мы сами оказываемся свободнее.",
     "And we discover that in letting go, we ourselves become freer."),
    ("Это не быстрая работа. Она занимает годы, и это нормально.",
     "This is not quick work. It takes years, and that is all right."),
    ("Но начинается она сегодня, здесь, с одного решения.",
     "But it begins today, here, with a single decision."),
)

CLOSING: Tuple[Tuple[str, str], ...] = (
    ("Встанем для заключительного благословения.",
     "Let us stand for the closing benediction."),
    ("Идите с миром, и пусть милость идёт с вами.",
     "Go in peace, and may mercy go with you."),
    ("На следующей неделе мы собираемся в обычное время.",
     "Next week we gather at the usual time."),
    ("Чай и кофе накрыты в соседнем зале. До встречи.",
     "Tea and coffee are laid out in the next room. See you then."),
)


class Section:
    """A stretch of the service: what is said, how fast, and whether music is playing."""

    __slots__ = ("gap_s", "label", "lines", "music", "repeats")

    def __init__(self, label: str, lines: Sequence[Tuple[str, str]], repeats: int = 1,
                 gap_s: float = 1.4, music: bool = False) -> None:
        self.label = label
        self.lines = lines
        self.repeats = repeats
        self.gap_s = gap_s
        self.music = music


def default_script() -> List[Section]:
    """The arc of an ordinary service, long enough to contain a real sermon block."""
    # Repeat counts are chosen so the sermon runs past the eight minutes the
    # summariser needs before it will treat a stretch as a sermon at all.
    return [
        Section("Opening", WELCOME, repeats=3, gap_s=1.6),
        Section("Singing", SINGING, repeats=5, gap_s=1.0, music=True),
        Section("Prayer", PRAYER, repeats=3, gap_s=1.8),
        Section("Reading", READING, repeats=3, gap_s=1.5),
        Section("Sermon", SERMON, repeats=9, gap_s=1.2),
        Section("Singing", SINGING, repeats=4, gap_s=1.0, music=True),
        Section("Closing", CLOSING, repeats=2, gap_s=1.6),
    ]


def _speaking_seconds(text: str, rng: random.Random) -> float:
    """Roughly how long a line takes to say, at a measured speaking pace."""
    words = max(len(text.split()), 1)
    return max(words / rng.uniform(2.2, 3.0), 0.8)


def build_words_json(text: str, start_s: float, end_s: float,
                     rng: random.Random) -> str:
    """Word timings spread across the line, in the shape the live preview reads."""
    tokens = text.split()
    if not tokens:
        return "[]"
    start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
    span = max(end_ms - start_ms, len(tokens))
    step = span / len(tokens)
    words: List[Dict[str, Any]] = []
    cursor = float(start_ms)
    for token in tokens:
        word_end = cursor + step * rng.uniform(0.75, 0.95)
        words.append({
            "w": " " + token,
            "s_ms": int(cursor),
            "e_ms": int(word_end),
            "c": round(rng.uniform(0.62, 0.99), 4),
        })
        cursor += step
    return json.dumps(words, ensure_ascii=False)


def generate_rows(script: Optional[Sequence[Section]] = None, seed: int = 20260823,
                  started_at: Optional[float] = None) -> List[Dict[str, Any]]:
    """Every caption of a generated service, ready to insert."""
    rng = random.Random(seed)
    sections = list(script if script is not None else default_script())
    origin = float(started_at if started_at is not None else 1_756_000_000)

    rows: List[Dict[str, Any]] = []
    cursor_s = 0.0
    row_id = 1
    for section in sections:
        for _repeat in range(section.repeats):
            for source, translation in section.lines:
                duration = _speaking_seconds(source, rng)
                start_s = cursor_s
                end_s = start_s + duration
                ts_ms = int((origin + end_s) * 1000)
                rows.append({
                    "id": row_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S",
                                               time.localtime(origin + end_s)),
                    "text": source,
                    "start_time": round(start_s, 3),
                    "end_time": round(end_s, 3),
                    "confidence": round(rng.uniform(0.72, 0.98), 4),
                    "original_text": None,
                    "corrected_by": None,
                    "needs_review": 0,
                    "translated_text": translation,
                    "translation_language": TARGET_LANGUAGE,
                    "speech_type": "Music" if section.music else "Speaking",
                    "audio_tag": "Music" if section.music else "Speech",
                    "music_prob": round(rng.uniform(0.62, 0.94) if section.music
                                        else rng.uniform(0.01, 0.12), 4),
                    "denied": 0,
                    "ts_ms": ts_ms,
                    "words_json": build_words_json(source, start_s, end_s, rng),
                    "is_final": 1,
                    "partial_seq": None,
                    "source_language": SOURCE_LANGUAGE,
                    "segment_id": str(row_id),
                    "words_source": "whisper",
                    "session_id": None,
                    "denied_reason": None,
                    "marked": 0,
                    "translation_ts_ms": ts_ms,
                    "asr_model": ASR_MODEL,
                    "mt_engine": MT_ENGINE,
                    "mt_model": MT_MODEL,
                })
                row_id += 1
                cursor_s = end_s + section.gap_s * rng.uniform(0.7, 1.5)
            # A breath between repeats, so the timeline is not perfectly regular.
            cursor_s += rng.uniform(1.0, 3.0)
    return rows


def duration_minutes(rows: Sequence[Dict[str, Any]]) -> float:
    """How long the generated service runs."""
    if not rows:
        return 0.0
    return (rows[-1]["ts_ms"] - rows[0]["ts_ms"]) / 60000.0


def write(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    """Write the generated service as a session database the demo can replay."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        if rows:
            columns = list(rows[0].keys())
            conn.executemany(
                f"INSERT INTO transcriptions ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                [tuple(row[name] for name in columns) for row in rows])
        started = rows[0]["ts_ms"] / 1000.0 if rows else time.time()
        conn.executemany(
            "INSERT OR REPLACE INTO session_meta (key, value) VALUES (?, ?)",
            [
                ("session_started",
                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))),
                ("asr_model", ASR_MODEL),
                ("mt_engine", MT_ENGINE),
                ("mt_model", MT_MODEL),
                ("source_language", SOURCE_LANGUAGE),
                ("target_language", TARGET_LANGUAGE),
                ("synthetic", "1"),
            ])
        conn.commit()
    finally:
        conn.close()
    return path


def generate(path: str, seed: int = 20260823,
             script: Optional[Sequence[Section]] = None) -> str:
    """Generate a service and write it to ``path``."""
    return write(path, generate_rows(script=script, seed=seed))
