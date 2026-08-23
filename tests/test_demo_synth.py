"""Generating a service to demonstrate, without recording anyone."""

from __future__ import annotations

import json
import sqlite3

from stt import demo_playback, demo_synth


def test_the_same_seed_gives_the_same_service():
    first = demo_synth.generate_rows(seed=7)
    second = demo_synth.generate_rows(seed=7)

    assert [row["text"] for row in first] == [row["text"] for row in second]
    assert [row["ts_ms"] for row in first] == [row["ts_ms"] for row in second]
    assert [row["confidence"] for row in first] == [row["confidence"] for row in second]


def test_a_different_seed_gives_different_timings():
    first = demo_synth.generate_rows(seed=1)
    second = demo_synth.generate_rows(seed=2)

    assert [row["ts_ms"] for row in first] != [row["ts_ms"] for row in second]


def test_captions_run_forwards_and_do_not_overlap():
    rows = demo_synth.generate_rows()

    stamps = [row["ts_ms"] for row in rows]
    assert stamps == sorted(stamps)
    for earlier, later in zip(rows, rows[1:]):
        assert later["start_time"] >= earlier["end_time"]


def test_every_caption_has_a_translation():
    """Otherwise the replay would try to translate live, which a demo cannot do."""
    rows = demo_synth.generate_rows()

    assert rows
    assert all(row["translated_text"] for row in rows)


def test_singing_is_marked_as_music_and_speech_is_not():
    rows = demo_synth.generate_rows()

    music = [row for row in rows if row["speech_type"] == "Music"]
    speech = [row for row in rows if row["speech_type"] == "Speaking"]

    assert music and speech
    assert all(row["music_prob"] > 0.5 for row in music)
    assert all(row["music_prob"] < 0.5 for row in speech)


def test_the_service_is_long_enough_to_contain_a_sermon():
    """The summariser ignores a stretch shorter than eight minutes."""
    rows = demo_synth.generate_rows()
    sermon_lines = dict(demo_synth.SERMON)
    sermon = [row for row in rows if row["text"] in sermon_lines]

    assert sermon
    assert (sermon[-1]["ts_ms"] - sermon[0]["ts_ms"]) / 60000.0 > 8.0


def test_word_timings_cover_the_caption_and_stay_inside_it():
    rows = demo_synth.generate_rows()
    row = next(r for r in rows if len(r["text"].split()) > 3)

    words = json.loads(row["words_json"])

    assert len(words) == len(row["text"].split())
    assert words[0]["s_ms"] >= int(row["start_time"] * 1000) - 1
    assert words[-1]["e_ms"] <= int(row["end_time"] * 1000) + 1
    assert all(w["s_ms"] <= w["e_ms"] for w in words)
    assert all(0.0 <= w["c"] <= 1.0 for w in words)


def test_word_timings_of_an_empty_line_are_empty():
    assert demo_synth.build_words_json("", 0.0, 1.0, __import__("random").Random(1)) == "[]"


def test_confidences_look_like_a_model_rather_than_a_constant():
    rows = demo_synth.generate_rows()

    values = {row["confidence"] for row in rows}

    assert len(values) > 10
    assert all(0.5 < value <= 1.0 for value in values)


# --- as a database ---------------------------------------------------------


def test_the_written_database_replays(tmp_path):
    path = demo_synth.generate(str(tmp_path / "demo.db"))

    schedule = demo_playback.load_schedule(path)

    assert len(schedule) == len(demo_synth.generate_rows())
    assert schedule[0].offset_s == 0.0
    assert schedule[0].words_json


def test_the_written_database_has_the_tables_the_server_expects(tmp_path):
    path = demo_synth.generate(str(tmp_path / "demo.db"))

    conn = sqlite3.connect(path)
    try:
        tables = {row[0] for row in
                  conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()

    assert {"transcriptions", "session_meta", "service_phase_bins",
            "service_phase_blocks", "sermon_summaries"} <= tables


def test_the_written_database_records_what_produced_it(tmp_path):
    path = demo_synth.generate(str(tmp_path / "demo.db"))

    conn = sqlite3.connect(path)
    try:
        meta = dict(conn.execute("SELECT key, value FROM session_meta").fetchall())
    finally:
        conn.close()

    assert meta["synthetic"] == "1"
    assert meta["asr_model"] == demo_synth.ASR_MODEL


def test_writing_twice_replaces_rather_than_appends(tmp_path):
    path = str(tmp_path / "demo.db")
    demo_synth.generate(path)
    demo_synth.generate(path)

    conn = sqlite3.connect(path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0]
    finally:
        conn.close()

    assert count == len(demo_synth.generate_rows())


def test_a_custom_script_is_honoured(tmp_path):
    script = [demo_synth.Section("Opening", (("Привет", "Hello"),), repeats=2)]

    rows = demo_synth.generate_rows(script=script)

    assert len(rows) == 2
    assert all(row["text"] == "Привет" for row in rows)


def test_an_empty_script_produces_an_empty_service(tmp_path):
    path = demo_synth.generate(str(tmp_path / "empty.db"), script=[])

    assert demo_synth.generate_rows(script=[]) == []
    assert demo_playback.load_schedule(path) == []
    assert demo_synth.duration_minutes([]) == 0.0
