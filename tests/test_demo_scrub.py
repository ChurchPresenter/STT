"""Turning a recorded service into one that can leave the building.

Every caption here is constructed. The real shapes this was written against are
described in prose: a service where the speaker greets people by first name, where
Russian patronymics appear in address, and where the transcript occasionally carries
a phone number read aloud from the announcements.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest

from stt import demo_scrub, demo_synth


@pytest.fixture()
def rules():
    return demo_scrub.build_rules(["Мария", "Пётр"])


def make_session(path, rows):
    """A session database holding ``rows`` of (text, translated_text, words_json, ts_ms)."""
    conn = sqlite3.connect(str(path))
    conn.executescript(demo_synth.SCHEMA)
    for index, (text, translated, words_json, ts_ms) in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO transcriptions (id, text, original_text, translated_text, "
            "words_json, ts_ms, start_time, end_time, is_final, denied, confidence, "
            "music_prob, speech_type, corrected_by, segment_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0, 0.9, 0.02, 'Speaking', 'operator', ?)",
            (index, text, text, translated, words_json, ts_ms,
             index * 3.0, index * 3.0 + 2.5, str(index)))
    conn.execute("INSERT INTO session_meta (key, value) VALUES ('host', 'MacStudio-AV')")
    conn.commit()
    conn.close()
    return str(path)


def read_rows(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT id, text, translated_text, words_json, ts_ms, corrected_by, "
            "confidence, music_prob FROM transcriptions ORDER BY id").fetchall()
    finally:
        conn.close()


# --- replacement rules -----------------------------------------------------


def test_a_listed_name_is_replaced_wherever_it_appears(rules):
    cleaned, counts = demo_scrub.scrub_text("Спасибо, Мария, за музыку.", rules)

    assert "Мария" not in cleaned
    assert counts["name:Мария"] == 1


def test_replacement_is_whole_word_only():
    rules = demo_scrub.build_rules(["Ann"])

    cleaned, _ = demo_scrub.scrub_text("Announcements follow", rules)

    assert cleaned == "Announcements follow"


def test_a_longer_name_is_replaced_before_the_shorter_one_inside_it():
    """Replacing "Анна" first would leave the tail of "Анастасия" behind."""
    rules = demo_scrub.build_rules(["Анна", "Анна-Мария"])

    cleaned, _ = demo_scrub.scrub_text("Анна-Мария и Анна", rules)

    assert "Анна" not in cleaned


def test_the_same_name_always_becomes_the_same_substitute(rules):
    first, _ = demo_scrub.scrub_text("Мария сказала", rules)
    second, _ = demo_scrub.scrub_text("потом Мария ушла", rules)

    assert first.split()[0] == second.split()[1]


def test_a_number_read_aloud_is_removed():
    rules = demo_scrub.build_rules([])

    cleaned, counts = demo_scrub.scrub_text("Телефон 555 123 4567 для записи", rules)

    assert "555" not in cleaned
    assert counts["digits"] == 1


def test_scrubbing_nothing_returns_nothing(rules):
    assert demo_scrub.scrub_text("", rules) == ("", {})
    assert demo_scrub.scrub_text(None, rules) == (None, {})


# --- word timings ----------------------------------------------------------


def _words(tokens, start_ms=1000, step=400):
    return json.dumps([{"w": " " + token, "s_ms": start_ms + i * step,
                        "e_ms": start_ms + i * step + 300, "c": 0.9}
                       for i, token in enumerate(tokens)], ensure_ascii=False)


def test_word_timings_survive_a_same_length_replacement():
    words = _words(["Спасибо,", "Мария,", "за", "музыку."])

    rebuilt, dropped = demo_scrub.scrub_words_json(
        words, "Спасибо, Мария, за музыку.", "Спасибо, Alex, за музыку.")

    assert dropped is False
    parsed = json.loads(rebuilt)
    original = json.loads(words)
    assert [w["s_ms"] for w in parsed] == [w["s_ms"] for w in original]
    assert [w["e_ms"] for w in parsed] == [w["e_ms"] for w in original]
    assert parsed[1]["w"] == " Alex,"


def test_word_timings_are_dropped_rather_than_guessed_when_the_count_changes():
    """A mis-stamped word makes the live preview reveal at the wrong moment."""
    words = _words(["Телефон", "555", "123", "4567"])

    rebuilt, dropped = demo_scrub.scrub_words_json(
        words, "Телефон 555 123 4567", "Телефон [number]")

    assert dropped is True
    assert rebuilt is None


def test_word_timings_are_left_alone_when_nothing_changed():
    words = _words(["ничего", "не", "изменилось"])

    rebuilt, dropped = demo_scrub.scrub_words_json(words, "a b c", "a b c")

    assert rebuilt == words
    assert dropped is False


def test_unparseable_word_timings_are_dropped():
    rebuilt, dropped = demo_scrub.scrub_words_json("not json", "before", "after")

    assert rebuilt is None and dropped is True


def test_the_leading_space_convention_is_preserved():
    words = json.dumps([{"w": "Мария", "s_ms": 0, "e_ms": 100}])

    rebuilt, _ = demo_scrub.scrub_words_json(words, "Мария", "Alex")

    assert json.loads(rebuilt)[0]["w"] == "Alex"


# --- residual flags --------------------------------------------------------


def test_a_patronymic_is_flagged_as_a_personal_name():
    flags = demo_scrub.flag_residuals("Спасибо, Иван Петрович, за слово.")

    assert any("patronymic" in flag for flag in flags)


def test_being_addressed_as_brother_or_sister_is_flagged():
    flags = demo_scrub.flag_residuals("Приветствуем брата Николая среди нас.")

    assert any("named directly" in flag for flag in flags)


def test_a_run_of_capitalised_words_is_flagged():
    flags = demo_scrub.flag_residuals("Today Sarah Wilson leads the singing.")

    assert any("capitalised run" in flag for flag in flags)


def test_ordinary_text_raises_nothing():
    assert demo_scrub.flag_residuals("Давайте склоним головы в молитве.") == []
    assert demo_scrub.flag_residuals("") == []


# --- the whole database ----------------------------------------------------


def test_the_source_recording_is_never_written_to(tmp_path, rules):
    src = make_session(tmp_path / "src.db", [("Спасибо, Мария.", "Thank you, Maria.", None, 1000)])
    digest = hashlib.sha256(open(src, "rb").read()).hexdigest()

    demo_scrub.scrub_session(src, str(tmp_path / "out.db"), rules)

    assert hashlib.sha256(open(src, "rb").read()).hexdigest() == digest
    assert not os.path.exists(src + "-wal")


def test_captions_are_rewritten_in_place(tmp_path, rules):
    src = make_session(tmp_path / "src.db", [
        ("Спасибо, Мария.", "Thank you, Мария.", None, 1000),
    ])
    out = str(tmp_path / "out.db")

    demo_scrub.scrub_session(src, out, rules)

    row = read_rows(out)[0]
    assert "Мария" not in row[1]
    assert "Мария" not in row[2]


def test_timings_and_signals_are_preserved(tmp_path, rules):
    """These are what make the replay look like a service rather than a text dump."""
    src = make_session(tmp_path / "src.db", [
        ("Спасибо, Мария.", "Thanks.", None, 1_700_000),
    ])
    out = str(tmp_path / "out.db")

    demo_scrub.scrub_session(src, out, rules)

    row = read_rows(out)[0]
    assert row[4] == 1_700_000     # ts_ms
    assert row[6] == 0.9           # confidence
    assert row[7] == 0.02          # music_prob


def test_who_corrected_a_line_is_cleared(tmp_path, rules):
    src = make_session(tmp_path / "src.db", [("Привет.", "Hello.", None, 1000)])
    out = str(tmp_path / "out.db")

    demo_scrub.scrub_session(src, out, rules)

    assert read_rows(out)[0][5] is None


def test_the_machine_a_service_ran_on_is_no_longer_named(tmp_path, rules):
    src = make_session(tmp_path / "src.db", [("Привет.", "Hello.", None, 1000)])
    out = str(tmp_path / "out.db")

    demo_scrub.scrub_session(src, out, rules)

    conn = sqlite3.connect(out)
    try:
        host = conn.execute("SELECT value FROM session_meta WHERE key='host'").fetchone()[0]
    finally:
        conn.close()

    assert host != "MacStudio-AV"


def test_a_caption_with_no_translation_is_dropped(tmp_path, rules):
    """Otherwise the replay would try to translate it live, which a demo cannot do."""
    src = make_session(tmp_path / "src.db", [
        ("Есть перевод.", "Has a translation.", None, 1000),
        ("Нет перевода.", "", None, 2000),
    ])
    out = str(tmp_path / "out.db")

    report = demo_scrub.scrub_session(src, out, rules)

    assert report.rows_out == 1
    assert report.rows_dropped == 1


def test_untranslated_captions_can_be_kept_on_request(tmp_path, rules):
    src = make_session(tmp_path / "src.db", [("Нет перевода.", "", None, 2000)])
    out = str(tmp_path / "out.db")

    report = demo_scrub.scrub_session(src, out, rules, require_translation=False)

    assert report.rows_out == 1


def test_a_window_keeps_only_the_excerpt_asked_for(tmp_path, rules):
    src = make_session(tmp_path / "src.db", [
        ("Первое.", "First.", None, 1_000_000),
        ("Второе.", "Second.", None, 1_030_000),     # +30s
        ("Третье.", "Third.", None, 1_600_000),      # +600s
    ])
    out = str(tmp_path / "out.db")

    report = demo_scrub.scrub_session(src, out, rules, window=(0, 60_000))

    assert report.rows_out == 2
    assert report.rows_dropped == 1


def test_the_report_counts_what_it_changed(tmp_path, rules):
    src = make_session(tmp_path / "src.db", [
        ("Спасибо, Мария.", "Thanks.", None, 1000),
        ("И Пётр тоже.", "And Peter too.", None, 2000),
    ])
    out = str(tmp_path / "out.db")

    report = demo_scrub.scrub_session(src, out, rules)

    assert report.rows_in == 2
    # Two hits per name: these fixtures store the caption in both text and
    # original_text, and both columns are rewritten.
    assert report.replacements["name:Мария"] == 2
    assert report.replacements["name:Пётр"] == 2
    assert len(report.changes) == 4


def test_the_report_records_what_it_is_unsure_about(tmp_path):
    src = make_session(tmp_path / "src.db", [
        ("Спасибо, Иван Петрович.", "Thanks.", None, 1000),
    ])
    out = str(tmp_path / "out.db")

    report = demo_scrub.scrub_session(src, out, demo_scrub.build_rules([]))

    assert report.residual_flags
    assert any("patronymic" in flag for _, _, flag in report.residual_flags)


def test_the_scrubbed_recording_still_replays(tmp_path, rules):
    from stt import demo_playback

    src = make_session(tmp_path / "src.db", [
        ("Спасибо, Мария.", "Thank you.", _words(["Спасибо,", "Мария."]), 1000),
        ("Начнём молитву.", "Let us pray.", None, 4000),
    ])
    out = str(tmp_path / "out.db")

    demo_scrub.scrub_session(src, out, rules)
    schedule = demo_playback.load_schedule(out)

    assert len(schedule) == 2
    assert schedule[0].offset_s == 0.0
    assert "Мария" not in schedule[0].text


# --- the review file -------------------------------------------------------


def test_the_review_file_says_what_changed_and_what_is_uncertain(tmp_path, rules):
    src = make_session(tmp_path / "src.db", [
        ("Спасибо, Мария.", "Thanks.", None, 1000),
        ("Слово брату Николаю.", "A word to brother Nikolai.", None, 2000),
    ])
    out = str(tmp_path / "out.db")
    report = demo_scrub.scrub_session(src, out, rules)

    review = demo_scrub.write_review(str(tmp_path / "out.review.txt"), report)
    text = open(review, encoding="utf-8").read()

    assert "Спасибо, Мария." in text          # the before, so a reader can judge
    assert "residual flags" in text
    assert "captions kept" in text
    # It must not claim to have certified anything.
    assert "cannot say" in text


def test_the_excerpt_window_starts_at_the_first_real_caption(tmp_path, rules):
    """A recording opens before the service does.

    Real sessions begin with a long stretch of partials and hallucinated filler from
    the minutes before anyone speaks. Anchoring the window on the first row of any
    kind made a thirty-minute excerpt contain none of the service.
    """
    src = make_session(tmp_path / "src.db", [
        ("Настоящее начало.", "The real beginning.", None, 2_000_000),
        ("Дальше по службе.", "Further into the service.", None, 2_060_000),  # +60s
    ])
    conn = sqlite3.connect(src)
    # Half an hour of partials before the first caption, as a real recording has.
    for index, ts in enumerate(range(200_000, 2_000_000, 100_000), start=100):
        conn.execute("INSERT INTO transcriptions (id, text, ts_ms, is_final, denied) "
                     "VALUES (?, 'Продолжение следует...', ?, 0, 0)", (index, ts))
    conn.commit()
    conn.close()
    out = str(tmp_path / "out.db")

    report = demo_scrub.scrub_session(src, out, rules, window=(0, 120_000))

    assert report.rows_out == 2
    assert [row[1] for row in read_rows(out)] == ["Настоящее начало.", "Дальше по службе."]


def test_partials_and_hidden_rows_are_dropped_rather_than_scrubbed(tmp_path, rules):
    """The demo never displays them, so carrying them only bulks out the file."""
    src = make_session(tmp_path / "src.db", [("Настоящая строка.", "A real line.", None, 1000)])
    conn = sqlite3.connect(src)
    conn.execute("INSERT INTO transcriptions (id, text, ts_ms, is_final, denied, "
                 "translated_text) VALUES (90, 'частичная', 2000, 0, 0, 'partial')")
    conn.execute("INSERT INTO transcriptions (id, text, ts_ms, is_final, denied, "
                 "translated_text) VALUES (91, 'галлюцинация', 3000, 1, 1, 'hallucination')")
    conn.execute("INSERT INTO transcriptions (id, text, ts_ms, is_final, denied) "
                 "VALUES (92, ' ', NULL, 1, 0)")
    conn.commit()
    conn.close()
    out = str(tmp_path / "out.db")

    report = demo_scrub.scrub_session(src, out, rules)

    assert report.rows_out == 1
    assert report.rows_dropped == 3
