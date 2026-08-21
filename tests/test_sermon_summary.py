"""Sermon summary and chapters (stt/sermon_summary.py).

The property under test throughout is that no timestamp reaches the operator unless a
transcript row actually carries it. A chapter marker is published content, and a model
asked for times will always produce plausible ones — so the tests that matter here are the
ones that feed a deliberately wrong reply through and check what survives.

Captions are constructed, never real service text (see AGENTS.md).
"""

import sqlite3

import pytest

from stt.sermon_summary import (
    STATUS_DONE,
    STATUS_PENDING,
    Chapter,
    Row,
    build_map_prompt,
    build_reduce_prompt,
    chunk_rows,
    delete_summary,
    ensure_tables,
    explain_no_sermons,
    fingerprint,
    format_offset,
    has_summaries,
    load_summaries,
    load_summary,
    mark_error,
    parse_chapters,
    parse_offset,
    parse_sections,
    read_sermon_rows,
    ready_sermons,
    render_markdown,
    save_summary,
    snap_chapters,
    supersede,
    transcript_text,
)

BASE = 1_700_000_000_000
MIN = 60_000


def rows(count, *, base=BASE, step=MIN, text="a caption of several words here"):
    """``count`` rows one step apart, starting at ``base``."""
    return [Row(i + 1, base + i * step, f"{text} {i}") for i in range(count)]


class TestChunking:
    def test_packs_rows_up_to_the_budget(self):
        chunks = chunk_rows(rows(10), 40, counter=lambda s: 10)
        assert len(chunks) > 1
        for c in chunks:
            assert sum(10 + 1 for _ in c.rows) <= 40 or len(c.rows) == 1

    def test_never_splits_a_row(self):
        source = rows(9)
        chunks = chunk_rows(source, 33, counter=lambda s: 10)
        rejoined = [r for c in chunks for r in c.rows]
        assert rejoined == source

    def test_keeps_rows_in_order_and_covers_all_of_them(self):
        source = rows(25)
        chunks = chunk_rows(source, 50, counter=lambda s: 7)
        ids = [r.id for c in chunks for r in c.rows]
        assert ids == sorted(ids)
        assert len(ids) == 25

    def test_an_oversized_row_gets_its_own_chunk_rather_than_being_dropped(self):
        # Declining it would silently lose that stretch of the sermon.
        source = [Row(1, BASE, "short"), Row(2, BASE + MIN, "enormous"), Row(3, BASE + 2 * MIN, "short")]
        chunks = chunk_rows(source, 20, counter=lambda s: 100 if s == "enormous" else 5)
        big = [c for c in chunks if any(r.id == 2 for r in c.rows)]
        assert len(big) == 1 and len(big[0].rows) == 1

    def test_chunk_carries_the_range_it_covers(self):
        chunks = chunk_rows(rows(6), 30, counter=lambda s: 10)
        assert chunks[0].start_ms == BASE
        for c in chunks:
            assert c.start_ms == c.rows[0].ts_ms
            assert c.end_ms == c.rows[-1].ts_ms

    def test_empty_input_or_no_budget_yields_nothing(self):
        assert chunk_rows([], 100) == []
        assert chunk_rows(rows(3), 0) == []


class TestOffsets:
    @pytest.mark.parametrize("ms,expected", [
        (0, "0:00"), (61_000, "1:01"), (599_000, "9:59"), (3_600_000, "1:00:00"),
    ])
    def test_format(self, ms, expected):
        assert format_offset(BASE + ms, BASE) == expected

    def test_format_never_goes_negative(self):
        assert format_offset(BASE - 5000, BASE) == "0:00"

    @pytest.mark.parametrize("text,expected", [
        ("0:00", 0), ("1:01", 61_000), ("12:30", 750_000), ("1:00:00", 3_600_000),
    ])
    def test_parse(self, text, expected):
        assert parse_offset(text) == expected

    @pytest.mark.parametrize("text", ["", "later", "12", "1:60", "abc:de"])
    def test_parse_rejects_non_timestamps(self, text):
        assert parse_offset(text) is None


class TestParsing:
    def test_splits_sections_case_insensitively(self):
        out = parse_sections("### Summary\nA point was made.\n\n### Chapters\n0:00 Opening")
        assert out["summary"] == "A point was made."
        assert out["chapters"] == "0:00 Opening"

    def test_text_before_any_heading_is_recoverable(self):
        assert parse_sections("no headings at all")[""] == "no headings at all"

    def test_tolerates_heading_level_and_decoration(self):
        out = parse_sections("## **Summary**\nbody\n# Chapters\n0:00 One")
        assert out["summary"] == "body" and "chapters" in out

    @pytest.mark.parametrize("line", [
        "0:00 Opening", "- 0:00 Opening", "* 0:00 — Opening",
        "[0:00] Opening", "0:00 - Opening", "0:00: Opening",
    ])
    def test_chapter_line_shapes(self, line):
        assert parse_chapters(line) == [(0, "Opening")]

    def test_lines_without_a_timestamp_are_skipped_not_guessed(self):
        parsed = parse_chapters("Opening remarks\n2:00 The turn\nA closing thought")
        assert parsed == [(120_000, "The turn")]

    def test_strips_title_decoration(self):
        assert parse_chapters("1:00 **The turn**") == [(60_000, "The turn")]

    def test_a_timestamp_with_no_real_title_is_not_a_chapter(self):
        assert parse_chapters("1:00 **") == []


class TestSnapChapters:
    """The anti-fabrication step."""

    def test_snaps_to_a_real_row_stamp(self):
        source = rows(10)
        # 2:30 falls between the rows at 2:00 and 3:00.
        out = snap_chapters([(150_000, "The turn")], source, start_ms=BASE)
        assert [c.ts_ms for c in out] == [BASE]  # first chapter is moved to the first row
        out = snap_chapters([(0, "Opening"), (150_000, "The turn")], source, start_ms=BASE)
        assert out[1].ts_ms in {r.ts_ms for r in source}

    def test_every_returned_stamp_belongs_to_a_row(self):
        source = rows(12)
        stamps = {r.ts_ms for r in source}
        proposed = [(0, "A"), (95_000, "B"), (301_000, "C"), (517_000, "D")]
        for chapter in snap_chapters(proposed, source, start_ms=BASE):
            assert chapter.ts_ms in stamps

    def test_a_fabricated_time_past_the_end_is_dropped_not_clamped(self):
        source = rows(5)  # sermon is 4 minutes long
        out = snap_chapters([(0, "Opening"), (45 * 60_000, "Invented")], source, start_ms=BASE)
        assert [c.title for c in out] == ["Opening"]

    def test_two_proposals_on_the_same_row_collapse_to_one(self):
        source = rows(4)
        out = snap_chapters([(0, "First"), (1000, "Second"), (2000, "Third")],
                            source, start_ms=BASE)
        assert [c.title for c in out] == ["First"]

    def test_result_is_strictly_increasing(self):
        source = rows(20)
        proposed = [(0, "A"), (600_000, "C"), (180_000, "B"), (900_000, "D")]
        out = snap_chapters(proposed, source, start_ms=BASE)
        assert [c.ts_ms for c in out] == sorted(c.ts_ms for c in out)
        assert len({c.ts_ms for c in out}) == len(out)

    def test_first_chapter_covers_the_opening(self):
        source = rows(10)
        out = snap_chapters([(120_000, "Late start")], source, start_ms=BASE)
        assert out[0].ts_ms == source[0].ts_ms

    def test_caps_the_chapter_count(self):
        source = rows(30)
        proposed = [(i * 60_000, f"Point {i}") for i in range(20)]
        assert len(snap_chapters(proposed, source, start_ms=BASE, max_chapters=6)) == 6

    def test_no_rows_means_no_chapters(self):
        assert snap_chapters([(0, "Opening")], [], start_ms=BASE) == []


class TestFingerprint:
    def test_stable_for_the_same_rows(self):
        assert fingerprint(rows(5)) == fingerprint(rows(5))

    def test_changes_when_a_caption_is_corrected(self):
        source = rows(5)
        edited = list(source)
        edited[2] = Row(edited[2].id, edited[2].ts_ms, "a corrected caption")
        assert fingerprint(edited) != fingerprint(source)

    def test_changes_when_rows_are_added(self):
        assert fingerprint(rows(5)) != fingerprint(rows(6))

    def test_same_words_under_new_row_ids_is_new_material(self):
        a = [Row(1, BASE, "one"), Row(2, BASE + MIN, "two")]
        b = [Row(9, BASE, "one"), Row(10, BASE + MIN, "two")]
        assert fingerprint(a) != fingerprint(b)


class TestReadySermons:
    def block(self, **kw):
        base = {"label": "Sermon 1", "ongoing": False, "minutes": 30,
                "end_ms": BASE, "start_ms": BASE - 30 * MIN}
        base.update(kw)
        return base

    def test_a_settled_closed_sermon_is_ready(self):
        now = BASE + 300_000
        assert ready_sermons([self.block()], now_ms=now, settle_seconds=180)

    def test_an_ongoing_sermon_is_never_ready(self):
        now = BASE + 300_000
        assert ready_sermons([self.block(ongoing=True)], now_ms=now) == []

    def test_a_block_that_just_closed_waits_out_the_settle_window(self):
        # track_blocks back-dates end_ms, so a fresh close is still moving.
        assert ready_sermons([self.block()], now_ms=BASE + 10_000, settle_seconds=180) == []

    def test_non_sermon_phases_are_ignored(self):
        now = BASE + 300_000
        for label in ("Songs 2", "Opening", "Communion", None):
            assert ready_sermons([self.block(label=label)], now_ms=now) == []

    def test_numbered_sermons_all_match(self):
        now = BASE + 300_000
        assert len(ready_sermons([self.block(label="Sermon 2")], now_ms=now)) == 1

    def test_too_short_to_be_a_sermon_is_skipped(self):
        now = BASE + 300_000
        assert ready_sermons([self.block(minutes=3)], now_ms=now, min_minutes=8) == []

    def test_the_operator_can_reach_a_still_running_sermon(self):
        # The detector calls the last block ongoing, so a sermon that is still the final
        # block is unreachable by any automatic rule — but the operator can see it ended.
        block = self.block(ongoing=True)
        assert ready_sermons([block], now_ms=BASE, include_ongoing=True) == [block]

    def test_an_ongoing_block_has_nothing_to_settle(self):
        # Its end is "now", so the settle window would otherwise never elapse.
        block = self.block(ongoing=True)
        assert ready_sermons([block], now_ms=BASE, settle_seconds=180,
                             include_ongoing=True) == [block]

    def test_include_ongoing_still_respects_label_and_length(self):
        assert ready_sermons([self.block(ongoing=True, label="Songs 1")],
                             now_ms=BASE, include_ongoing=True) == []
        assert ready_sermons([self.block(ongoing=True, minutes=2)],
                             now_ms=BASE, include_ongoing=True, min_minutes=8) == []

    def test_a_block_with_no_end_is_never_ready(self):
        assert ready_sermons([self.block(end_ms=0)], now_ms=BASE + 10 ** 7) == []


class TestPrompts:
    def test_map_prompt_states_the_range_and_carries_the_text(self):
        chunk = chunk_rows(rows(3), 1000, counter=lambda s: 5)[0]
        system, user = build_map_prompt(chunk, base_ms=BASE)
        assert "0:00" in user and chunk.rows[0].text in user
        assert "same language" in system

    def test_reduce_prompt_names_both_sections_and_the_cap(self):
        system, user = build_reduce_prompt([("[0:00-2:00]", "A point.")], max_chapters=6)
        assert "### Summary" in system and "### Chapters" in system
        assert "6" in system and "Never invent a time." in system
        assert "A point." in user

    def test_reduce_prompt_skips_empty_gists(self):
        _, user = build_reduce_prompt([("[0:00-1:00]", ""), ("[1:00-2:00]", "Kept.")])
        assert "Kept." in user and user.count("[") == 1


class TestDegradedInputs:
    """The paths that keep a summary from taking down a service."""

    def test_a_raising_tokenizer_falls_back_to_the_estimate(self):
        def broken(_text):
            raise RuntimeError("tokenizer gone")

        chunks = chunk_rows(rows(4), 500, counter=broken)
        assert sum(len(c.rows) for c in chunks) == 4

    def test_corrupt_stored_chapters_read_as_none(self, tmp_path):
        c = sqlite3.connect(tmp_path / "session.db")
        ensure_tables(c)
        save_summary(c, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + MIN, status=STATUS_DONE)
        c.execute("UPDATE sermon_summaries SET chapters_json = ? WHERE fingerprint = ?",
                  ("{not json", "fp1"))
        c.commit()
        assert load_summary(c, "fp1")["chapters"] == []
        c.close()

    def test_writes_survive_a_missing_table(self, tmp_path):
        c = sqlite3.connect(tmp_path / "old.db")
        assert delete_summary(c, "fp1") == 0
        assert supersede(c, label="Sermon 1", start_ms=BASE, keep="fp1") == 0
        assert mark_error(c, "fp1", "boom") == 0
        c.close()


class TestPersistence:
    @pytest.fixture()
    def conn(self, tmp_path):
        c = sqlite3.connect(tmp_path / "session.db")
        ensure_tables(c)
        yield c
        c.close()

    def test_round_trips(self, conn):
        chapters = [Chapter(BASE, "Opening"), Chapter(BASE + MIN, "The turn")]
        save_summary(conn, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE, transcript="text",
                     summary="A summary.", chapters=chapters, model="gemma",
                     generated_at="2026-08-20 11:00:00")
        loaded = load_summary(conn, "fp1")
        assert loaded["summary"] == "A summary."
        assert loaded["chapters"] == [{"ts_ms": BASE, "title": "Opening"},
                                      {"ts_ms": BASE + MIN, "title": "The turn"}]
        assert loaded["status"] == STATUS_DONE

    def test_same_fingerprint_updates_in_place(self, conn):
        save_summary(conn, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + MIN, status=STATUS_PENDING)
        save_summary(conn, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + MIN, status=STATUS_DONE, summary="Done.")
        assert len(load_summaries(conn)) == 1
        assert load_summary(conn, "fp1")["status"] == STATUS_DONE

    def test_a_changed_fingerprint_is_a_separate_row(self, conn):
        save_summary(conn, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + MIN, status=STATUS_DONE)
        save_summary(conn, fingerprint="fp2", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 2 * MIN, status=STATUS_DONE)
        assert len(load_summaries(conn)) == 2

    def test_summaries_come_back_in_service_order(self, conn):
        save_summary(conn, fingerprint="b", label="Sermon 2", start_ms=BASE + 60 * MIN,
                     end_ms=BASE + 90 * MIN, status=STATUS_DONE)
        save_summary(conn, fingerprint="a", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE)
        assert [s["label"] for s in load_summaries(conn)] == ["Sermon 1", "Sermon 2"]

    def test_delete_allows_regeneration(self, conn):
        save_summary(conn, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + MIN, status=STATUS_DONE)
        assert delete_summary(conn, "fp1") == 1
        assert load_summary(conn, "fp1") is None

    def test_supersede_replaces_a_partial_summary_of_the_same_sermon(self, conn):
        # Summarised on request while still preaching, then again once the block closed.
        save_summary(conn, fingerprint="partial", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 10 * MIN, status=STATUS_DONE, summary="Half of it.")
        save_summary(conn, fingerprint="complete", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE, summary="All of it.")
        assert supersede(conn, label="Sermon 1", start_ms=BASE, keep="complete") == 1
        assert [x["summary"] for x in load_summaries(conn)] == ["All of it."]

    def test_mark_error_keeps_the_sermon_it_describes(self, conn):
        # The failure handler knows only the fingerprint; an upsert from its defaults
        # would replace the label and range with placeholders.
        save_summary(conn, fingerprint="fp1", label="Sermon 2", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_PENDING, transcript="text")
        assert mark_error(conn, "fp1", "RuntimeError: boom") == 1
        got = load_summary(conn, "fp1")
        assert got["status"] == "error" and got["error"] == "RuntimeError: boom"
        assert got["label"] == "Sermon 2" and got["start_ms"] == BASE
        assert got["end_ms"] == BASE + 30 * MIN and got["transcript"] == "text"

    def test_mark_error_on_an_unknown_fingerprint_writes_nothing(self, conn):
        assert mark_error(conn, "nope", "boom") == 0
        assert load_summaries(conn) == []

    def test_supersede_leaves_a_different_sermon_alone(self, conn):
        save_summary(conn, fingerprint="one", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE)
        save_summary(conn, fingerprint="two", label="Sermon 2", start_ms=BASE + 60 * MIN,
                     end_ms=BASE + 90 * MIN, status=STATUS_DONE)
        assert supersede(conn, label="Sermon 2", start_ms=BASE + 60 * MIN, keep="two") == 0
        assert len(load_summaries(conn)) == 2

    def test_supersede_keeps_a_rerun_of_a_moved_block(self, conn):
        # Same label, different start: the boundaries moved, so it is other material.
        save_summary(conn, fingerprint="a", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE)
        save_summary(conn, fingerprint="b", label="Sermon 1", start_ms=BASE + 5 * MIN,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE)
        assert supersede(conn, label="Sermon 1", start_ms=BASE + 5 * MIN, keep="b") == 0
        assert len(load_summaries(conn)) == 2

    def test_has_summaries_is_the_archive_filter(self, conn):
        assert has_summaries(conn) is False
        save_summary(conn, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + MIN, status=STATUS_DONE)
        assert has_summaries(conn) is True

    def test_a_session_without_the_table_reads_as_empty(self, tmp_path):
        c = sqlite3.connect(tmp_path / "old.db")
        assert load_summaries(c) == [] and has_summaries(c) is False
        assert load_summary(c, "fp1") is None
        c.close()

    def test_ensure_tables_is_idempotent(self, conn):
        ensure_tables(conn)
        ensure_tables(conn)
        assert load_summaries(conn) == []


class TestReadSermonRows:
    @pytest.fixture()
    def conn(self, tmp_path):
        c = sqlite3.connect(tmp_path / "session.db")
        c.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, ts_ms INTEGER, "
                  "text TEXT, is_final INTEGER, denied INTEGER)")
        yield c
        c.close()

    def insert(self, conn, rows_):
        conn.executemany("INSERT INTO transcriptions (id, ts_ms, text, is_final, denied) "
                         "VALUES (?, ?, ?, ?, ?)", rows_)
        conn.commit()

    def test_reads_only_inside_the_block(self, conn):
        self.insert(conn, [(1, BASE - MIN, "before", 1, 0), (2, BASE, "inside", 1, 0),
                           (3, BASE + 10 * MIN, "after", 1, 0)])
        got = read_sermon_rows(conn, BASE, BASE + MIN)
        assert [r.text for r in got] == ["inside"]

    def test_excludes_partials_and_denied_rows(self, conn):
        # Denied covers auto-denied music and hallucination flags; a summary is prose.
        self.insert(conn, [(1, BASE, "kept", 1, 0), (2, BASE + 1000, "partial", 0, 0),
                           (3, BASE + 2000, "music", 1, 1)])
        assert [r.text for r in read_sermon_rows(conn, BASE, BASE + MIN)] == ["kept"]

    def test_skips_blank_text(self, conn):
        self.insert(conn, [(1, BASE, "kept", 1, 0), (2, BASE + 1000, "   ", 1, 0),
                           (3, BASE + 2000, None, 1, 0)])
        assert len(read_sermon_rows(conn, BASE, BASE + MIN)) == 1

    def test_ordered_oldest_first(self, conn):
        self.insert(conn, [(1, BASE + 2000, "b", 1, 0), (2, BASE + 1000, "a", 1, 0)])
        assert [r.text for r in read_sermon_rows(conn, BASE, BASE + MIN)] == ["a", "b"]

    def test_a_missing_table_reads_as_empty(self, tmp_path):
        c = sqlite3.connect(tmp_path / "empty.db")
        assert read_sermon_rows(c, BASE, BASE + MIN) == []
        c.close()


class TestRendering:
    def test_transcript_text_joins_rows(self):
        assert transcript_text([Row(1, BASE, "one"), Row(2, BASE + 1, "two")]) == "one two"

    def test_markdown_renders_chapters_relative_to_the_sermon_start(self):
        out = render_markdown({
            "label": "Sermon 1", "start_ms": BASE, "summary": "A summary.",
            "chapters": [{"ts_ms": BASE, "title": "Opening"},
                         {"ts_ms": BASE + 125_000, "title": "The turn"}],
        })
        assert "# Sermon 1" in out and "A summary." in out
        assert "- 0:00 Opening" in out and "- 2:05 The turn" in out

    def test_markdown_without_chapters_omits_the_heading(self):
        out = render_markdown({"label": "Sermon 1", "start_ms": BASE, "summary": "Only prose."})
        assert "## Chapters" not in out


class TestExplainNoSermons:
    """Why nothing was queued. One message for four causes sent the operator hunting."""

    def block(self, **kw):
        base = {"label": "Sermon 1", "kind": "S", "minutes": 30, "ongoing": False,
                "start_ms": BASE, "end_ms": BASE + 30 * MIN}
        base.update(kw)
        return base

    def test_a_service_the_detector_never_ran_over_says_so(self):
        # The common archived case, and the one the old message hid completely.
        msg = explain_no_sermons([])
        assert "no detected phases" in msg and "Re-run & save" in msg

    def test_a_service_with_no_sermon_names_what_it_did_find(self):
        msg = explain_no_sermons([self.block(label="Songs 1"), self.block(label="Opening")])
        assert "Songs 1" in msg and "Opening" in msg

    def test_a_sermon_under_the_minimum_reports_the_numbers(self):
        msg = explain_no_sermons([self.block(minutes=4)], min_minutes=8)
        assert "4 min" in msg and "8 min" in msg

    def test_otherwise_everything_is_already_summarised(self):
        assert "already" in explain_no_sermons([self.block()], min_minutes=8)

    def test_an_unnamed_block_is_described_rather_than_dropped(self):
        msg = explain_no_sermons([self.block(label=None)])
        assert "unnamed" in msg

