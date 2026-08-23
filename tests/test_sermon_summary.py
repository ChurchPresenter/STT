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
    dominant_source_language,
    row_signature,
    same_language,
    Chapter,
    Row,
    build_map_prompt,
    build_reduce_prompt,
    build_translate_prompt,
    chapter_range,
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
    parse_translation,
    progress_text,
    read_sermon_rows,
    ready_sermons,
    render_markdown,
    save_summary,
    sermon_ranges,
    set_progress,
    snap_chapters,
    supersede,
    transcript_text,
    unfinished,
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

    def test_reduce_prompt_names_both_sections_and_the_range(self):
        system, user = build_reduce_prompt([("[0:00-2:00]", "A point.")], floor=3, ceiling=6)
        assert "### Summary" in system and "### Chapters" in system
        assert "between 3 and 6" in system and "Never invent a time." in system
        assert "A point." in user

    def test_the_reduce_prompt_asks_for_a_range_not_a_target(self):
        # Asked for a number a model reaches it, and the last movements of a shorter sermon
        # become invented divisions.
        system, _ = build_reduce_prompt([("[0:00-2:00]", "A point.")], floor=3, ceiling=9)
        assert "do not " in system and "reach a number" in system

    def test_a_degenerate_range_reads_as_one_number(self):
        system, _ = build_reduce_prompt([("[0:00-2:00]", "A point.")], floor=2, ceiling=2)
        assert "between" not in system and "2 lines" in system

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
        assert supersede(c, label="Sermon 1", start_ms=BASE, end_ms=BASE + MIN,
                         keep="fp1") == 0
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
        assert loaded["chapters"] == [
            {"ts_ms": BASE, "title": "Opening", "title_translated": ""},
            {"ts_ms": BASE + MIN, "title": "The turn", "title_translated": ""}]
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
        assert supersede(conn, label="Sermon 1", start_ms=BASE, end_ms=BASE + 30 * MIN,
                         keep="complete") == 1
        assert [x["summary"] for x in load_summaries(conn)] == ["All of it."]

    def test_a_corrected_boundary_supersedes_the_summary_it_replaced(self, conn):
        """Measured on .62: correcting the start produced a second summary beside the first.

        The old one describes a range the operator has said is wrong, so leaving it gives one
        sermon two summaries that disagree and no way to tell which is current.
        """
        save_summary(conn, fingerprint="detector", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 9 * MIN, status=STATUS_DONE, summary="Nine minutes.")
        # The operator moved the start two minutes earlier; the ranges overlap.
        save_summary(conn, fingerprint="corrected", label="Sermon 1", start_ms=BASE - 2 * MIN,
                     end_ms=BASE + 9 * MIN, status=STATUS_DONE, summary="Thirteen minutes.")
        assert supersede(conn, label="Sermon 1", start_ms=BASE - 2 * MIN,
                         end_ms=BASE + 9 * MIN, keep="corrected") == 1
        assert [x["summary"] for x in load_summaries(conn)] == ["Thirteen minutes."]



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
        assert supersede(conn, label="Sermon 2", start_ms=BASE + 60 * MIN,
                         end_ms=BASE + 90 * MIN, keep="two") == 0
        assert len(load_summaries(conn)) == 2

    def test_supersede_replaces_a_moved_block_that_still_overlaps(self, conn):
        # Same label, moved start: the same sermon seen differently, not other material.
        save_summary(conn, fingerprint="a", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE)
        save_summary(conn, fingerprint="b", label="Sermon 1", start_ms=BASE + 5 * MIN,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE)
        # Overlapping ranges now supersede, which is the point of the change.
        assert supersede(conn, label="Sermon 1", start_ms=BASE + 5 * MIN,
                         end_ms=BASE + 30 * MIN, keep="b") == 1
        assert len(load_summaries(conn)) == 1

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


class TestChapterRange:
    """Density follows the sermon, because speakers do not share a shape.

    One preacher works through eight points in twenty minutes; another develops three across
    forty. A single fixed band makes the model split a movement to reach a floor it cannot
    fill, or merge two to stay under a ceiling that is not this sermon's.
    """

    def test_a_longer_sermon_may_have_more_chapters(self):
        assert chapter_range(12)[1] < chapter_range(37)[1] < chapter_range(80)[1]

    def test_the_floor_rises_with_length_too(self):
        assert chapter_range(12)[0] <= chapter_range(37)[0] <= chapter_range(80)[0]

    @pytest.mark.parametrize("minutes", [0, 1, 5, 9, 12, 20, 37, 45, 60, 80, 120, 400])
    def test_the_floor_never_exceeds_the_ceiling(self, minutes):
        floor, ceiling = chapter_range(minutes)
        assert floor <= ceiling, (minutes, floor, ceiling)

    @pytest.mark.parametrize("minutes", [0, 5, 37, 400])
    def test_at_least_two_are_always_offered(self, minutes):
        # One "chapter" is not a chapter list; snap_chapters already forces a first marker.
        assert chapter_range(minutes)[0] >= 2

    def test_the_hard_cap_holds_however_long_the_service(self):
        # Past a dozen it is a table of contents again, which is what the cap prevents.
        assert chapter_range(600)[1] == 12
        assert chapter_range(600, hard_max=6)[1] == 6

    def test_the_intervals_are_configurable(self):
        wide = chapter_range(60, min_minutes_per_chapter=20, max_minutes_per_chapter=30)
        tight = chapter_range(60, min_minutes_per_chapter=3, max_minutes_per_chapter=6)
        assert wide[1] < tight[1]

    def test_a_zero_interval_does_not_divide_by_zero(self):
        assert chapter_range(37, min_minutes_per_chapter=0, max_minutes_per_chapter=0)


class TestUnfinished:
    """What the end-of-session catch-up still owes."""

    def block(self, **kw):
        base = {"label": "Sermon 1", "minutes": 30, "start_ms": BASE,
                "end_ms": BASE + 30 * MIN, "ongoing": False}
        base.update(kw)
        return base

    def stored(self, status, **kw):
        base = {"label": "Sermon 1", "start_ms": BASE, "status": status}
        base.update(kw)
        return base

    def test_a_sermon_with_no_row_is_unfinished(self):
        assert unfinished([self.block()], []) == [self.block()]

    def test_a_done_sermon_is_left_alone(self):
        assert unfinished([self.block()], [self.stored("done")]) == []

    @pytest.mark.parametrize("status", ["pending", "running", "error"])
    def test_anything_short_of_done_is_unfinished(self, status):
        # The process that was going to finish a pending run is the one that just stopped,
        # and an error is usually a model that was unreachable then and is not now.
        assert unfinished([self.block()], [self.stored(status)]) == [self.block()]

    def test_a_row_for_a_different_sermon_does_not_count(self):
        other = self.stored("done", label="Sermon 2", start_ms=BASE + 60 * MIN)
        assert unfinished([self.block()], [other]) == [self.block()]

    def test_a_row_at_a_different_start_does_not_count(self):
        # The block moved, so the stored summary describes other material.
        assert unfinished([self.block()], [self.stored("done", start_ms=BASE + 5 * MIN)]) \
            == [self.block()]

    def test_non_sermon_blocks_are_never_owed(self):
        assert unfinished([self.block(label="Songs 1")], []) == []

    def test_a_block_under_the_minimum_is_not_owed(self):
        assert unfinished([self.block(minutes=3)], [], min_minutes=8) == []

    def test_it_reports_every_outstanding_sermon(self):
        two = self.block(label="Sermon 2", start_ms=BASE + 60 * MIN, end_ms=BASE + 90 * MIN)
        out = unfinished([self.block(), two], [self.stored("done")])
        assert [b["label"] for b in out] == ["Sermon 2"]


class TestProgress:
    """A run takes minutes; silence for all of them looks the same as stuck."""

    def test_it_counts_parts_from_one(self):
        assert progress_text(0, 15) == "part 1 of 15"
        assert progress_text(14, 15) == "part 15 of 15"

    def test_it_does_not_run_past_the_total(self):
        # The reduce step reports done == total; it must not read "part 16 of 15".
        assert progress_text(15, 15) == "part 15 of 15"

    def test_waiting_on_the_peer_says_so(self):
        # During a service the machine holding the model defers to its captions, so a part
        # can sit for minutes having been told to come back. That is correct, and identical
        # to a hang unless it says which it is.
        assert "waiting for the paired machine" in progress_text(3, 15, waiting=True)
        assert "part 4 of 15" in progress_text(3, 15, waiting=True)

    def test_the_reduce_step_has_its_own_wording(self):
        assert progress_text(15, 15, reducing=True) == "writing the summary"

    def test_the_translation_step_says_so_too(self):
        # Six minutes of summarising followed by silent translating reads as a stall.
        assert progress_text(15, 15, translating=True) == "translating the summary"

    def test_an_unknown_total_still_says_something(self):
        assert progress_text(0, 0) == "starting"


class TestSetProgress:
    @pytest.fixture()
    def conn(self, tmp_path):
        c = sqlite3.connect(tmp_path / "session.db")
        ensure_tables(c)
        save_summary(c, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_PENDING, transcript="text")
        yield c
        c.close()

    def test_it_records_progress(self, conn):
        assert set_progress(conn, "fp1", "part 3 of 15") == 1
        assert load_summary(conn, "fp1")["progress"] == "part 3 of 15"

    def test_it_touches_nothing_else(self, conn):
        set_progress(conn, "fp1", "part 3 of 15")
        got = load_summary(conn, "fp1")
        assert got["label"] == "Sermon 1" and got["start_ms"] == BASE
        assert got["transcript"] == "text" and got["status"] == STATUS_PENDING

    def test_finishing_clears_it(self, conn):
        # A finished run has no progress left to report, and a stale note beside a summary
        # reads as though it were still going.
        set_progress(conn, "fp1", "part 15 of 15")
        save_summary(conn, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE, summary="Done.")
        assert load_summary(conn, "fp1")["progress"] == ""

    def test_an_unknown_fingerprint_writes_nothing(self, conn):
        assert set_progress(conn, "nope", "part 1 of 2") == 0

    def test_a_migration_that_cannot_run_does_not_stop_a_summary(self, tmp_path):
        # A progress note is never worth failing a summary over, so a database that refuses
        # the ALTER degrades to having no progress rather than raising.
        class Refuses(sqlite3.Connection):
            def execute(self, sql, *a):
                if sql.startswith("PRAGMA table_info"):
                    raise sqlite3.OperationalError("locked")
                return sqlite3.Connection.execute(self, sql, *a)

        c = sqlite3.connect(tmp_path / "awkward.db", factory=Refuses)
        ensure_tables(c)  # must not raise
        assert load_summaries(c) == []
        c.close()

    def test_progress_on_a_table_without_the_column_reads_as_empty(self, tmp_path):
        c = sqlite3.connect(tmp_path / "bare.db")
        assert set_progress(c, "fp1", "part 1 of 2") == 0
        c.close()

    def test_a_session_summarised_before_the_column_existed_still_reads(self, tmp_path):
        """.62 already holds rows written without this column.

        CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so without the migration
        every write against those sessions would fail on the missing column.
        """
        path = tmp_path / "older.db"
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE sermon_summaries ("
                  "id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, start_ms INTEGER, "
                  "end_ms INTEGER, fingerprint TEXT UNIQUE, transcript TEXT, summary TEXT, "
                  "chapters_json TEXT, model TEXT, status TEXT, error TEXT, generated_at TEXT)")
        c.execute("INSERT INTO sermon_summaries (fingerprint, label, status) "
                  "VALUES ('old', 'Sermon 1', 'done')")
        c.commit()

        ensure_tables(c)
        assert set_progress(c, "old", "part 1 of 3") == 1
        assert load_summary(c, "old")["progress"] == "part 1 of 3"
        c.close()


class TestSermonRanges:
    """What an operator corrects is what gets summarised.

    The detector finds the structure from audio and is right about it, but only approximate
    about the edges — dwell starts a block a minute or two after the preaching does. For a
    caption timeline that is fine; for a summary it decides whether the model reads the
    introduction or the end of the song before it.
    """

    def block(self, **kw):
        base = {"index": 0, "kind": "S", "label": "Sermon 1", "minutes": 30,
                "start_ms": BASE, "end_ms": BASE + 30 * MIN, "ongoing": False}
        base.update(kw)
        return base

    def test_with_no_corrections_it_is_the_detector(self):
        got = sermon_ranges([self.block()], [])
        assert len(got) == 1 and got[0]["start_ms"] == BASE
        assert got[0]["source"] == "detector"

    def test_a_relabel_makes_a_speaking_block_a_sermon(self):
        # The detector calls a quiet preacher "Speaking"; the operator knows better.
        blocks = [self.block(label="Speaking")]
        assert sermon_ranges(blocks, []) == []
        got = sermon_ranges(blocks, [{"block_index": 0, "label": "Sermon 1",
                                      "start_ms": None, "end_ms": None}])
        assert len(got) == 1 and got[0]["source"] == "correction"

    def test_a_relabel_can_also_take_a_sermon_away(self):
        got = sermon_ranges([self.block()], [{"block_index": 0, "label": "Songs 1",
                                              "start_ms": None, "end_ms": None}])
        assert got == []

    def test_a_drawn_span_sets_the_range_outright(self):
        # This is the fine-tuning: the operator moves the start and end, and the summary
        # reads exactly that.
        got = sermon_ranges([self.block()], [
            {"block_index": None, "label": "Sermon 1", "kind": "S",
             "start_ms": BASE + 2 * MIN, "end_ms": BASE + 26 * MIN}])
        assert len(got) == 1
        assert got[0]["start_ms"] == BASE + 2 * MIN
        assert got[0]["end_ms"] == BASE + 26 * MIN
        assert got[0]["source"] == "correction"

    def test_a_drawn_span_replaces_the_block_underneath_it(self):
        # Otherwise the same sermon is summarised twice, once per definition.
        got = sermon_ranges([self.block()], [
            {"block_index": None, "label": "Sermon 1", "kind": "S",
             "start_ms": BASE + 2 * MIN, "end_ms": BASE + 26 * MIN}])
        assert len(got) == 1

    def test_a_span_elsewhere_leaves_other_blocks_alone(self):
        later = self.block(index=1, label="Sermon 2", start_ms=BASE + 60 * MIN,
                           end_ms=BASE + 90 * MIN)
        got = sermon_ranges([self.block(), later], [
            {"block_index": None, "label": "Sermon 3", "kind": "S",
             "start_ms": BASE + 120 * MIN, "end_ms": BASE + 150 * MIN}])
        assert [b["label"] for b in got] == ["Sermon 1", "Sermon 2", "Sermon 3"]

    def test_results_come_back_in_service_order(self):
        got = sermon_ranges([self.block()], [
            {"block_index": None, "label": "Sermon 0", "kind": "S",
             "start_ms": BASE - 40 * MIN, "end_ms": BASE - 10 * MIN}])
        assert [b["start_ms"] for b in got] == sorted(b["start_ms"] for b in got)

    def test_a_corrected_span_still_has_to_be_long_enough(self):
        got = sermon_ranges([], [{"block_index": None, "label": "Sermon 1", "kind": "S",
                                  "start_ms": BASE, "end_ms": BASE + 3 * MIN}],
                            min_minutes=8)
        assert got == []

    def test_a_span_with_no_range_is_ignored(self):
        # A relabel without a range is handled by block_index, not as a span.
        got = sermon_ranges([self.block()], [{"block_index": None, "label": "Sermon 9",
                                              "start_ms": None, "end_ms": None}])
        assert [b["label"] for b in got] == ["Sermon 1"]

    def test_adjusting_the_same_boundary_twice_keeps_only_the_latest(self):
        """A grouping correction is always an insert, so both adjustments are on record.

        Taking both would summarise one sermon twice, from two ranges the operator has
        already superseded — which is what makes the boundary un-adjustable in practice.
        """
        got = sermon_ranges([self.block()], [
            {"id": 1, "block_index": None, "label": "Sermon 1", "kind": "S",
             "start_ms": BASE - 2 * MIN, "end_ms": BASE + 30 * MIN},
            {"id": 2, "block_index": None, "label": "Sermon 1", "kind": "S",
             "start_ms": BASE - 4 * MIN, "end_ms": BASE + 30 * MIN},
        ])
        assert len(got) == 1
        assert got[0]["start_ms"] == BASE - 4 * MIN, "the older adjustment won"

    def test_two_spans_that_do_not_overlap_are_both_kept(self):
        got = sermon_ranges([], [
            {"id": 1, "block_index": None, "label": "Sermon 1", "kind": "S",
             "start_ms": BASE, "end_ms": BASE + 30 * MIN},
            {"id": 2, "block_index": None, "label": "Sermon 2", "kind": "S",
             "start_ms": BASE + 60 * MIN, "end_ms": BASE + 90 * MIN},
        ])
        assert [b["label"] for b in got] == ["Sermon 1", "Sermon 2"]

    def test_the_shape_matches_a_block_so_callers_cannot_tell(self):
        got = sermon_ranges([], [{"block_index": None, "label": "Sermon 1", "kind": "S",
                                  "start_ms": BASE, "end_ms": BASE + 30 * MIN}])[0]
        for key in ("label", "start_ms", "end_ms", "minutes", "ongoing", "kind"):
            assert key in got


class TestTranslationPrompt:
    def test_it_names_the_language_and_the_count(self):
        system, user = build_translate_prompt("A summary.", ["One", "Two", "Three"], "Spanish")
        assert "Spanish" in system and "3" in system
        assert "One" in user and "Three" in user

    def test_titles_are_numbered_so_they_can_be_matched_back(self):
        # By position, because a translated title against the wrong timestamp is worse than
        # no translation at all.
        _, user = build_translate_prompt("s", ["First", "Second"], "German")
        assert "1. First" in user and "2. Second" in user

    def test_it_asks_for_a_translation_not_another_summary(self):
        system, _ = build_translate_prompt("s", ["One"], "French")
        assert "Translate only" in system and "Do not summarise" in system


class TestParseTranslation:
    def test_it_reads_the_summary_and_the_titles(self):
        raw = "### Summary\nUn resumen.\n\n### Chapters\n1. Primero\n2. Segundo"
        summary, titles = parse_translation(raw, 2)
        assert summary == "Un resumen." and titles == ["Primero", "Segundo"]

    def test_it_keeps_the_order_the_numbers_give_not_the_order_they_arrive(self):
        raw = "### Summary\nX\n\n### Chapters\n2. Segundo\n1. Primero"
        _, titles = parse_translation(raw, 2)
        assert titles == ["Primero", "Segundo"]

    @pytest.mark.parametrize("chapters", [
        "1. Only one",                      # short
        "1. One\n3. Three",                 # a gap
        "Primero\nSegundo",                 # unnumbered
        "",                                 # nothing
    ])
    def test_a_list_that_does_not_line_up_is_discarded_whole(self, chapters):
        """All or nothing.

        A short or gapped list would pair translations with the wrong timestamps, which is
        the failure this feature exists to avoid. Better to publish one language than two
        that disagree about when something was said.
        """
        raw = "### Summary\nUn resumen.\n\n### Chapters\n" + chapters
        summary, titles = parse_translation(raw, 2)
        assert titles == []
        assert summary == "Un resumen."   # the summary is still usable on its own

    def test_extra_titles_are_tolerated_if_the_expected_ones_are_all_there(self):
        raw = "### Summary\nX\n\n### Chapters\n1. One\n2. Two\n3. Spare"
        _, titles = parse_translation(raw, 2)
        assert titles == ["One", "Two"]

    def test_decoration_is_stripped(self):
        raw = "### Summary\nX\n\n### Chapters\n1. **Primero**\n2) Segundo"
        _, titles = parse_translation(raw, 2)
        assert titles == ["Primero", "Segundo"]


class TestTranslationStorage:
    @pytest.fixture()
    def conn(self, tmp_path):
        c = sqlite3.connect(tmp_path / "session.db")
        ensure_tables(c)
        yield c
        c.close()

    def test_a_translation_rides_with_its_chapter(self, conn):
        save_summary(conn, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE, summary="Original.",
                     chapters=[Chapter(BASE, "Opening"), Chapter(BASE + MIN, "The turn")],
                     summary_translated="Traducido.",
                     titles_translated=["Apertura", "El giro"])
        got = load_summary(conn, "fp1")
        assert got["summary_translated"] == "Traducido."
        assert got["chapters"][0] == {"ts_ms": BASE, "title": "Opening",
                                      "title_translated": "Apertura"}
        assert got["chapters"][1]["title_translated"] == "El giro"

    def test_without_a_translation_the_field_is_simply_empty(self, conn):
        save_summary(conn, fingerprint="fp1", label="Sermon 1", start_ms=BASE,
                     end_ms=BASE + 30 * MIN, status=STATUS_DONE, summary="Original.",
                     chapters=[Chapter(BASE, "Opening")])
        got = load_summary(conn, "fp1")
        assert got["summary_translated"] == ""
        assert got["chapters"][0]["title_translated"] == ""

    def test_a_session_summarised_before_the_column_existed_still_reads(self, tmp_path):
        c = sqlite3.connect(tmp_path / "older.db")
        c.execute("CREATE TABLE sermon_summaries ("
                  "id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, start_ms INTEGER, "
                  "end_ms INTEGER, fingerprint TEXT UNIQUE, transcript TEXT, summary TEXT, "
                  "chapters_json TEXT, model TEXT, status TEXT, error TEXT, generated_at TEXT)")
        c.execute("INSERT INTO sermon_summaries (fingerprint, label, status) "
                  "VALUES ('old', 'Sermon 1', 'done')")
        c.commit()
        ensure_tables(c)
        assert load_summary(c, "old")["summary_translated"] == ""
        c.close()

    def test_markdown_carries_both_under_one_timestamp(self):
        out = render_markdown({
            "label": "Sermon 1", "start_ms": BASE, "summary": "Original.",
            "summary_translated": "Traducido.",
            "chapters": [{"ts_ms": BASE, "title": "Opening", "title_translated": "Apertura"}],
        })
        assert "Original." in out and "Traducido." in out
        assert "- 0:00 Opening" in out and "0:00 Apertura" in out


LEGACY_DDL = ("CREATE TABLE sermon_summaries ("
              "id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, start_ms INTEGER, "
              "end_ms INTEGER, fingerprint TEXT UNIQUE, transcript TEXT, summary TEXT, "
              "chapters_json TEXT, model TEXT, status TEXT, error TEXT, generated_at TEXT)")


class TestReadingAnOlderSchema:
    """Reading a session whose table predates a column, without migrating it first.

    This is how production reads: the archive is opened read-only through _archive_open_ro,
    so ensure_tables cannot run and the reader meets whatever columns that database happens to
    have. An earlier test built the old table, migrated it, and *then* read — which proves the
    migration works and says nothing about the case that breaks. It passed on .62 while every
    session summarised before summary_translated existed reported "No sermon has been
    summarised for this service".
    """

    def legacy(self, tmp_path, name="older.db", extra=()):
        c = sqlite3.connect(tmp_path / name)
        c.execute(LEGACY_DDL)
        for column in extra:
            c.execute(f"ALTER TABLE sermon_summaries ADD COLUMN {column} TEXT")
        c.execute("INSERT INTO sermon_summaries (fingerprint, label, start_ms, end_ms, "
                  "summary, chapters_json, status) VALUES "
                  "('old', 'Sermon 1', 100, 200, 'It was summarised.', "
                  "'[{\"ts_ms\": 100, \"title\": \"Opening\"}]', 'done')")
        c.commit()
        return c

    def test_a_table_missing_every_later_column_still_reads(self, tmp_path):
        # No ensure_tables: exactly what the read-only archive path meets.
        c = self.legacy(tmp_path)
        got = load_summaries(c)
        assert len(got) == 1, "the row is there; the reader refused to see it"
        assert got[0]["summary"] == "It was summarised."
        assert got[0]["label"] == "Sermon 1"
        c.close()

    def test_the_shape_that_actually_shipped(self, tmp_path):
        # .62: progress present, summary_translated absent.
        c = self.legacy(tmp_path, extra=("progress",))
        assert len(load_summaries(c)) == 1
        c.close()

    def test_columns_it_does_not_have_read_as_empty(self, tmp_path):
        c = self.legacy(tmp_path)
        got = load_summaries(c)[0]
        assert got["progress"] == "" and got["summary_translated"] == ""
        assert got["chapters"] == [{"ts_ms": 100, "title": "Opening", "title_translated": ""}]
        c.close()

    def test_one_summary_reads_the_same_way(self, tmp_path):
        c = self.legacy(tmp_path)
        got = load_summary(c, "old")
        assert got is not None and got["summary"] == "It was summarised."
        c.close()

    def test_the_listing_and_the_read_agree(self, tmp_path):
        """has_summaries said True while load_summaries said empty.

        That disagreement is what made the symptom baffling: the picker offered a service and
        the page then said nothing had been summarised for it.
        """
        c = self.legacy(tmp_path)
        assert has_summaries(c) is True
        assert len(load_summaries(c)) == 1
        c.close()

    def test_a_column_added_tomorrow_cannot_break_reading_today(self, tmp_path):
        # The other direction: a database ahead of the reader.
        c = sqlite3.connect(tmp_path / "newer.db")
        c.execute(LEGACY_DDL)
        for column in ("progress", "summary_translated", "some_future_column"):
            c.execute(f"ALTER TABLE sermon_summaries ADD COLUMN {column} TEXT")
        c.execute("INSERT INTO sermon_summaries (fingerprint, label, summary, status) "
                  "VALUES ('new', 'Sermon 1', 'From the future.', 'done')")
        c.commit()
        got = load_summaries(c)
        assert len(got) == 1 and got[0]["summary"] == "From the future."
        c.close()



# ─── the tick's change detector ──────────────────────────────────────
#
# A sermon stays "ready" for the rest of the service, so the scan asked the same
# question every twenty seconds and paid a full read plus a sha to hear the same
# answer — inside the loop that also pushes captions to the UI.


class TestRowSignature:
    @pytest.fixture()
    def conn(self, tmp_path):
        c = sqlite3.connect(tmp_path / "session.db")
        c.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, ts_ms INTEGER, "
                  "text TEXT, is_final INTEGER, denied INTEGER, source_language TEXT)")
        yield c
        c.close()

    def insert(self, conn, rows_):
        conn.executemany(
            "INSERT INTO transcriptions (id, ts_ms, text, is_final, denied, source_language) "
            "VALUES (?, ?, ?, ?, ?, ?)", rows_)
        conn.commit()

    def test_unchanged_rows_give_the_same_signature(self, conn):
        self.insert(conn, [(1, BASE, "one", 1, 0, "ru"), (2, BASE + MIN, "two", 1, 0, "ru")])
        first = row_signature(conn, BASE, BASE + 10 * MIN)
        assert row_signature(conn, BASE, BASE + 10 * MIN) == first

    def test_a_new_caption_moves_it(self, conn):
        self.insert(conn, [(1, BASE, "one", 1, 0, "ru")])
        before = row_signature(conn, BASE, BASE + 10 * MIN)
        self.insert(conn, [(2, BASE + MIN, "two", 1, 0, "ru")])
        assert row_signature(conn, BASE, BASE + 10 * MIN) != before

    def test_denying_a_caption_moves_it(self, conn):
        """The summariser excludes denied rows, so the material really did change."""
        self.insert(conn, [(1, BASE, "one", 1, 0, "ru"), (2, BASE + MIN, "two", 1, 0, "ru")])
        before = row_signature(conn, BASE, BASE + 10 * MIN)
        conn.execute("UPDATE transcriptions SET denied = 1 WHERE id = 2")
        conn.commit()
        assert row_signature(conn, BASE, BASE + 10 * MIN) != before

    def test_an_edit_that_changes_the_length_moves_it(self, conn):
        self.insert(conn, [(1, BASE, "one", 1, 0, "ru")])
        before = row_signature(conn, BASE, BASE + 10 * MIN)
        conn.execute("UPDATE transcriptions SET text = 'one more' WHERE id = 1")
        conn.commit()
        assert row_signature(conn, BASE, BASE + 10 * MIN) != before

    def test_an_empty_stretch_is_stable_rather_than_an_error(self, conn):
        assert row_signature(conn, BASE, BASE + MIN) == (0, 0, 0)

    def test_an_unreadable_database_reads_as_changed(self, tmp_path):
        """Never skip on an error: the fingerprint is the one allowed to decide."""
        c = sqlite3.connect(tmp_path / "empty.db")   # no transcriptions table at all
        try:
            assert row_signature(c, BASE, BASE + MIN) == (-1, -1, -1)
        finally:
            c.close()


# ─── skipping a translation that cannot say anything new ─────────────


class TestSourceLanguage:
    @pytest.fixture()
    def conn(self, tmp_path):
        c = sqlite3.connect(tmp_path / "session.db")
        c.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, ts_ms INTEGER, "
                  "text TEXT, is_final INTEGER, denied INTEGER, source_language TEXT)")
        yield c
        c.close()

    def insert(self, conn, rows_):
        conn.executemany(
            "INSERT INTO transcriptions (id, ts_ms, text, is_final, denied, source_language) "
            "VALUES (?, ?, ?, ?, ?, ?)", rows_)
        conn.commit()

    def test_the_majority_language_wins(self, conn):
        """One misdetected caption must not make a Russian sermon English."""
        self.insert(conn, [(1, BASE, "a", 1, 0, "ru"), (2, BASE + MIN, "b", 1, 0, "ru"),
                           (3, BASE + 2 * MIN, "c", 1, 0, "en")])
        assert dominant_source_language(conn, BASE, BASE + 10 * MIN) == "ru"

    def test_nothing_detected_reads_as_unknown(self, conn):
        self.insert(conn, [(1, BASE, "a", 1, 0, None), (2, BASE + MIN, "b", 1, 0, "")])
        assert dominant_source_language(conn, BASE, BASE + 10 * MIN) == ""

    def test_a_missing_column_is_not_an_error(self, tmp_path):
        c = sqlite3.connect(tmp_path / "old.db")
        c.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, ts_ms INTEGER, "
                  "text TEXT, is_final INTEGER, denied INTEGER)")
        try:
            assert dominant_source_language(c, BASE, BASE + MIN) == ""
        finally:
            c.close()

    @pytest.mark.parametrize("source,target,expected", [
        ("en", "en", True),
        ("en-US", "en", True),
        ("en", "en_GB", True),
        ("ru", "en", False),
        ("", "en", False),      # undetected: translate anyway
        ("en", "", False),
    ])
    def test_same_language(self, source, target, expected):
        assert same_language(source, target) is expected
