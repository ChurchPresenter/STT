"""Service phase detection (stt/service_phase.py).

The numbers asserted here come from ten real services in the archive, not from taste:
binned per minute, PANNs speech_type/music_prob reproduces a service's structure, and the
dwell settings were chosen by measuring recall and lag against an offline segmenter.

The property that matters most is causality — the answer produced live at minute N must
equal the answer a replay produces at minute N, or a recorded session is not a fair test of
the live detector. TestCausality pins that directly.
"""

import sqlite3

import pytest

from stt.service_phase import (
    MUSIC,
    QUIET,
    SPEECH,
    Bin,
    analyze,
    bin_rows,
    classify_bin,
    compile_cues,
    tag_class,
    delete_correction,
    delete_correction_by_id,
    ensure_tables,
    label_blocks,
    load_analysis,
    load_corrections,
    read_rows,
    save_analysis,
    save_correction,
    save_group_correction,
    track_blocks,
)

MIN = 60_000


def rows(spec, start=1_000_000, text=""):
    """Rows from a compact spec: 'MMSSS' -> one row per minute of that type."""
    out = []
    for i, c in enumerate(spec):
        stype = {"M": "Music", "S": "Speaking", "_": "Quiet"}[c]
        prob = 0.9 if c == "M" else 0.0
        out.append((start + i * MIN, stype, prob, text))
    return out


def bins_for(spec):
    return bin_rows(rows(spec))


def classes_for(spec):
    return [classify_bin(b) for b in bins_for(spec)]


class TestBinRows:
    def test_buckets_by_minute(self):
        b = bin_rows(rows("SSS"))
        assert len(b) == 3
        assert [x.speech for x in b] == [1, 1, 1]

    def test_music_by_speech_type_or_probability(self):
        # Both routes matter: the pipeline tags Music from a smoothed probability, but the
        # stored music_prob is the raw instantaneous value, so either can be the evidence.
        b = bin_rows([(0, "Speaking", 0.9, ""), (1, "Music", 0.0, "")])
        assert b[0].music == 2

    def test_threshold_is_configurable(self):
        b = bin_rows([(0, "Speaking", 0.6, "")], music_prob_threshold=0.8)
        assert b[0].music == 0 and b[0].speech == 1

    def test_silent_stretches_become_empty_bins_not_missing_ones(self):
        # The tracker needs the gap to exist in order to see it.
        b = bin_rows([(0, "Speaking", 0.0, ""), (5 * MIN, "Speaking", 0.0, "")])
        assert len(b) == 6
        assert b[2].rows == 0

    def test_counts_words_and_cues(self):
        cues = compile_cues({"amen": [r"амин[ья]"]})
        b = bin_rows([(0, "Speaking", 0.0, "Аминь."), (1, "Speaking", 0.0, "два слова тут")],
                     cues=cues)
        assert b[0].cues["amen"] == 1
        assert b[0].words == 4

    def test_no_rows_is_no_bins(self):
        assert bin_rows([]) == []
        assert bin_rows([(None, "Speaking", 0.0, "")]) == []


class TestCompileCues:
    def test_matches_at_word_boundaries(self):
        pat = compile_cues({"c": [r"хлеб"]})["c"]
        assert pat.search("дал хлеб им")
        assert not pat.search("хлебозавод")

    def test_case_insensitive(self):
        assert compile_cues({"a": ["аминь"]})["a"].search("АМИНЬ")

    def test_a_broken_phrase_list_is_skipped_not_fatal(self):
        # A bad regex in config must not take the detector down mid-service.
        out = compile_cues({"bad": ["(unclosed"], "good": ["аминь"]})
        assert "bad" not in out and "good" in out

    def test_empty_lists_are_dropped(self):
        assert compile_cues({"a": [], "b": ["  "], "c": None}) == {}

    def test_config_comment_keys_are_skipped(self):
        # config.default.json documents every block with sibling _comment keys. Iterating
        # one as a phrase list would walk the comment's characters and compile each letter.
        out = compile_cues({"_comment": "Regex fragments counted per bucket.", "amen": ["аминь"]})
        assert list(out) == ["amen"]

    def test_a_bare_string_is_not_treated_as_a_phrase_list(self):
        assert compile_cues({"amen": "аминь"}) == {}


class TestClassifyBin:
    def test_dominance_not_majority(self):
        # A minute of preaching still contains pauses tagged Quiet; requiring an outright
        # majority would classify most of a real service as nothing at all.
        b = Bin(0, 0, MIN)
        b.speech, b.quiet = 4, 6
        assert classify_bin(b) == SPEECH

    def test_empty_bin_is_quiet(self):
        assert classify_bin(Bin(0, 0, MIN)) == QUIET

    def test_music_wins_ties_with_speech(self):
        b = Bin(0, 0, MIN)
        b.music, b.speech = 5, 5
        assert classify_bin(b) == MUSIC

    def test_below_dominance_is_quiet(self):
        b = Bin(0, 0, MIN)
        b.speech, b.music, b.quiet = 1, 1, 20
        assert classify_bin(b) == QUIET


class TestTrackBlocks:
    def test_a_one_minute_blip_does_not_split_a_block(self):
        # The single most important behaviour: an announcement inside a song set, or a
        # musical interlude inside a sermon, must not end the block.
        spec = "SSSSSMSSSSS"
        blocks = track_blocks(classes_for(spec), bins_for(spec))
        assert len(blocks) == 1
        assert blocks[0].kind == SPEECH

    def test_a_sustained_change_does_split(self):
        spec = "SSSSSMMMMM"
        blocks = track_blocks(classes_for(spec), bins_for(spec), enter_minutes=2)
        assert [b.kind for b in blocks] == [SPEECH, MUSIC]

    def test_a_block_starts_where_its_run_started_not_where_we_became_sure(self):
        # Otherwise every boundary is reported enter_minutes too late, and the saved
        # timeline would be systematically skewed against the recording.
        spec = "SSSSMMMM"
        blocks = track_blocks(classes_for(spec), bins_for(spec), enter_minutes=3)
        assert blocks[1].start_bin == 4

    def test_quiet_needs_a_longer_dwell_than_music(self):
        # A lull inside a sermon is common and must not end it.
        spec = "SSSSS__SSSSS"
        blocks = track_blocks(classes_for(spec), bins_for(spec), enter_minutes=2, exit_minutes=3)
        assert len(blocks) == 1

    def test_the_final_block_is_marked_ongoing(self):
        blocks = track_blocks(classes_for("SSSSS"), bins_for("SSSSS"))
        assert blocks[-1].ongoing is True
        assert all(not b.ongoing for b in blocks[:-1])

    def test_blocks_are_contiguous_and_cover_every_bin(self):
        spec = "MMMMSSSSS__MMMM"
        blocks = track_blocks(classes_for(spec), bins_for(spec))
        assert blocks[0].start_bin == 0
        assert blocks[-1].end_bin == len(spec) - 1
        for a, b in zip(blocks, blocks[1:]):
            assert b.start_bin == a.end_bin + 1

    def test_minutes_counts_inclusively(self):
        blocks = track_blocks(classes_for("SSSSS"), bins_for("SSSSS"))
        assert blocks[0].minutes == 5

    def test_no_classes_is_no_blocks(self):
        assert track_blocks([], []) == []


class TestCausality:
    """The live answer at minute N must equal the replayed answer at minute N."""

    SPEC = "MMMMMMSSSSSSSSSS__MMMMSSSSSSSSSSSS_MMMM"

    def test_a_boundary_never_moves_once_emitted(self):
        settled = None
        for n in range(1, len(self.SPEC) + 1):
            prefix = self.SPEC[:n]
            blocks = track_blocks(classes_for(prefix), bins_for(prefix))
            closed = [(b.kind, b.start_bin, b.end_bin) for b in blocks if not b.ongoing]
            if settled is not None:
                assert closed[:len(settled)] == settled, (
                    f"a settled boundary changed at minute {n}")
            settled = closed

    def test_future_data_never_revises_a_closed_block(self):
        # What a mid-service operator saw must still be what the saved timeline says.
        full = track_blocks(classes_for(self.SPEC), bins_for(self.SPEC))
        full_closed = [(b.kind, b.start_bin, b.end_bin) for b in full if not b.ongoing]
        for n in range(1, len(self.SPEC) + 1):
            prefix = self.SPEC[:n]
            closed = [(b.kind, b.start_bin, b.end_bin)
                      for b in track_blocks(classes_for(prefix), bins_for(prefix))
                      if not b.ongoing]
            assert closed == full_closed[:len(closed)], f"history was rewritten at minute {n}"

    def test_the_spec_actually_exercises_several_boundaries(self):
        # Guards the two tests above from silently passing on a degenerate fixture.
        blocks = track_blocks(classes_for(self.SPEC), bins_for(self.SPEC))
        assert len(blocks) >= 4


class TestLabelBlocks:
    CFG = {"sermon_min_minutes": 8, "songs_min_minutes": 3, "communion_min_hits": 12}

    def labelled(self, spec, cfg=None, **kw):
        b = bins_for(spec)
        blocks = track_blocks(classes_for(spec), b)
        return label_blocks(blocks, b, cfg or self.CFG, **kw)

    def test_sermons_are_numbered_in_order(self):
        spec = "S" * 12 + "M" * 5 + "S" * 12
        assert [b.label for b in self.labelled(spec) if b.kind == SPEECH] == ["Sermon 1", "Sermon 2"]

    def test_song_sets_are_numbered_in_order(self):
        spec = "M" * 5 + "S" * 12 + "M" * 5
        assert [b.label for b in self.labelled(spec) if b.kind == MUSIC] == ["Songs 1", "Songs 2"]

    def test_a_short_opening_talk_is_not_a_sermon(self):
        spec = "S" * 4 + "M" * 6 + "S" * 12
        labels = [b.label for b in self.labelled(spec) if b.kind == SPEECH]
        assert labels == ["Opening", "Sermon 1"]

    def test_the_song_count_starts_at_the_opening_not_the_recording(self):
        # The shape of a real Sunday: the band rehearses to an empty room long before the
        # service starts, so the first song of the service must not be called Songs 3.
        spec = "M" * 6 + "_" * 4 + "M" * 5 + "_" * 4 + "S" * 4 + "M" * 6 + "S" * 12
        blocks = self.labelled(spec)
        assert [b.label for b in blocks if b.kind == MUSIC] == ["Music", "Music", "Songs 1"]
        assert [b.label for b in blocks if b.kind == SPEECH] == ["Opening", "Sermon 1"]

    def test_rehearsal_keeps_the_plain_music_name_and_its_confidence(self):
        # Deliberately no new category: "Music" is what the detector already calls music it
        # will not number, and the page offers it in the correction dropdown.
        spec = "M" * 6 + "_" * 4 + "S" * 4 + "M" * 6 + "S" * 12
        rehearsal = next(b for b in self.labelled(spec) if b.kind == MUSIC)
        assert rehearsal.label == "Music" and rehearsal.confidence == 0.4

    def test_with_no_opening_the_count_still_starts_at_the_first_song(self):
        # No anchor to work from, so counting from the recording is the best guess left.
        spec = "M" * 5 + "S" * 12 + "M" * 5
        assert [b.label for b in self.labelled(spec) if b.kind == MUSIC] == ["Songs 1", "Songs 2"]

    def test_music_after_the_opening_is_numbered_even_when_short_music_precedes_it(self):
        spec = "M" * 3 + "_" * 4 + "S" * 4 + "M" * 6 + "_" * 4 + "M" * 6 + "S" * 12
        assert [b.label for b in self.labelled(spec) if b.kind == MUSIC] == [
            "Music", "Songs 1", "Songs 2"]

    def test_short_music_after_the_opening_is_still_not_a_song_set(self):
        # The songs_min threshold keeps applying past the anchor; only the count restarts.
        spec = "M" * 6 + "_" * 4 + "S" * 4 + "M" * 2 + "_" * 4 + "M" * 6 + "S" * 12
        assert [b.label for b in self.labelled(spec) if b.kind == MUSIC] == [
            "Music", "Music", "Songs 1"]

    def test_quiet_blocks_are_never_named(self):
        spec = "S" * 10 + "_" * 6 + "S" * 10
        for b in self.labelled(spec):
            if b.kind == QUIET:
                assert b.label is None and b.confidence == 0.0

    def test_communion_needs_volume_not_a_single_mention(self):
        # A sermon merely discussing communion scored 8 hits across 84 minutes in the
        # archive; the actual communion scored 34 with 15 in one quarter-hour.
        cues = compile_cues({"communion": [r"причасти\w*"]})
        spec = "S" * 12
        b = bin_rows(rows(spec, text="причастие"), cues=cues)   # 1 hit/min = 12
        blocks = label_blocks(track_blocks([classify_bin(x) for x in b], b), b, self.CFG)
        assert blocks[0].label == "Communion"

        b2 = bin_rows(rows(spec), cues=cues)                    # no hits
        blocks2 = label_blocks(track_blocks([classify_bin(x) for x in b2], b2), b2, self.CFG)
        assert blocks2[0].label == "Sermon 1"

    def test_communion_outranks_the_sermon_ordinal(self):
        # It replaces a sermon slot rather than sitting beside one, so it must not consume
        # a sermon number — otherwise every later sermon is misnumbered.
        cues = compile_cues({"communion": [r"причасти\w*"]})
        b = bin_rows(rows("S" * 12, text="причастие") + rows("M" * 5, start=1_000_000 + 12 * MIN)
                     + rows("S" * 12, start=1_000_000 + 17 * MIN), cues=cues)
        blocks = label_blocks(track_blocks([classify_bin(x) for x in b], b), b, self.CFG)
        speech = [x.label for x in blocks if x.kind == SPEECH]
        assert speech == ["Communion", "Sermon 1"]

    def test_first_sunday_raises_communion_confidence(self):
        cues = compile_cues({"communion": [r"причасти\w*"]})
        b = bin_rows(rows("S" * 12, text="причастие"), cues=cues)
        plain = label_blocks(track_blocks([classify_bin(x) for x in b], b), b, self.CFG)[0].confidence
        first = label_blocks(track_blocks([classify_bin(x) for x in b], b), b, self.CFG,
                             first_sunday=True)[0].confidence
        assert first > plain

    def test_an_ongoing_short_talk_stays_low_confidence(self):
        # It may still grow into a sermon; the page must not claim otherwise.
        blocks = self.labelled("S" * 4)
        assert blocks[-1].ongoing and blocks[-1].confidence <= 0.3

    def test_cues_are_summed_onto_the_block(self):
        cues = compile_cues({"amen": [r"амин[ья]"]})
        b = bin_rows(rows("S" * 10, text="Аминь."), cues=cues)
        blocks = label_blocks(track_blocks([classify_bin(x) for x in b], b), b, self.CFG)
        assert blocks[0].cues["amen"] == 10

    def test_thresholds_come_from_config(self):
        spec = "S" * 10
        assert self.labelled(spec, {"sermon_min_minutes": 20})[0].label != "Sermon 1"
        assert self.labelled(spec, {"sermon_min_minutes": 5})[0].label == "Sermon 1"


class TestAnalyze:
    def test_end_to_end_shape(self):
        out = analyze(rows("M" * 6 + "S" * 12), {"sermon_min_minutes": 8})
        assert out["current"]["kind"] == SPEECH
        assert out["current"]["ongoing"] is True
        assert len(out["bins"]) == 18
        assert set(out["classes"]) <= set("MS_")

    def test_empty_input_is_survivable(self):
        out = analyze([])
        assert out["current"] is None and out["blocks"] == [] and out["bins"] == []

    def test_cue_phrases_come_from_config(self):
        cfg = {"cue_phrases": {"amen": [r"амин[ья]"]}}
        out = analyze(rows("S" * 5, text="Аминь."), cfg)
        assert out["bins"][0]["cues"]["amen"] == 1

    def test_no_cue_config_means_no_cues(self):
        out = analyze(rows("S" * 5, text="Аминь."), {})
        assert out["bins"][0]["cues"] == {}

    @pytest.mark.parametrize("bad", [None, {}, {"bin_seconds": 0}])
    def test_unusable_config_still_produces_a_result(self, bad):
        assert analyze(rows("S" * 5), bad)["current"] is not None


class TestPersistence:
    """The detector rewrites its own output every tick; corrections must survive that."""

    CFG = {"sermon_min_minutes": 8, "songs_min_minutes": 3}

    def db(self, tmp_path, name="2026-03-15_090615.db"):
        path = str(tmp_path / name)
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "ts_ms INTEGER, speech_type TEXT, music_prob REAL, text TEXT, "
                     "is_final INTEGER DEFAULT 1)")
        conn.commit()
        return conn

    def fill(self, conn, spec, text=""):
        conn.executemany(
            "INSERT INTO transcriptions (ts_ms, speech_type, music_prob, text, is_final) "
            "VALUES (?, ?, ?, ?, 1)", rows(spec, text=text))
        conn.commit()

    def test_read_rows_returns_detector_input_in_order(self, tmp_path):
        conn = self.db(tmp_path)
        self.fill(conn, "MMSS")
        got = read_rows(conn)
        assert len(got) == 4
        assert [r[1] for r in got] == ["Music", "Music", "Speaking", "Speaking"]

    def test_read_rows_keeps_denied_music_rows(self, tmp_path):
        # Music is auto-denied from the transcript; excluding denied rows would erase
        # exactly the evidence that a song is playing.
        conn = self.db(tmp_path)
        conn.execute("ALTER TABLE transcriptions ADD COLUMN denied INTEGER DEFAULT 0")
        self.fill(conn, "MMMM")
        conn.execute("UPDATE transcriptions SET denied = 1")
        conn.commit()
        assert len(read_rows(conn)) == 4

    def test_read_rows_skips_partials(self, tmp_path):
        conn = self.db(tmp_path)
        self.fill(conn, "SS")
        conn.execute("INSERT INTO transcriptions (ts_ms, speech_type, text, is_final) "
                     "VALUES (99, 'Speaking', 'partial', 0)")
        conn.commit()
        assert len(read_rows(conn)) == 2

    def test_round_trip_through_save_and_load(self, tmp_path):
        conn = self.db(tmp_path)
        self.fill(conn, "M" * 5 + "S" * 12)
        analysis = analyze(read_rows(conn), self.CFG)
        save_analysis(conn, analysis)
        loaded = load_analysis(conn)
        assert [b["kind"] for b in loaded["blocks"]] == [b["kind"] for b in analysis["blocks"]]
        assert [b["label"] for b in loaded["blocks"]] == [b["label"] for b in analysis["blocks"]]
        assert len(loaded["bins"]) == len(analysis["bins"])

    def test_a_later_tick_replaces_rather_than_accumulates(self, tmp_path):
        # Blocks merge and renumber as a service runs; a stale row would survive as a
        # phantom block on the timeline.
        conn = self.db(tmp_path)
        self.fill(conn, "S" * 20)
        save_analysis(conn, analyze(read_rows(conn), self.CFG))
        first = len(load_analysis(conn)["blocks"])
        save_analysis(conn, analyze(read_rows(conn), self.CFG))
        assert len(load_analysis(conn)["blocks"]) == first

    def test_corrections_survive_a_detector_rewrite(self, tmp_path):
        # The whole point of the separate table.
        conn = self.db(tmp_path)
        self.fill(conn, "S" * 20)
        save_analysis(conn, analyze(read_rows(conn), self.CFG))
        save_correction(conn, 0, kind="S", label="Communion", note="was a communion service",
                        corrected_at="2026-07-31T10:00:00")
        save_analysis(conn, analyze(read_rows(conn), self.CFG))
        got = load_corrections(conn)
        assert len(got) == 1 and got[0]["label"] == "Communion"

    def test_correcting_a_block_twice_keeps_one_answer(self, tmp_path):
        conn = self.db(tmp_path)
        save_correction(conn, 3, kind="S", label="Sermon 2")
        save_correction(conn, 3, kind="S", label="Communion")
        got = load_corrections(conn)
        assert len(got) == 1 and got[0]["label"] == "Communion"

    def test_a_correction_without_a_block_is_always_an_insert(self, tmp_path):
        # A boundary the detector missed entirely has no block to replace.
        conn = self.db(tmp_path)
        save_correction(conn, None, kind="S", label="Sermon 3", start_ms=1, end_ms=2)
        save_correction(conn, None, kind="M", label="Songs 4", start_ms=3, end_ms=4)
        assert len(load_corrections(conn)) == 2

    def test_deleting_a_correction_leaves_the_others(self, tmp_path):
        conn = self.db(tmp_path)
        save_correction(conn, 1, kind="S", label="Opening")
        save_correction(conn, 2, kind="M", label="Songs 1")
        assert delete_correction(conn, 1) == 1
        got = load_corrections(conn)
        assert [c["block_index"] for c in got] == [2]

    def test_deleting_a_correction_that_is_not_there_is_not_an_error(self, tmp_path):
        # The page can race a re-render against a click; a no-op beats a 500.
        conn = self.db(tmp_path)
        assert delete_correction(conn, 7) == 0

    def test_delete_leaves_blockless_corrections_alone(self, tmp_path):
        # block_index NULL never matches `= ?`, and nothing on the page can undo it.
        conn = self.db(tmp_path)
        save_correction(conn, None, kind="S", label="Sermon 3", start_ms=1, end_ms=2)
        assert delete_correction(conn, 0) == 0
        assert len(load_corrections(conn)) == 1

    def test_a_deleted_correction_can_be_made_again(self, tmp_path):
        conn = self.db(tmp_path)
        save_correction(conn, 0, kind="S", label="Opening")
        delete_correction(conn, 0)
        save_correction(conn, 0, kind="S", label="Closing")
        got = load_corrections(conn)
        assert len(got) == 1 and got[0]["label"] == "Closing"

    def test_a_group_is_stored_as_a_span_with_no_block(self, tmp_path):
        conn = self.db(tmp_path)
        save_group_correction(conn, 1_000, 5_000, kind="M", label="Worship set")
        got = load_corrections(conn)
        assert len(got) == 1
        assert got[0]["block_index"] is None
        assert (got[0]["start_ms"], got[0]["end_ms"]) == (1_000, 5_000)
        assert got[0]["label"] == "Worship set"

    def test_regrouping_the_same_span_replaces_it(self, tmp_path):
        # Otherwise a reviewer renaming a group stacks a second claim on the same minutes.
        conn = self.db(tmp_path)
        save_group_correction(conn, 1_000, 5_000, kind="M", label="Worship set")
        save_group_correction(conn, 1_000, 5_000, kind="M", label="Opening songs")
        got = load_corrections(conn)
        assert len(got) == 1 and got[0]["label"] == "Opening songs"

    def test_groups_on_different_spans_coexist(self, tmp_path):
        conn = self.db(tmp_path)
        save_group_correction(conn, 1_000, 5_000, kind="M", label="Worship set")
        save_group_correction(conn, 6_000, 9_000, kind="S", label="Teaching")
        assert len(load_corrections(conn)) == 2

    def test_a_group_leaves_per_block_corrections_alone(self, tmp_path):
        conn = self.db(tmp_path)
        save_correction(conn, 0, kind="M", label="Other")
        save_group_correction(conn, 1_000, 5_000, kind="M", label="Worship set")
        assert sorted(c["block_index"] for c in load_corrections(conn)
                      if c["block_index"] is not None) == [0]
        assert len(load_corrections(conn)) == 2

    def test_delete_by_id_removes_only_that_row(self, tmp_path):
        conn = self.db(tmp_path)
        keep = save_group_correction(conn, 1_000, 5_000, kind="M", label="Worship set")
        drop = save_group_correction(conn, 6_000, 9_000, kind="S", label="Teaching")
        assert delete_correction_by_id(conn, drop) == 1
        assert [c["id"] for c in load_corrections(conn)] == [keep]

    def test_delete_by_id_reaches_a_group_that_block_index_cannot(self, tmp_path):
        # The whole reason the id path exists: a group has no block_index to name it by.
        conn = self.db(tmp_path)
        row_id = save_group_correction(conn, 1_000, 5_000, kind="M", label="Worship set")
        assert delete_correction(conn, 0) == 0
        assert delete_correction_by_id(conn, row_id) == 1
        assert load_corrections(conn) == []

    def test_delete_by_id_of_a_missing_row_is_not_an_error(self, tmp_path):
        conn = self.db(tmp_path)
        assert delete_correction_by_id(conn, 4242) == 0

    def test_a_group_survives_a_detector_rewrite(self, tmp_path):
        conn = self.db(tmp_path)
        self.fill(conn, "S" * 20)
        save_analysis(conn, analyze(read_rows(conn), self.CFG))
        save_group_correction(conn, 1_000, 5_000, kind="S", label="Whole talk")
        save_analysis(conn, analyze(read_rows(conn), self.CFG))
        assert [c["label"] for c in load_corrections(conn)] == ["Whole talk"]

    def test_missing_tables_read_as_empty_not_an_error(self, tmp_path):
        # Reviewing a session recorded before this feature existed.
        conn = self.db(tmp_path)
        assert load_corrections(conn) == []
        assert load_analysis(conn)["blocks"] == []

    def test_ensure_tables_is_idempotent(self, tmp_path):
        conn = self.db(tmp_path)
        ensure_tables(conn)
        ensure_tables(conn)
        save_correction(conn, 1, kind="S", label="x")
        assert len(load_corrections(conn)) == 1

    def test_cues_survive_the_json_round_trip(self, tmp_path):
        conn = self.db(tmp_path)
        self.fill(conn, "S" * 10, text="Аминь.")
        save_analysis(conn, analyze(read_rows(conn), {"cue_phrases": {"amen": [r"амин[ья]"]}}))
        assert load_analysis(conn)["blocks"][0]["cues"]["amen"] == 10


class TestDualLanguageCues:
    """Transcript and translation together: either can miss what the other catches.

    Measured over the archive, the union raised communion hits 17% and opening-phrase hits
    28% over the transcript alone — with only 53% of rows carrying a translation at all.
    """

    RU = compile_cues({"communion": [r"хлеб\w*"]})
    EN = compile_cues({"communion": ["the bread"]})

    def dual(self, pairs):
        rs = [(1_000_000 + i * MIN, "Speaking", 0.0, t, tr) for i, (t, tr) in enumerate(pairs)]
        return bin_rows(rs, cues=self.RU, cues_translated=self.EN)

    def test_the_translation_catches_what_the_transcript_missed(self):
        b = self.dual([("нечто иное", "he broke the bread")])
        assert b[0].cues["communion"] == 1

    def test_the_transcript_catches_what_the_translation_missed(self):
        b = self.dual([("преломил хлеб", "he shared it with them")])
        assert b[0].cues["communion"] == 1

    def test_a_row_matching_both_counts_once_not_twice(self):
        # Summing would weight translated rows above untranslated ones, which is exactly
        # the comparison the communion threshold rests on.
        b = self.dual([("преломил хлеб", "he broke the bread")])
        assert b[0].cues["communion"] == 1

    def test_multiple_mentions_in_one_row_still_count(self):
        b = self.dual([("хлеб и хлеб", "")])
        assert b[0].cues["communion"] == 2

    def test_the_larger_side_wins(self):
        b = self.dual([("хлеб", "the bread and the bread")])
        assert b[0].cues["communion"] == 2

    def test_a_missing_translation_degrades_to_the_transcript(self):
        for missing in (None, ""):
            b = self.dual([("преломил хлеб", missing)])
            assert b[0].cues["communion"] == 1

    def test_four_tuple_rows_still_work(self):
        # Sessions recorded before translation existed have no fifth column.
        b = bin_rows(rows("S" * 3, text="хлеб"), cues=self.RU, cues_translated=self.EN)
        assert b[0].cues["communion"] == 1

    def test_a_cue_defined_only_for_the_translation_still_counts(self):
        b = bin_rows([(0, "Speaking", 0.0, "ничего", "welcome everyone")],
                     cues=self.RU, cues_translated=compile_cues({"opening": ["welcome"]}))
        assert b[0].cues["opening"] == 1

    def test_analyze_reads_both_phrase_lists_from_config(self):
        cfg = {"cue_phrases": {"communion": [r"хлеб\w*"]},
               "cue_phrases_translated": {"communion": ["the bread"]}}
        out = analyze([(0, "Speaking", 0.0, "ничего", "the bread"),
                       (MIN, "Speaking", 0.0, "хлеб", None)], cfg)
        assert sum(b["cues"].get("communion", 0) for b in out["bins"]) == 2

    def test_the_translation_widens_the_communion_margin(self):
        """The shipped threshold depends on this, so pin it.

        In the archive the one real communion block scores 17 hits counting transcript and
        translation together, against 6 for the highest non-communion block. On the
        transcript alone the peak is 12 — exactly the shipped threshold, one missed keyword
        from failing. This asserts the mechanism that buys the headroom.
        """
        cfg_both = {"cue_phrases": {"communion": [r"хлеб\w*"]},
                    "cue_phrases_translated": {"communion": ["the bread"]},
                    "communion_min_hits": 12, "sermon_min_minutes": 8}
        cfg_src = dict(cfg_both, cue_phrases_translated={})
        # 12 minutes: transcript names it in 8, the translation carries the other 4.
        pairs = ([("хлеб", None)] * 8) + ([("нечто иное", "the bread")] * 4)
        rs = [(1_000_000 + i * MIN, "Speaking", 0.0, t, tr) for i, (t, tr) in enumerate(pairs)]
        assert analyze(rs, cfg_both)["blocks"][0]["label"] == "Communion"
        assert analyze(rs, cfg_src)["blocks"][0]["label"] == "Sermon 1"


class TestUnusualDurations:
    """Duration flags departures from the archive; it never overrides the audio.

    The bands come from ten ordinary services — music p50 7 min / longest 27, speaking
    p50 12 / longest 54, music share 27%-61%. None of the exceptions the operator named
    (music services, Christmas plays) is in that sample, which is exactly why an
    implausible duration lowers confidence and asks for review rather than relabelling.
    """

    CFG = {"sermon_min_minutes": 8, "songs_min_minutes": 3,
           "typical_music_max_minutes": 30, "typical_speaking_max_minutes": 60}

    def blocks_for(self, spec, cfg=None):
        return analyze(rows(spec), cfg or self.CFG)["blocks"]

    def test_an_hour_of_music_is_flagged(self):
        b = self.blocks_for("M" * 60 + "S" * 10)[0]
        assert b["unusual"] and "music runs 60 min" in b["unusual"][0]

    def test_but_it_is_still_music(self):
        # The audio said music. A play or a carol night is real; relabelling it would be
        # wrong on precisely the service most worth getting right.
        b = self.blocks_for("M" * 60 + "S" * 10)[0]
        assert b["kind"] == MUSIC
        assert b["label"].startswith("Songs")

    def test_flagging_lowers_confidence(self):
        long_b = self.blocks_for("M" * 60 + "S" * 10)[0]
        normal = self.blocks_for("M" * 10 + "S" * 10)[0]
        assert long_b["confidence"] < normal["confidence"]

    def test_a_normal_length_song_set_is_not_flagged(self):
        # The longest real song set in the archive is 27 minutes.
        assert self.blocks_for("M" * 25 + "S" * 10)[0]["unusual"] == []

    def test_a_long_sermon_is_flagged_only_past_the_observed_maximum(self):
        # The longest real sermon is 54 minutes; 44 must stay unremarkable.
        assert self.blocks_for("S" * 44)[0]["unusual"] == []
        assert self.blocks_for("S" * 70 + "M" * 5)[0]["unusual"] != []

    def test_an_ongoing_block_that_has_already_overrun_is_flagged(self):
        # Exceeding a maximum is monotone: finishing later cannot make it untrue. A sermon
        # already past anything in the archive is exactly what an operator wants to see
        # flagged while it is still running, not afterwards.
        blocks = self.blocks_for("S" * 90)
        assert blocks[-1]["ongoing"] and blocks[-1]["unusual"] != []

    def test_a_short_ongoing_block_is_not_flagged(self):
        # It has not finished growing; calling it implausible would be premature.
        blocks = self.blocks_for("S" * 12)
        assert blocks[-1]["ongoing"] and blocks[-1]["unusual"] == []

    def test_the_bands_come_from_config(self):
        cfg = dict(self.CFG, typical_music_max_minutes=5)
        assert self.blocks_for("M" * 10 + "S" * 10, cfg)[0]["unusual"] != []


class TestServiceNotes:
    CFG = {"sermon_min_minutes": 8, "typical_music_share_min": 0.20,
           "typical_music_share_max": 0.70}

    def test_a_music_heavy_service_is_noted(self):
        notes = analyze(rows("M" * 80 + "S" * 10), self.CFG)["notes"]
        assert notes and "more musical than usual" in notes[0].lower()

    def test_a_typical_balance_is_not_noted(self):
        # 43% music is the archive median.
        assert analyze(rows("M" * 40 + "S" * 55), self.CFG)["notes"] == []

    def test_a_sermon_only_service_is_noted(self):
        notes = analyze(rows("S" * 90), self.CFG)["notes"]
        assert notes and "less musical" in notes[0].lower()

    def test_a_service_too_short_to_characterise_is_not_judged(self):
        assert analyze(rows("M" * 10), self.CFG)["notes"] == []

    def test_notes_survive_an_empty_session(self):
        assert analyze([], self.CFG)["notes"] == []


class TestBlocksSchemaMigration:
    def test_a_table_from_an_earlier_build_gains_the_new_column(self, tmp_path):
        # CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so without the
        # migration the next tick's INSERT would fail on a session already running.
        conn = sqlite3.connect(str(tmp_path / "old.db"))
        conn.execute("CREATE TABLE service_phase_blocks (block_index INTEGER PRIMARY KEY, "
                     "kind TEXT, start_bin INTEGER, end_bin INTEGER, start_ms INTEGER, "
                     "end_ms INTEGER, minutes INTEGER, label TEXT, confidence REAL, "
                     "cues_json TEXT, ongoing INTEGER)")
        conn.commit()
        ensure_tables(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(service_phase_blocks)")}
        assert "unusual_json" in cols
        save_analysis(conn, analyze(rows("M" * 60 + "S" * 10), {"sermon_min_minutes": 8}))
        assert load_analysis(conn)["blocks"][0]["unusual"] != []
        conn.close()


class TestWriteChurn:
    """The tick re-derives the whole session but must only write what moved.

    Rewriting both tables wholesale on a 20-second tick cost ~46,000 row-writes across a
    real 167-minute service to record ~500 real changes, on the same disk the session
    recording is being written to. Measured on that session, exactly one bin changes per
    minute (max two) and the block list changes far more rarely.
    """

    CFG = {"sermon_min_minutes": 8, "songs_min_minutes": 3}

    def db(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "churn.db"))
        ensure_tables(conn)
        return conn

    def test_an_unchanged_analysis_writes_nothing(self, tmp_path):
        conn = self.db(tmp_path)
        a = analyze(rows("M" * 6 + "S" * 12), self.CFG)
        assert save_analysis(conn, a) == {"bins": 18, "blocks": 2, "spans": 0}
        assert save_analysis(conn, a) == {"bins": 0, "blocks": 0, "spans": 0}

    def test_only_the_changed_bin_is_written(self, tmp_path):
        conn = self.db(tmp_path)
        save_analysis(conn, analyze(rows("S" * 20), self.CFG))
        # One more minute of audio: one new bin, plus the ongoing block whose end moved.
        written = save_analysis(conn, analyze(rows("S" * 21), self.CFG))
        assert written["bins"] == 1
        assert written["blocks"] <= 1

    def test_a_growing_session_does_not_rewrite_settled_bins(self, tmp_path):
        conn = self.db(tmp_path)
        total = 0
        for n in range(10, 40):
            total += save_analysis(conn, analyze(rows("S" * n), self.CFG))["bins"]
        # 30 steps, one new bin each — not 30 x the whole table.
        assert total <= 60, f"settled bins are being rewritten ({total} writes)"

    def test_the_result_is_still_correct_after_incremental_writes(self, tmp_path):
        conn = self.db(tmp_path)
        spec = "M" * 8 + "S" * 14 + "M" * 6 + "S" * 12
        for n in range(5, len(spec) + 1):
            save_analysis(conn, analyze(rows(spec[:n]), self.CFG))
        fresh = analyze(rows(spec), self.CFG)
        stored = load_analysis(conn)
        assert [b["label"] for b in stored["blocks"]] == [b["label"] for b in fresh["blocks"]]
        assert [(b["index"], b["music"], b["speech"], b["cues"]) for b in stored["bins"]] == \
               [(b["index"], b["music"], b["speech"], b["cues"]) for b in fresh["bins"]]

    def test_merged_blocks_do_not_leave_a_phantom(self, tmp_path):
        # Blocks renumber and merge as a service runs; a leftover row would show on the
        # timeline as a block that never happened.
        conn = self.db(tmp_path)
        save_analysis(conn, analyze(rows("M" * 6 + "S" * 12 + "M" * 6), self.CFG))
        assert len(load_analysis(conn)["blocks"]) == 3
        save_analysis(conn, analyze(rows("S" * 12), self.CFG))
        assert len(load_analysis(conn)["blocks"]) == 1

    def test_a_shorter_session_truncates_stale_bins(self, tmp_path):
        conn = self.db(tmp_path)
        save_analysis(conn, analyze(rows("S" * 30), self.CFG))
        save_analysis(conn, analyze(rows("S" * 10), self.CFG))
        assert len(load_analysis(conn)["bins"]) == 10

    def test_a_late_translation_updates_a_settled_bin(self, tmp_path):
        """Translation lands via a later async UPDATE on a row the bin already counted.

        The cue appears with the row count, word count and audio mix all unchanged, so a
        diff on those alone silently loses it — and an operator correcting a misheard word
        does the same. cues_json has to be part of the comparison.
        """
        conn = self.db(tmp_path)
        cfg = {"cue_phrases": {}, "cue_phrases_translated": {"amen": ["amen"]},
               "sermon_min_minutes": 8}
        # Identical transcript both times; only translated_text arrives on the second pass.
        before = [(1_000_000 + i * MIN, "Speaking", 0.0, "одно слово", None) for i in range(10)]
        after = [(1_000_000 + i * MIN, "Speaking", 0.0, "одно слово", "amen") for i in range(10)]
        save_analysis(conn, analyze(before, cfg))
        assert load_analysis(conn)["bins"][0]["cues"] == {}
        written = save_analysis(conn, analyze(after, cfg))
        assert written["bins"] == 10, "the late translation was not persisted"
        assert load_analysis(conn)["bins"][0]["cues"] == {"amen": 1}

    def test_an_operator_correction_that_keeps_the_word_count_still_writes(self, tmp_path):
        conn = self.db(tmp_path)
        cfg = {"cue_phrases": {"amen": [r"амин[ья]"]}, "sermon_min_minutes": 8}
        typo = [(1_000_000 + i * MIN, "Speaking", 0.0, "аминт", None) for i in range(6)]
        fixed = [(1_000_000 + i * MIN, "Speaking", 0.0, "аминь", None) for i in range(6)]
        save_analysis(conn, analyze(typo, cfg))
        assert save_analysis(conn, analyze(fixed, cfg))["bins"] == 6
        assert load_analysis(conn)["bins"][0]["cues"] == {"amen": 1}


class TestAudioTagFallback:
    """PANNs' own label, consulted where the derived one gives up.

    speech_type is not the classifier's answer but a derivation of it: Music only above
    music_prob_threshold, otherwise Quiet or Speaking on whether audio_db clears
    quiet_db_threshold. Measured on one live service whose input sat at about -40 dB — the
    threshold itself — 27 rows tagged "Speech" and 14 tagged "Music" were all recorded as
    Quiet, and nine minutes of a service became a Quiet block. The tag was right throughout.
    """

    @pytest.mark.parametrize("tag", [
        "Music", "music", "Singing", "Choir", "Musical instrument", "Electric guitar",
        "Piano", "Organ", "Drum kit",
    ])
    def test_music_tags_read_as_music(self, tag):
        assert tag_class(tag) == MUSIC

    @pytest.mark.parametrize("tag", [
        "Speech", "speech", "Narration, monologue", "Conversation", "Speech synthesizer",
    ])
    def test_speech_tags_read_as_speech(self, tag):
        assert tag_class(tag) == SPEECH

    @pytest.mark.parametrize("tag", [
        "Silence", "Inside, small room", "Heart sounds, heartbeat", "Sniff", "Patter",
        "Animal", "", None, "   ",
    ])
    def test_everything_else_says_nothing(self, tag):
        # A tag that means neither must not be forced into one; the row stays Quiet.
        assert tag_class(tag) is None

    def row(self, ts, speech_type, prob, tag):
        return (ts, speech_type, prob, "some words here", None, tag)

    def test_singing_the_db_gate_silenced_is_counted_as_music(self):
        # The live shape: tag Music, probability well under the threshold, label Quiet.
        rows = [self.row(1_000_000 + i * 1000, "Quiet", 0.13, "Music") for i in range(5)]
        b = bin_rows(rows)[0]
        assert (b.music, b.speech, b.quiet) == (5, 0, 0)

    def test_speech_the_db_gate_silenced_is_counted_as_speech(self):
        rows = [self.row(1_000_000 + i * 1000, "Quiet", 0.05, "Speech") for i in range(5)]
        b = bin_rows(rows)[0]
        assert (b.music, b.speech, b.quiet) == (0, 5, 0)

    def test_genuine_silence_is_still_quiet(self):
        rows = [self.row(1_000_000 + i * 1000, "Quiet", 0.04, "Silence") for i in range(5)]
        b = bin_rows(rows)[0]
        assert (b.music, b.speech, b.quiet) == (0, 0, 5)

    def test_a_confident_speech_type_is_never_overruled(self):
        """Where the derived label is sure, it stays authoritative.

        The dwell settings were measured against it, so the tag only fills the gap it
        leaves — it does not get a vote on rows the pipeline already classified.
        """
        rows = [self.row(1_000_000, "Speaking", 0.02, "Music"),
                self.row(1_001_000, "Music", 0.9, "Speech")]
        b = bin_rows(rows)[0]
        assert (b.music, b.speech) == (1, 1)

    def test_it_can_be_switched_off(self):
        rows = [self.row(1_000_000 + i * 1000, "Quiet", 0.13, "Music") for i in range(5)]
        b = bin_rows(rows, use_audio_tag=False)[0]
        assert (b.music, b.speech, b.quiet) == (0, 0, 5)

    def test_rows_without_a_tag_column_still_work(self):
        # Sessions recorded before the column, and the four-tuple fallback in read_rows.
        rows = [(1_000_000 + i * 1000, "Quiet", 0.1, "text") for i in range(3)]
        b = bin_rows(rows)[0]
        assert b.quiet == 3

