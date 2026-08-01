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
    ensure_tables,
    label_blocks,
    load_analysis,
    load_corrections,
    read_rows,
    save_analysis,
    save_correction,
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
