"""Scoring the phase detector against what a human said about a service.

The harness exists because every accuracy claim about phase detection so far has been a
sentence rather than a number — the config comments quote figures nothing in the tree can
reproduce. These tests pin the measuring instrument itself: that it reads a session without
touching it, that it judges only the stretches a human named, and that it can count how
often a label settled late, which is the cost of any rule that ranks blocks across a whole
service.

Every fixture here is constructed. Real services stay on the machines that recorded them.
"""

import json
import os
import sqlite3

import pytest

from stt.phase_replay import (
    SOURCE_BINS,
    SOURCE_ROWS,
    Run,
    TruthSpan,
    base_label,
    compare,
    load_recording,
    main,
    progressive,
    render_comparison,
    render_score,
    replay,
    score,
    shipped_run,
)
from stt.phase_rules import parse_rules

MIN = 60_000
BASE = 1_700_000_000_000

# The shipped shape, minus the parts these tests do not exercise: a long speech block is a
# sermon, a short one after it is Speaking, music is Songs.
RULES = parse_rules({"phases": [
    {"name": "Sermon", "number": True, "confidence": 0.7,
     "match": {"kind": "S", "min_minutes": 8}},
    {"name": "Speaking", "confidence": 0.3, "match": {"kind": "S"}},
    {"name": "Songs", "number": True, "confidence": 0.7,
     "match": {"kind": "M", "min_minutes": 3}},
    {"name": "Music", "confidence": 0.4, "match": {"kind": "M"}},
]})


def build_session(tmp_path, spec, *, corrections=(), marks=(), blocks=None, name="s.db",
                  profile=None):
    """A session database shaped like a real one, from a compact 'MMSSS' spec.

    One character per minute, the same shorthand tests/test_service_phase.py uses: M music,
    S speech, _ quiet. Bins and transcript rows are written so a replay can run from either.
    """
    path = str(tmp_path / name)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE service_phase_bins (bin_index INTEGER PRIMARY KEY, "
                 "start_ms INTEGER, end_ms INTEGER, music INTEGER, speech INTEGER, "
                 "quiet INTEGER, words INTEGER, cues_json TEXT)")
    conn.execute("CREATE TABLE service_phase_blocks (block_index INTEGER PRIMARY KEY, "
                 "kind TEXT, start_bin INTEGER, end_bin INTEGER, start_ms INTEGER, "
                 "end_ms INTEGER, minutes INTEGER, label TEXT, confidence REAL, "
                 "cues_json TEXT, ongoing INTEGER, unusual_json TEXT)")
    conn.execute("CREATE TABLE service_phase_corrections (id INTEGER PRIMARY KEY "
                 "AUTOINCREMENT, block_index INTEGER, start_ms INTEGER, end_ms INTEGER, "
                 "kind TEXT, label TEXT, note TEXT, corrected_at TEXT)")
    conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "ts_ms INTEGER, speech_type TEXT, music_prob REAL, text TEXT, "
                 "translated_text TEXT, audio_tag TEXT, is_final INTEGER, denied INTEGER)")
    conn.execute("CREATE TABLE session_meta (key TEXT PRIMARY KEY, value TEXT)")
    for i, char in enumerate(spec):
        start = BASE + i * MIN
        music = 1 if char == "M" else 0
        speech = 1 if char == "S" else 0
        quiet = 1 if char == "_" else 0
        conn.execute("INSERT INTO service_phase_bins VALUES (?,?,?,?,?,?,?,?)",
                     (i, start, start + MIN, music, speech, quiet, 5 if speech else 0, "{}"))
        speech_type = {"M": "Music", "S": "Speaking", "_": "Quiet"}[char]
        conn.execute("INSERT INTO transcriptions (ts_ms, speech_type, music_prob, text, "
                     "translated_text, audio_tag, is_final, denied) VALUES (?,?,?,?,?,?,1,0)",
                     (start + 1000, speech_type, 0.9 if music else 0.0,
                      "word " * 5, "", "", ))
    for block in (blocks or []):
        conn.execute("INSERT INTO service_phase_blocks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (block["index"], block["kind"], block["start_bin"], block["end_bin"],
                      block["start_ms"], block["end_ms"], block["minutes"], block["label"],
                      block.get("confidence", 0.7), "{}", 0, "[]"))
    for c in corrections:
        conn.execute("INSERT INTO service_phase_corrections (block_index, start_ms, end_ms, "
                     "kind, label, note, corrected_at) VALUES (?,?,?,?,?,?,?)",
                     (c.get("block_index"), c.get("start_ms"), c.get("end_ms"),
                      c.get("kind", "S"), c.get("label"), "", "2026-08-30T12:00:00"))
    for m in marks:
        conn.execute("INSERT INTO service_phase_corrections (block_index, start_ms, end_ms, "
                     "kind, label, note, corrected_at) VALUES (NULL,?,NULL,?,?,?,?)",
                     (m["start_ms"], m.get("kind", "S"), m["label"], "",
                      "2026-08-30T12:00:00"))
    if profile:
        conn.execute("INSERT INTO session_meta VALUES (?,?)",
                     ("service_phase.profile", profile))
    conn.commit()
    conn.close()
    return path


def block(index, kind, start_bin, end_bin, label):
    return {"index": index, "kind": kind, "start_bin": start_bin, "end_bin": end_bin,
            "start_ms": BASE + start_bin * MIN, "end_ms": BASE + (end_bin + 1) * MIN,
            "minutes": end_bin - start_bin + 1, "label": label}


class TestBaseLabel:
    def test_an_ordinal_is_dropped(self):
        assert base_label("Sermon 2") == "Sermon"

    def test_a_plain_name_survives(self):
        assert base_label("Communion") == "Communion"

    def test_a_name_ending_in_a_word_is_untouched(self):
        assert base_label("Songs Reprise") == "Songs Reprise"

    def test_nothing_reads_as_empty(self):
        assert base_label(None) == "" and base_label("") == ""


class TestLoadRecording:
    def test_it_reads_bins_rows_and_blocks(self, tmp_path):
        path = build_session(tmp_path, "MMSSSSSSSSSS",
                             blocks=[block(0, "M", 0, 1, "Songs 1")])
        rec = load_recording(path)
        assert len(rec.stored_bins) == 12
        assert len(rec.rows) == 12
        assert [b["label"] for b in rec.stored_blocks] == ["Songs 1"]

    def test_reading_leaves_the_database_untouched(self, tmp_path):
        # A session database may be opened while it is still being written to.
        path = build_session(tmp_path, "SSSS")
        before = (os.path.getsize(path), os.path.getmtime(path))
        load_recording(path)
        assert (os.path.getsize(path), os.path.getmtime(path)) == before
        assert not os.path.exists(path + "-shm")
        assert not os.path.exists(path + "-wal")

    def test_the_recorded_profile_comes_back(self, tmp_path):
        path = build_session(tmp_path, "SSSS", profile="sunday-morning")
        assert load_recording(path).profile == "sunday-morning"

    def test_a_session_with_no_profile_reads_as_none(self, tmp_path):
        assert load_recording(build_session(tmp_path, "SSSS")).profile is None


class TestTruth:
    def test_a_correction_with_a_span_is_truth(self, tmp_path):
        path = build_session(tmp_path, "S" * 12, corrections=[
            {"block_index": 0, "start_ms": BASE, "end_ms": BASE + 12 * MIN,
             "label": "Closing"}])
        truth = load_recording(path).truth
        assert [(t.label, t.source) for t in truth] == [("Closing", "correction")]

    def test_a_mark_is_truth_too(self, tmp_path):
        # The best boundary evidence there is: a human in the room saying "now".
        blocks = [block(0, "S", 0, 11, "Sermon 1")]
        path = build_session(tmp_path, "S" * 12, blocks=blocks,
                             marks=[{"start_ms": BASE + 2 * MIN, "label": "Sermon"}])
        truth = load_recording(path).truth
        assert [t.source for t in truth] == ["mark"]
        assert truth[0].start_ms == BASE + 2 * MIN

    def test_a_correction_beats_an_overlapping_mark(self, tmp_path):
        # The correction was made afterwards, with the timeline in view.
        blocks = [block(0, "S", 0, 11, "Sermon 1")]
        path = build_session(tmp_path, "S" * 12, blocks=blocks,
                             marks=[{"start_ms": BASE + 2 * MIN, "label": "Sermon"}],
                             corrections=[{"block_index": 0, "start_ms": BASE,
                                           "end_ms": BASE + 12 * MIN, "label": "Closing"}])
        assert [t.source for t in load_recording(path).truth] == ["correction"]

    def test_a_service_nobody_corrected_has_no_truth(self, tmp_path):
        assert load_recording(build_session(tmp_path, "S" * 12)).truth == []


class TestScore:
    def truth(self, label="Sermon", start_bin=0, end_bin=12):
        return [TruthSpan(BASE + start_bin * MIN, BASE + end_bin * MIN, label, "correction")]

    def test_a_run_that_matches_scores_one(self):
        run = Run("r", [block(0, "S", 0, 11, "Sermon 1")])
        assert score(run, self.truth()).agreement == 1.0

    def test_a_run_that_disagrees_scores_zero(self):
        run = Run("r", [block(0, "S", 0, 11, "Speaking")])
        got = score(run, self.truth())
        assert got.agreement == 0.0
        assert [s["label"] for s in got.spurious] == ["Speaking"]
        assert [m["label"] for m in got.missed] == ["Sermon"]

    def test_the_ordinal_does_not_matter(self):
        # Sermon 3 over a stretch a human called Sermon 1 is right about the thing that counts.
        run = Run("r", [block(0, "S", 0, 11, "Sermon 3")])
        assert score(run, self.truth()).agreement == 1.0

    def test_only_named_stretches_are_judged(self):
        # Half the run is over ground nobody corrected; it is neither credited nor blamed.
        run = Run("r", [block(0, "S", 0, 5, "Sermon 1"), block(1, "S", 6, 11, "Speaking")])
        got = score(run, self.truth(end_bin=6))
        assert got.judged_minutes == 6
        assert got.agreement == 1.0
        assert got.spurious == []

    def test_partial_agreement_is_counted_in_minutes(self):
        run = Run("r", [block(0, "S", 0, 5, "Sermon 1"), block(1, "S", 6, 11, "Speaking")])
        got = score(run, self.truth(end_bin=12))
        assert got.judged_minutes == 12 and got.agreed_minutes == 6
        assert got.agreement == 0.5

    def test_per_label_precision_and_recall(self):
        run = Run("r", [block(0, "S", 0, 5, "Sermon 1"), block(1, "S", 6, 11, "Speaking")])
        got = score(run, self.truth(end_bin=12))
        assert got.per_label["Sermon"].recall == 0.5
        assert got.per_label["Sermon"].precision == 1.0

    def test_counts_report_how_many_of_each_phase(self):
        run = Run("r", [block(0, "S", 0, 5, "Sermon 1"), block(1, "S", 6, 11, "Sermon 2")])
        assert score(run, self.truth(end_bin=12)).counts["Sermon"] == 2

    def test_nothing_judged_is_not_a_perfect_score(self):
        assert score(Run("r", []), []).agreement == 0.0


class TestCompare:
    def truth(self):
        return [TruthSpan(BASE, BASE + 12 * MIN, "Closing", "correction")]

    def test_a_fix_is_reported(self):
        before = score(Run("a", [block(0, "S", 0, 11, "Sermon 1")]), self.truth())
        after = score(Run("b", [block(0, "S", 0, 11, "Closing")]), self.truth())
        got = compare(before, after)
        assert [f.get("label") for f in got.fixed if f.get("label") == "Sermon 1"]
        assert got.broken == []
        assert got.agreement_delta > 0

    def test_a_regression_is_reported_separately(self):
        # The whole point: a candidate that fixes one thing and breaks another must not
        # read as an improvement.
        before = score(Run("a", [block(0, "S", 0, 11, "Closing")]), self.truth())
        after = score(Run("b", [block(0, "S", 0, 11, "Sermon 1")]), self.truth())
        got = compare(before, after)
        assert got.broken and not [f for f in got.fixed if f.get("label") == "Sermon 1"]
        assert got.agreement_delta < 0

    def test_no_change_reads_as_no_change(self):
        before = score(Run("a", [block(0, "S", 0, 11, "Closing")]), self.truth())
        after = score(Run("b", [block(0, "S", 0, 11, "Closing")]), self.truth())
        got = compare(before, after)
        assert got.fixed == [] and got.broken == []
        assert "nothing changed" in render_comparison(got)


class TestReplay:
    def test_the_shipped_run_is_what_the_service_stored(self, tmp_path):
        blocks = [block(0, "S", 0, 11, "Sermon 1")]
        rec = load_recording(build_session(tmp_path, "S" * 12, blocks=blocks))
        assert [b["label"] for b in shipped_run(rec).blocks] == ["Sermon 1"]

    def test_replaying_from_bins_reproduces_the_labels(self, tmp_path):
        rec = load_recording(build_session(tmp_path, "M" * 4 + "S" * 12))
        run = replay(rec, {}, RULES, source=SOURCE_BINS)
        assert [b["label"] for b in run.blocks] == ["Songs 1", "Sermon 1"]

    def test_replaying_from_rows_agrees_with_bins(self, tmp_path):
        # The two paths must not disagree, or a candidate's result would depend on which
        # one the caller happened to pick.
        rec = load_recording(build_session(tmp_path, "M" * 4 + "S" * 12))
        from_bins = replay(rec, {}, RULES, source=SOURCE_BINS)
        from_rows = replay(rec, {}, RULES, source=SOURCE_ROWS)
        assert [b["label"] for b in from_bins.blocks] == [b["label"] for b in from_rows.blocks]

    def test_a_candidate_threshold_changes_the_answer(self, tmp_path):
        rec = load_recording(build_session(tmp_path, "S" * 12))
        long_rules = parse_rules({"phases": [
            {"name": "Sermon", "number": True, "match": {"kind": "S", "min_minutes": 20}},
            {"name": "Speaking", "match": {"kind": "S"}},
        ]})
        assert [b["label"] for b in replay(rec, {}, RULES).blocks] == ["Sermon 1"]
        assert [b["label"] for b in replay(rec, {}, long_rules).blocks] == ["Speaking"]


class TestProgressive:
    def test_a_detector_that_never_revises_reports_nothing(self, tmp_path):
        rec = load_recording(build_session(tmp_path, "M" * 4 + "S" * 12))
        assert progressive(rec, {}, RULES) == []

    def test_todays_detector_never_revises_a_settled_label(self, tmp_path):
        """The baseline any ranking rule will be measured against.

        Worth stating as a test rather than a claim: the shipped rules only ever change a
        block's name while it is still running, so the churn a whole-service rule
        introduces is entirely attributable to that rule.
        """
        rec = load_recording(build_session(tmp_path, "M" * 4 + "S" * 12 + "M" * 4))
        assert progressive(rec, {}, RULES) == []

    def test_a_late_relabel_is_counted(self, tmp_path):
        """A settled block renamed minutes later, which is what the instrument is for.

        Songs are numbered from the Opening, so a music block that closed as "Songs 1"
        while no Opening had been seen becomes plain "Music" once one arrives. The quiet
        stretch between them is what makes the two events separate: the block closes at
        minute 8 and is renamed at minute 12.
        """
        rules = parse_rules({"phases": [
            {"name": "Opening", "confidence": 0.5,
             "match": {"kind": "S", "before_first": "Sermon"}},
            {"name": "Songs", "number": True, "number_from": "Opening",
             "match": {"kind": "M", "min_minutes": 3}},
            {"name": "Music", "confidence": 0.4, "match": {"kind": "M"}},
        ]})
        rec = load_recording(build_session(tmp_path, "M" * 5 + "_" * 5 + "S" * 5))
        changes = progressive(rec, {}, rules)
        assert [(c.was, c.now) for c in changes] == [("Songs 1", "Music")]
        assert changes[0].at_minute == 12


class TestRendering:
    def test_a_score_renders_its_headline_and_labels(self):
        got = score(Run("candidate", [block(0, "S", 0, 11, "Sermon 1")]),
                    [TruthSpan(BASE, BASE + 12 * MIN, "Sermon 1", "correction")])
        text = render_score(got)
        assert "candidate" in text and "Sermon" in text and "12 judged minutes" in text


class TestDegradedDatabases:
    """A session that is missing what the harness wants must not raise.

    The archive holds sessions written by every past build, and half of them predate a
    table the current one takes for granted. A tool that reads the archive has to survive
    that, or it can only ever measure the recent past.
    """

    def bare(self, tmp_path, name="bare.db"):
        path = str(tmp_path / name)
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
        conn.commit()
        conn.close()
        return path

    def test_a_session_with_no_phase_tables_reads_as_empty(self, tmp_path):
        rec = load_recording(self.bare(tmp_path))
        assert rec.stored_bins == [] and rec.stored_blocks == [] and rec.truth == []

    def test_replaying_a_session_with_no_bins_yields_no_blocks(self, tmp_path):
        rec = load_recording(self.bare(tmp_path))
        assert replay(rec, {}, RULES).blocks == []

    def test_progressive_on_an_empty_session_is_empty(self, tmp_path):
        assert progressive(load_recording(self.bare(tmp_path)), {}, RULES) == []

    def test_unreadable_cue_json_does_not_raise(self, tmp_path):
        path = build_session(tmp_path, "S" * 12)
        conn = sqlite3.connect(path)
        conn.execute("UPDATE service_phase_bins SET cues_json = 'not json'")
        conn.commit()
        conn.close()
        rec = load_recording(path)
        assert [b["label"] for b in replay(rec, {}, RULES).blocks] == ["Sermon 1"]


class TestCommandLine:
    def test_it_scores_a_session_and_says_so(self, tmp_path, capsys):
        path = build_session(tmp_path, "S" * 12, corrections=[
            {"block_index": 0, "start_ms": BASE, "end_ms": BASE + 12 * MIN,
             "label": "Sermon 1"}])
        rules_path = tmp_path / "rules.json"
        rules_path.write_text(json.dumps({"phases": [
            {"name": "Sermon", "number": True, "match": {"kind": "S", "min_minutes": 8}},
            {"name": "Speaking", "match": {"kind": "S"}},
        ]}), encoding="utf-8")
        assert main([path, "--rules", str(rules_path), "--progressive"]) == 0
        out = capsys.readouterr().out
        assert "shipped" in out and "candidate" in out
        assert "settled labels changed" in out

    def test_a_session_nobody_corrected_says_there_is_nothing_to_score(self, tmp_path,
                                                                      capsys):
        # Honest rather than a made-up 100%: with no correction there is no truth.
        path = build_session(tmp_path, "S" * 12)
        assert main([path]) == 0
        assert "no ground truth" in capsys.readouterr().out

    def test_a_config_file_is_read(self, tmp_path, capsys):
        path = build_session(tmp_path, "S" * 12, corrections=[
            {"block_index": 0, "start_ms": BASE, "end_ms": BASE + 12 * MIN,
             "label": "Speaking"}])
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"limits": {"Sermon": {"max": 0}}}), encoding="utf-8")
        rules_path = tmp_path / "rules.json"
        rules_path.write_text(json.dumps({"phases": [
            {"name": "Sermon", "number": True, "match": {"kind": "S", "min_minutes": 8}},
            {"name": "Speaking", "match": {"kind": "S"}},
        ]}), encoding="utf-8")
        assert main([path, "--rules", str(rules_path), "--config", str(cfg)]) == 0
        out = capsys.readouterr().out
        assert "candidate: 100.0%" in out


class TestRealServiceDatabases:
    """Runs against whatever real sessions this machine happens to hold.

    Service recordings never enter the repository, so these skip on a clean checkout and in
    CI. The assertions are deliberately about shape — that a real database loads, scores and
    is left untouched — never about a congregation's afternoon.
    """

    @pytest.fixture()
    def databases(self):
        import glob
        found = sorted(glob.glob("tests/fixtures/sessions/*.db"))
        if not found:
            pytest.skip("no local session databases; see tests/fixtures/sessions/ in .gitignore")
        return found

    def test_every_local_session_loads(self, databases):
        for path in databases:
            rec = load_recording(path)
            assert rec.session
            assert isinstance(rec.stored_bins, list)

    def test_scoring_a_real_session_does_not_raise(self, databases):
        for path in databases:
            rec = load_recording(path)
            got = score(shipped_run(rec), rec.truth)
            assert 0.0 <= got.agreement <= 1.0

    def test_a_real_session_is_left_byte_identical(self, databases):
        for path in databases:
            before = (os.path.getsize(path), os.path.getmtime(path))
            load_recording(path)
            assert (os.path.getsize(path), os.path.getmtime(path)) == before
