"""Learning detector settings from corrected services (stt/phase_learn.py).

The behaviour that matters most is what happens with almost no data, because that is the
state every installation starts in and the state this one is in today: one archived session
and two corrections. A learner that proposes confidently from two examples is worse than no
learner, so the zero- and few-sample cases are pinned first.
"""

import sqlite3

import pytest

from stt.phase_marks import resolve as resolve_marks
from stt.phase_learn import (
    MIN_SAMPLES,
    CorrectedPhase,
    ServiceShape,
    apply_proposals,
    collect,
    group_by_service,
    propose_all,
    propose_counts,
    propose_durations,
    propose_fragments,
    propose_service_length,
    read_corrected_phases,
    read_marked_phases,
)

MIN = 60_000
BASE = 1_700_000_000_000
BASELINE = {"sermon_min_minutes": 8, "songs_min_minutes": 3,
            "typical_music_max_minutes": 30, "typical_speaking_max_minutes": 60}


def session(tmp_path, name, corrections, blocks=(), rows=()):
    """A session database carrying corrections, blocks and transcript rows."""
    path = str(tmp_path / name)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "ts_ms INTEGER, text TEXT, is_final INTEGER)")
    conn.execute("CREATE TABLE service_phase_blocks (block_index INTEGER PRIMARY KEY, "
                 "kind TEXT, start_bin INTEGER, end_bin INTEGER, start_ms INTEGER, "
                 "end_ms INTEGER, minutes INTEGER, label TEXT, confidence REAL, "
                 "cues_json TEXT, ongoing INTEGER, unusual_json TEXT)")
    conn.execute("CREATE TABLE service_phase_corrections (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "block_index INTEGER, start_ms INTEGER, end_ms INTEGER, kind TEXT, "
                 "label TEXT, note TEXT, corrected_at TEXT)")
    for index, kind, start_ms, minutes in blocks:
        conn.execute("INSERT INTO service_phase_blocks (block_index, kind, start_ms, end_ms, "
                     "minutes) VALUES (?, ?, ?, ?, ?)",
                     (index, kind, start_ms, start_ms + minutes * MIN, minutes))
    for block_index, start_ms, end_ms, kind, label in corrections:
        conn.execute("INSERT INTO service_phase_corrections (block_index, start_ms, end_ms, "
                     "kind, label) VALUES (?, ?, ?, ?, ?)",
                     (block_index, start_ms, end_ms, kind, label))
    for ts_ms, text in rows:
        conn.execute("INSERT INTO transcriptions (ts_ms, text, is_final) VALUES (?, ?, 1)",
                     (ts_ms, text))
    conn.commit()
    return path, conn


class TestZeroAndFewSamples:
    def test_no_corrections_proposes_nothing_actionable(self, tmp_path):
        _, conn = session(tmp_path, "a.db", corrections=[])
        result = propose_all(collect([("a.db", conn)]), {}, BASELINE)
        assert result["corrections"] == 0
        assert result["actionable"] == 0
        assert all(not p["actionable"] for p in result["proposals"])

    def test_todays_state_two_corrections_proposes_nothing(self, tmp_path):
        # The state a fresh installation is in: one session, two corrections.
        _, conn = session(
            tmp_path, "2026-03-01_092547.db",
            corrections=[(0, None, None, "M", "Other"), (7, None, None, "_", "Prayer")],
            blocks=[(0, "M", 1_000_000, 11), (7, "_", 1_000_000 + 65 * MIN, 6)])
        result = propose_all(collect([("s", conn)]), {}, BASELINE)
        assert result["corrections"] == 2
        assert result["actionable"] == 0

    def test_below_the_minimum_the_current_value_is_kept(self, tmp_path):
        corrections = [(i, None, None, "S", "Sermon 1") for i in range(MIN_SAMPLES - 1)]
        blocks = [(i, "S", 1_000_000 + i * 30 * MIN, 20) for i in range(MIN_SAMPLES - 1)]
        _, conn = session(tmp_path, "a.db", corrections=corrections, blocks=blocks)
        cfg = {"sermon_min_minutes": 8}
        proposals = propose_durations(collect([("a", conn)]), cfg, BASELINE)
        sermon = next(p for p in proposals if p.key == "sermon_min_minutes")
        assert sermon.suggested == 8 and not sermon.actionable
        assert "%d" % (MIN_SAMPLES - 1) in sermon.evidence

    def test_the_evidence_says_so_rather_than_going_silent(self, tmp_path):
        _, conn = session(tmp_path, "a.db", corrections=[])
        proposals = propose_durations(collect([("a", conn)]), {}, BASELINE)
        assert all("no corrected" in p.evidence for p in proposals)


class TestDurationProposals:
    def sermons(self, tmp_path, lengths):
        corrections = [(i, None, None, "S", "Sermon %d" % (i + 1)) for i in range(len(lengths))]
        blocks = [(i, "S", 1_000_000 + i * 60 * MIN, n) for i, n in enumerate(lengths)]
        _, conn = session(tmp_path, "a.db", corrections=corrections, blocks=blocks)
        return collect([("a", conn)])

    def test_the_threshold_lands_at_or_below_the_shortest_real_sermon(self, tmp_path):
        # A threshold above the shortest corrected sermon would have misnamed that sermon.
        phases = self.sermons(tmp_path, [6, 9, 14, 22, 25])
        proposal = next(p for p in propose_durations(phases, {"sermon_min_minutes": 8}, BASELINE)
                        if p.key == "sermon_min_minutes")
        assert proposal.suggested <= 6
        assert proposal.samples == 5 and proposal.actionable

    def test_a_proposal_equal_to_the_current_value_is_not_actionable(self, tmp_path):
        phases = self.sermons(tmp_path, [8, 8, 12, 30, 44])
        proposal = next(p for p in propose_durations(phases, {"sermon_min_minutes": 8}, BASELINE)
                        if p.key == "sermon_min_minutes")
        assert proposal.suggested == 8 and not proposal.actionable

    def test_the_baseline_is_reported_beside_the_suggestion(self, tmp_path):
        phases = self.sermons(tmp_path, [6, 9, 14, 22])
        proposal = next(p for p in propose_durations(phases, {"sermon_min_minutes": 15}, BASELINE)
                        if p.key == "sermon_min_minutes")
        assert proposal.baseline == 8 and proposal.current == 15

    def test_a_longer_service_raises_the_unusual_flags(self, tmp_path):
        # A congregation whose sermons really run 70 minutes should stop being told they are
        # unusual — that is the whole point of not shipping one church's maxima.
        phases = self.sermons(tmp_path, [55, 62, 70, 74])
        proposal = next(p for p in propose_durations(phases, {}, BASELINE)
                        if p.key == "typical_speaking_max_minutes")
        assert proposal.suggested >= 74

    def test_ordinals_are_folded_together(self, tmp_path):
        # "Sermon 1" and "Sermon 3" are the same phase for measuring purposes.
        phases = self.sermons(tmp_path, [10, 12, 14, 16])
        proposal = next(p for p in propose_durations(phases, {}, BASELINE)
                        if p.key == "sermon_min_minutes")
        assert proposal.samples == 4


class TestGroupedSpans:
    def test_a_grouped_correction_counts_by_its_own_span(self, tmp_path):
        # A group has no block index; its span is what says how long the phase was.
        start = 1_000_000
        corrections = [(None, start + i * 60 * MIN, start + i * 60 * MIN + 26 * MIN, "_",
                        "Communion") for i in range(MIN_SAMPLES)]
        _, conn = session(tmp_path, "a.db", corrections=corrections)
        phases = collect([("a", conn)])
        assert len(phases) == MIN_SAMPLES
        assert all(p.minutes == 26 for p in phases)

    def test_a_correction_with_neither_a_block_nor_a_span_is_ignored(self, tmp_path):
        _, conn = session(tmp_path, "a.db",
                          corrections=[(99, None, None, "S", "Sermon 1")], blocks=[])
        assert read_corrected_phases(conn, "a") == []

    def test_a_blank_label_is_not_a_correction(self, tmp_path):
        _, conn = session(tmp_path, "a.db", corrections=[(0, None, None, "S", "")],
                          blocks=[(0, "S", 1_000_000, 12)])
        assert read_corrected_phases(conn, "a") == []


class TestFragmentMining:
    def communions(self, tmp_path, texts, other_texts=()):
        start = 1_000_000
        corrections, rows = [], []
        for i, text in enumerate(texts):
            at = start + i * 120 * MIN
            corrections.append((None, at, at + 20 * MIN, "_", "Communion"))
            rows.append((at + MIN, text))
        for j, text in enumerate(other_texts):
            at = start + (len(texts) + j) * 120 * MIN
            corrections.append((None, at, at + 20 * MIN, "S", "Sermon 1"))
            rows.append((at + MIN, text))
        _, conn = session(tmp_path, "a.db", corrections=corrections, rows=rows)
        return collect([("a", conn)], with_text=True)

    def test_a_phrase_in_every_communion_and_no_sermon_is_proposed(self, tmp_path):
        phases = self.communions(
            tmp_path,
            ["чаша есть новый завет в крови"] * MIN_SAMPLES,
            ["сегодня мы говорим о вере и терпении"] * 3)
        proposal = propose_fragments(phases, "communion_verse", "Communion")[0]
        assert proposal.samples == MIN_SAMPLES
        assert any("чаша есть новый" in s for s in proposal.suggested)

    def test_a_phrase_the_sermons_also_use_is_not_proposed(self, tmp_path):
        shared = "во имя отца и сына"
        phases = self.communions(tmp_path, [shared] * MIN_SAMPLES, [shared] * 3)
        proposal = propose_fragments(phases, "communion_verse", "Communion")[0]
        assert proposal.suggested == []

    def test_phrases_already_configured_are_not_proposed_again(self, tmp_path):
        phases = self.communions(tmp_path, ["чаша есть новый завет"] * MIN_SAMPLES)
        known = {"cup": ["чаша есть"]}
        proposal = propose_fragments(phases, "communion_verse", "Communion", known=known)[0]
        assert "чаша есть" not in proposal.suggested

    def test_too_few_corrected_examples_proposes_nothing(self, tmp_path):
        phases = self.communions(tmp_path, ["чаша есть новый завет"] * 2)
        proposal = propose_fragments(phases, "communion_verse", "Communion")[0]
        assert proposal.suggested == [] and not proposal.actionable
        assert str(MIN_SAMPLES) in proposal.evidence


class TestApplying:
    def proposal(self, key, suggested, actionable=True):
        return {"key": key, "suggested": suggested, "actionable": actionable}

    def test_only_the_named_keys_are_taken(self):
        out = apply_proposals(
            {"sermon_min_minutes": 8, "songs_min_minutes": 3},
            [self.proposal("sermon_min_minutes", 6), self.proposal("songs_min_minutes", 5)],
            ["sermon_min_minutes"])
        assert out["sermon_min_minutes"] == 6
        assert out["songs_min_minutes"] == 3

    def test_a_proposal_that_was_not_actionable_is_refused(self):
        out = apply_proposals({"sermon_min_minutes": 8},
                              [self.proposal("sermon_min_minutes", 6, actionable=False)],
                              ["sermon_min_minutes"])
        assert out["sermon_min_minutes"] == 8

    def test_the_caller_s_config_is_not_mutated(self):
        cfg = {"sermon_min_minutes": 8}
        apply_proposals(cfg, [self.proposal("sermon_min_minutes", 6)], ["sermon_min_minutes"])
        assert cfg["sermon_min_minutes"] == 8

    def test_mined_phrases_land_as_escaped_fragments(self):
        # A mined phrase is literal text, not a pattern an operator wrote: a stray ( in it
        # would otherwise compile to a broken regex and take its own cue group down.
        out = apply_proposals({}, [self.proposal("cue_fragments.communion_verse",
                                                 ["чаша (нового) завета"])],
                              ["cue_fragments.communion_verse"])
        fragments = out["cue_fragments"]["communion_verse"]
        assert len(fragments) == 1
        pattern = next(iter(fragments.values()))[0]
        assert "\\(" in pattern

    def test_mined_phrases_do_not_replace_the_existing_ones(self):
        cfg = {"cue_fragments": {"communion_verse": {"cup": ["чаша"]}}}
        out = apply_proposals(cfg, [self.proposal("cue_fragments.communion_verse", ["новый завет"])],
                              ["cue_fragments.communion_verse"])
        assert "cup" in out["cue_fragments"]["communion_verse"]
        assert len(out["cue_fragments"]["communion_verse"]) == 2


class TestAnchoredCorrectionsBeatTheIndex:
    """What the learner is taught when the blocks have moved under a correction.

    read_corrected_phases used to look a block_index up in service_phase_blocks and take that
    block's *current* times. Blocks renumber whenever the detector's output changes shape —
    one real session went from five blocks to six when the audio tag started being read — so
    a correction could hand the learner a phase of the wrong length under the operator's name,
    and phase durations are exactly what it measures.
    """

    def db(self, tmp_path, correction, blocks):
        path = tmp_path / "session.db"
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE service_phase_corrections (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "block_index INTEGER, start_ms INTEGER, end_ms INTEGER, kind TEXT, "
                  "label TEXT, note TEXT, corrected_at TEXT)")
        c.execute("CREATE TABLE service_phase_blocks (block_index INTEGER PRIMARY KEY, kind TEXT, "
                  "start_ms INTEGER, end_ms INTEGER, minutes INTEGER)")
        c.execute("INSERT INTO service_phase_corrections (block_index, start_ms, end_ms, kind, "
                  "label) VALUES (?, ?, ?, ?, ?)", correction)
        c.executemany("INSERT INTO service_phase_blocks (block_index, kind, start_ms, end_ms, "
                      "minutes) VALUES (?, ?, ?, ?, ?)", blocks)
        c.commit()
        return c

    def test_a_drifted_index_does_not_change_the_length_learned(self, tmp_path):
        # The correction named a 17-minute sermon. After the re-run, block 4 is 2 minutes of
        # music; the anchor is what keeps the learner from being told the sermon was 2.
        conn = self.db(
            tmp_path,
            correction=(4, 25 * 60000, 42 * 60000, "S", "Sermon 2"),
            blocks=[(4, "M", 23 * 60000, 25 * 60000, 2)])
        got = read_corrected_phases(conn, "s.db")
        conn.close()
        assert len(got) == 1
        assert got[0].minutes == 17, "the learner took the drifted block's length"
        assert got[0].start_ms == 25 * 60000

    def test_a_legacy_correction_still_reads_its_block(self, tmp_path):
        # No span recorded, so the block table is all there is — and remains correct when
        # nothing has moved.
        conn = self.db(
            tmp_path,
            correction=(1, None, None, None, "Sermon 1"),
            blocks=[(1, "S", 3 * 60000, 14 * 60000, 11)])
        got = read_corrected_phases(conn, "s.db")
        conn.close()
        assert len(got) == 1 and got[0].minutes == 11
        assert got[0].kind == "S", "the kind came from the block it named"


class TestMarksAreEvidence:
    """A live mark is the strongest boundary statement anyone makes, and it was discarded.

    A mark carries no end and no block index, so both branches of read_corrected_phases
    reject it — which meant the operator pressing "the sermon starts now" taught the
    detector nothing at all.
    """

    def session(self, tmp_path, marks=(), blocks=(), name="s.db"):
        path = tmp_path / name
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE service_phase_corrections (id INTEGER PRIMARY KEY "
                     "AUTOINCREMENT, block_index INTEGER, start_ms INTEGER, "
                     "end_ms INTEGER, kind TEXT, label TEXT, note TEXT, corrected_at TEXT)")
        conn.execute("CREATE TABLE service_phase_blocks (block_index INTEGER PRIMARY KEY, "
                     "kind TEXT, start_bin INTEGER, end_bin INTEGER, start_ms INTEGER, "
                     "end_ms INTEGER, minutes INTEGER, label TEXT, confidence REAL, "
                     "cues_json TEXT, ongoing INTEGER, unusual_json TEXT)")
        for b in blocks:
            conn.execute("INSERT INTO service_phase_blocks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (b["index"], b["kind"], 0, 0, b["start_ms"], b["end_ms"],
                          b["minutes"], b["label"], 0.7, "{}", int(b.get("ongoing", 0)), "[]"))
        for m in marks:
            conn.execute("INSERT INTO service_phase_corrections (block_index, start_ms, "
                         "end_ms, kind, label, note, corrected_at) "
                         "VALUES (NULL,?,NULL,?,?,'','2026-08-30T12:00:00')",
                         (m["start_ms"], m.get("kind", "S"), m["label"]))
        conn.commit()
        return conn

    def block(self, start, end, label="Sermon 1", ongoing=False):
        return {"index": 0, "kind": "S", "start_ms": start, "end_ms": end,
                "minutes": max(1, (end - start) // MIN), "label": label,
                "ongoing": ongoing}

    def test_a_closed_mark_becomes_a_corrected_phase(self, tmp_path):
        conn = self.session(tmp_path,
                            marks=[{"start_ms": BASE + 5 * MIN, "label": "Sermon"}],
                            blocks=[self.block(BASE, BASE + 40 * MIN)])
        got = read_marked_phases(conn, "s.db", resolve=resolve_marks)
        assert [(p.label, p.minutes) for p in got] == [("Sermon", 35)]

    def test_an_open_mark_is_not_evidence(self, tmp_path):
        # It was still running, so it says nothing about how long that phase was.
        conn = self.session(tmp_path,
                            marks=[{"start_ms": BASE + 5 * MIN, "label": "Sermon"}],
                            blocks=[self.block(BASE, BASE + 40 * MIN, ongoing=True)])
        assert read_marked_phases(conn, "s.db", resolve=resolve_marks,
                                  now_ms=BASE + 40 * MIN) == []

    def test_a_session_with_no_tables_yields_nothing(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "bare.db")
        assert read_marked_phases(conn, "bare.db", resolve=resolve_marks) == []


class TestServiceShape:
    def phase(self, session, label, minutes, start_minute=0):
        start = BASE + start_minute * MIN
        return CorrectedPhase(session, label, "S", start, start + minutes * MIN, minutes)

    def test_it_counts_what_each_service_contained(self):
        shapes = group_by_service([
            self.phase("a.db", "Sermon 1", 30, 10),
            self.phase("a.db", "Sermon 2", 47, 50),
            self.phase("b.db", "Sermon 1", 39, 9),
        ])
        assert [s.counts.get("Sermon") for s in shapes] == [2, 1]

    def test_it_measures_how_long_the_service_ran(self):
        shapes = group_by_service([self.phase("a.db", "Songs 1", 5, 0),
                                   self.phase("a.db", "Sermon 1", 30, 100)])
        assert shapes[0].end_minutes == 130

    def test_the_profile_travels_with_the_service(self):
        shapes = group_by_service([self.phase("a.db", "Sermon 1", 30)],
                                  {"a.db": "sunday-morning"})
        assert shapes[0].profile == "sunday-morning"


class TestProposeCounts:
    def shape(self, session, sermons, minutes=120):
        return ServiceShape(session, {"Sermon": sermons}, minutes)

    def test_it_proposes_the_usual_number(self):
        shapes = [self.shape("a", 2), self.shape("b", 2), self.shape("c", 2),
                  self.shape("d", 2)]
        got = propose_counts(shapes, {})[0]
        assert got.suggested == 2 and got.actionable

    def test_one_unusual_service_does_not_raise_the_cap(self):
        # A Christmas service with three sermons must not describe every other Sunday.
        shapes = [self.shape("a", 2), self.shape("b", 2), self.shape("c", 3),
                  self.shape("d", 2)]
        assert propose_counts(shapes, {})[0].suggested == 2

    def test_too_little_evidence_is_said_rather_than_guessed(self):
        got = propose_counts([self.shape("a", 2)], {})[0]
        assert not got.actionable
        assert "needs 4" in got.evidence

    def test_it_does_not_repropose_what_is_already_set(self):
        shapes = [self.shape(str(i), 2) for i in range(4)]
        assert not propose_counts(shapes, {"Sermon": {"max": 2}})[0].actionable

    def test_the_evidence_names_the_counts(self):
        shapes = [self.shape(str(i), 2) for i in range(4)]
        assert "2, 2, 2, 2" in propose_counts(shapes, {})[0].evidence


class TestProposeServiceLength:
    def test_it_proposes_a_length_from_the_services(self):
        shapes = [ServiceShape(str(i), {"Sermon": 2}, 120 + i) for i in range(4)]
        got = propose_service_length(shapes, {})[0]
        assert got.actionable and 120 <= got.suggested <= 123

    def test_too_little_evidence_is_said_rather_than_guessed(self):
        got = propose_service_length([ServiceShape("a", {}, 120)], {})[0]
        assert not got.actionable


class TestProposalsSayWhereTheyLand:
    """The dead end this closes: a number applied somewhere the detector never reads."""

    def test_a_threshold_is_aimed_at_the_profile(self):
        phases = [CorrectedPhase("s%d.db" % i, "Sermon 1", "S", BASE,
                                 BASE + 30 * MIN, 30) for i in range(4)]
        got = propose_all(phases, {}, {})
        targets = {p["key"]: p["target"] for p in got["proposals"]}
        assert targets["sermon_min_minutes"] == "profile"

    def test_cue_phrases_are_aimed_at_the_config(self):
        # The cue compiler genuinely reads them there.
        got = propose_all([], {}, {})
        targets = {p["key"]: p["target"] for p in got["proposals"]}
        assert targets["cue_fragments.communion_verse"] == "config"
