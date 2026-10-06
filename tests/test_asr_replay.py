"""Replaying a session through the sentence filters (stt/asr_replay.py)."""

import sqlite3

from stt import asr_replay
from stt.asr_replay import Row, Settings, compare, load_rows, replay, score, shipped_run

PHRASES = ("DimaTorzok", "Субтитры создавал")


def row(i, text, *, denied=False, reason=None, speech_type="Speaking", tag="Speech"):
    return Row(id=i, text=text, original_text=text, denied=denied, reason=reason,
               speech_type=speech_type, audio_tag=tag)


def settings(**kw):
    return Settings(phrases=PHRASES, music_reason="music:0.5", **kw)


class TestReplay:
    def test_spoken_row_over_music_is_rescued_by_the_speech_rule(self):
        rows = [row(1, "a reading under keyboards", denied=True, reason="music:0.5", speech_type="Music", tag="Speech")]
        before = shipped_run(rows)
        after = replay(rows, settings())
        cmp = compare(before, after)
        assert cmp.rescued == [1] and not cmp.newly_denied

    def test_sung_row_stays_denied(self):
        rows = [row(1, "a sung line", denied=True, reason="music:0.5", speech_type="Music", tag="Music")]
        assert replay(rows, settings()).outcomes == {1: "music:0.5"}

    def test_rule_can_be_replayed_off(self):
        rows = [row(1, "spoken", denied=True, reason="music:0.5", speech_type="Music", tag="Speech")]
        assert replay(rows, settings(speech_overrides_music=False)).outcomes == {1: "music:0.5"}

    def test_glued_credit_row_is_rescued(self):
        rows = [row(1, "Субтитры создавал DimaTorzok Мир вам", denied=True, reason="hallucination")]
        cmp = compare(shipped_run(rows), replay(rows, settings()))
        assert cmp.rescued == [1]

    def test_pure_credit_stays_denied(self):
        rows = [row(1, "Субтитры создавал DimaTorzok", denied=True, reason="hallucination")]
        assert replay(rows, settings()).outcomes == {1: "hallucination"}

    def test_dup_and_short_are_carried_not_replayed(self):
        rows = [row(1, "same line", denied=True, reason="dup"), row(2, "ok", denied=True, reason="short")]
        assert replay(rows, settings()).outcomes == {1: "dup", 2: "short"}

    def test_a_tighter_filter_reports_newly_denied(self):
        rows = [row(1, "welcome Subtitles by someone")]
        tight = Settings(phrases=("Subtitles by",))
        cmp = compare(shipped_run(rows), replay(rows, tight))
        assert cmp.rescued == [] and cmp.newly_denied == [] and cmp.unchanged == 1  # strip keeps the speech
        only = [row(2, "Subtitles by")]
        assert compare(shipped_run(only), replay(only, tight)).newly_denied == [2]


class TestScore:
    def test_speech_lost_and_junk_kept(self):
        run = asr_replay.Run("r", {1: "music:0.5", 2: None, 3: None, 4: "hallucination"})
        s = score(run, frozenset({1, 2}), frozenset({3, 4}))
        assert s.speech_lost == [1] and s.junk_kept == [3]
        assert s.speech_recall == 0.5 and s.junk_rejected == 0.5

    def test_empty_truth_is_not_a_failure(self):
        s = score(asr_replay.Run("r", {}), frozenset(), frozenset())
        assert s.speech_recall == 1.0 and s.junk_rejected == 1.0


class TestLoadAndReport:
    def _db(self, tmp_path):
        path = str(tmp_path / "s.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, text TEXT, original_text TEXT, denied INTEGER,"
                     " denied_reason TEXT, speech_type TEXT, audio_tag TEXT, is_final INTEGER)")
        conn.executemany("INSERT INTO transcriptions VALUES (?,?,?,?,?,?,?,?)", [
            (1, "spoken", None, 1, "music:0.5", "Music", "Speech", 1),
            (2, "partial", None, 0, None, None, None, 0),
            (3, "kept", "kept raw", 0, None, "Speaking", "Speech", 1),
        ])
        conn.commit()
        conn.close()
        return path

    def test_load_rows_reads_final_rows_only(self, tmp_path):
        rows = load_rows(self._db(tmp_path))
        assert [r.id for r in rows] == [1, 3]
        assert rows[1].original_text == "kept raw" and rows[0].original_text == "spoken"

    def test_report_names_ids_and_never_text(self, tmp_path, capsys):
        assert asr_replay.main([self._db(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "rescued 1" in out and ": 1" in out
        assert "spoken" not in out and "kept raw" not in out
