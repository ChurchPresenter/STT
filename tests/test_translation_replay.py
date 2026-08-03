"""Offline replay harness for translation changes (stt/translation_replay.py)."""

import glob
import json
import os
import sqlite3

import pytest

from stt import translation_replay
from stt.translation_replay import (
    CaptionPair,
    ReplayResult,
    as_dict,
    compare,
    engine_breakdown,
    from_dict,
    load_session_pairs,
    model_breakdown,
    resolve_model,
    render_comparison,
    render_summary,
    replay,
    session_settings,
    settings_mismatch,
    shipped_run,
    summarize,
)

SCHEMA = """
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, text TEXT, translated_text TEXT,
    denied INTEGER DEFAULT 0, is_final INTEGER DEFAULT 1,
    ts_ms INTEGER, translation_ts_ms INTEGER)
"""


def make_db(path, rows):
    """A session database holding ``rows`` of (text, translated, denied, final, ts, tts)."""
    conn = sqlite3.connect(str(path))
    conn.execute(SCHEMA)
    for index, (text, translated, denied, final, ts_ms, translation_ts_ms) in enumerate(rows):
        conn.execute(
            "INSERT INTO transcriptions (timestamp, text, translated_text, denied, is_final,"
            " ts_ms, translation_ts_ms) VALUES (?,?,?,?,?,?,?)",
            ("2026-08-02 10:%02d:00" % index, text, translated, denied, final,
             ts_ms, translation_ts_ms))
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture()
def session_db(tmp_path):
    return make_db(tmp_path / "session.db", [
        ("Мир вам.", "Peace be with you.", 0, 1, 1000, 1400),
        ("Субтитры подогнал «Симон»", None, 1, 1, 2000, None),      # denied: not speech
        ("Аминь.", "Amen.", 0, 1, 3000, 3600),
        ("частичный текст", None, 0, 0, 4000, None),                 # partial, not final
        ("   ", None, 0, 1, 5000, None),                             # blank
        ("Господь благ.", None, 0, 1, 6000, None),                   # never translated
    ])


class TestLoadSessionPairs:
    def test_keeps_only_final_undenied_speech(self, session_db):
        pairs = load_session_pairs(session_db)
        assert [p.source for p in pairs] == ["Мир вам.", "Аминь.", "Господь благ."]

    def test_carries_shipped_translation_and_latency(self, session_db):
        first = load_session_pairs(session_db)[0]
        assert first.shipped == "Peace be with you."
        assert first.shipped_latency_ms == 400

    def test_untranslated_row_has_no_shipped_text(self, session_db):
        last = load_session_pairs(session_db)[-1]
        assert last.shipped is None
        assert last.shipped_latency_ms is None

    def test_min_source_words_filter(self, session_db):
        pairs = load_session_pairs(session_db, min_source_words=2)
        assert [p.source for p in pairs] == ["Мир вам.", "Господь благ."]

    def test_open_does_not_write_to_the_database(self, tmp_path, session_db):
        before = (tmp_path / "session.db").stat().st_mtime_ns
        load_session_pairs(session_db)
        assert (tmp_path / "session.db").stat().st_mtime_ns == before
        # A read that journals would leave these behind next to the operator's file.
        assert not (tmp_path / "session.db-wal").exists()
        assert not (tmp_path / "session.db-shm").exists()


class TestShippedRun:
    def test_scores_shipped_captions_without_translating(self, session_db):
        results = shipped_run(load_session_pairs(session_db), "en")
        assert [r.accepted for r in results] == ["Peace be with you.", "Amen.", None]
        assert results[-1].reason == "untranslated"

    def test_flags_a_shipped_caption_that_todays_rules_reject(self, tmp_path):
        # The scripture failure mode: the reference's numbers vanish into a recited passage.
        db = make_db(tmp_path / "s.db", [
            ("Апостол Павел, 11 главе, с 23 стиха.",
             "I beseech you, brothers, not to be recipients of ignorance.", 0, 1, 1, 2),
        ])
        result = shipped_run(load_session_pairs(db), "en")[0]
        assert result.accepted is None
        assert result.reason == "numbers"


class TestReplay:
    def test_scores_each_caption(self, session_db):
        pairs = load_session_pairs(session_db)
        results = replay(pairs, lambda text: "translated", "en", clock=lambda: 0.0)
        assert len(results) == len(pairs)
        assert all(r.accepted == "translated" for r in results)

    def test_a_failed_call_scores_as_a_rejection_not_a_gap(self, session_db):
        pairs = load_session_pairs(session_db)
        results = replay(pairs, lambda text: None, "en", clock=lambda: 0.0)
        assert len(results) == len(pairs)
        assert all(r.reason == "empty" for r in results)

    def test_an_exception_does_not_lose_the_run(self, session_db):
        def explode(text):
            if "Аминь" in text:
                raise RuntimeError("endpoint died")
            return "ok"

        results = replay(load_session_pairs(session_db), explode, "en", clock=lambda: 0.0)
        assert [r.ok for r in results] == [True, False, True]

    def test_times_each_call_from_the_injected_clock(self, session_db):
        ticks = iter([0.0, 0.5, 1.0, 1.25, 2.0, 2.0])
        results = replay(load_session_pairs(session_db), lambda text: "ok", "en",
                         clock=lambda: next(ticks))
        assert [r.elapsed_ms for r in results] == [500.0, 250.0, 0.0]

    def test_reports_progress(self, session_db):
        seen = []
        replay(load_session_pairs(session_db), lambda text: "ok", "en",
               on_progress=lambda done, total: seen.append((done, total)), clock=lambda: 0.0)
        assert seen == [(1, 3), (2, 3), (3, 3)]


def result(source, accepted, reason=None, row_id=1, elapsed=None, shipped=None):
    pair = CaptionPair(row_id, "2026-08-02 10:00:00", source, shipped, None)
    return ReplayResult(pair, accepted, accepted, reason, elapsed)


class TestSummarize:
    def test_counts_acceptance_and_reasons(self):
        summary = summarize([
            result("Мир вам.", "Peace be with you."),
            result("Аминь.", None, "numbers"),
            result("Аминь.", None, "numbers"),
            result("Аминь.", None, "refusal"),
        ], "run")
        assert (summary.total, summary.accepted, summary.rejected) == (4, 1, 3)
        assert summary.by_reason == {"numbers": 2, "refusal": 1}

    def test_counts_outputs_far_shorter_than_their_source(self):
        long_source = "Третье, что мы видим в этом отрывке, когда он объяснил притчу, это радость."
        summary = summarize([
            result(long_source, "Thank you."),
            result(long_source, "The third thing we see in this passage, when he explained the parable, is joy."),
        ], "run")
        assert summary.short_outputs == 1

    def test_short_sources_are_left_out_of_the_ratio(self):
        # A three-word source expands unpredictably; the ratio only means something
        # once there is enough text for it to converge.
        summary = summarize([result("Аминь.", "Amen.")], "run")
        assert summary.short_outputs == 0

    def test_latency_percentiles(self):
        summary = summarize(
            [result("a b c d e f g h", "x", elapsed=ms) for ms in (100.0, 200.0, 900.0)], "run")
        assert summary.latency_p50_ms == 200.0
        assert summary.latency_p90_ms == 900.0

    def test_no_timings_leaves_latency_unset(self):
        assert summarize([result("Аминь.", "Amen.")], "run").latency_p50_ms is None


class TestCompare:
    def test_splits_captions_that_crossed_the_line(self):
        before = [result("a", "kept", row_id=1), result("b", None, "numbers", row_id=2),
                  result("c", "was good", row_id=3), result("d", None, "refusal", row_id=4)]
        after = [result("a", "kept", row_id=1), result("b", "now fine", row_id=2),
                 result("c", None, "numbers", row_id=3), result("d", None, "refusal", row_id=4)]
        outcome = compare(before, after)
        assert outcome.identical == 1
        assert [r.pair.row_id for r in outcome.fixed] == [2]
        assert [r.pair.row_id for r in outcome.broken] == [3]
        assert outcome.both_rejected == 1

    def test_different_wording_counts_as_changed_not_identical(self):
        outcome = compare([result("a", "one wording")], [result("a", "another wording")])
        assert (outcome.changed, outcome.identical) == (1, 0)

    def test_rows_present_in_only_one_run_are_reported(self):
        outcome = compare([result("a", "x", row_id=1)],
                          [result("a", "x", row_id=1), result("b", "y", row_id=2)])
        assert outcome.only_in_one == 1

    def test_matches_by_row_id_not_position(self):
        before = [result("a", "x", row_id=1), result("b", "y", row_id=2)]
        after = [result("b", "y", row_id=2), result("a", "x", row_id=1)]
        assert compare(before, after).identical == 2


class TestRoundTrip:
    def test_results_survive_json(self, tmp_path):
        original = [result("Мир вам.", "Peace be with you.", elapsed=12.5, shipped="Peace!"),
                    result("Аминь.", None, "numbers", row_id=2)]
        restored = from_dict(json.loads(json.dumps(as_dict(original))))
        assert [r.accepted for r in restored] == ["Peace be with you.", None]
        assert [r.reason for r in restored] == [None, "numbers"]
        assert restored[0].elapsed_ms == 12.5
        assert restored[0].pair.shipped == "Peace!"

    def test_a_restored_run_still_compares(self, tmp_path):
        original = [result("a", "x", row_id=1)]
        restored = from_dict(as_dict(original))
        assert compare(original, restored).identical == 1


class TestHttpTranslator:
    def test_posts_the_caption_and_returns_the_translation(self, monkeypatch):
        sent = {}

        def fake_urlopen(request, timeout=None):
            sent["url"] = request.full_url
            sent["body"] = json.loads(request.data.decode("utf-8"))
            sent["timeout"] = timeout
            return FakeResponse({"translated_text": "Peace be with you."})

        monkeypatch.setattr(translation_replay.urllib.request, "urlopen", fake_urlopen)
        translate = translation_replay.http_translator("http://192.168.2.52:8080/", "ru", "en")
        assert translate("Мир вам.") == "Peace be with you."
        assert sent["url"] == "http://192.168.2.52:8080/api/translate"
        assert sent["body"]["text"] == "Мир вам."
        assert sent["body"]["source_lang"] == "ru"
        assert sent["body"]["target_lang"] == "en"

    def test_bypasses_the_servers_text_cache(self, monkeypatch):
        # Without this the second run replays the first run's answers and every
        # comparison reports a perfect, fictional, zero-change result.
        sent = {}

        def fake_urlopen(request, timeout=None):
            sent.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse({"translated_text": "x"})

        monkeypatch.setattr(translation_replay.urllib.request, "urlopen", fake_urlopen)
        translation_replay.http_translator("http://h", "ru", "en")("text")
        assert sent["return_extras"] is True

    def test_an_empty_translation_is_none(self, monkeypatch):
        monkeypatch.setattr(translation_replay.urllib.request, "urlopen",
                            lambda request, timeout=None: FakeResponse({"translated_text": ""}))
        assert translation_replay.http_translator("http://h", "ru", "en")("text") is None


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestCli:
    def test_scores_the_shipped_captions_without_an_endpoint(self, session_db, capsys):
        assert translation_replay.main([session_db]) == 0
        out = capsys.readouterr().out
        assert "loaded 3 captions" in out
        assert "shipped: 3 captions, 2 accepted, 1 rejected" in out

    def test_replays_against_an_endpoint_and_compares(self, session_db, capsys, monkeypatch):
        monkeypatch.setattr(translation_replay.urllib.request, "urlopen",
                            lambda request, timeout=None: FakeResponse({"translated_text": "Amen."}))
        assert translation_replay.main([session_db, "--endpoint", "http://h", "--label", "cand"]) == 0
        out = capsys.readouterr().out
        assert "cand: 3 captions" in out
        assert "fixed: 1" in out  # the never-translated caption now has one

    def test_writes_and_reuses_a_run(self, session_db, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(translation_replay.urllib.request, "urlopen",
                            lambda request, timeout=None: FakeResponse({"translated_text": "Amen."}))
        saved = str(tmp_path / "run.json")
        translation_replay.main([session_db, "--endpoint", "http://h", "--out", saved])
        capsys.readouterr()

        translation_replay.main([session_db, "--endpoint", "http://h", "--baseline", saved])
        out = capsys.readouterr().out
        assert "identical: 3" in out  # same stub, so nothing should have moved

    def test_limit_caps_the_replay(self, session_db, capsys):
        translation_replay.main([session_db, "--limit", "1"])
        assert "loaded 1 captions" in capsys.readouterr().out

    def test_a_database_with_no_captions_fails(self, tmp_path, capsys):
        empty = make_db(tmp_path / "empty.db", [("Субтитры", None, 1, 1, 1, None)])
        assert translation_replay.main([empty]) == 1
        assert "no translatable captions" in capsys.readouterr().out


class TestRendering:
    def test_summary_mentions_the_counts(self):
        text = render_summary(summarize([result("a", "x"), result("b", None, "numbers")], "run"))
        assert "run: 2 captions, 1 accepted, 1 rejected" in text
        assert "numbers=1" in text

    def test_comparison_shows_a_broken_caption(self):
        before = [result("Псалом 23", "Psalm 23", row_id=1, shipped="Psalm 23")]
        after = [result("Псалом 23", None, "numbers", row_id=1, shipped="Psalm 23")]
        text = render_comparison(compare(before, after))
        assert "broken: 1" in text
        assert "Псалом 23" in text


PROVENANCE_SCHEMA = """
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, text TEXT, translated_text TEXT,
    denied INTEGER DEFAULT 0, is_final INTEGER DEFAULT 1,
    ts_ms INTEGER, translation_ts_ms INTEGER,
    asr_model TEXT, mt_engine TEXT, mt_model TEXT)
"""


def make_provenance_db(path, rows):
    """A session database recorded since rows carry the model that produced them."""
    conn = sqlite3.connect(str(path))
    conn.execute(PROVENANCE_SCHEMA)
    for index, (text, translated, engine, model) in enumerate(rows):
        conn.execute(
            "INSERT INTO transcriptions (timestamp, text, translated_text, denied, is_final,"
            " ts_ms, translation_ts_ms, mt_engine, mt_model) VALUES (?,?,?,0,1,?,?,?,?)",
            ("2026-08-02 10:%02d:00" % index, text, translated, 1000, 1400, engine, model))
    conn.commit()
    conn.close()
    return str(path)


class TestPerRowProvenance:
    """Which engine produced a caption, so a replay is not scored against the wrong model."""

    def test_engine_and_model_are_loaded(self, tmp_path):
        db = make_provenance_db(tmp_path / "p.db", [
            ("Мир вам.", "Peace be with you.", "llm", "gemma-3-4b.gguf"),
        ])
        pair = load_session_pairs(db)[0]
        assert pair.shipped_engine == "llm"
        assert pair.shipped_model == "gemma-3-4b.gguf"

    def test_the_breakdown_separates_the_two_engines(self, tmp_path):
        # The reason this exists: a session configured for the LLM contains NMT rows
        # wherever the LLM declined, and they used to be indistinguishable.
        db = make_provenance_db(tmp_path / "p.db", [
            ("a", "A", "llm", "gguf"), ("b", "B", "llm", "gguf"),
            ("c", "C", "nmt", "madlad"),
        ])
        assert engine_breakdown(load_session_pairs(db)) == {"llm": 2, "nmt": 1}

    def test_an_untranslated_row_counts_for_no_engine(self, tmp_path):
        db = make_provenance_db(tmp_path / "p.db", [("a", None, None, None)])
        assert engine_breakdown(load_session_pairs(db)) == {}

    def test_an_older_database_still_loads(self, session_db):
        # Sessions recorded before these columns existed must stay replayable.
        pairs = load_session_pairs(session_db)
        assert [p.source for p in pairs] == ["Мир вам.", "Аминь.", "Господь благ."]
        assert all(p.shipped_engine is None for p in pairs)

    def test_an_older_database_reports_unknown_rather_than_zero(self, session_db):
        # "not recorded" and "nothing fell back" must not look the same.
        assert engine_breakdown(load_session_pairs(session_db)) == {}

    def test_the_cli_says_when_provenance_is_missing(self, session_db, capsys):
        translation_replay.main([session_db])
        assert "predates per-row provenance" in capsys.readouterr().out

    def test_the_cli_reports_the_breakdown_when_present(self, tmp_path, capsys):
        db = make_provenance_db(tmp_path / "p.db", [
            ("a", "A", "llm", "gguf"), ("b", "B", "nmt", "madlad"),
        ])
        translation_replay.main([db])
        assert "shipped by: llm=1, nmt=1" in capsys.readouterr().out


class TestRealServiceDatabases:
    """The harness against real sessions, when the machine holding them runs the suite.

    The harness is public — it is just code. The sessions are not: a service database
    is verbatim congregation speech, and a replay run is that speech again alongside a
    model's translation of every line. Both stay on the machine that recorded them
    (see .gitignore), so these cases skip everywhere else and CI stays green.

    Drop or symlink session databases into tests/fixtures/sessions/ to enable them.
    What they check is what synthetic rows cannot: that a real schema — years of
    migrations, partial rows, denied rows, columns that postdate the file — still
    loads, and that reading one leaves it untouched.
    """

    DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "sessions")

    @pytest.fixture
    def databases(self):
        found = sorted(glob.glob(os.path.join(self.DIR, "*.db"))) if os.path.isdir(self.DIR) else []
        if not found:
            pytest.skip("no local session databases; see tests/fixtures/sessions/ in .gitignore")
        return found

    def test_every_local_session_loads(self, databases):
        for path in databases:
            pairs = load_session_pairs(path)
            assert pairs, f"{os.path.basename(path)} yielded no captions"
            assert all(p.source.strip() for p in pairs)
            assert all(p.row_id > 0 for p in pairs)

    def test_reading_leaves_the_database_untouched(self, databases):
        # An operator's recording is evidence. Reading it must not journal, and must
        # not leave sidecars beside a file this tool does not own.
        for path in databases:
            before = os.stat(path).st_mtime_ns, os.path.getsize(path)
            load_session_pairs(path)
            assert (os.stat(path).st_mtime_ns, os.path.getsize(path)) == before
            assert not os.path.exists(path + "-wal")
            assert not os.path.exists(path + "-shm")

    def test_shipped_captions_score_without_a_model(self, databases):
        for path in databases:
            pairs = load_session_pairs(path)
            results = shipped_run(pairs, "en")
            assert len(results) == len(pairs)
            summary = summarize(results, os.path.basename(path))
            # A real service is overwhelmingly captions that worked. A rule that
            # rejects a large share of one is a rule that is wrong, not a service
            # that was bad.
            translated = [r for r in results if r.pair.shipped]
            if translated:
                rejected = sum(1 for r in translated if not r.ok)
                assert rejected / len(translated) < 0.10, (
                    f"{os.path.basename(path)}: {rejected}/{len(translated)} shipped "
                    f"captions rejected — {summary.by_reason}")


META_SCHEMA = "CREATE TABLE session_meta (key TEXT PRIMARY KEY, value TEXT)"


def add_meta(path, values):
    conn = sqlite3.connect(str(path))
    conn.execute(META_SCHEMA)
    conn.executemany("INSERT INTO session_meta (key, value) VALUES (?,?)", list(values.items()))
    conn.commit()
    conn.close()
    return str(path)


class TestSessionSettings:
    """A replay configured by what the operator typed can disagree with the session
    for reasons nobody sees. The session knows what produced it; ask it."""

    def test_a_local_session_reports_its_own_settings(self, session_db):
        add_meta(session_db, {
            "mt.method": "llm", "mt.llm.model": "gemma.gguf", "mt.context_window": "1",
            "mt.llm.max_tokens": "160", "mt.llm.retry_on_reject": "true",
            "mt.llm.system_prompt": "You translate live captions.", "app.commit": "abc1234",
        })
        s = session_settings(session_db)
        assert s["method"] == "llm"
        assert s["model"] == "gemma.gguf"
        assert s["system_prompt"] == "You translate live captions."
        assert s["app_commit"] == "abc1234"

    def test_an_offloaded_session_prefers_the_translating_box(self, session_db):
        # The local mt.llm.* keys describe a model that never ran: on an offloaded
        # session the remote does the work, and its values are the only true ones.
        add_meta(session_db, {
            "mt.method": "madlad",
            "mt.model": "google/madlad400-3b-mt",
            "mt.remote.effective.method": "llm",
            "mt.remote.effective.model": "gemma-3-4b-it-Q4_K_M.gguf",
            "mt.remote.effective.llm_system_prompt": "Remote prompt.",
            "mt.remote.effective.llm_max_tokens": "160",
        })
        s = session_settings(session_db)
        assert s["method"] == "llm"
        assert s["model"] == "gemma-3-4b-it-Q4_K_M.gguf"
        assert s["system_prompt"] == "Remote prompt."
        assert s["max_tokens"] == "160"

    def test_a_session_without_the_table_reports_nothing(self, session_db):
        assert session_settings(session_db) == {}

    def test_blank_values_are_not_reported_as_recorded(self, session_db):
        # An empty string is "not recorded", not a setting worth comparing against.
        add_meta(session_db, {"mt.llm.system_prompt": "", "mt.method": "llm"})
        s = session_settings(session_db)
        assert "system_prompt" not in s
        assert s["method"] == "llm"

    def test_reading_settings_leaves_the_database_untouched(self, tmp_path, session_db):
        add_meta(session_db, {"mt.method": "llm"})
        before = (tmp_path / "session.db").stat().st_mtime_ns
        session_settings(session_db)
        assert (tmp_path / "session.db").stat().st_mtime_ns == before
        assert not (tmp_path / "session.db-wal").exists()


class TestSettingsMismatch:
    def test_a_difference_is_named(self):
        out = settings_mismatch({"model": "a.gguf", "context_window": "1"},
                                {"model": "b.gguf"})
        assert out == ["model: session='a.gguf', this run='b.gguf'"]

    def test_agreement_is_silent(self):
        assert settings_mismatch({"model": "a.gguf"}, {"model": "a.gguf"}) == []

    def test_values_are_compared_as_text(self):
        # session_meta stores everything as TEXT; an int candidate must still match.
        assert settings_mismatch({"max_tokens": "160"}, {"max_tokens": 160}) == []

    def test_a_setting_the_session_never_recorded_is_not_a_mismatch(self):
        # Absence is not disagreement — it is the older-session case, reported
        # separately so it is not mistaken for a change the operator made.
        assert settings_mismatch({}, {"model": "b.gguf"}) == []

    def test_every_difference_is_listed(self):
        out = settings_mismatch({"model": "a", "context_window": "1", "fallback": "nmt"},
                                {"model": "b", "context_window": "2", "fallback": "nmt"})
        assert len(out) == 2


class TestCliSettings:
    def test_the_cli_prints_recorded_settings(self, session_db, capsys):
        add_meta(session_db, {"mt.method": "llm", "mt.llm.model": "gemma.gguf",
                              "mt.llm.system_prompt": "P" * 80})
        translation_replay.main([session_db])
        out = capsys.readouterr().out
        assert "method           llm" in out
        assert "80 chars" in out

    def test_the_cli_says_when_the_prompt_was_not_recorded(self, session_db, capsys):
        add_meta(session_db, {"mt.method": "llm"})
        translation_replay.main([session_db])
        assert "the prompt that produced these captions is unknown" in capsys.readouterr().out

    def test_the_cli_says_when_nothing_was_recorded(self, session_db, capsys):
        translation_replay.main([session_db])
        assert "predates session_meta" in capsys.readouterr().out


class TestResolveModel:
    """A NULL model column means "what the session says", not "unknown".

    Rows carry a model name only once it stops matching what the session recorded at
    start. Reading the column alone would report a hot reload as the only
    attributable caption in a service and the rest as unattributed — backwards.
    """

    def pair(self, model=None, shipped="x"):
        return CaptionPair(1, "t", "src", shipped, None, shipped_engine="llm",
                           shipped_model=model)

    def test_null_resolves_to_the_session_model(self):
        assert resolve_model(self.pair(), {"model": "gemma.gguf"}) == "gemma.gguf"

    def test_a_row_value_wins_over_the_session(self):
        # The hot-reload case: this row is explicitly not what the session started with.
        assert resolve_model(self.pair("qwen.gguf"), {"model": "gemma.gguf"}) == "qwen.gguf"

    def test_an_untranslated_row_has_no_model(self):
        assert resolve_model(self.pair(shipped=None), {"model": "gemma.gguf"}) is None

    def test_no_session_model_leaves_it_unknown(self):
        assert resolve_model(self.pair(), {}) is None

    def test_the_breakdown_counts_the_session_model_for_null_rows(self, tmp_path):
        db = make_provenance_db(tmp_path / "p.db", [
            ("a", "A", "llm", None), ("b", "B", "llm", None),
        ])
        pairs = load_session_pairs(db)
        assert model_breakdown(pairs, {"model": "gemma.gguf"}) == {"gemma.gguf": 2}

    def test_a_mid_session_change_shows_as_two_models(self, tmp_path):
        # The point of keeping the column: a service that changed model is a service
        # measured as one configuration when it was two.
        db = make_provenance_db(tmp_path / "p.db", [
            ("a", "A", "llm", None), ("b", "B", "llm", "qwen.gguf"),
        ])
        pairs = load_session_pairs(db)
        assert model_breakdown(pairs, {"model": "gemma.gguf"}) == \
            {"gemma.gguf": 1, "qwen.gguf": 1}

    def test_the_cli_warns_when_the_model_changed_mid_session(self, tmp_path, capsys):
        db = make_provenance_db(tmp_path / "p.db", [
            ("a", "A", "llm", None), ("b", "B", "llm", "qwen.gguf"),
        ])
        add_meta(db, {"mt.llm.model": "gemma.gguf", "mt.method": "llm"})
        translation_replay.main([db])
        assert "the model changed mid-session" in capsys.readouterr().out
