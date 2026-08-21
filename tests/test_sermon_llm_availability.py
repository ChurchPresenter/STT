"""Whether this machine can summarise at all, and what it says when it cannot.

The rule is only "is an LLM configured". Offload is deliberately not part of it: it decides
where *captions* are translated, and a machine translating with MADLAD while an LLM sits
configured beside it summarises perfectly well. An earlier version refused a local model
whenever the machine offloaded, which turned a caption-routing setting into a summariser
kill switch.

The message matters as much as the verdict. .62 spent a run reporting "the model returned
nothing for any part" — a symptom, naming no fix — when the actual state was that nothing
was configured.
"""

import os
import sqlite3

import pytest

from conftest import extract_definitions

OFFLOADED = {"remote": {"enabled": True, "endpoint": "http://192.168.2.52:8080"}}
LOCAL_ONLY = {"remote": {"enabled": False, "endpoint": ""}}


def target_ns(live_translation=None, *, models_dir="/models", llama_installed=True):
    return extract_definitions(
        "speech_to_text.py",
        ["_sermon_peer_endpoint", "_sermon_llm_target", "_sermon_llm_unavailable",
         "_sermon_model_label"],
        extra_globals={
            "os": os,
            "config": {"live_translation": live_translation or LOCAL_ONLY},
            "MODELS_DIR": models_dir,
            "local_llm_available": lambda: llama_installed,
            # The real helper adds the scheme and fills in a missing port; the summariser
            # must go through it rather than reading the raw config value.
            "_get_remote_endpoint_safe": lambda: (
                (lambda ep: ep if "://" in ep else "http://" + ep)(
                    ((live_translation or LOCAL_ONLY).get("remote", {}) or {}).get("endpoint", "").strip())
                or None),
            "_llm_local_model_path": lambda d, repo, name: os.path.join(
                d, repo.replace("/", "--"), name),
        })


def reason_for(llm_cfg, live_translation=None, **kw):
    return target_ns(live_translation, **kw)["_sermon_llm_unavailable"](llm_cfg)


def target_for(llm_cfg, live_translation=None, **kw):
    return target_ns(live_translation, **kw)["_sermon_llm_target"](llm_cfg)


def label_for(llm_cfg, live_translation=None, **kw):
    return target_ns(live_translation, **kw)["_sermon_model_label"](llm_cfg)


@pytest.fixture()
def gguf(tmp_path):
    """A configured, existing local model."""
    path = tmp_path / "gemma-4-12b-it-Q4_K_M.gguf"
    path.write_bytes(b"GGUF")
    return str(path)


class TestNothingConfigured:
    """.62's actual state: provider=endpoint, endpoint empty, no GGUF."""

    @pytest.mark.parametrize("llm_cfg", [
        {},
        {"provider": "endpoint", "endpoint": ""},
        {"provider": "endpoint", "endpoint": "   "},
        {"provider": "local"},
        {"provider": "local", "gguf_path": "", "gguf_repo": "", "gguf_file": ""},
    ])
    def test_it_says_no_llm_is_configured(self, llm_cfg):
        reason = reason_for(llm_cfg)
        assert reason and "No LLM is configured" in reason

    def test_the_message_names_the_fix_not_the_symptom(self):
        reason = reason_for({})
        assert "translation settings" in reason
        assert "returned nothing" not in reason

    def test_an_empty_block_is_never_read_as_load_the_local_model(self):
        # provider defaults to endpoint; defaulting to local would mean an unconfigured
        # machine trying to load a GGUF it was never told about.
        assert "No LLM is configured" in reason_for({})


class TestPrecedence:
    """Explicit configuration wins; offload is the fallback.

    A machine told to use a particular model keeps using it whether or not its captions go
    elsewhere — offload names the peer, it does not overrule a local choice.
    """

    def test_a_configured_local_model_beats_the_peer(self, gguf):
        assert target_for({"provider": "local", "gguf_path": gguf}, OFFLOADED) == ("local", gguf)

    def test_a_configured_endpoint_beats_the_peer(self):
        assert target_for({"provider": "endpoint", "endpoint": "http://x/v1/chat"},
                          OFFLOADED) == ("endpoint", "http://x/v1/chat")

    def test_an_unconfigured_machine_falls_back_to_the_peer(self):
        # .62 exactly: provider endpoint, endpoint empty, no GGUF, captions offloaded.
        kind, detail = target_for({"provider": "endpoint", "endpoint": ""}, OFFLOADED)
        assert kind == "peer" and detail == OFFLOADED["remote"]["endpoint"]

    def test_a_bare_host_and_port_is_given_a_scheme(self):
        """The shape .62 actually stores, and the one that broke.

        requests refuses a URL with no scheme outright, so reading the config value directly
        produced a call that could never be made. Every other peer call site goes through the
        helper that fixes this.
        """
        bare = {"remote": {"enabled": True, "endpoint": "192.168.2.52:8080"}}
        kind, detail = target_for({}, bare)
        assert kind == "peer"
        assert detail.startswith("http://"), detail

    def test_the_label_keeps_the_readable_form(self):
        bare = {"remote": {"enabled": True, "endpoint": "192.168.2.52:8080"}}
        assert label_for({}, bare) == "peer:192.168.2.52:8080"

    def test_a_local_provider_with_no_model_falls_back_to_the_peer(self):
        kind, _ = target_for({"provider": "local"}, OFFLOADED)
        assert kind == "peer"

    def test_offloading_makes_an_otherwise_unusable_machine_usable(self):
        assert reason_for({}, LOCAL_ONLY) is not None
        assert reason_for({}, OFFLOADED) is None

    def test_a_disabled_remote_is_not_a_peer(self):
        cfg = {"remote": {"enabled": False, "endpoint": "http://192.168.2.52:8080"}}
        assert target_for({}, cfg)[0] is None

    def test_a_remote_with_no_endpoint_is_not_a_peer(self):
        cfg = {"remote": {"enabled": True, "endpoint": ""}}
        assert target_for({}, cfg)[0] is None


class TestProvenance:
    def test_a_peer_summary_records_the_machine_that_ran_it(self):
        # Months later, "which model wrote this" and "which machine ran it" are one question.
        assert label_for({}, OFFLOADED) == "peer:" + OFFLOADED["remote"]["endpoint"].split("://", 1)[-1]

    def test_a_local_summary_records_the_model_file(self, gguf):
        assert label_for({"provider": "local", "gguf_path": gguf}) == os.path.basename(gguf)


class TestLocalModel:
    def test_an_explicit_path_that_exists_is_usable(self, gguf):
        assert reason_for({"provider": "local", "gguf_path": gguf}) is None

    def test_a_path_that_does_not_exist_is_not_configured(self, tmp_path):
        reason = reason_for({"provider": "local", "gguf_path": str(tmp_path / "absent.gguf")})
        assert reason and "No LLM is configured" in reason

    def test_repo_and_file_resolve_through_the_model_directory(self, tmp_path):
        models = tmp_path / "models"
        (models / "google--gemma").mkdir(parents=True)
        (models / "google--gemma" / "m.gguf").write_bytes(b"GGUF")
        cfg = {"provider": "local", "gguf_repo": "google/gemma", "gguf_file": "m.gguf"}
        assert reason_for(cfg, models_dir=str(models)) is None

    def test_an_explicit_path_wins_over_repo_and_file(self, gguf, tmp_path):
        # Mirrors get_local_llm's load order; reporting on the repo when a path overrode
        # it would name a file that never loads.
        cfg = {"provider": "local", "gguf_path": gguf,
               "gguf_repo": "nope/nope", "gguf_file": "missing.gguf"}
        assert reason_for(cfg, models_dir=str(tmp_path / "empty")) is None

    def test_a_missing_runtime_is_reported_distinctly(self, gguf):
        # The operator picked a model and the runtime for it is absent — a different thing
        # to go and fix than having configured nothing.
        reason = reason_for({"provider": "local", "gguf_path": gguf}, llama_installed=False)
        assert reason and "llama-cpp-python" in reason
        assert "No LLM is configured" not in reason

    def test_a_missing_runtime_is_not_reported_when_no_model_is_set(self):
        # Nothing configured is the more useful complaint of the two.
        reason = reason_for({"provider": "local"}, llama_installed=False)
        assert "No LLM is configured" in reason


class TestBlockedRunDoesNoWork:
    """The defect this file was written for.

    The worker asked per chunk instead of once, so a 27k-character sermon produced ~30
    refused calls with a sleep between each, ~30 identical log lines, and a stored error
    describing the symptom rather than the cause. The call count is the only thing that
    pins it: a version that checks inside the loop still ends with status=error.
    """

    def session(self, tmp_path):
        from stt.sermon_summary import STATUS_PENDING, ensure_tables, save_summary
        path = str(tmp_path / "2026-08-16_225506.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "ts_ms INTEGER, text TEXT, is_final INTEGER, denied INTEGER)")
        conn.executemany(
            "INSERT INTO transcriptions (ts_ms, text, is_final, denied) VALUES (?, ?, 1, 0)",
            [(1_700_000_000_000 + i * 60_000, f"a constructed caption number {i}")
             for i in range(60)])
        conn.commit()
        ensure_tables(conn)
        from stt.sermon_summary import fingerprint, read_sermon_rows
        rows = read_sermon_rows(conn, 1_700_000_000_000, 1_700_000_000_000 + 60 * 60_000)
        fp = fingerprint(rows)
        save_summary(conn, fingerprint=fp, label="Sermon 1", start_ms=rows[0].ts_ms,
                     end_ms=rows[-1].ts_ms, status=STATUS_PENDING)
        conn.close()
        return path, fp

    def run_worker(self, path, fp, llm_cfg, generate=None):
        import stt.sermon_summary as ss

        calls = []
        boom = {}
        ns = extract_definitions(
            "speech_to_text.py",
            ["_sermon_peer_endpoint", "_sermon_llm_target", "_sermon_llm_unavailable",
             "_SermonUnavailable", "_sermon_summarize_one"],
            extra_globals={
                "os": os, "sqlite3": sqlite3, "time": __import__("time"),
                "datetime": __import__("datetime").datetime,
                "config": {"live_translation": {"llm": llm_cfg}},
                "MODELS_DIR": "/models",
                "local_llm_available": lambda: True,
                "_llm_local_model_path": lambda d, r, n: os.path.join(d, r, n),
                "_sermon_summary_config": lambda: {},
                "coerce_int": __import__("stt.coercion", fromlist=["x"]).coerce_int,
                "_sermon_load": ss.load_summary,
                "_sermon_save": ss.save_summary,
                "_sermon_read_rows": ss.read_sermon_rows,
                "_sermon_fingerprint": ss.fingerprint,
                "_sermon_transcript_text": ss.transcript_text,
                "_sermon_delete": ss.delete_summary,
                "_sermon_mark_error": ss.mark_error,
                "_sermon_chunk_rows": ss.chunk_rows,
                "_sermon_map_prompt": ss.build_map_prompt,
                "_sermon_reduce_prompt": ss.build_reduce_prompt,
                "_sermon_parse_sections": ss.parse_sections,
                "_sermon_parse_chapters": ss.parse_chapters,
                "_sermon_snap_chapters": ss.snap_chapters,
                "_sermon_format_offset": ss.format_offset,
                "_SERMON_PENDING": ss.STATUS_PENDING,
                "_SERMON_RUNNING": ss.STATUS_RUNNING,
                "_SERMON_DONE": ss.STATUS_DONE,
                "_SERMON_ERROR": ss.STATUS_ERROR,
                "_SERMON_MAP_SYSTEM": ss.MAP_SYSTEM,
                "_sermon_model_label": lambda cfg: "test-model",
                # Small on purpose: the fixture has to chunk into several parts, or
                # "stopped at the first" and "ran them all" are the same number.
                "_sermon_budget": lambda cfg, sys_p, n: (60, None),
                "_sermon_generate_waiting": (
                    generate(calls, boom) if generate
                    else (lambda *a, **k: calls.append(a) or "a gist")),
                "_sermon_fold": lambda g, *a, **k: g,
                "_sermon_emit": lambda *a, **k: None,
                "_archive_write_done": lambda *a, **k: None,
                "_sermon_yield": lambda: None,
                "_server_shutting_down": type("E", (), {"is_set": staticmethod(lambda: False)})(),
            })
        boom["exc"] = ns["_SermonUnavailable"]
        ns["_sermon_summarize_one"](path, fp, True)
        return calls

    def test_an_unconfigured_machine_makes_no_generate_calls(self, tmp_path):
        path, fp = self.session(tmp_path)
        calls = self.run_worker(path, fp, {})
        assert calls == [], f"asked a model that cannot run {len(calls)} times"

    def test_it_stores_the_cause_not_the_symptom(self, tmp_path):
        from stt.sermon_summary import STATUS_ERROR, load_summary
        path, fp = self.session(tmp_path)
        self.run_worker(path, fp, {})
        conn = sqlite3.connect(path)
        entry = load_summary(conn, fp)
        conn.close()
        assert entry["status"] == STATUS_ERROR
        assert "No LLM is configured" in entry["error"]
        assert "returned nothing" not in entry["error"]

    def test_a_configured_machine_does_reach_the_model(self, tmp_path, gguf):
        # The counter has to be able to go up, or the first test proves nothing.
        path, fp = self.session(tmp_path)
        calls = self.run_worker(path, fp, {"provider": "local", "gguf_path": gguf})
        assert calls, "a configured machine never called the model"

    def test_a_call_that_cannot_be_made_stops_at_the_first_part(self, tmp_path, gguf):
        """The bug this replaced: an unusable endpoint asked ~30 times.

        The count is the only thing that separates the fix from the bug — both end at
        status=error either way.
        """
        path, fp = self.session(tmp_path)

        def raiser(calls, boom):
            def _gen(*a, **k):
                calls.append(a)
                raise boom["exc"]("could not reach the paired machine at http://x: boom")
            return _gen

        calls = self.run_worker(path, fp, {"provider": "local", "gguf_path": gguf},
                                generate=raiser)
        assert len(calls) == 1, f"kept going after a fatal failure: {len(calls)} calls"

    def test_it_stores_the_transport_reason_not_the_symptom(self, tmp_path, gguf):
        from stt.sermon_summary import STATUS_ERROR, load_summary
        path, fp = self.session(tmp_path)

        def raiser(calls, boom):
            def _gen(*a, **k):
                calls.append(a)
                raise boom["exc"]("could not reach the paired machine at http://x: boom")
            return _gen

        self.run_worker(path, fp, {"provider": "local", "gguf_path": gguf}, generate=raiser)
        conn = sqlite3.connect(path)
        entry = load_summary(conn, fp)
        conn.close()
        assert entry["status"] == STATUS_ERROR
        assert "could not reach the paired machine" in entry["error"]
        assert "returned nothing" not in entry["error"]

    def test_an_empty_reply_still_reports_returning_nothing(self, tmp_path, gguf):
        # The other half of the distinction: a model that answers with nothing is not the
        # same as a call that could not be made, and must still say so.
        from stt.sermon_summary import STATUS_ERROR, load_summary
        path, fp = self.session(tmp_path)

        def empty(calls, boom):
            return lambda *a, **k: calls.append(a) or None

        calls = self.run_worker(path, fp, {"provider": "local", "gguf_path": gguf},
                                generate=empty)
        conn = sqlite3.connect(path)
        entry = load_summary(conn, fp)
        conn.close()
        assert len(calls) > 1, "an empty reply is per-part, so the run should continue"
        assert entry["status"] == STATUS_ERROR
        assert "returned nothing" in entry["error"]

