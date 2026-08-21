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

OFFLOADED = {"remote": {"enabled": True, "endpoint": "http://192.168.2.52:8080/api/translate"}}
LOCAL_ONLY = {"remote": {"enabled": False, "endpoint": ""}}


def reason_for(llm_cfg, live_translation=None, *, models_dir="/models",
               llama_installed=True):
    ns = extract_definitions(
        "speech_to_text.py",
        ["_sermon_llm_unavailable"],
        extra_globals={
            "os": os,
            "config": {"live_translation": live_translation or LOCAL_ONLY},
            "MODELS_DIR": models_dir,
            "local_llm_available": lambda: llama_installed,
            "_llm_local_model_path": lambda d, repo, name: os.path.join(
                d, repo.replace("/", "--"), name),
        })
    return ns["_sermon_llm_unavailable"](llm_cfg)


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

    def test_offloading_does_not_change_the_message(self):
        assert reason_for({}, OFFLOADED) == reason_for({}, LOCAL_ONLY)

    def test_an_empty_block_is_never_read_as_load_the_local_model(self):
        # provider defaults to endpoint; defaulting to local would mean an unconfigured
        # machine trying to load a GGUF it was never told about.
        assert "No LLM is configured" in reason_for({})


class TestOffloadIsNotConsulted:
    """The rule that replaced the one this file used to pin."""

    def test_a_configured_endpoint_works_while_offloading(self):
        assert reason_for({"provider": "endpoint", "endpoint": "http://x/v1/chat"},
                          OFFLOADED) is None

    def test_a_configured_local_model_works_while_offloading(self, gguf):
        assert reason_for({"provider": "local", "gguf_path": gguf}, OFFLOADED) is None

    def test_the_verdict_is_identical_offloaded_or_not(self, gguf):
        cfg = {"provider": "local", "gguf_path": gguf}
        assert reason_for(cfg, OFFLOADED) == reason_for(cfg, LOCAL_ONLY) is None


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

    def run_worker(self, path, fp, llm_cfg):
        import stt.sermon_summary as ss

        calls = []
        ns = extract_definitions(
            "speech_to_text.py",
            ["_sermon_llm_unavailable", "_sermon_summarize_one"],
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
                "_sermon_budget": lambda cfg, sys_p, n: (2000, None),
                "_sermon_llm_generate": lambda *a, **k: calls.append(a) or "unreachable",
                "_sermon_fold": lambda g, *a, **k: g,
                "_sermon_emit": lambda *a, **k: None,
                "_archive_write_done": lambda *a, **k: None,
                "_sermon_yield": lambda: None,
                "_server_shutting_down": type("E", (), {"is_set": staticmethod(lambda: False)})(),
            })
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

