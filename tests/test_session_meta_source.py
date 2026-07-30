"""Session provenance must describe the config the session actually runs on.

The transcription worker outlives a Start/Stop cycle and is reused, so its
module-level `config` dates from when the process spawned. It reloads a fresh copy
into process_config at each session start and runs the session on that — but the
provenance write read the stale global. A session that offloaded every caption to
a paired machine running an LLM was recorded as translating locally with MADLAD,
and that record is what a transcript is attributed from weeks later.

Provenance that is confidently wrong is worse than absent, so this pins both
halves: the builder honours the config it is handed, and the worker hands it the
fresh one.
"""

import ast
from pathlib import Path

import pytest

from conftest import extract_definitions

REPO = Path(__file__).resolve().parent.parent
SRC = (REPO / "speech_to_text.py").read_text(encoding="utf-8")


def _current_session_meta(*, global_config, session_config, state=None):
    captured = {}

    def build(cfg, version, commit, describe, hostname, madlad_default):
        captured["config"] = cfg
        return {"mt.method": (cfg.get("live_translation") or {}).get("translation_method", ""),
                "mt.offloaded": str(bool(((cfg.get("live_translation") or {})
                                          .get("remote") or {}).get("enabled"))).lower()}

    ns = extract_definitions(
        "speech_to_text.py", ["_current_session_meta", "_session_meta_enabled"],
        extra_globals={
            "config": global_config,
            "_build_session_meta": build,
            "SERVER_VERSION": "1", "SERVER_COMMIT": "c", "SERVER_DESCRIBE": "d",
            "MADLAD_DEFAULT_MODEL": "google/madlad400-3b-mt",
            "socket": type("S", (), {"gethostname": staticmethod(lambda: "host")}),
            "transcription_state": state,
        })
    # _current_session_meta calls _session_meta_enabled from the same namespace.
    ns["_session_meta_enabled"] = ns["_session_meta_enabled"]
    return ns["_current_session_meta"](session_config), captured


STALE = {"live_translation": {"translation_method": "madlad", "remote": {"enabled": False}},
         "session_meta": {"enabled": True}}
FRESH = {"live_translation": {"translation_method": "madlad",
                              "remote": {"enabled": True, "endpoint": "192.168.2.52:8080"}},
         "session_meta": {"enabled": True}}


class TestDescribesTheSessionsOwnConfig:
    def test_the_passed_config_wins_over_the_module_global(self):
        """The production failure, reduced: worker global says local, session offloads."""
        meta, captured = _current_session_meta(global_config=STALE, session_config=FRESH)
        assert captured["config"] is FRESH
        assert meta["mt.offloaded"] == "true", (
            "a session that offloads every caption must not be recorded as local")

    def test_falls_back_to_the_global_when_no_config_is_passed(self):
        """The live pipeline passes nothing, so its behaviour must be unchanged."""
        meta, captured = _current_session_meta(global_config=STALE, session_config=None)
        assert captured["config"] is STALE
        assert meta["mt.offloaded"] == "false"

    def test_recording_can_be_disabled_from_the_session_config(self):
        off = {"live_translation": {}, "session_meta": {"enabled": False}}
        meta, _ = _current_session_meta(global_config=STALE, session_config=off)
        assert meta == {}

    def test_a_disabled_global_does_not_silence_an_enabled_session(self):
        off_global = {"live_translation": {}, "session_meta": {"enabled": False}}
        meta, _ = _current_session_meta(global_config=off_global, session_config=FRESH)
        assert meta, "the session's own setting governs the session's provenance"


class TestTheWorkerPassesItsFreshConfig:
    """Structural: the call site is inside a 2000-line worker loop that cannot be
    exec'd, but passing the wrong object there is exactly what caused the bug."""

    def _func(self, name):
        return next(n for n in ast.walk(ast.parse(SRC))
                    if isinstance(n, ast.FunctionDef) and n.name == name)

    def test_initialize_database_accepts_a_session_config(self):
        args = [a.arg for a in self._func("initialize_database").args.args]
        assert "session_config" in args, (
            "without this the database and its provenance are built from whatever "
            "config the worker process happened to spawn with")

    def test_the_worker_passes_its_reloaded_config(self):
        call = next((n for n in ast.walk(ast.parse(SRC))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id == "initialize_database"), None)
        assert call is not None, "initialize_database is never called"
        assert call.args, "initialize_database() called with no config — the bug"
        assert isinstance(call.args[0], ast.Name) and call.args[0].id == "process_config", (
            "the worker must pass the config it reloaded for this session, not "
            "inherit the one it spawned with")

    def test_the_worker_reloads_before_it_initialises_the_database(self):
        reload_line = next(n.lineno for n in ast.walk(ast.parse(SRC))
                           if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "process_config"
                                   for t in n.targets))
        init_line = next(n.lineno for n in ast.walk(ast.parse(SRC))
                         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                         and n.func.id == "initialize_database")
        assert reload_line < init_line, (
            "the fresh config must exist before the session is described by it")


@pytest.mark.parametrize("name", ["_current_session_meta", "_session_meta_enabled",
                                  "initialize_database"])
def test_the_functions_under_test_still_exist(name):
    """A rename would otherwise turn the structural tests into silent no-ops."""
    assert any(isinstance(n, ast.FunctionDef) and n.name == name
               for n in ast.walk(ast.parse(SRC)))
