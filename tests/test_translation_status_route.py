"""/api/translation/status must describe the engine that is actually running.

This endpoint is not only a dashboard. A paired machine reads it to record what
translated its captions, so a wrong answer is written into a session database and
survives the server that produced it. That happened: a session whose captions came
from a GGUF on a GPU was recorded as "google/madlad400-3b-mt, device: cpu", because
the payload always described the NMT model.

The route is exercised by extracting it from the monolith and running it against a
stub namespace (see tests/conftest.py) — the module cannot be imported, and CI
installs no Flask, so booting the app is not an option. jsonify is stubbed to
return the mapping unchanged, which is what these assertions care about.
"""

import threading

import pytest

from conftest import extract_definitions

NLLB = "facebook/nllb-200-distilled-600M"
MADLAD = "google/madlad400-3b-mt"
GGUF = "gemma-3-4b-it-Q4_K_M.gguf"


def _cache_stub(size=0):
    return type("C", (), {
        "get_size": staticmethod(lambda: size),
        "get_stats": staticmethod(lambda: {"size": size, "hits": 0, "misses": 0, "hit_rate": 0.0}),
    })()


def call_status(live_translation, *, local_llm=None, device=None, is_ct2=False,
                model_loaded=True, whisper_model="large-v3"):
    """Run the route and return its JSON body as a plain dict."""
    ns = extract_definitions(
        "speech_to_text.py", ["get_translation_status"],
        extra_globals={
            "config": {"live_translation": live_translation,
                       "model": {"whisper": {"model": whisper_model}}},
            "request": type("R", (), {"remote_addr": "127.0.0.1"})(),
            "jsonify": lambda payload: payload,
            "check_ip_whitelist": lambda: True,
            "_is_trusted_translation_client": lambda ip: False,
            "_translation_clients": {},
            "_translation_clients_lock": threading.Lock(),
            "_trusted_translation_clients": set(),
            "_pending_pair_requests": {},
            "_live_translation_model": None,
            "_live_translation_device": device,
            "_live_translation_is_ct2": is_ct2,
            "_local_llm": local_llm,
            "_local_translate_ms_ema": None,
            "_remote_translate_ms_ema": None,
            "_llm_device_label": lambda: "metal" if local_llm is not None else None,
            "is_live_translation_model_loaded": lambda: model_loaded,
            "is_live_translation_model_loading": lambda: False,
            "get_translation_cache": lambda: _cache_stub(),
            "get_server_text_cache": lambda: _cache_stub(),
            "_get_remote_endpoint_safe": lambda: "",
            "_check_remote_reachable": lambda ep: None,
            "TRANSLATION_LANGUAGES": {"en": "English", "es": "Spanish"},
            "transcription_state": {"running": False},
            "time": __import__("time"),
        })
    return ns["get_translation_status"]()


LLM_LOCAL = {
    "enabled": True, "translation_method": "llm", "target_language": "en",
    "translation_model": MADLAD,          # the standby NMT model
    "use_fp16": False, "use_ctranslate2": True, "ct2_compute_type": "auto",
    "llm": {"provider": "local", "gguf_repo": "ggml-org/g", "gguf_file": GGUF},
}

NMT = {
    "enabled": True, "translation_method": "madlad", "target_language": "en",
    "translation_model": MADLAD,
    "use_fp16": True, "use_ctranslate2": True, "ct2_compute_type": "int8",
}


class TestLlmSession:
    """The regression: an LLM session described by the NMT model's fields."""

    def test_names_the_gguf_not_the_standby_nmt_model(self):
        body = call_status(LLM_LOCAL, local_llm=object())
        assert body["translation_model"] == GGUF
        assert MADLAD not in str(body["translation_model"])

    def test_reports_the_llm_provider_and_model(self):
        body = call_status(LLM_LOCAL, local_llm=object())
        assert body["llm_provider"] == "local"
        assert body["llm_model"] == GGUF

    def test_reports_where_the_llm_runs(self):
        assert call_status(LLM_LOCAL, local_llm=object())["model_device"] == "metal"

    def test_model_loaded_tracks_the_llm_not_the_nmt_model(self):
        """model_loaded must follow the engine in use, not the idle NMT one."""
        assert call_status(LLM_LOCAL, local_llm=object(), model_loaded=False)["model_loaded"] is True
        assert call_status(LLM_LOCAL, local_llm=None, model_loaded=True)["model_loaded"] is False

    @pytest.mark.parametrize("field", ["use_fp16", "use_ctranslate2", "ct2_compute_type",
                                       "is_ctranslate2"])
    def test_carries_no_nmt_precision_claims(self, field):
        """These were reported from config even on an LLM session.

        A paired machine copies them into a transcript, so it would have recorded
        "CTranslate2, int8" for captions produced by llama.cpp.
        """
        assert call_status(LLM_LOCAL, local_llm=object())[field] is None

    def test_endpoint_provider_reports_its_model_and_url(self):
        cfg = dict(LLM_LOCAL, llm={"provider": "endpoint", "model": "qwen2.5:7b",
                                   "endpoint": "http://host:11434/api/chat"})
        body = call_status(cfg)
        assert body["translation_model"] == "qwen2.5:7b"
        assert body["llm_endpoint"] == "http://host:11434/api/chat"

    def test_endpoint_provider_claims_no_local_device(self):
        """The device belongs to the other machine; naming one would be a guess."""
        cfg = dict(LLM_LOCAL, llm={"provider": "endpoint", "model": "m", "endpoint": "u"})
        assert call_status(cfg)["model_device"] is None


class TestNmtSessionUnchanged:
    """The LLM work must not have altered what an NMT session reports."""

    def test_names_the_nmt_model(self):
        assert call_status(NMT, device="cuda")["translation_model"] == MADLAD

    def test_keeps_its_precision_and_backend_fields(self):
        body = call_status(NMT, device="cuda", is_ct2=True)
        assert body["use_fp16"] is True
        assert body["use_ctranslate2"] is True
        assert body["ct2_compute_type"] == "int8"
        assert body["is_ctranslate2"] is True

    def test_reports_the_device_the_nmt_model_landed_on(self):
        # 'cpu' on a machine meant to accelerate is the classic cause of the
        # seconds-per-sentence problem, so this field has to keep working.
        assert call_status(NMT, device="cpu")["model_device"] == "cpu"

    @pytest.mark.parametrize("field", ["llm_provider", "llm_model", "llm_endpoint"])
    def test_carries_no_llm_claims(self, field):
        assert call_status(NMT, device="cuda")[field] is None


class TestWhisperSessionUnchanged:
    def test_names_the_whisper_model(self):
        cfg = dict(NMT, translation_method="whisper_translate")
        assert call_status(cfg)["translation_model"] == "whisper/large-v3"

    def test_carries_no_llm_claims(self):
        cfg = dict(NMT, translation_method="whisper_translate")
        assert call_status(cfg)["llm_provider"] is None
