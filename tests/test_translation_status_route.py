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
from stt.coercion import coerce_int
from stt.llm_translate import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
    uses_local_llm,
)

NLLB = "facebook/nllb-200-distilled-600M"
MADLAD = "google/madlad400-3b-mt"
GGUF = "gemma-3-4b-it-Q4_K_M.gguf"


def _cache_stub(size=0):
    return type("C", (), {
        "get_size": staticmethod(lambda: size),
        "get_stats": staticmethod(lambda: {"size": size, "hits": 0, "misses": 0, "hit_rate": 0.0}),
    })()


def call_status(live_translation, *, local_llm=None, device=None, is_ct2=False,
                model_loaded=True, whisper_model="large-v3", trusted=(), a_pushed=None):
    """Run the route and return its JSON body as a plain dict."""
    ns = extract_definitions(
        "speech_to_text.py", ["get_translation_status", "_llm_retry_enabled"],
        extra_globals={
            "config": {"live_translation": live_translation,
                       "model": {"whisper": {"model": whisper_model}}},
            "request": type("R", (), {"remote_addr": "127.0.0.1"})(),
            "jsonify": lambda payload: payload,
            "check_ip_whitelist": lambda: True,
            "_paired_client_ok": lambda ip=None: False,
            "_pair_tokens": lambda: {},
            "_translation_clients": {},
            "_translation_clients_lock": threading.Lock(),
            "_trusted_translation_clients": set(trusted),
            "_a_pushed": dict(a_pushed or {"language": False, "glossary": False}),
            "_translation_client_ports": {},
            "_pending_pair_requests": {},
            "_live_translation_model": None,
            "_live_translation_device": device,
            "_live_translation_is_ct2": is_ct2,
            "_local_llm": local_llm,
            "_local_translate_ms_ema": None,
            "_remote_translate_ms_ema": None,
            "_llm_device_label": lambda: "metal" if local_llm is not None else None,
            "_uses_local_llm": uses_local_llm,
            "is_live_translation_model_loaded": lambda: model_loaded,
            "is_live_translation_model_loading": lambda: False,
            "get_translation_cache": lambda: _cache_stub(),
            "get_server_text_cache": lambda: _cache_stub(),
            "_get_remote_endpoint_safe": lambda: "",
            "_check_remote_reachable": lambda ep: None,
            "TRANSLATION_LANGUAGES": {"en": "English", "es": "Spanish"},
            # The payload now also reports *how* the LLM runs, so a paired machine
            # can record the configuration that shaped its captions and not only
            # the model's name. These are the helpers that answer that.
            "coerce_int": coerce_int,
            "_LLM_MIN_N_CTX": 1024,
            "_llm_system_prompt": build_system_prompt,
            "_DEFAULT_LLM_SYSTEM_PROMPT": DEFAULT_SYSTEM_PROMPT_TEMPLATE,
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


class TestWhatMachineAControls:
    """a_pushed — which settings a paired Machine A has actually taken over.

    Machine B's settings page locks controls from this. It used to lock every
    tagged control the moment any client paired, which shut B's operator out of
    the LLM settings A never sends — on the box that is the only one running the
    LLM. Being paired and being controlled are different things.
    """

    def test_a_pairing_alone_controls_nothing(self):
        body = call_status(NMT, trusted=["192.168.2.62"])
        assert body["trusted_clients"] == ["192.168.2.62"]
        assert body["a_pushed"] == []

    def test_reports_only_what_was_pushed(self):
        body = call_status(NMT, trusted=["192.168.2.62"],
                           a_pushed={"language": True, "glossary": False})
        assert body["a_pushed"] == ["language"]

    def test_the_model_is_never_something_a_can_take_over(self):
        # A cannot set this machine's model or engine at all, so no pairing state
        # may make B's settings page lock those controls.
        for pushed in ({"language": True, "glossary": True}, {"language": False, "glossary": False}):
            body = call_status(NMT, trusted=["192.168.2.62"], a_pushed=pushed)
            assert "model" not in body["a_pushed"]

    def test_reports_several_and_stays_sorted(self):
        body = call_status(NMT, trusted=["192.168.2.62"],
                           a_pushed={"language": True, "glossary": True})
        assert body["a_pushed"] == ["glossary", "language"]

    def test_unpaired_machine_reports_nothing_controlled(self):
        # No caller is paired, so the field must not leak a stale claim.
        assert call_status(NMT)["a_pushed"] == []


class TestWhisperSessionUnchanged:
    def test_names_the_whisper_model(self):
        cfg = dict(NMT, translation_method="whisper_translate")
        assert call_status(cfg)["translation_model"] == "whisper/large-v3"

    def test_carries_no_llm_claims(self):
        cfg = dict(NMT, translation_method="whisper_translate")
        assert call_status(cfg)["llm_provider"] is None


class TestLlmParametersReported:
    """The payload must describe *how* the LLM runs, not only which model.

    A paired machine records this endpoint's answer into its session database. On an
    offloaded session these settings exist nowhere else — the transcript names the
    translator but not the configuration that shaped every caption — so a service
    recorded without them cannot be replayed against a changed setting later.
    """

    def test_the_generation_settings_are_reported(self):
        cfg = dict(LLM_LOCAL, context_window=2,
                   llm=dict(LLM_LOCAL["llm"], max_tokens=200, n_ctx=4096))
        status = call_status(cfg, local_llm=object())
        assert status["llm_max_tokens"] == 200
        assert status["llm_n_ctx"] == 4096
        assert status["llm_context_window"] == 2

    def test_the_rejection_behaviour_is_reported(self):
        cfg = dict(LLM_LOCAL, llm=dict(LLM_LOCAL["llm"],
                                       retry_on_reject=False, fallback="skip"))
        status = call_status(cfg, local_llm=object())
        assert status["llm_retry_on_reject"] is False
        assert status["llm_fallback"] == "skip"

    def test_the_rejection_defaults_are_reported_when_unset(self):
        status = call_status(LLM_LOCAL, local_llm=object())
        assert status["llm_retry_on_reject"] is True
        assert status["llm_fallback"] == "nmt"

    def test_the_prompt_is_the_one_actually_sent(self):
        # Effective, not configured: the configured value is usually blank, and the
        # target language is substituted into the template before the model sees it.
        status = call_status(LLM_LOCAL, local_llm=object())
        assert "English Bibles" in status["llm_system_prompt"]
        assert "{language}" not in status["llm_system_prompt"]

    def test_a_custom_prompt_is_reported_as_built(self):
        cfg = dict(LLM_LOCAL, llm=dict(LLM_LOCAL["llm"],
                                       system_prompt="Render вечеря as communion."))
        status = call_status(cfg, local_llm=object())
        assert status["llm_system_prompt"].startswith("Render вечеря as communion.")
        assert "Translate into English." in status["llm_system_prompt"]

    def test_n_ctx_is_clamped_to_the_usable_floor(self):
        # Below the floor the prompt leaves no room for the caption, so reporting a
        # configured 512 would describe a session that could not have happened.
        cfg = dict(LLM_LOCAL, llm=dict(LLM_LOCAL["llm"], n_ctx=512))
        assert call_status(cfg, local_llm=object())["llm_n_ctx"] == 1024

    @pytest.mark.parametrize("field", ["llm_max_tokens", "llm_n_ctx", "llm_retry_on_reject",
                                       "llm_fallback", "llm_context_window", "llm_system_prompt"])
    def test_an_nmt_session_claims_none_of_them(self, field):
        assert call_status(NMT)[field] is None

    def test_an_endpoint_provider_reports_no_local_context_size(self):
        # n_ctx sizes the in-process GGUF and says nothing about a remote model's window.
        cfg = dict(LLM_LOCAL, llm={"provider": "endpoint", "endpoint": "http://h", "model": "m"})
        status = call_status(cfg)
        assert status["llm_n_ctx"] is None
        assert status["llm_max_tokens"] == 160
