"""The summariser must never load a local GGUF on a machine that offloads.

A machine that hands its captions to a paired server keeps its live_translation.llm
block configured but unused, so `provider` can still read 'local' with a GGUF named
beside it. On a machine that offloads *because* it cannot run that model, loading it is
not a degraded summary — it is that machine falling over mid-service.

The NLLB path already refuses to load a local model while offloading (_offload_no_local
in speech_to_text.py). This pins the same refusal for the LLM path, which reached it by
a different route and did not inherit the check.
"""

import pytest

from conftest import extract_definitions
from stt.session_meta import is_offloaded

OFFLOADED = {"remote": {"enabled": True, "endpoint": "http://192.168.2.52:8080/api/translate"}}
LOCAL_ONLY = {"remote": {"enabled": False, "endpoint": ""}}


def reason_for(llm_cfg, live_translation, *, llama_installed=True):
    ns = extract_definitions(
        "speech_to_text.py",
        ["_sermon_llm_unavailable"],
        extra_globals={
            "config": {"live_translation": live_translation},
            "_translation_is_offloaded": is_offloaded,
            "local_llm_available": lambda: llama_installed,
        })
    return ns["_sermon_llm_unavailable"](llm_cfg)


class TestOffloadedMachine:
    def test_a_local_provider_is_refused_while_offloading(self):
        reason = reason_for({"provider": "local", "gguf_file": "gemma.gguf"}, OFFLOADED)
        assert reason and "offload" in reason

    def test_the_refusal_points_at_the_machine_that_can(self):
        reason = reason_for({"provider": "local"}, OFFLOADED)
        assert "machine that holds it" in reason

    def test_an_endpoint_provider_is_still_allowed_while_offloading(self):
        # Offload is about captions; a reachable LLM endpoint is a separate arrangement
        # and does not load anything on this box.
        assert reason_for({"provider": "endpoint", "endpoint": "http://x/v1/chat"},
                          OFFLOADED) is None

    def test_a_remote_configured_but_disabled_is_not_offloading(self):
        cfg = {"remote": {"enabled": False, "endpoint": "http://192.168.2.52:8080/x"}}
        assert reason_for({"provider": "local"}, cfg) is None

    def test_a_remote_enabled_with_no_endpoint_is_not_offloading(self):
        cfg = {"remote": {"enabled": True, "endpoint": ""}}
        assert reason_for({"provider": "local"}, cfg) is None


class TestLocalMachine:
    def test_a_local_provider_is_allowed(self):
        assert reason_for({"provider": "local", "gguf_file": "gemma.gguf"}, LOCAL_ONLY) is None

    def test_a_missing_runtime_is_reported(self):
        reason = reason_for({"provider": "local"}, LOCAL_ONLY, llama_installed=False)
        assert reason and "llama-cpp-python" in reason

    @pytest.mark.parametrize("endpoint", ["", "   ", None])
    def test_an_endpoint_provider_without_an_endpoint_is_reported(self, endpoint):
        reason = reason_for({"provider": "endpoint", "endpoint": endpoint}, LOCAL_ONLY)
        assert reason and "endpoint" in reason

    def test_the_default_provider_is_endpoint_not_local(self):
        # An empty llm block must not be read as "load the local model".
        reason = reason_for({}, LOCAL_ONLY)
        assert reason and "endpoint" in reason
