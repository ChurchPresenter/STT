"""/api/translate/unload must free whichever translation engine is resident.

Machine A posts to this route when it stops transcription, so that the machine it
offloads to releases the weights. It did not. The route gated on the NMT flag
alone, which is false during an LLM session, so it took the "Model not loaded"
branch and returned success while the in-process GGUF stayed resident for the
life of the process — A logged that reply as a completed unload, and the failure
was invisible from either side.

The route is exercised by extracting it from the monolith and running it against a
stub namespace (see tests/conftest.py) — the module cannot be imported, and CI
installs no Flask, so booting the app is not an option. jsonify is stubbed to
return the mapping unchanged, which is what these assertions care about.
"""

import threading

from conftest import extract_definitions
from stt.llm_translate import uses_local_llm

GGUF = "gemma-3-4b-it-Q4_K_M.gguf"
PAIRED = "192.168.2.62"

LLM_LOCAL = {
    "enabled": True, "translation_method": "llm",
    "translation_model": "google/madlad400-3b-mt",   # the standby NMT model
    "llm": {"provider": "local", "gguf_repo": "ggml-org/g", "gguf_file": GGUF},
}

LLM_ENDPOINT = {
    "enabled": True, "translation_method": "llm",
    "llm": {"provider": "endpoint", "endpoint": "http://elsewhere:11434", "model": "gemma3"},
}

NMT = {"enabled": True, "translation_method": "nllb"}


class Unloads:
    """Records which releasers ran, and lets the test wait for the threads."""

    def __init__(self):
        self.called = []
        self._done = threading.Event()
        self._lock = threading.Lock()
        self.expected = 0

    def _record(self, name):
        with self._lock:
            self.called.append(name)
            if len(self.called) >= self.expected:
                self._done.set()

    def nmt(self):
        self._record("nmt")

    def llm(self):
        self._record("llm")

    def settled(self, expected):
        """Block until `expected` releasers have run, or fail on the deadline."""
        self.expected = expected
        if expected == 0:
            return sorted(self.called)
        assert self._done.wait(5), f"unload threads did not run: {self.called}"
        return sorted(self.called)


def call_unload(live_translation, *, nmt_loaded=False, llm_loaded=False,
                caller=PAIRED, trusted=(PAIRED,), clients=None):
    """Run the route; returns (body, status_or_None, Unloads recorder)."""
    unloads = Unloads()
    ns = extract_definitions(
        "speech_to_text.py", ["translate_unload"],
        extra_globals={
            "config": {"live_translation": live_translation},
            "request": type("R", (), {"remote_addr": caller})(),
            "jsonify": lambda payload: payload,
            "_translation_clients": dict(clients or {}),
            "_translation_clients_lock": threading.Lock(),
            "_trusted_translation_clients": set(trusted),
            "_is_trusted_translation_client": lambda ip: ip in set(trusted),
            "_uses_local_llm": uses_local_llm,
            "is_live_translation_model_loaded": lambda: nmt_loaded,
            "is_local_llm_loaded": lambda: llm_loaded,
            "unload_live_translation_model": unloads.nmt,
            "unload_local_llm": unloads.llm,
        })
    result = ns["translate_unload"]()
    if isinstance(result, tuple):
        return result[0], result[1], unloads
    return result, None, unloads


class TestLocalLlmSession:
    """The regression: the GGUF survived every stop A asked for."""

    def test_frees_the_gguf(self):
        _, status, unloads = call_unload(LLM_LOCAL, llm_loaded=True)
        assert status is None
        assert unloads.settled(1) == ["llm"]

    def test_does_not_report_an_unloaded_model_as_absent(self):
        body, _, unloads = call_unload(LLM_LOCAL, llm_loaded=True)
        unloads.settled(1)
        assert body["success"] is True
        assert body["message"] != "Model not loaded"
        assert body["unloaded"] == ["LLM"]

    def test_leaves_the_standby_nmt_model_alone(self):
        _, _, unloads = call_unload(LLM_LOCAL, llm_loaded=True, nmt_loaded=False)
        assert "nmt" not in unloads.settled(1)

    def test_frees_both_when_both_are_resident(self):
        _, _, unloads = call_unload(LLM_LOCAL, llm_loaded=True, nmt_loaded=True)
        assert unloads.settled(2) == ["llm", "nmt"]


class TestOtherEngines:
    def test_nmt_session_unloads_the_nmt_model(self):
        body, _, unloads = call_unload(NMT, nmt_loaded=True)
        assert unloads.settled(1) == ["nmt"]
        assert body["unloaded"] == ["translation model"]

    def test_endpoint_provider_owns_no_local_weights(self):
        """The model lives on a third machine; this one has nothing to release."""
        body, _, unloads = call_unload(LLM_ENDPOINT, llm_loaded=True)
        assert unloads.settled(0) == []
        assert body["message"] == "Model not loaded"

    def test_nothing_resident_is_not_an_error(self):
        body, status, unloads = call_unload(LLM_LOCAL)
        assert status is None and body["success"] is True
        assert body["message"] == "Model not loaded"
        assert unloads.settled(0) == []


class TestGuards:
    """Both predate this fix and must survive it."""

    def test_unpaired_caller_is_refused(self):
        _, status, unloads = call_unload(LLM_LOCAL, llm_loaded=True,
                                         caller="10.0.0.9", trusted=(PAIRED,))
        assert status == 403
        assert unloads.settled(0) == []

    def test_another_active_client_blocks_the_unload(self):
        import time
        body, _, unloads = call_unload(
            LLM_LOCAL, llm_loaded=True,
            clients={"192.168.2.70": time.time()})
        assert body["success"] is False
        assert body["active"] == 1
        assert unloads.settled(0) == []

    def test_a_long_idle_client_does_not_block_the_unload(self):
        import time
        _, _, unloads = call_unload(
            LLM_LOCAL, llm_loaded=True,
            clients={"192.168.2.70": time.time() - 600})
        assert unloads.settled(1) == ["llm"]
