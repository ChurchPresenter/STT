"""Start-button readiness: is the *selected* transcription model actually on disk?

speech_to_text.py cannot be imported (ML libraries, Flask app, background threads at
import time), so these extract the individual functions and exec them against a stub
namespace — see tests/conftest.py:extract_definitions.

This gates the Start button in both the web UI and the watchdog GUI, and since the
shipped config now selects no model at all, "nothing selected" is the state a fresh
install is in — not an edge case.
"""

import os

import pytest

from conftest import extract_definitions


def _ns(models_dir):
    return extract_definitions(
        "speech_to_text.py", ["_selected_model_downloaded", "_setup_status"],
        {"MODELS_DIR": str(models_dir), "config": {}})


def _cfg(**model):
    return {"model": model}


class TestNothingSelected:
    """The fresh-install state: the config names no model."""

    @pytest.mark.parametrize("name", ["", "   ", None])
    def test_an_unset_whisper_model_is_not_ready(self, tmp_path, name):
        ns = _ns(tmp_path)
        cfg = _cfg(type="whisper", backend="faster-whisper", whisper={"model": name})
        assert ns["_selected_model_downloaded"](cfg) is False

    @pytest.mark.parametrize("model_id", ["", "   ", None])
    def test_an_unset_huggingface_model_is_not_ready(self, tmp_path, model_id):
        # The bug this pins: "" joined to MODELS_DIR is the models directory itself,
        # which exists — so an unset model reported as downloaded and the UI offered
        # Start for a model nobody had chosen.
        ns = _ns(tmp_path)
        cfg = _cfg(type="huggingface", huggingface={"model_id": model_id})
        assert ns["_selected_model_downloaded"](cfg) is False

    def test_a_missing_model_section_is_not_ready(self, tmp_path):
        ns = _ns(tmp_path)
        assert ns["_selected_model_downloaded"]({}) is False


class TestSelectedAndPresent:
    def test_a_downloaded_faster_whisper_model_is_ready(self, tmp_path):
        (tmp_path / "faster-whisper-small").mkdir()
        ns = _ns(tmp_path)
        cfg = _cfg(type="whisper", backend="faster-whisper", whisper={"model": "small"})
        assert ns["_selected_model_downloaded"](cfg) is True

    def test_a_different_downloaded_model_does_not_count(self, tmp_path):
        # "downloaded AND selected": the worker loads exactly what is configured.
        (tmp_path / "faster-whisper-small").mkdir()
        ns = _ns(tmp_path)
        cfg = _cfg(type="whisper", backend="faster-whisper", whisper={"model": "medium"})
        assert ns["_selected_model_downloaded"](cfg) is False

    def test_a_downloaded_huggingface_model_is_ready(self, tmp_path):
        (tmp_path / "openai--whisper-tiny").mkdir()
        ns = _ns(tmp_path)
        cfg = _cfg(type="huggingface", huggingface={"model_id": "openai/whisper-tiny"})
        assert ns["_selected_model_downloaded"](cfg) is True

    def test_a_custom_path_must_exist(self, tmp_path):
        present = tmp_path / "my-model.bin"
        present.write_bytes(b"x")
        ns = _ns(tmp_path)
        assert ns["_selected_model_downloaded"](_cfg(type="custom",
                                                     custom={"model_path": str(present)})) is True
        assert ns["_selected_model_downloaded"](_cfg(type="custom",
                                                     custom={"model_path": ""})) is False


class TestSetupStatus:
    """What a new operator sees on first run."""

    def test_an_unconfigured_install_is_not_ready_and_says_why(self, tmp_path):
        ns = _ns(tmp_path)
        ns["config"] = {"model": {"type": "whisper", "backend": "faster-whisper",
                                  "whisper": {"model": ""}},
                        "audio": {}}
        ns["_mic_explicitly_selected"] = lambda cfg: False
        status = ns["_setup_status"]()
        assert status["ready"] is False
        assert status["model_ready"] is False
        assert "Model Manager" in status["model_hint"]
        assert "microphone" in status["mic_hint"].lower()

    def test_a_fully_set_up_install_reports_ready_with_no_hints(self, tmp_path):
        (tmp_path / "faster-whisper-small").mkdir()
        ns = _ns(tmp_path)
        ns["config"] = {"model": {"type": "whisper", "backend": "faster-whisper",
                                  "whisper": {"model": "small"}}}
        ns["_mic_explicitly_selected"] = lambda cfg: True
        status = ns["_setup_status"]()
        assert status == {"model_ready": True, "mic_ready": True, "ready": True,
                          "model_hint": "", "mic_hint": ""}


class TestShippedDefaultIsNotReady:
    def test_the_shipped_config_needs_a_model_chosen_first(self, tmp_path):
        # End to end on the real template: copy it as a fresh install's config and
        # confirm the operator is told to pick a model rather than being offered a
        # Start that would fail in the worker.
        import json
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "config.default.json")
        with open(path, encoding="utf-8") as fh:
            shipped = json.load(fh)
        ns = _ns(tmp_path)
        assert ns["_selected_model_downloaded"](shipped) is False
