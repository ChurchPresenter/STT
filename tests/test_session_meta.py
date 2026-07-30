"""Session provenance mapping (stt/session_meta.py:build_session_meta and helpers)."""

import pytest

from stt.session_meta import (
    build_session_meta,
    changed_keys,
    read_history,
    resolve_asr_implementation,
    resolve_asr_model,
)

MADLAD_DEFAULT = "google/madlad400-3b-mt"


def build(config, **kwargs):
    """build_session_meta with fixed identity so tests assert on config handling."""
    kwargs.setdefault("version", "26.1.168")
    kwargs.setdefault("commit", "abc1234")
    kwargs.setdefault("describe", "26.1.168-3-gabc1234")
    kwargs.setdefault("hostname", "stt-box")
    kwargs.setdefault("madlad_default", MADLAD_DEFAULT)
    kwargs.setdefault("started_at", "2026-05-20T18:39:19")
    return build_session_meta(config, **kwargs)


class TestIdentity:
    def test_records_build_and_host(self):
        meta = build({})
        assert meta["app.version"] == "26.1.168"
        assert meta["app.commit"] == "abc1234"
        assert meta["app.describe"] == "26.1.168-3-gabc1234"
        assert meta["host.name"] == "stt-box"
        assert meta["session.started_at"] == "2026-05-20T18:39:19"

    def test_started_at_defaults_to_now(self):
        meta = build_session_meta({}, "v", "c", "d", "h", MADLAD_DEFAULT)
        # ISO 8601 seconds precision, e.g. 2026-05-20T18:39:19
        assert meta["session.started_at"].count(":") == 2
        assert "T" in meta["session.started_at"]


class TestValuesAreStrings:
    def test_every_value_is_a_string(self):
        config = {
            "model": {"type": "whisper", "backend": "faster-whisper", "whisper": {"model": "small"}},
            "whisper_decoding": {"live_transcription": {"beam_size": 3, "temperature": 0}},
            "vad": {"enabled": True, "threshold": 0.5},
            "live_translation": {"enabled": True, "context_window": 1},
            "hallucination_filter": {"enabled": True, "phrases": ["a", "b"]},
        }
        meta = build(config)
        assert meta, "expected a populated mapping"
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items())

    def test_booleans_are_lowercase_json_style(self):
        meta = build({"vad": {"enabled": True}, "live_translation": {"enabled": False}})
        assert meta["asr.vad.enabled"] == "true"
        assert meta["mt.enabled"] == "false"

    def test_none_becomes_empty_not_the_word_none(self):
        meta = build({"live_translation": {"target_language": None}})
        assert meta["mt.target_language"] == ""

    def test_unserialisable_value_degrades_to_str(self):
        # A config value JSON can't encode must still round-trip as *something*
        # rather than blowing up provenance for the whole session.
        meta = build({"vad": {"threshold": {1, 2}}})
        assert meta["asr.vad.threshold"] in ("{1, 2}", "{2, 1}")

    def test_list_temperature_is_json_not_python_repr(self):
        config = {"whisper_decoding": {"file_transcription": {"temperature": [0.0, 0.2]}},
                  "file_transcription": {"model": {}}}
        meta = build(config)
        assert meta["asr.file.decode.temperature"] == "[0.0,0.2]"


class TestAsrResolution:
    def test_whisper_type_reads_the_whisper_block(self):
        cfg = {"type": "whisper", "whisper": {"model": "large-v3"}}
        assert resolve_asr_model(cfg) == "large-v3"

    def test_huggingface_type_reads_the_model_id(self):
        cfg = {"type": "huggingface", "whisper": {"model": "small"},
               "huggingface": {"model_id": "openai/whisper-tiny"}}
        assert resolve_asr_model(cfg) == "openai/whisper-tiny"

    def test_custom_type_reads_the_model_path(self):
        cfg = {"type": "custom", "whisper": {"model": "small"},
               "custom": {"model_path": "/models/mine"}}
        assert resolve_asr_model(cfg) == "/models/mine"

    def test_missing_type_defaults_to_whisper(self):
        assert resolve_asr_model({"whisper": {"model": "base"}}) == "base"

    def test_faster_whisper_and_openai_whisper_are_distinguished(self):
        # backend only means anything for type "whisper", and the two backends
        # are different implementations - "whisper" must not read as faster-whisper.
        assert resolve_asr_implementation(
            {"type": "whisper", "backend": "faster-whisper"}) == "faster-whisper"
        assert resolve_asr_implementation(
            {"type": "whisper", "backend": "whisper"}) == "openai-whisper"
        assert resolve_asr_implementation({"type": "whisper"}) == "openai-whisper"

    def test_non_whisper_type_ignores_backend(self):
        assert resolve_asr_implementation(
            {"type": "huggingface", "backend": "faster-whisper"}) == "huggingface"

    def test_full_live_decode_block_is_captured(self):
        decode = {
            "beam_size": 3,
            "best_of": 1,
            "temperature": 0,
            "condition_on_previous_text": False,
            "logprob_threshold": -0.5,
            "no_speech_threshold": 0.6,
            "compression_ratio_threshold": 1.8,
        }
        meta = build({"whisper_decoding": {"live_transcription": decode}})
        for key in decode:
            assert f"asr.decode.{key}" in meta, f"{key} not recorded"
        # The tuned thresholds are the whole point - they must survive verbatim.
        assert meta["asr.decode.logprob_threshold"] == "-0.5"
        assert meta["asr.decode.compression_ratio_threshold"] == "1.8"
        assert meta["asr.decode.condition_on_previous_text"] == "false"

    def test_absent_decode_keys_are_omitted_not_blanked(self):
        meta = build({"whisper_decoding": {"live_transcription": {"beam_size": 5}}})
        assert meta["asr.decode.beam_size"] == "5"
        assert "asr.decode.best_of" not in meta


class TestFileTranscription:
    def test_empty_backend_inherits_the_main_model(self):
        config = {
            "model": {"type": "whisper", "backend": "faster-whisper",
                      "whisper": {"model": "small"}},
            "file_transcription": {"language": "auto",
                                   "model": {"backend": "", "type": "whisper"}},
        }
        meta = build(config)
        assert meta["asr.file.backend"] == "faster-whisper"
        assert meta["asr.file.implementation"] == "faster-whisper"

    def test_explicit_backend_overrides_the_main_model(self):
        config = {
            "model": {"type": "whisper", "backend": "faster-whisper",
                      "whisper": {"model": "small"}},
            "file_transcription": {"model": {"backend": "whisper", "type": "whisper",
                                             "whisper": {"model": "large-v3"}}},
        }
        meta = build(config)
        assert meta["asr.file.implementation"] == "openai-whisper"
        assert meta["asr.file.model"] == "large-v3"

    def test_file_model_falls_back_to_live_model(self):
        config = {
            "model": {"type": "whisper", "whisper": {"model": "medium"}},
            "file_transcription": {"model": {"backend": "", "type": "whisper"}},
        }
        assert build(config)["asr.file.model"] == "medium"

    def test_omitted_entirely_when_unconfigured(self):
        meta = build({"model": {"type": "whisper", "whisper": {"model": "small"}}})
        assert not any(k.startswith("asr.file.") for k in meta)


class TestTranslation:
    def test_madlad_engine_with_stale_nllb_id_records_what_actually_runs(self):
        config = {"live_translation": {"translation_method": "madlad",
                                       "translation_model": "facebook/nllb-200-distilled-600M"}}
        meta = build(config)
        assert meta["mt.model"] == MADLAD_DEFAULT
        # the stale configured value is kept alongside so the mismatch is visible
        assert meta["mt.model_configured"] == "facebook/nllb-200-distilled-600M"

    def test_nllb_engine_keeps_its_configured_model(self):
        config = {"live_translation": {"translation_method": "nllb",
                                       "translation_model": "facebook/nllb-200-distilled-600M"}}
        assert build(config)["mt.model"] == "facebook/nllb-200-distilled-600M"

    def test_madlad_engine_with_madlad_id_is_untouched(self):
        config = {"live_translation": {"translation_method": "madlad",
                                       "translation_model": "google/madlad400-7b-mt"}}
        assert build(config)["mt.model"] == "google/madlad400-7b-mt"

    def test_generation_params_are_flattened(self):
        config = {"live_translation": {"generation_params": {
            "num_beams": 5, "no_repeat_ngram_size": 4,
            "repetition_penalty": 1.1, "length_penalty": 1.0}}}
        meta = build(config)
        assert meta["mt.gen.num_beams"] == "5"
        assert meta["mt.gen.no_repeat_ngram_size"] == "4"
        assert meta["mt.gen.repetition_penalty"] == "1.1"
        assert meta["mt.gen.length_penalty"] == "1.0"

    def test_context_window_is_recorded(self):
        # The setting most implicated in fragment mistranslation, and invisible
        # in the transcript.
        assert build({"live_translation": {"context_window": 1}})["mt.context_window"] == "1"

    def test_remote_offload_split_is_recorded(self):
        config = {"live_translation": {"remote": {
            "enabled": True, "endpoint": "http://192.168.2.52:8080",
            "model": "", "fallback": "skip"}}}
        meta = build(config)
        assert meta["mt.remote.enabled"] == "true"
        assert meta["mt.remote.endpoint"] == "http://192.168.2.52:8080"
        assert meta["mt.remote.fallback"] == "skip"

    def test_remote_block_omitted_when_absent(self):
        meta = build({"live_translation": {"enabled": True}})
        assert not any(k.startswith("mt.remote.") for k in meta)

    def test_ct2_and_device_settings_are_recorded(self):
        config = {"live_translation": {"use_ctranslate2": True, "ct2_compute_type": "auto",
                                       "use_gpu": True, "use_fp16": False}}
        meta = build(config)
        assert meta["mt.use_ctranslate2"] == "true"
        assert meta["mt.ct2_compute_type"] == "auto"
        assert meta["mt.use_gpu"] == "true"
        assert meta["mt.use_fp16"] == "false"


class TestFilters:
    def test_hallucination_filter_state_and_phrase_count(self):
        config = {"hallucination_filter": {"enabled": True, "phrases": ["a", "b", "c"],
                                            "cjk_filter_enabled": True}}
        meta = build(config)
        assert meta["filter.hallucination_enabled"] == "true"
        assert meta["filter.hallucination_phrase_count"] == "3"
        assert meta["filter.cjk_enabled"] == "true"

    def test_phrase_count_zero_when_absent_or_wrong_type(self):
        assert build({})["filter.hallucination_phrase_count"] == "0"
        assert build({"hallucination_filter": {"phrases": "oops"}})[
            "filter.hallucination_phrase_count"] == "0"

    def test_glossary_revision_is_recorded(self):
        config = {"custom_dictionary": {"nllb_glossary_enabled": True,
                                         "file": "custom_dictionary.json"}}
        meta = build(config)
        assert meta["glossary.enabled"] == "true"
        assert meta["glossary.file"] == "custom_dictionary.json"


class TestBestEffort:
    """A session must still start when the config is partial or malformed."""

    @pytest.mark.parametrize("config", [
        None,
        {},
        {"model": None},
        {"model": "not-a-dict"},
        {"live_translation": []},
        {"whisper_decoding": {"live_transcription": None}},
        {"whisper_decoding": "not-a-dict-either"},
        {"model": {"type": "whisper", "whisper": None}},
        {"hallucination_filter": None},
        {"file_transcription": {"model": None}},
    ])
    def test_never_raises_and_still_records_identity(self, config):
        meta = build(config)
        assert meta["app.version"] == "26.1.168"
        assert all(isinstance(v, str) for v in meta.values())

    def test_missing_translation_block_still_records_transcription(self):
        meta = build({"model": {"type": "whisper", "whisper": {"model": "small"}}})
        assert meta["asr.model"] == "small"


class TestChangedKeys:
    def test_only_differing_keys(self):
        prev = {"mt.target_language": "en", "mt.method": "madlad"}
        curr = {"mt.target_language": "es", "mt.method": "madlad"}
        assert changed_keys(prev, curr) == {"mt.target_language": "es"}

    def test_new_key_counts_as_changed(self):
        assert changed_keys({}, {"mt.context_window": "2"}) == {"mt.context_window": "2"}

    def test_disappearing_key_is_not_a_change(self):
        # Inventing an empty row would read as though the setting was cleared.
        assert changed_keys({"mt.method": "madlad"}, {}) == {}

    def test_identical_config_yields_nothing(self):
        same = {"mt.method": "madlad"}
        assert changed_keys(same, dict(same)) == {}


class TestReadHistory:
    def test_single_entry_means_never_changed(self):
        assert read_history({"mt.target_language": "en"}, "mt.target_language") == [("", "en")]

    def test_appended_changes_are_ordered_oldest_first(self):
        meta = {
            "mt.target_language": "en",
            "mt.target_language@2026-05-20T19:10:00": "es",
            "mt.target_language@2026-05-20T20:05:00": "de",
        }
        assert read_history(meta, "mt.target_language") == [
            ("", "en"),
            ("2026-05-20T19:10:00", "es"),
            ("2026-05-20T20:05:00", "de"),
        ]

    def test_unrelated_keys_with_shared_prefix_are_excluded(self):
        meta = {"mt.model": "a", "mt.model_configured": "b", "mt.model@2026-05-20T19:00:00": "c"}
        assert read_history(meta, "mt.model") == [("", "a"), ("2026-05-20T19:00:00", "c")]

    def test_absent_key_is_empty(self):
        assert read_history({}, "mt.target_language") == []
