"""Session provenance mapping (stt/session_meta.py:build_session_meta and helpers)."""

import pytest

from stt.session_meta import (
    build_session_meta,
    changed_keys,
    content_digest,
    glossary_provenance,
    is_offloaded,
    latest_values,
    read_history,
    remote_provenance,
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


class TestOffloadedTranslation:
    """On an offloaded session the remote's model translates, not the local one."""

    def offload(self, remote_model="", **lt):
        cfg = {"live_translation": {"translation_method": "nllb",
                                    "translation_model": "facebook/nllb-200-distilled-600M",
                                    "remote": {"enabled": True,
                                               "endpoint": "192.168.2.52:8080",
                                               "model": remote_model}}}
        cfg["live_translation"].update(lt)
        return cfg

    def test_offloaded_flag_is_recorded(self):
        assert build(self.offload())["mt.offloaded"] == "true"
        assert build({"live_translation": {"enabled": True}})["mt.offloaded"] == "false"

    def test_enabled_without_endpoint_is_not_offloaded(self):
        cfg = {"live_translation": {"remote": {"enabled": True, "endpoint": ""}}}
        assert build(cfg)["mt.offloaded"] == "false"

    def test_endpoint_without_enabled_is_not_offloaded(self):
        cfg = {"live_translation": {"remote": {"enabled": False,
                                               "endpoint": "192.168.2.52:8080"}}}
        assert build(cfg)["mt.offloaded"] == "false"

    def test_blank_remote_model_does_not_claim_the_local_model(self):
        # A blank remote model means "use Machine B's own model" - asserting the
        # local id here would name a model that never ran.
        meta = build(self.offload(remote_model=""))
        assert meta["mt.model"] == ""
        # ...but the local config value stays visible for reference.
        assert meta["mt.model_configured"] == "facebook/nllb-200-distilled-600M"

    def test_explicit_remote_model_is_dictated_so_local_id_stands(self):
        meta = build(self.offload(remote_model="google/madlad400-3b-mt"))
        assert meta["mt.model"] == "facebook/nllb-200-distilled-600M"

    def test_local_session_still_records_its_model(self):
        cfg = {"live_translation": {"translation_method": "nllb",
                                    "translation_model": "facebook/nllb-200-distilled-600M"}}
        assert build(cfg)["mt.model"] == "facebook/nllb-200-distilled-600M"

    def test_is_offloaded_helper(self):
        assert is_offloaded({"remote": {"enabled": True, "endpoint": "h:8080"}}) is True
        assert is_offloaded({"remote": {"enabled": True}}) is False
        assert is_offloaded({}) is False


class TestAudioShapingAndSegmentation:
    def test_context_prompt_is_recorded_as_a_decode_input(self):
        """Prior captions are passed to Whisper as initial_prompt, so the same
        audio decodes differently depending on what preceded it."""
        meta = build({"audio": {"context_prompt": {"enabled": True, "max_chars": 200}}})
        assert meta["asr.context_prompt.enabled"] == "true"
        assert meta["asr.context_prompt.max_chars"] == "200"

    def test_loudness_normalization_is_recorded(self):
        meta = build({"audio": {"loudness_normalization": {
            "enabled": True, "target_rms_dbfs": -20, "max_gain": 10}}})
        assert meta["asr.normalize.enabled"] == "true"
        assert meta["asr.normalize.target_rms_dbfs"] == "-20"
        assert meta["asr.normalize.max_gain"] == "10"

    def test_segment_boundaries_are_recorded(self):
        meta = build({"audio": {"energy_threshold": 100, "phrase_timeout": 2,
                                 "same_output_threshold": 7, "stabilize_live_text": True,
                                 "pending_buffer": {"enabled": True, "max_words": 30,
                                                    "max_age_seconds": 10}}})
        assert meta["asr.segment.energy_threshold"] == "100"
        assert meta["asr.segment.phrase_timeout"] == "2"
        assert meta["asr.segment.same_output_threshold"] == "7"
        assert meta["asr.segment.stabilize_live_text"] == "true"
        assert meta["asr.segment.pending_buffer.enabled"] == "true"
        assert meta["asr.segment.pending_buffer.max_words"] == "30"
        assert meta["asr.segment.pending_buffer.max_age_seconds"] == "10"

    def test_gpu_intent_is_recorded(self):
        assert build({"performance": {"use_gpu": True}})["asr.use_gpu"] == "true"

    def test_custom_model_type_only_for_a_custom_model(self):
        custom = build({"model": {"type": "custom", "custom": {"model_path": "/m",
                                                                "model_type": "whisper"}}})
        assert custom["asr.custom.model_type"] == "whisper"
        assert "asr.custom.model_type" not in build({"model": {"type": "whisper"}})


class TestTtsAndFileTranslation:
    def test_tts_voice_is_recorded_per_backend(self):
        edge = build({"live_translation": {"tts": {"enabled": True, "backend": "edge",
                                                    "edge_voice": "en-US-AriaNeural",
                                                    "piper_model": "en_US-lessac-medium",
                                                    "speed": 1.0}}})
        assert edge["mt.tts.voice"] == "en-US-AriaNeural"
        piper = build({"live_translation": {"tts": {"enabled": True, "backend": "piper",
                                                     "edge_voice": "en-US-AriaNeural",
                                                     "piper_model": "en_US-lessac-medium"}}})
        assert piper["mt.tts.voice"] == "en_US-lessac-medium"

    def test_disabled_tts_names_no_voice(self):
        meta = build({"live_translation": {"tts": {"enabled": False,
                                                    "edge_voice": "en-US-AriaNeural"}}})
        assert meta["mt.tts.enabled"] == "false"
        assert "mt.tts.voice" not in meta

    def test_file_translation_stage_is_recorded(self):
        """asr.file.* covered how a file was transcribed, not how it was translated."""
        meta = build({"file_transcription": {"translate_enabled": True, "translate_to": "en",
                                              "translation_model": "facebook/nllb-200-distilled-600M"}})
        assert meta["mt.file.enabled"] == "true"
        assert meta["mt.file.target_language"] == "en"
        assert meta["mt.file.model"] == "facebook/nllb-200-distilled-600M"

    def test_offload_cache_settings_are_recorded(self):
        # A cache hit can return text translated under an earlier prompt or glossary.
        meta = build({"live_translation": {"remote": {"enabled": True, "endpoint": "h:8080",
                                                       "server_cache_enabled": True,
                                                       "server_cache_size": 512,
                                                       "sync_dictionary_on_edit": True}}})
        assert meta["mt.remote.server_cache_enabled"] == "true"
        assert meta["mt.remote.server_cache_size"] == "512"
        assert meta["mt.remote.sync_dictionary_on_edit"] == "true"

    def test_output_artifact_toggles_are_recorded(self):
        meta = build({"live_translation": {"warmup": True, "srt_enabled": True,
                                            "html_enabled": False, "max_entries_to_send": 100}})
        assert meta["mt.warmup"] == "true"
        assert meta["mt.srt_enabled"] == "true"
        assert meta["mt.html_enabled"] == "false"
        assert meta["mt.max_entries_to_send"] == "100"


class TestLlmSettings:
    """mt.llm.* - what an LLM session needs in order to be reproducible.

    Everything that decides an LLM caption lives in live_translation.llm, and none
    of it is visible in the transcript: the same service captioned by the same
    GGUF reads differently after one prompt edit.
    """

    @staticmethod
    def llm(**llm_cfg):
        cfg = {"translation_method": "llm", "target_language": "es",
               "translation_model": "facebook/nllb-200-distilled-600M",
               "llm": llm_cfg}
        return {"live_translation": cfg}

    def test_local_provider_records_the_gguf_and_its_load_settings(self):
        meta = build(self.llm(provider="local", gguf_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
                              gguf_file="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                              n_gpu_layers="auto", n_ctx=2048))
        assert meta["mt.llm.provider"] == "local"
        assert meta["mt.llm.gguf_repo"] == "bartowski/Qwen2.5-7B-Instruct-GGUF"
        assert meta["mt.llm.gguf_file"] == "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        assert meta["mt.llm.n_gpu_layers"] == "auto"
        assert meta["mt.llm.n_ctx"] == "2048"

    def test_an_explicit_path_names_the_file_that_actually_loads(self):
        # get_local_llm() prefers gguf_path, so recording the repo pair would name
        # a file the session never opened.
        meta = build(self.llm(provider="local", gguf_path="/models/gemma-3-4b-it-Q4_K_M.gguf",
                              gguf_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
                              gguf_file="Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
        assert meta["mt.llm.model"] == "/models/gemma-3-4b-it-Q4_K_M.gguf"

    def test_llm_session_records_the_llm_as_the_model_that_translated(self):
        """The NMT id is the fallback on an LLM session, not what captioned it."""
        meta = build(self.llm(provider="local", gguf_repo="r", gguf_file="m.gguf"))
        assert meta["mt.model"] == "r/m.gguf"
        assert meta["mt.model_configured"] == "facebook/nllb-200-distilled-600M"

    def test_endpoint_provider_records_where_and_what(self):
        meta = build(self.llm(provider="endpoint", endpoint="http://127.0.0.1:11434/api/chat",
                              model="qwen2.5:7b-instruct"))
        assert meta["mt.llm.endpoint"] == "http://127.0.0.1:11434/api/chat"
        assert meta["mt.llm.model"] == "qwen2.5:7b-instruct"
        assert meta["mt.model"] == "qwen2.5:7b-instruct"

    def test_provider_specific_keys_do_not_cross_over(self):
        local = build(self.llm(provider="local", gguf_file="m.gguf", endpoint="http://x/api/chat"))
        assert "mt.llm.endpoint" not in local
        endpoint = build(self.llm(provider="endpoint", model="m", n_ctx=2048))
        assert "mt.llm.n_ctx" not in endpoint
        assert "mt.llm.gguf_file" not in endpoint

    def test_provider_defaults_to_endpoint_when_unset(self):
        assert build(self.llm(model="m"))["mt.llm.provider"] == "endpoint"

    def test_budget_and_timeout_settings_are_recorded(self):
        # timeout_ms decides whether a caption came from the LLM at all: on timeout
        # it falls back to NMT without a trace in the transcript.
        meta = build(self.llm(model="m", max_tokens=160, keep_alive=-1,
                              timeout_ms=8000, warmup_timeout_ms=180000))
        assert meta["mt.llm.max_tokens"] == "160"
        assert meta["mt.llm.keep_alive"] == "-1"
        assert meta["mt.llm.timeout_ms"] == "8000"
        assert meta["mt.llm.warmup_timeout_ms"] == "180000"

    def test_api_key_is_never_recorded(self):
        """Session databases are delivered to a NAS; a bearer token must not ride along."""
        meta = build(self.llm(model="m", api_key="sk-secret-value"))
        assert "sk-secret-value" not in "".join(meta.values())
        assert not [k for k in meta if "api_key" in k and not k.endswith("_set")]
        assert meta["mt.llm.api_key_set"] == "true"

    def test_absent_api_key_is_recorded_as_unset(self):
        assert build(self.llm(model="m"))["mt.llm.api_key_set"] == "false"

    def test_blank_prompt_records_the_template_the_model_received(self):
        """An empty override means the shipped template ran - with the language filled in.

        Recording the configured "" would leave a reader unable to tell what the
        model was told, which is the whole point of storing the prompt.
        """
        meta = build(self.llm(model="m", system_prompt=""))
        assert meta["mt.llm.system_prompt_custom"] == "false"
        prompt = meta["mt.llm.system_prompt"]
        assert "{language}" not in prompt
        assert "Spanish" in prompt
        assert prompt.endswith("Translate into Spanish. Output only Spanish.")

    def test_custom_prompt_is_recorded_with_the_target_directive_appended(self):
        meta = build(self.llm(model="m", system_prompt="Render вечеря as communion."))
        assert meta["mt.llm.system_prompt_custom"] == "true"
        prompt = meta["mt.llm.system_prompt"]
        assert prompt.startswith("Render вечеря as communion.")
        assert "Translate into Spanish" in prompt

    def test_prompt_follows_the_target_language(self):
        cfg = self.llm(model="m")
        cfg["live_translation"]["target_language"] = "uk"
        assert "Ukrainian" in build(cfg)["mt.llm.system_prompt"]

    def test_nmt_session_carries_no_llm_keys(self):
        """Inert settings recorded as fact invite the wrong explanation for a caption."""
        cfg = {"live_translation": {"translation_method": "nllb",
                                    "translation_model": "facebook/nllb-200-distilled-600M",
                                    "llm": {"provider": "local", "gguf_file": "m.gguf"}}}
        assert not [k for k in build(cfg) if k.startswith("mt.llm.")]

    def test_offloaded_llm_does_not_claim_the_local_gguf(self):
        # Machine B translates; this box's GGUF is not what ran. The remote's model
        # arrives separately under mt.remote.effective.* once it is probed.
        cfg = self.llm(provider="local", gguf_file="m.gguf")
        cfg["live_translation"]["remote"] = {"enabled": True, "endpoint": "192.168.2.52:8080",
                                             "model": ""}
        meta = build(cfg)
        assert meta["mt.model"] == ""
        assert meta["mt.llm.gguf_file"] == "m.gguf"  # configured locally, still readable

    def test_nmt_engine_settings_are_not_claimed_for_a_local_llm_session(self):
        """Nothing loads a torch or CT2 model, so recording fp16/beams asserts a
        model that never ran - the same reason the status route nulls them."""
        meta = build(self.llm(provider="local", gguf_file="m.gguf"))
        for key in ("mt.use_fp16", "mt.use_ctranslate2", "mt.ct2_compute_type",
                    "mt.ct2_inter_threads", "mt.ct2_intra_threads"):
            assert key not in meta
        assert not [k for k in meta if k.startswith("mt.gen.")]

    def test_offloaded_llm_keeps_the_nmt_settings_the_fallback_would_use(self):
        cfg = self.llm(model="m")
        cfg["live_translation"].update(use_fp16=True, use_ctranslate2=True,
                                       remote={"enabled": True, "endpoint": "h:8080"})
        assert build(cfg)["mt.use_fp16"] == "true"

    def test_nmt_session_is_unaffected(self):
        cfg = {"live_translation": {"translation_method": "nllb", "use_fp16": True,
                                    "generation_params": {"num_beams": 2}}}
        meta = build(cfg)
        assert meta["mt.use_fp16"] == "true"
        assert meta["mt.gen.num_beams"] == "2"

    def test_a_prompt_edit_mid_session_is_a_recordable_change(self):
        before = build(self.llm(model="m", system_prompt=""))
        after = build(self.llm(model="m", system_prompt="Render вечеря as communion."))
        changes = changed_keys(before, after)
        assert "mt.llm.system_prompt" in changes
        assert changes["mt.llm.system_prompt_custom"] == "true"


class TestRemoteProvenance:
    STATUS = {
        "success": True,
        "translation_model": "google/madlad400-3b-mt",
        "translation_method": "madlad",
        "model_device": "mps",
        "model_dtype": "float32",
        "is_ctranslate2": True,
        "ct2_compute_type": "auto",
    }

    def test_records_an_llm_remote(self):
        """The remote translates with an LLM, so the transcript must say so.

        Reporting the standby NMT model here is how a session whose captions came
        from a GGUF on the GPU was recorded as "google/madlad400-3b-mt, cpu".
        """
        meta = remote_provenance({
            "success": True,
            "translation_model": "gemma-3-4b-it-Q4_K_M.gguf",
            "translation_method": "llm",
            "model_device": "metal",
            "llm_provider": "local",
        })
        assert meta["mt.remote.effective.method"] == "llm"
        assert meta["mt.remote.effective.model"] == "gemma-3-4b-it-Q4_K_M.gguf"
        assert meta["mt.remote.effective.device"] == "metal"
        assert meta["mt.remote.effective.llm_provider"] == "local"

    def test_nmt_remote_carries_no_llm_keys(self):
        meta = remote_provenance(self.STATUS)
        assert not [k for k in meta if "llm" in k]

    def test_maps_the_remote_status_payload(self):
        meta = remote_provenance(self.STATUS)
        assert meta["mt.remote.effective.model"] == "google/madlad400-3b-mt"
        assert meta["mt.remote.effective.method"] == "madlad"
        assert meta["mt.remote.effective.device"] == "mps"
        assert meta["mt.remote.effective.dtype"] == "float32"
        assert meta["mt.remote.effective.ct2"] == "true"
        assert meta["mt.remote.effective.ct2_compute_type"] == "auto"

    def test_null_fields_are_omitted_not_blanked(self):
        status = dict(self.STATUS, model_device=None, model_dtype=None)
        meta = remote_provenance(status)
        assert "mt.remote.effective.device" not in meta
        assert "mt.remote.effective.dtype" not in meta
        assert meta["mt.remote.effective.model"] == "google/madlad400-3b-mt"

    @pytest.mark.parametrize("status", [
        None,
        {},
        {"success": False, "error": "unreachable"},
        "not a mapping",
        [1, 2, 3],
    ])
    def test_unusable_response_leaves_no_claim(self, status):
        # An unreachable remote must record nothing rather than something wrong.
        assert remote_provenance(status) == {}

    def test_success_absent_is_treated_as_success(self):
        # Older remotes may omit the flag; the payload is still usable.
        assert remote_provenance({"translation_model": "x"})[
            "mt.remote.effective.model"] == "x"

    def test_only_known_fields_are_lifted(self):
        meta = remote_provenance(dict(self.STATUS, cache_size=99, remote_clients=["a"]))
        assert all(k.startswith("mt.remote.effective.") for k in meta)
        assert len(meta) == 6


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

    def test_phrase_digest_distinguishes_equal_length_lists(self):
        """Equal counts are not equal filters; only the digest says so."""
        one = build({"hallucination_filter": {"phrases": ["a", "b"]}})
        two = build({"hallucination_filter": {"phrases": ["a", "c"]}})
        assert one["filter.hallucination_phrase_count"] == two["filter.hallucination_phrase_count"]
        assert one["filter.hallucination_phrase_digest"] != two["filter.hallucination_phrase_digest"]

    def test_phrase_digest_is_empty_when_there_are_no_phrases(self):
        assert build({})["filter.hallucination_phrase_digest"] == ""

    def test_word_count_filters_are_recorded(self):
        """A caption below min_words is never saved, leaving no trace in the rows."""
        meta = build({"audio": {"min_words": 3, "fuzzy_duplicate_threshold": 0.85}})
        assert meta["filter.min_words"] == "3"
        assert meta["filter.fuzzy_duplicate_threshold"] == "0.85"

    def test_profanity_filter_is_recorded_without_the_word_list(self):
        meta = build({"profanity_filter": {"enabled": True, "replacement": "****",
                                            "words": ["a", "b", "c"]}})
        assert meta["filter.profanity_enabled"] == "true"
        assert meta["filter.profanity_replacement"] == "****"
        assert meta["filter.profanity_word_count"] == "3"

    def test_music_drop_is_recorded_as_its_consequence(self):
        detecting = {"speech_type_detection": {"enabled": True,
                                                "transcribe_detected_music": False}}
        assert build(detecting)["filter.music_dropped"] == "true"

    def test_music_is_not_dropped_when_transcribed_or_undetected(self):
        transcribed = {"speech_type_detection": {"enabled": True,
                                                  "transcribe_detected_music": True}}
        assert build(transcribed)["filter.music_dropped"] == "false"
        off = {"speech_type_detection": {"enabled": False}}
        assert build(off)["filter.music_dropped"] == "false"


class TestContentDigest:
    def test_empty_content_has_no_digest(self):
        for value in (None, [], {}, ""):
            assert content_digest(value) == ""

    def test_stable_across_ordering_but_not_across_content(self):
        assert content_digest({"a": 1, "b": 2}) == content_digest({"b": 2, "a": 1})
        assert content_digest(["a", "b"]) != content_digest(["a", "c"])

    def test_unserialisable_content_still_digests(self):
        # A digest is provenance, not a feature: it must not be able to fail a start.
        assert content_digest([object()])


class TestAudioTypeDetection:
    STD = {"speech_type_detection": {
        "enabled": True, "method": "panns", "device": "cpu",
        "music_threshold": 0.5, "music_prob_threshold": 0.5,
        "quiet_db_threshold": -40, "smoothing_window": 4,
        "transcribe_detected_music": False}}

    def test_classifier_settings_are_recorded(self):
        """The per-segment Speaking/Music/Quiet tag is in the transcript; its thresholds were not."""
        meta = build(self.STD)
        assert meta["audio_type.enabled"] == "true"
        assert meta["audio_type.method"] == "panns"
        assert meta["audio_type.device"] == "cpu"
        assert meta["audio_type.music_threshold"] == "0.5"
        assert meta["audio_type.music_prob_threshold"] == "0.5"
        assert meta["audio_type.quiet_db_threshold"] == "-40"
        assert meta["audio_type.smoothing_window"] == "4"
        assert meta["audio_type.transcribe_music"] == "false"

    def test_omitted_entirely_when_unconfigured(self):
        assert not [k for k in build({}) if k.startswith("audio_type.")]


class TestCorrections:
    def test_review_and_delay_settings_are_recorded(self):
        """With auto_publish off, a caption never approved never reached the screen."""
        meta = build({"corrections": {"enabled": True, "confidence_threshold": 0.7,
                                       "output_delay": {"enabled": True, "delay_seconds": 7,
                                                        "auto_publish": False}}})
        assert meta["corrections.enabled"] == "true"
        assert meta["corrections.confidence_threshold"] == "0.7"
        assert meta["corrections.output_delay.enabled"] == "true"
        assert meta["corrections.output_delay.seconds"] == "7"
        assert meta["corrections.output_delay.auto_publish"] == "false"

    def test_omitted_entirely_when_unconfigured(self):
        assert not [k for k in build({}) if k.startswith("corrections.")]


class TestGlossaryProvenance:
    DICT = {"glossary": {"uk_to_en": {"вечеря": "communion", "громада": "congregation"},
                         "en_to_es": {"grace": "gracia"}}}

    def test_terms_are_counted_and_digested(self):
        """config records the file name; the terms that rewrite captions live in it."""
        meta = glossary_provenance(self.DICT)
        assert meta["glossary.term_count"] == "3"
        assert meta["glossary.pairs"] == "en_to_es,uk_to_en"
        assert meta["glossary.digest"]
        assert meta["glossary.source"] == "local"

    def test_an_edit_changes_the_digest(self):
        edited = {"glossary": {"uk_to_en": {"вечеря": "the Lord's supper",
                                             "громада": "congregation"},
                               "en_to_es": {"grace": "gracia"}}}
        before, after = glossary_provenance(self.DICT), glossary_provenance(edited)
        assert before["glossary.term_count"] == after["glossary.term_count"]
        assert before["glossary.digest"] != after["glossary.digest"]

    def test_key_order_does_not_change_the_digest(self):
        reordered = {"glossary": {"en_to_es": {"grace": "gracia"},
                                  "uk_to_en": {"громада": "congregation",
                                               "вечеря": "communion"}}}
        assert glossary_provenance(reordered)["glossary.digest"] == \
            glossary_provenance(self.DICT)["glossary.digest"]

    def test_a_pushed_client_glossary_says_so(self):
        # On an offloaded session the client's table applies, and naming this
        # machine's file would point at terms that never ran.
        meta = glossary_provenance(self.DICT, source="paired-client")
        assert meta["glossary.source"] == "paired-client"

    @pytest.mark.parametrize("dictionary", [None, {}, {"glossary": None},
                                            {"glossary": "not-a-dict"}, {"other": {}}])
    def test_unusable_dictionary_leaves_no_claim(self, dictionary):
        assert glossary_provenance(dictionary) == {}

    def test_non_mapping_pairs_are_ignored_not_counted(self):
        meta = glossary_provenance({"glossary": {"uk_to_en": {"a": "b"}, "junk": "oops"}})
        assert meta["glossary.pairs"] == "uk_to_en"
        assert meta["glossary.term_count"] == "1"

    def test_llm_session_records_no_glossary(self):
        """apply_glossary() runs only in the NMT decode paths."""
        cfg = {"live_translation": {"translation_method": "llm", "llm": {"model": "m"}},
               "custom_dictionary": {"nllb_glossary_enabled": True, "file": "d.json"}}
        assert not [k for k in build(cfg) if k.startswith("glossary.")]

    def test_offloaded_llm_still_records_it(self):
        # Machine B may translate with NMT and apply the glossary this box syncs to it.
        cfg = {"live_translation": {"translation_method": "llm", "llm": {"model": "m"},
                                    "remote": {"enabled": True, "endpoint": "h:8080"}},
               "custom_dictionary": {"nllb_glossary_enabled": True, "file": "d.json"}}
        assert build(cfg)["glossary.enabled"] == "true"


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


class TestLatestValues:
    """What config must be diffed against, so a change isn't re-appended forever."""

    def test_unchanged_keys_pass_through(self):
        assert latest_values({"mt.method": "madlad"}) == {"mt.method": "madlad"}

    def test_appended_change_supersedes_the_base_key(self):
        meta = {"mt.target_language": "en", "mt.target_language@2026-05-20T19:10:00": "es"}
        assert latest_values(meta)["mt.target_language"] == "es"

    def test_last_change_wins(self):
        meta = {
            "mt.target_language": "en",
            "mt.target_language@2026-05-20T19:10:00": "es",
            "mt.target_language@2026-05-20T20:05:00": "de",
        }
        assert latest_values(meta)["mt.target_language"] == "de"

    def test_a_change_back_to_the_original_is_the_latest(self):
        # Flipped away and back: latest is the original value, so nothing more
        # should be appended on the next config write.
        meta = {
            "asr.language": "ru",
            "asr.language@2026-05-20T23:09:16": "en",
            "asr.language@2026-05-20T23:12:07": "ru",
        }
        assert latest_values(meta)["asr.language"] == "ru"

    def test_change_only_key_still_appears(self):
        assert latest_values({"mt.x@2026-05-20T19:00:00": "v"}) == {"mt.x": "v"}

    def test_keys_sharing_a_prefix_are_not_confused(self):
        meta = {"mt.model": "a", "mt.model_configured": "b",
                "mt.model@2026-05-20T19:00:00": "c"}
        result = latest_values(meta)
        assert result["mt.model"] == "c"
        assert result["mt.model_configured"] == "b"

    def test_empty(self):
        assert latest_values({}) == {}


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
