"""TTS backend registry + the pure voice-switch decision (stt/tts_backends.py)."""

import pytest

from stt.tts_backends import (
    BACKENDS,
    DEFAULT_SUPERTONIC_STEPS,
    DEFAULT_SUPERTONIC_VOICE,
    EDGE,
    PIPER,
    SUPERTONIC,
    SUPERTONIC_LANGUAGES,
    SUPERTONIC_UNKNOWN_LANG,
    SUPERTONIC_VOICES,
    backend_ids,
    clamp_supertonic_speed,
    clamp_supertonic_steps,
    get_backend,
    local_backend_ids,
    select_voice,
    supertonic_lang,
    supertonic_supports,
    supertonic_voice_for_lang,
)


class TestRegistry:
    def test_known_ids_resolve_to_themselves(self):
        for backend_id in backend_ids():
            assert get_backend(backend_id).id == backend_id

    def test_unknown_id_falls_back_to_edge(self):
        # A stale or hand-edited config must not take TTS down.
        assert get_backend("kokoro").id == EDGE
        assert get_backend("").id == EDGE
        assert get_backend(None).id == EDGE

    def test_id_lookup_ignores_case_and_whitespace(self):
        assert get_backend("  Piper ").id == PIPER

    def test_voice_and_prefs_keys_are_distinct_per_backend(self):
        voice_keys = [b.voice_key for b in BACKENDS.values()]
        prefs_keys = [b.prefs_key for b in BACKENDS.values()]
        assert len(set(voice_keys)) == len(voice_keys)
        assert len(set(prefs_keys)) == len(prefs_keys)
        assert not set(voice_keys) & set(prefs_keys)

    def test_only_local_backends_are_listed_as_local(self):
        assert set(local_backend_ids()) == {PIPER, SUPERTONIC}
        assert get_backend(EDGE).is_local is False

    def test_audio_format_is_declared_not_inferred(self):
        # The emit path reads this instead of guessing from the backend name.
        assert get_backend(EDGE).audio_format == "mp3"
        assert get_backend(PIPER).audio_format == "wav"
        assert get_backend(SUPERTONIC).audio_format == "wav"

    def test_only_piper_needs_a_model_per_language(self):
        assert get_backend(PIPER).needs_per_language_model is True
        assert get_backend(SUPERTONIC).needs_per_language_model is False
        assert get_backend(EDGE).needs_per_language_model is False


class TestSelectVoice:
    def test_stored_preference_beats_the_default(self):
        section = {
            "edge_voice": "en-US-AriaNeural",
            "edge_voice_preferences": {"ru": "ru-RU-SvetlanaNeural"},
        }
        chosen = select_voice(EDGE, "en", "ru", section, lambda lang: "ru-RU-DmitryNeural")
        assert chosen == "ru-RU-SvetlanaNeural"
        assert section["edge_voice"] == "ru-RU-SvetlanaNeural"

    def test_falls_back_to_the_injected_default(self):
        section = {"edge_voice": "en-US-AriaNeural"}
        chosen = select_voice(EDGE, "en", "ru", section, lambda lang: "ru-RU-DmitryNeural")
        assert chosen == "ru-RU-DmitryNeural"

    def test_remembers_the_outgoing_voice_for_its_language(self):
        section = {"piper_model": "en_US-amy-medium"}
        select_voice(PIPER, "en", "de", section, lambda lang: "de_DE-thorsten-medium")
        assert section["piper_model_preferences"]["en"] == "en_US-amy-medium"

    def test_switching_back_restores_the_earlier_choice(self):
        section = {"piper_model": "en_US-amy-medium"}
        select_voice(PIPER, "en", "de", section, lambda lang: "de_DE-thorsten-medium")
        back = select_voice(PIPER, "de", "en", section, lambda lang: "en_US-lessac-medium")
        assert back == "en_US-amy-medium"  # not the default the picker offered

    def test_no_voice_available_leaves_the_current_one_alone(self):
        # Piper with nothing downloaded for the new language: better to keep
        # speaking in the old voice than to blank the setting.
        section = {"piper_model": "en_US-amy-medium"}
        assert select_voice(PIPER, "en", "vi", section, lambda lang: None) is None
        assert section["piper_model"] == "en_US-amy-medium"

    def test_uses_the_registry_keys_for_each_backend(self):
        section = {"supertonic_voice": "M4"}
        select_voice(SUPERTONIC, "en", "pl", section, lambda lang: "M4")
        assert section["supertonic_voice_preferences"] == {"en": "M4"}
        assert "edge_voice_preferences" not in section

    def test_survives_a_corrupt_preferences_value(self):
        section = {"edge_voice": "en-US-AriaNeural", "edge_voice_preferences": "not-a-dict"}
        chosen = select_voice(EDGE, "en", "de", section, lambda lang: "de-DE-KatjaNeural")
        assert chosen == "de-DE-KatjaNeural"
        assert section["edge_voice_preferences"] == {"en": "en-US-AriaNeural"}

    def test_unset_current_voice_is_not_stashed(self):
        section = {"edge_voice": ""}
        select_voice(EDGE, "en", "de", section, lambda lang: "de-DE-KatjaNeural")
        assert section["edge_voice_preferences"] == {}


class TestSupertonicLanguages:
    def test_trained_language_passes_through(self):
        assert supertonic_lang("pl") == "pl"
        assert supertonic_lang("UK") == "uk"

    def test_untrained_language_falls_back_to_language_agnostic(self):
        # Chinese is the live case: piper covers it, supertonic does not.
        assert supertonic_supports("zh") is False
        assert supertonic_lang("zh") == SUPERTONIC_UNKNOWN_LANG

    def test_voice_picker_signals_unsupported_language(self):
        assert supertonic_voice_for_lang("zh") is None
        assert supertonic_voice_for_lang("ko") == DEFAULT_SUPERTONIC_VOICE

    def test_voice_picker_keeps_the_current_preset(self):
        # Voices are language-independent, so a language switch shouldn't
        # silently change who is speaking.
        assert supertonic_voice_for_lang("ja", current="M4") == "M4"

    def test_voice_ids_are_unique_and_shaped_for_the_ui(self):
        ids = [v["id"] for v in SUPERTONIC_VOICES]
        assert len(ids) == len(set(ids))
        assert all({"id", "name"} <= set(v) for v in SUPERTONIC_VOICES)

    def test_language_list_matches_the_installed_package(self):
        # The frozenset is duplicated so stt/ stays stdlib-only; when the real
        # package is around, prove the copy hasn't drifted from it.
        supertonic = pytest.importorskip("supertonic")
        assert SUPERTONIC_LANGUAGES == frozenset(supertonic.SUPPORTED_LANGUAGES)
        assert SUPERTONIC_UNKNOWN_LANG == supertonic.UNKNOWN_LANGUAGE


class TestClamping:
    def test_steps_clamped_to_model_range(self):
        assert clamp_supertonic_steps(1) == 5
        assert clamp_supertonic_steps(99) == 12
        assert clamp_supertonic_steps(8) == 8

    def test_steps_tolerate_junk(self):
        assert clamp_supertonic_steps("nope") == DEFAULT_SUPERTONIC_STEPS
        assert clamp_supertonic_steps(None) == DEFAULT_SUPERTONIC_STEPS

    def test_speed_clamped_to_model_range(self):
        # The shared UI slider goes 0.5-2.0; the model accepts 0.7-2.0.
        assert clamp_supertonic_speed(0.5) == 0.7
        assert clamp_supertonic_speed(3.0) == 2.0
        assert clamp_supertonic_speed(1.2) == pytest.approx(1.2)

    def test_speed_tolerates_junk(self):
        assert clamp_supertonic_speed("fast") == 1.0
