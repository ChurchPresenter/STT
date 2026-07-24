"""Piper catalog + pure TTS voice matching (stt/tts_catalog.py)."""

from stt.tts_catalog import (
    PIPER_MODELS_CATALOG,
    edge_voice_for_lang,
    piper_model_for_lang,
)


class TestPiperCatalog:
    def test_entries_have_required_keys(self):
        for m in PIPER_MODELS_CATALOG:
            assert {"id", "name", "language", "quality", "size"} <= set(m)

    def test_ids_are_unique(self):
        ids = [m["id"] for m in PIPER_MODELS_CATALOG]
        assert len(ids) == len(set(ids))


class TestEdgeVoiceForLang:
    VOICES = [
        {"Locale": "en-US", "ShortName": "en-US-AriaNeural"},
        {"Locale": "ru-RU", "ShortName": "ru-RU-DmitryNeural"},
        {"Locale": "ru-RU", "ShortName": "ru-RU-SvetlanaNeural"},
    ]

    def test_matches_locale_prefix_case_insensitive(self):
        assert edge_voice_for_lang("RU", self.VOICES) == "ru-RU-DmitryNeural"  # first match

    def test_no_match_returns_none(self):
        assert edge_voice_for_lang("fr", self.VOICES) is None

    def test_empty_voices_returns_none(self):
        assert edge_voice_for_lang("en", []) is None

    def test_tolerates_missing_locale(self):
        voices = [{"ShortName": "x"}, {"Locale": "es-ES", "ShortName": "es-ES-ElviraNeural"}]
        assert edge_voice_for_lang("es", voices) == "es-ES-ElviraNeural"


class TestPiperModelForLang:
    CATALOG = [
        {"id": "en_US-lessac-medium", "language": "en"},
        {"id": "en_US-amy-medium", "language": "en"},
        {"id": "ru_RU-ruslan-medium", "language": "ru"},
    ]

    def test_returns_first_downloaded_match(self):
        downloaded = {"en_US-amy-medium"}  # first en model NOT downloaded, second is
        result = piper_model_for_lang("en", self.CATALOG, lambda mid: mid in downloaded)
        assert result == "en_US-amy-medium"

    def test_none_when_language_absent(self):
        assert piper_model_for_lang("de", self.CATALOG, lambda mid: True) is None

    def test_none_when_match_not_downloaded(self):
        assert piper_model_for_lang("ru", self.CATALOG, lambda mid: False) is None

    def test_case_insensitive_language(self):
        assert piper_model_for_lang("RU", self.CATALOG, lambda mid: True) == "ru_RU-ruslan-medium"

    def test_works_against_the_real_catalog(self):
        # 'zh' exists in the shipped catalog; treat everything as downloaded.
        assert piper_model_for_lang("zh", PIPER_MODELS_CATALOG, lambda mid: True) == "zh_CN-huayan-medium"
