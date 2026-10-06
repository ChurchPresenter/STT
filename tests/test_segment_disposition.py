"""One sentence's fate: keep, keep with a credit cut out, or deny (stt/segment_disposition.py)."""

from stt.segment_disposition import decide

PHRASES = ["DimaTorzok", "Субтитры создавал", "Продолжение следует"]


class TestDecide:
    def test_clean_speech_is_kept(self):
        d = decide("Господи, благодарим Тебя", phrases=PHRASES)
        assert (d.denied, d.reason, d.text) == (False, None, "Господи, благодарим Тебя")
        assert not d.stripped and not d.hallucination

    def test_pure_hallucination_is_denied(self):
        d = decide("Субтитры создавал DimaTorzok", phrases=PHRASES)
        assert d.denied and d.reason == "hallucination" and d.hallucination and not d.stripped

    def test_glued_credit_keeps_the_speech(self):
        d = decide("Субтитры создавал DimaTorzok Мир вам, дорогая церковь!", phrases=PHRASES)
        assert not d.denied and d.reason is None
        assert d.text == "Мир вам, дорогая церковь!"
        assert d.stripped and d.hallucination

    def test_music_denies_when_not_transcribed(self):
        d = decide("все мои источники в Тебе", phrases=PHRASES, music_label="Music", music_reason="music:0.5")
        assert d.denied and d.reason == "music:0.5" and d.music

    def test_music_kept_when_transcribing_music(self):
        d = decide("все мои источники в Тебе", phrases=PHRASES, music_label="Music", transcribe_music=True)
        assert not d.denied and not d.music

    def test_speaking_label_is_not_music(self):
        assert not decide("слово", phrases=PHRASES, music_label="Speaking").denied

    def test_hallucination_outranks_music_and_cjk(self):
        d = decide("Продолжение следует", phrases=PHRASES, music_label="Music", cjk_deny=True, music_reason="music:0.5")
        assert d.reason == "hallucination" and not d.music

    def test_cjk_outranks_music(self):
        d = decide("слово", phrases=PHRASES, music_label="Music", cjk_deny=True, music_reason="music:0.5")
        assert d.reason == "cjk" and not d.music

    def test_stripped_speech_can_still_be_music(self):
        d = decide("Слава Богу Субтитры создавал DimaTorzok", phrases=PHRASES, music_label="Music", music_reason="music:0.5")
        assert d.denied and d.reason == "music:0.5" and d.stripped and d.text == "Слава Богу"

    def test_no_phrases_means_no_hallucination(self):
        d = decide("Субтитры создавал DimaTorzok", phrases=[])
        assert not d.denied and not d.hallucination
