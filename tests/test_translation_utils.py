"""Glossary post-processing and the live translation cache (stt/translation_utils.py)."""

import threading

from stt.translation_utils import (
    TextTranslationCache,
    TranslationCache,
    apply_glossary,
    should_cache_translation,
    should_use_fp16,
    translation_device,
    translation_load_dtype,
)


def glossary(mapping, key="en_to_es"):
    return {"glossary": {key: mapping}}


class TestApplyGlossary:
    def test_no_dictionary_passthrough(self):
        assert apply_glossary("hello", "en", "es", None) == "hello"
        assert apply_glossary("hello", "en", "es", {}) == "hello"

    def test_wrong_language_pair_passthrough(self):
        d = glossary({"church": "iglesia"}, key="en_to_fr")
        assert apply_glossary("the church", "en", "es", d) == "the church"

    def test_simple_replacement_case_insensitive(self):
        d = glossary({"church": "iglesia"})
        assert apply_glossary("The Church is open", "en", "es", d) == "The iglesia is open"

    def test_word_boundaries(self):
        d = glossary({"art": "arte"})
        assert apply_glossary("the start of art", "en", "es", d) == "the start of arte"

    def test_longest_term_wins(self):
        d = glossary({"holy": "santo", "holy spirit": "espíritu santo"})
        assert apply_glossary("the holy spirit", "en", "es", d) == "the espíritu santo"

    def test_punctuation_edged_terms(self):
        # \b would fail on terms starting/ending with punctuation; lookarounds must not
        d = glossary({"St. Paul": "San Pablo"})
        assert apply_glossary("read St. Paul today", "en", "es", d) == "read San Pablo today"

    def test_backslashes_in_target_are_literal(self):
        d = glossary({"path": r"C:\new\1"})
        assert apply_glossary("the path here", "en", "es", d) == r"the C:\new\1 here"

    def test_bad_dictionary_shape_fails_open(self):
        assert apply_glossary("hello", "en", "es", {"glossary": "not-a-dict"}) == "hello"


class TestTranslationCache:
    def test_get_miss(self):
        c = TranslationCache()
        assert c.get(1, "hello", "es") is None

    def test_set_and_get(self):
        c = TranslationCache()
        c.set(1, "hello", "hola", "es")
        assert c.get(1, "hello", "es") == "hola"

    def test_changed_original_misses(self):
        c = TranslationCache()
        c.set(1, "hello", "hola", "es")
        assert c.get(1, "hello there", "es") is None

    def test_changed_language_misses_unless_stale_accepted(self):
        c = TranslationCache()
        c.set(1, "hello", "hola", "es")
        assert c.get(1, "hello", "fr") is None
        # Hot language switch: old segments may keep their stale-language text
        assert c.get(1, "hello", "fr", accept_stale_lang=True) == "hola"

    def test_invalidate(self):
        c = TranslationCache()
        c.set(1, "hello", "hola", "es")
        c.invalidate(1)
        assert c.get(1, "hello", "es") is None
        c.invalidate(99)  # unknown id is a no-op

    def test_clear_and_size(self):
        c = TranslationCache()
        c.set(1, "a", "x", "es")
        c.set(2, "b", "y", "es")
        assert c.get_size() == 2
        c.clear()
        assert c.get_size() == 0

    def test_eviction_drops_oldest_hundred(self):
        c = TranslationCache(max_size=150)
        for i in range(150):
            c.set(i, f"t{i}", f"x{i}", "es")
        c.set(150, "t150", "x150", "es")  # triggers eviction of ids 0..99
        assert c.get(0, "t0", "es") is None
        assert c.get(99, "t99", "es") is None
        assert c.get(100, "t100", "es") == "x100"
        assert c.get(150, "t150", "es") == "x150"

    def test_extras_round_trip(self):
        c = TranslationCache()
        c.set_with_extras(5, "hello", "hola", "es", confidence=0.9, alternatives=["buenas"])
        assert c.get(5, "hello", "es") == "hola"
        assert c.get_extras(5) == {"confidence": 0.9, "alternatives": ["buenas"]}

    def test_extras_default_empty(self):
        c = TranslationCache()
        c.set(5, "hello", "hola", "es")
        assert c.get_extras(5) == {"confidence": None, "alternatives": []}
        assert c.get_extras(99) is None

    def test_max_segment_id_ignores_non_int_keys(self):
        c = TranslationCache()
        assert c.max_segment_id() == 0
        c.set(3, "a", "x", "es")
        c.set("live", "b", "y", "es")
        c.set(7, "c", "z", "es")
        assert c.max_segment_id() == 7

    def test_min_segment_id_ignores_non_int_keys(self):
        c = TranslationCache()
        assert c.min_segment_id() == 0
        c.set(7, "c", "z", "es")
        c.set("live", "b", "y", "es")
        c.set(3, "a", "x", "es")
        assert c.min_segment_id() == 3

    def test_translated_segments_returns_int_keyed_pairs(self):
        c = TranslationCache()
        c.set(3, "a", "uno", "es")
        c.set("live", "b", "dos", "es")
        c.set(7, "c", "tres", "es")
        assert sorted(c.translated_segments()) == [(3, "uno"), (7, "tres")]

    def test_translated_segments_omits_blank_translations(self):
        c = TranslationCache()
        c.set(1, "a", "uno", "es")
        c.set(2, "b", "   ", "es")
        assert c.translated_segments() == [(1, "uno")]


class TestTextTranslationCache:
    RESULT = {"text": "hola", "confidence": None, "alternatives": []}

    def test_miss_returns_none(self):
        c = TextTranslationCache()
        assert c.get("hello", "en", "es", 2) is None

    def test_set_then_get_roundtrip(self):
        c = TextTranslationCache()
        c.set("hello", "en", "es", 2, self.RESULT)
        assert c.get("hello", "en", "es", 2) == self.RESULT

    def test_get_returns_a_copy_not_the_stored_dict(self):
        c = TextTranslationCache()
        c.set("hello", "en", "es", 2, self.RESULT)
        got = c.get("hello", "en", "es", 2)
        got["text"] = "mutated"
        assert c.get("hello", "en", "es", 2)["text"] == "hola"  # store untouched

    def test_text_is_stripped_for_keying(self):
        c = TextTranslationCache()
        c.set("hello", "en", "es", 2, self.RESULT)
        assert c.get("  hello  ", "en", "es", 2) == self.RESULT

    def test_key_sensitivity(self):
        c = TextTranslationCache()
        c.set("hello", "en", "es", 2, self.RESULT)
        assert c.get("hello", "en", "es", 2) is not None  # exact hit
        assert c.get("goodbye", "en", "es", 2) is None     # different text
        assert c.get("hello", "de", "es", 2) is None       # different source
        assert c.get("hello", "en", "fr", 2) is None       # different target
        assert c.get("hello", "en", "es", 5) is None       # different num_beams

    def test_key_sensitivity_generation_params(self):
        c = TextTranslationCache()
        c.set("hello", "en", "es", 2, self.RESULT,
              length_penalty=1.0, no_repeat_ngram_size=0, repetition_penalty=1.0)
        # exact hit with matching gen params
        assert c.get("hello", "en", "es", 2,
                     length_penalty=1.0, no_repeat_ngram_size=0, repetition_penalty=1.0) is not None
        # each differing param misses
        assert c.get("hello", "en", "es", 2, length_penalty=1.5) is None
        assert c.get("hello", "en", "es", 2, no_repeat_ngram_size=3) is None
        assert c.get("hello", "en", "es", 2, repetition_penalty=1.2) is None

    def test_float_params_rounded_for_keying(self):
        c = TextTranslationCache()
        c.set("hello", "en", "es", 2, self.RESULT, length_penalty=1.0001)
        # 1.0001 and 1.0002 both round to 1.0 → same key
        assert c.get("hello", "en", "es", 2, length_penalty=1.0002) == self.RESULT

    def test_lru_eviction_bounds_size_and_drops_oldest(self):
        c = TextTranslationCache(max_size=3)
        for i in range(5):
            c.set(f"t{i}", "en", "es", 2, {"text": f"x{i}"})
        assert c.get_size() == 3
        assert c.get("t0", "en", "es", 2) is None  # evicted
        assert c.get("t1", "en", "es", 2) is None  # evicted
        assert c.get("t4", "en", "es", 2) == {"text": "x4"}  # newest kept

    def test_get_marks_recency_protecting_from_eviction(self):
        c = TextTranslationCache(max_size=2)
        c.set("a", "en", "es", 2, {"text": "A"})
        c.set("b", "en", "es", 2, {"text": "B"})
        assert c.get("a", "en", "es", 2) == {"text": "A"}  # 'a' now most-recent
        c.set("c", "en", "es", 2, {"text": "C"})           # evicts LRU = 'b'
        assert c.get("a", "en", "es", 2) == {"text": "A"}  # survived
        assert c.get("b", "en", "es", 2) is None           # evicted
        assert c.get("c", "en", "es", 2) == {"text": "C"}

    def test_clear_and_size(self):
        c = TextTranslationCache()
        c.set("a", "en", "es", 2, self.RESULT)
        assert c.get_size() == 1
        c.clear()
        assert c.get_size() == 0
        assert c.get("a", "en", "es", 2) is None

    def test_concurrent_access_stays_bounded_and_safe(self):
        c = TextTranslationCache(max_size=50)

        def worker(base):
            for i in range(200):
                c.set(f"k{base}-{i}", "en", "es", 2, {"text": str(i)})
                c.get(f"k{base}-{i}", "en", "es", 2)

        threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.get_size() <= 50


class TestTextTranslationCacheStats:
    def test_hits_misses_and_rate(self):
        c = TextTranslationCache()
        assert c.get_stats() == {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}
        c.get("x", "en", "es", 2)                       # miss
        c.set("x", "en", "es", 2, {"text": "y"})
        c.get("x", "en", "es", 2)                       # hit
        c.get("x", "en", "es", 2)                       # hit
        st = c.get_stats()
        assert st["size"] == 1
        assert st["hits"] == 2
        assert st["misses"] == 1
        assert st["hit_rate"] == round(2 / 3, 3)

    def test_clear_resets_counters(self):
        c = TextTranslationCache()
        c.set("x", "en", "es", 2, {"text": "y"})
        c.get("x", "en", "es", 2)
        c.get("z", "en", "es", 2)
        c.clear()
        assert c.get_stats() == {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}


class TestShouldCacheTranslation:
    def test_echo_is_not_cached(self):
        assert should_cache_translation("hello", "hello") is False
        assert should_cache_translation("hello", "  hello  ") is False  # whitespace-only diff

    def test_real_translation_is_cached(self):
        assert should_cache_translation("hello", "hola") is True


class TestShouldUseFp16:
    def test_enabled_on_gpu_devices(self):
        assert should_use_fp16(True, "cuda") is True
        assert should_use_fp16(True, "mps") is True

    def test_never_on_cpu(self):
        assert should_use_fp16(True, "cpu") is False

    def test_disabled_flag_is_false_everywhere(self):
        assert should_use_fp16(False, "cuda") is False
        assert should_use_fp16(False, "mps") is False
        assert should_use_fp16(False, "cpu") is False


class TestTranslationDevice:
    def test_cuda_preferred_when_both_present(self):
        assert translation_device(True, has_cuda=True, has_mps=True) == "cuda"

    def test_mps_when_no_cuda(self):
        assert translation_device(True, has_cuda=False, has_mps=True) == "mps"

    def test_cpu_when_no_accelerator(self):
        assert translation_device(True, has_cuda=False, has_mps=False) == "cpu"

    def test_cpu_when_gpu_disabled(self):
        assert translation_device(False, has_cuda=True, has_mps=True) == "cpu"


class TestTranslationLoadDtype:
    """Loading straight to fp16 keeps the fp32 copy from ever existing.

    MADLAD-3B is ~11.8 GB of fp32 weights; load-then-.half() peaks at ~17.7 GB,
    which does not fit in a 16 GB unified-memory Mac even though the fp16 weights
    are only ~5.9 GB.
    """

    def test_fp16_on_mps(self):
        # The case that makes MADLAD-3B reachable on the Mac at all.
        assert translation_load_dtype(True, True, has_cuda=False, has_mps=True) == "float16"

    def test_fp16_on_cuda(self):
        assert translation_load_dtype(True, True, has_cuda=True, has_mps=False) == "float16"

    def test_none_when_fp16_not_requested(self):
        assert translation_load_dtype(False, True, has_cuda=False, has_mps=True) is None
        assert translation_load_dtype(False, True, has_cuda=True, has_mps=False) is None

    def test_none_on_cpu_even_when_requested(self):
        # fp16 on CPU is slow/unsupported for many ops.
        assert translation_load_dtype(True, True, has_cuda=False, has_mps=False) is None

    def test_none_when_gpu_disabled(self):
        assert translation_load_dtype(True, False, has_cuda=True, has_mps=True) is None

    def test_agrees_with_should_use_fp16_across_the_matrix(self):
        """The two must never disagree, or a model loads fp16 but isn't counted as fp16."""
        for use_fp16 in (True, False):
            for use_gpu in (True, False):
                for has_cuda in (True, False):
                    for has_mps in (True, False):
                        device = translation_device(use_gpu, has_cuda, has_mps)
                        dtype = translation_load_dtype(use_fp16, use_gpu, has_cuda, has_mps)
                        assert (dtype == "float16") is should_use_fp16(use_fp16, device), (
                            use_fp16, use_gpu, has_cuda, has_mps)
