"""Static NLLB-200 translation catalog: language codes, UI names, model list.

Pure data + lookups shared by the translation routes and the live pipeline.
Extracted from speech_to_text.py so it can be imported and unit-tested without
the monolith's import-time side effects. Stdlib-only; no config needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# UI short code -> NLLB FLORES-200 code. "auto"/unknown fall back to English.
NLLB_LANG_CODES: Dict[str, str] = {
    "auto": "eng_Latn",  # Fallback to English if auto
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "ro": "ron_Latn",
    "sk": "slk_Latn",
    "bg": "bul_Cyrl",
    "hr": "hrv_Latn",
    "sr": "srp_Cyrl",
    "sl": "slv_Latn",
    "et": "est_Latn",
    "lv": "lvs_Latn",
    "lt": "lit_Latn",
    "fa": "pes_Arab",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
}

# Human-readable language names for UI
TRANSLATION_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese (Simplified)",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "el": "Greek",
    "he": "Hebrew",
    "hu": "Hungarian",
    "id": "Indonesian",
    "ms": "Malay",
    "ro": "Romanian",
    "sk": "Slovak",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "fa": "Persian",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
}

_NLLB_MODEL_DESCRIPTIONS: Dict[str, str] = {
    "facebook/nllb-200-distilled-600M": "Distilled 600M - Fast, good quality. Recommended for most users.",
    "facebook/nllb-200-distilled-1.3B": "Distilled 1.3B - Better quality, moderate speed.",
    "facebook/nllb-200-1.3B": "Full 1.3B - High quality, slower.",
    "facebook/nllb-200-3.3B": "Full 3.3B - Best quality, requires significant VRAM.",
    "facebook/nllb-moe-54b": "MoE 54B - Mixture of Experts, requires 80GB+ VRAM.",
}


def get_nllb_model_description(model_id: str) -> str:
    """Human-readable blurb for a known NLLB model id, or a generic fallback."""
    return _NLLB_MODEL_DESCRIPTIONS.get(
        model_id, "NLLB translation model - 200+ languages supported")


def get_default_nllb_models() -> List[Dict[str, Any]]:
    """The known NLLB models, used when the HuggingFace API is unreachable.

    Shape matches the entries the /api/models/nllb-list route builds from the
    HF API (model_id, name, size, size_order, downloads, likes, description).
    """
    return [
        {
            "model_id": "facebook/nllb-200-distilled-600M",
            "name": "nllb-200-distilled-600M",
            "size": "~1.2 GB",
            "size_order": 1,
            "downloads": 0,
            "likes": 0,
            "description": "Distilled 600M - Fast, good quality. Recommended for most users."
        },
        {
            "model_id": "facebook/nllb-200-distilled-1.3B",
            "name": "nllb-200-distilled-1.3B",
            "size": "~2.6 GB",
            "size_order": 2,
            "downloads": 0,
            "likes": 0,
            "description": "Distilled 1.3B - Better quality, moderate speed."
        },
        {
            "model_id": "facebook/nllb-200-1.3B",
            "name": "nllb-200-1.3B",
            "size": "~5.2 GB",
            "size_order": 3,
            "downloads": 0,
            "likes": 0,
            "description": "Full 1.3B - High quality, slower."
        },
        {
            "model_id": "facebook/nllb-200-3.3B",
            "name": "nllb-200-3.3B",
            "size": "~13 GB",
            "size_order": 4,
            "downloads": 0,
            "likes": 0,
            "description": "Full 3.3B - Best quality, requires significant VRAM."
        }
    ]


# --- MADLAD-400 (Google) -----------------------------------------------------
# MADLAD is a T5 MT model selected with a "<2xx>" target-language prefix on the
# input text (no source language, no forced_bos_token). Codes are short and
# mostly identical to our UI short codes; Hebrew uses Google's "iw" convention.
# "auto"/unknown fall back to English. This map also serves as the supported-
# target set for validation when the MADLAD engine is active.
MADLAD_LANG_CODES: Dict[str, str] = {
    "auto": "en",
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
    "nl": "nl", "pl": "pl", "ru": "ru", "ja": "ja", "ko": "ko", "zh": "zh",
    "ar": "ar", "hi": "hi", "tr": "tr", "vi": "vi", "th": "th", "uk": "uk",
    "cs": "cs", "sv": "sv", "da": "da", "fi": "fi", "no": "no", "el": "el",
    "he": "iw",  # MADLAD/mC4 uses "iw" for Hebrew
    "hu": "hu", "id": "id", "ms": "ms", "ro": "ro", "sk": "sk", "bg": "bg",
    "hr": "hr", "sr": "sr", "sl": "sl", "et": "et", "lv": "lv", "lt": "lt",
    "fa": "fa", "bn": "bn", "ta": "ta", "te": "te", "mr": "mr", "gu": "gu",
    "kn": "kn", "ml": "ml", "pa": "pa", "ur": "ur",
}


def madlad_target_code(target_lang: str) -> str:
    """UI short code -> MADLAD language code (fallback English)."""
    return MADLAD_LANG_CODES.get(target_lang, "en")


def build_madlad_input(text: str, target_lang: str) -> str:
    """MADLAD input is the text prefixed with a '<2xx>' target tag."""
    return f"<2{madlad_target_code(target_lang)}> {text}"


def is_madlad_model(model_id: str) -> bool:
    """Whether a model id refers to a MADLAD-400 model (vs NLLB)."""
    return "madlad" in (model_id or "").lower()


def supported_target(lang: str, method: str) -> bool:
    """Whether a UI target-language short code is valid for the active engine."""
    table = MADLAD_LANG_CODES if method == "madlad" else NLLB_LANG_CODES
    return lang in table


def madlad_anti_repetition_defaults(repetition_penalty: float, no_repeat_ngram_size: int) -> Tuple[float, int]:
    """Anti-repetition generation params for MADLAD, which can fall into
    repetition loops. Nudges the neutral defaults to safe values but keeps any
    operator-tuned settings: repetition_penalty 1.0 -> 1.1, no_repeat 0 -> 4.
    Returns (repetition_penalty, no_repeat_ngram_size)."""
    rep = repetition_penalty if (repetition_penalty and repetition_penalty != 1.0) else 1.1
    nr = no_repeat_ngram_size if (no_repeat_ngram_size and no_repeat_ngram_size > 0) else 4
    return rep, nr


def get_default_madlad_models() -> List[Dict[str, Any]]:
    """Known MADLAD-400 models, used when the HuggingFace API is unreachable.
    Same shape as get_default_nllb_models()."""
    return [
        {
            "model_id": "google/madlad400-3b-mt",
            "name": "madlad400-3b-mt",
            "size": "~12 GB",
            "size_order": 1,
            "downloads": 0,
            "likes": 0,
            "description": "MADLAD-400 3B - 400+ languages, high quality. ~12 GB (use fp16 on GPU)."
        },
        {
            "model_id": "google/madlad400-7b-mt",
            "name": "madlad400-7b-mt",
            "size": "~27 GB",
            "size_order": 2,
            "downloads": 0,
            "likes": 0,
            "description": "MADLAD-400 7B - Best quality, requires significant VRAM."
        }
    ]
