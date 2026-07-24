"""Static Piper TTS catalog + pure voice-selection matching.

The catalog is static data; the two matchers are the pure lang-code -> voice
decision, with their IO dependencies (the live edge-tts voice list, the
"is this piper model downloaded?" check) injected by the caller. Extracted from
speech_to_text.py so the matching logic is importable and unit-testable without
the monolith's import-time side effects. Stdlib-only.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# Bundled Piper voices offered in the model manager (id, name, language, quality, size).
PIPER_MODELS_CATALOG: List[Dict[str, str]] = [
    {"id": "en_US-lessac-medium", "name": "Lessac (US English)", "language": "en", "quality": "medium", "size": "75MB"},
    {"id": "en_US-amy-medium", "name": "Amy (US English)", "language": "en", "quality": "medium", "size": "75MB"},
    {"id": "en_US-ryan-medium", "name": "Ryan (US English)", "language": "en", "quality": "medium", "size": "75MB"},
    {"id": "en_GB-alba-medium", "name": "Alba (British English)", "language": "en", "quality": "medium", "size": "75MB"},
    {"id": "de_DE-thorsten-medium", "name": "Thorsten (German)", "language": "de", "quality": "medium", "size": "75MB"},
    {"id": "es_ES-davefx-medium", "name": "Davefx (Spanish)", "language": "es", "quality": "medium", "size": "75MB"},
    {"id": "fr_FR-siwis-medium", "name": "Siwis (French)", "language": "fr", "quality": "medium", "size": "75MB"},
    {"id": "ru_RU-ruslan-medium", "name": "Ruslan (Russian)", "language": "ru", "quality": "medium", "size": "75MB"},
    {"id": "uk_UA-ukrainian_tts-medium", "name": "Ukrainian TTS", "language": "uk", "quality": "medium", "size": "75MB"},
    {"id": "zh_CN-huayan-medium", "name": "Huayan (Chinese)", "language": "zh", "quality": "medium", "size": "75MB"},
]


def edge_voice_for_lang(lang_code: str, voices: List[Dict[str, Any]]) -> Optional[str]:
    """ShortName of the first edge-tts voice whose Locale starts with ``lang_code``.

    e.g. ('ru', [...]) -> 'ru-RU-DmitryNeural'. Case-insensitive. Returns None
    when ``voices`` is empty or nothing matches. ``voices`` is the live edge-tts
    voice list, passed in so this stays pure.
    """
    if not voices:
        return None
    lang_lower = lang_code.lower()
    for v in voices:
        locale = (v.get("Locale") or "").lower()
        if locale.startswith(lang_lower):
            return v.get("ShortName")
    return None


def piper_model_for_lang(
    lang_code: str,
    catalog: List[Dict[str, str]],
    is_downloaded: Callable[[str], bool],
) -> Optional[str]:
    """Id of the first *downloaded* piper model whose language equals ``lang_code``.

    ``is_downloaded`` is injected (it hits the filesystem in production) so the
    matching order/logic can be tested in isolation. Returns None if no matching
    model is downloaded.
    """
    lang_lower = lang_code.lower()
    for m in catalog:
        if m["language"].lower() == lang_lower and is_downloaded(m["id"]):
            return m["id"]
    return None
