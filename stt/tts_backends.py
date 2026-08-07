"""TTS backend registry + the pure voice-switch decision.

The server speaks translated captions through one of three backends: ``edge``
(Microsoft cloud), ``piper`` (local, one model per language) and ``supertonic``
(local, one model covering 30 languages). Everything that differs between them
and is *data* — which config key holds the selected voice, which per-language
preference map it stashes into, what audio container the synthesizer emits —
lives in :data:`BACKENDS` rather than in ``if backend == ...`` chains scattered
through the monolith.

:func:`select_voice` is the one non-trivial decision: when the operator changes
target language mid-service, which voice does the new language get? It is pure,
with the "pick a sensible default for this language" step injected, so it can be
tested without edge-tts's network call or piper's filesystem.

Stdlib-only, as required of every ``stt/`` logic module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, MutableMapping, Optional

EDGE = "edge"
PIPER = "piper"
SUPERTONIC = "supertonic"

#: The backend assumed when config is missing or names one we don't know.
DEFAULT_BACKEND = EDGE


@dataclass(frozen=True)
class TtsBackend:
    """Everything about a backend that is data rather than behaviour."""

    id: str
    label: str
    #: Runs on this machine (has a model to load/unload) vs. a cloud call.
    is_local: bool
    #: Container the synthesizer returns, as the browser's decoder needs it.
    audio_format: str
    #: Key under ``live_translation.tts`` holding the selected voice/model.
    voice_key: str
    #: Key under ``live_translation.tts`` holding the per-language voice map.
    prefs_key: str
    #: True when each language needs its own downloaded model (piper).
    needs_per_language_model: bool
    #: Import name probed to report availability in /api/tts/status.
    import_name: str


BACKENDS: Dict[str, TtsBackend] = {
    EDGE: TtsBackend(
        id=EDGE,
        label="Edge TTS (Cloud - Microsoft, 400+ voices)",
        is_local=False,
        audio_format="mp3",
        voice_key="edge_voice",
        prefs_key="edge_voice_preferences",
        needs_per_language_model=False,
        import_name="edge_tts",
    ),
    PIPER: TtsBackend(
        id=PIPER,
        label="Piper (Local/Offline, one model per language)",
        is_local=True,
        audio_format="wav",
        voice_key="piper_model",
        prefs_key="piper_model_preferences",
        needs_per_language_model=True,
        import_name="piper",
    ),
    SUPERTONIC: TtsBackend(
        id=SUPERTONIC,
        label="Supertonic (Local/Offline, 30 languages in one model)",
        is_local=True,
        audio_format="wav",
        voice_key="supertonic_voice",
        prefs_key="supertonic_voice_preferences",
        needs_per_language_model=False,
        import_name="supertonic",
    ),
}


def get_backend(backend_id: Optional[str]) -> TtsBackend:
    """The backend descriptor for ``backend_id``, falling back to edge.

    An unknown id is a stale or hand-edited config, not a crash: the caller
    gets the default backend, which is what the server did before the registry
    existed.
    """
    return BACKENDS.get((backend_id or "").strip().lower(), BACKENDS[DEFAULT_BACKEND])


def backend_ids() -> List[str]:
    """Registered backend ids, in presentation order."""
    return list(BACKENDS)


def local_backend_ids() -> List[str]:
    """Ids of the backends that load a model on this machine."""
    return [b.id for b in BACKENDS.values() if b.is_local]


def select_voice(
    backend_id: Optional[str],
    old_lang: str,
    new_lang: str,
    tts_section: MutableMapping[str, object],
    pick_default: Callable[[str], Optional[str]],
) -> Optional[str]:
    """Choose the voice for ``new_lang``, remembering the one ``old_lang`` used.

    Mutates ``tts_section`` in place — the outgoing voice is stashed under
    ``old_lang`` in the backend's preference map, so switching back later
    restores the operator's choice instead of re-picking a default. Returns the
    voice for ``new_lang`` (a stored preference wins over ``pick_default``), or
    None when nothing is available — the caller decides whether that is a
    warning or a silent no-op, since it means different things per backend.
    """
    backend = get_backend(backend_id)

    prefs_obj = tts_section.setdefault(backend.prefs_key, {})
    if not isinstance(prefs_obj, dict):  # a corrupt config shouldn't crash the switch
        prefs_obj = {}
        tts_section[backend.prefs_key] = prefs_obj

    current = tts_section.get(backend.voice_key) or ""
    if current and old_lang:
        prefs_obj[old_lang] = current

    chosen = prefs_obj.get(new_lang) or pick_default(new_lang)
    if chosen:
        tts_section[backend.voice_key] = chosen
    return chosen


# ─── Supertonic ─────────────────────────────────────────────────────────────
#
# One ~260MB model covers every language below; there is nothing per-language to
# download. Mirrors supertonic.SUPPORTED_LANGUAGES for the pinned model revision
# (duplicated rather than imported because this module must stay stdlib-only —
# tests/test_tts_backends.py asserts the two agree when the package is present).

SUPERTONIC_LANGUAGES = frozenset(
    {
        "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi",
        "fr", "hi", "hr", "hu", "id", "it", "ja", "ko", "lt", "lv",
        "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
        "vi",
    }
)

#: Language code meaning "work it out from the text" — used for anything the
#: model wasn't trained on (notably Chinese, which piper still covers).
SUPERTONIC_UNKNOWN_LANG = "na"

#: The presets shipped with the model. Voice choice is independent of language.
SUPERTONIC_VOICES: List[Dict[str, str]] = [
    {"id": "F1", "name": "F1 (female)"},
    {"id": "F2", "name": "F2 (female)"},
    {"id": "F3", "name": "F3 (female)"},
    {"id": "F4", "name": "F4 (female)"},
    {"id": "F5", "name": "F5 (female)"},
    {"id": "M1", "name": "M1 (male)"},
    {"id": "M2", "name": "M2 (male)"},
    {"id": "M3", "name": "M3 (male)"},
    {"id": "M4", "name": "M4 (male)"},
    {"id": "M5", "name": "M5 (male)"},
]

DEFAULT_SUPERTONIC_VOICE = "F1"

#: Denoising steps: higher is better and slower. The model's own bounds.
MIN_SUPERTONIC_STEPS = 5
MAX_SUPERTONIC_STEPS = 12
DEFAULT_SUPERTONIC_STEPS = 8

#: Speed multiplier bounds the model accepts; ours is a 0.5-2.0 slider.
MIN_SUPERTONIC_SPEED = 0.7
MAX_SUPERTONIC_SPEED = 2.0


def supertonic_lang(lang_code: str) -> str:
    """The ``lang=`` argument for a target language.

    Untrained languages fall back to ``"na"`` (language-agnostic) rather than
    failing — the pronunciation is worse, but the operator hears something.
    """
    return lang_code.lower() if lang_code.lower() in SUPERTONIC_LANGUAGES else SUPERTONIC_UNKNOWN_LANG


def supertonic_supports(lang_code: str) -> bool:
    """Whether the model was actually trained on this language."""
    return lang_code.lower() in SUPERTONIC_LANGUAGES


def supertonic_voice_for_lang(lang_code: str, current: Optional[str] = None) -> Optional[str]:
    """Voice for a language: the current preset, since voices are multilingual.

    Shaped like the other ``*_for_lang`` pickers so :func:`select_voice` can
    take it as ``pick_default``. Returns None only for a language the model
    can't speak, which lets the caller warn instead of emitting silence.
    """
    if not supertonic_supports(lang_code):
        return None
    return current or DEFAULT_SUPERTONIC_VOICE


def clamp_supertonic_steps(steps: object) -> int:
    """Coerce a configured step count into the model's supported range."""
    try:
        value = int(steps)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return DEFAULT_SUPERTONIC_STEPS
    return max(MIN_SUPERTONIC_STEPS, min(MAX_SUPERTONIC_STEPS, value))


def clamp_supertonic_speed(speed: object) -> float:
    """Coerce the shared 0.5-2.0 speed slider into what the model accepts."""
    try:
        value = float(speed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    return max(MIN_SUPERTONIC_SPEED, min(MAX_SUPERTONIC_SPEED, value))
