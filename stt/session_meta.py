"""Session provenance: which models and decode settings produced a transcript.

A session database records what was said but, historically, nothing about what
produced it. When a transcript turns out to have quality defects, attributing
them to the transcription stage or the translation stage requires knowing the
models and decode parameters that were live at the time — and those are
recoverable only while the server happens to still be running the same config.

This module builds the flat ``key -> value`` mapping stored in each session
database's ``session_meta`` table, and owns the sqlite reads/writes for it.

Extracted from speech_to_text.py so it can be imported (and unit-tested)
without the monolith's import-time side effects. Stdlib-only; config, identity
strings, and paths are passed in explicitly — never read from the monolith's
globals.

Two invariants the rest of the system depends on:

* **Effective, not merely intended.** The recorded model is the one that
  actually ran. ``resolve_translation_model_id`` can substitute the MADLAD
  default for a stale NLLB id, and the ASR model id lives in a different config
  sub-block per model type; both are resolved here so a reader gets one
  trustworthy key instead of three ambiguous ones. For an LLM session that
  extends to the prompt: what is recorded is the text the model received, not
  the (usually empty) configured override.
* **Never fatal.** Recording provenance must not be able to break a service
  that is about to start transcribing, nor a live language switch. Every
  public function here swallows and logs rather than raising.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from stt.llm_translate import DEFAULT_SYSTEM_PROMPT_TEMPLATE, build_system_prompt
from stt.nllb_catalog import TRANSLATION_LANGUAGES, resolve_translation_model_id

# Keys whose value can change while a session is running are appended under
# "<key>@<ISO timestamp>" rather than overwritten, so the base key always means
# "at session start" and the suffixed rows form an audit trail.
CHANGE_SUFFIX = "@"

# Live-decode parameters recorded verbatim. Several of these are routinely
# tuned away from Whisper's defaults (logprob_threshold and
# compression_ratio_threshold in particular, which trade recall for precision),
# and their values are invisible in the transcript itself.
_DECODE_KEYS = (
    "beam_size",
    "best_of",
    "temperature",
    "condition_on_previous_text",
    "logprob_threshold",
    "no_speech_threshold",
    "compression_ratio_threshold",
)

# live_translation.generation_params, flattened to mt.gen.*
_GEN_KEYS = (
    "num_beams",
    "no_repeat_ngram_size",
    "repetition_penalty",
    "length_penalty",
)

# live_translation.llm keys that apply to either provider, flattened to mt.llm.*.
# timeout_ms earns its place: on timeout the caption falls back to the NMT model
# silently, so a session captioned partly by a model this table would otherwise
# name as the only one running.
_LLM_KEYS = (
    "max_tokens",
    "keep_alive",
    "timeout_ms",
    "warmup_timeout_ms",
    # Both decide what a *rejected* caption becomes, which is the part of a
    # transcript that cannot be explained from the text alone. Without them a
    # reader cannot tell a caption the LLM produced on its second attempt from one
    # it got right first time, nor an untranslated row that means "the LLM declined
    # and this box has no NMT model loaded" from one that means "translation was
    # still warming up".
    "retry_on_reject",
    "fallback",
)

# Provider-specific keys: recording an endpoint for a local session (or GPU
# offload for a hosted one) would assert a setting that had no effect.
_LLM_LOCAL_KEYS = ("gguf_repo", "gguf_file", "gguf_path", "n_gpu_layers", "n_ctx")
_LLM_ENDPOINT_KEYS = ("endpoint",)


def _stringify(value: Any) -> str:
    """Render a config value as a stable string.

    The table is TEXT/TEXT so it stays schema-free as settings come and go.
    Booleans become "true"/"false" (not Python's "True"/"False") so the values
    read back as JSON-ish; containers become compact JSON, which matters for
    ``temperature`` — a scalar for live decoding but a fallback list for file
    decoding. None becomes "" so an unset setting is distinguishable from the
    string "None".
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _put(meta: Dict[str, str], key: str, value: Any) -> None:
    meta[key] = _stringify(value)


def content_digest(value: Any) -> str:
    """Short stable digest of a list or mapping, or "" when there is nothing.

    For content that decides the output but is far too large to store: filter
    phrase lists, glossary term tables. Two sessions with the same digest ran the
    same content; a different digest is proof they did not, which is what "the
    captions changed and nobody remembers editing anything" needs. Twelve hex
    characters — this identifies an edit, it does not defend against one.
    """
    if not value:
        return ""
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _section(config: Optional[Mapping[str, Any]], *path: str) -> Dict[str, Any]:
    """Fetch a nested config dict, returning {} for any missing or non-dict hop.

    Lets the builders below read a whole block without guarding each level —
    a partial config yields empty sections rather than an AttributeError.
    """
    node: Any = config or {}
    for part in path:
        if not isinstance(node, Mapping):
            return {}
        node = node.get(part, {})
    return dict(node) if isinstance(node, Mapping) else {}


def resolve_asr_implementation(model_cfg: Mapping[str, Any]) -> str:
    """Which ASR implementation the configured model type/backend selects.

    ``model.backend`` only distinguishes faster-whisper from OpenAI Whisper, and
    only when ``model.type`` is "whisper" — for the huggingface and custom types
    it carries no meaning. Collapsing that into one key stops a reader from
    concluding "backend: whisper" meant faster-whisper (they are different
    implementations, and faster-whisper is the default, so a plain "whisper"
    means it was deliberately switched off).
    """
    model_type = model_cfg.get("type") or "whisper"
    if model_type != "whisper":
        return str(model_type)
    backend = model_cfg.get("backend") or ""
    return "faster-whisper" if backend == "faster-whisper" else "openai-whisper"


def resolve_asr_model(model_cfg: Mapping[str, Any]) -> str:
    """The configured ASR model id, from whichever sub-block the type selects."""
    model_type = model_cfg.get("type") or "whisper"
    if model_type == "huggingface":
        return _stringify(_section(model_cfg, "huggingface").get("model_id"))
    if model_type == "custom":
        return _stringify(_section(model_cfg, "custom").get("model_path"))
    return _stringify(_section(model_cfg, "whisper").get("model"))


# --- Per-row provenance ----------------------------------------------------------
#
# session_meta records what a session was configured with. That is not the same as
# what produced a given caption, and the gap is not academic: when the LLM declines a
# caption it is translated by the NMT model instead, so a session recorded as "llm"
# contains rows from two engines with nothing to tell them apart. Measuring an LLM
# change against such a transcript silently compares its output to the other model's
# on exactly the rows where they differ most — which is how a replay first read as a
# regression when it was not one.
#
# Hence three columns on every row, filled by whichever leg actually ran.

# Engines that can produce a caption's translated_text.
MT_ENGINE_LLM = "llm"
MT_ENGINE_NMT = "nmt"
MT_ENGINE_REMOTE = "remote"      # another machine translated it; its own engine is in the label
MT_ENGINE_WHISPER = "whisper"    # whisper_translate decoded the audio straight to the target
MT_ENGINE_NONE = "none"          # nothing translated it; the caption is the source text


def asr_row_label(config: Optional[Mapping[str, Any]]) -> str:
    """What transcribed a row, as "implementation/model".

    Both halves, because neither identifies a decode on its own: "large-v3" says
    nothing about whether faster-whisper or openai-whisper produced it, and the two
    do not decode identically.
    """
    model_cfg = _section(config, "model")
    implementation = resolve_asr_implementation(model_cfg)
    model = resolve_asr_model(model_cfg)
    if not model:
        return implementation
    return "%s/%s" % (implementation, model)


def mt_row_label(lt_cfg: Optional[Mapping[str, Any]], engine: str,
                 *, remote_status: Optional[Mapping[str, Any]] = None,
                 model: str = "") -> str:
    """What translated a row, named the way that engine identifies its model.

    ``engine`` is passed by the leg that ran rather than derived from config, which
    is the whole point: config says which engine was *asked*, and the interesting
    rows are the ones where a different one answered.

    ``model`` is the caller's own model id, used by the engines this module cannot
    name from config alone: the NMT model actually resident, and the Whisper model
    that decoded straight to the target language.

    For a remote translation the label names the machine and, when its status has
    been seen, the model it reported — a paired box can be reconfigured without this
    one restarting, so local config is not evidence about what ran over there.
    """
    lt = lt_cfg or {}
    if engine == MT_ENGINE_LLM:
        return llm_model_id(_section(lt, "llm"))
    if engine == MT_ENGINE_REMOTE:
        endpoint = _stringify(_section(lt, "remote").get("endpoint")).strip()
        remote_model = _stringify((remote_status or {}).get("model")).strip()
        if endpoint and remote_model:
            return "%s (%s)" % (endpoint, remote_model)
        return endpoint or MT_ENGINE_REMOTE
    return _stringify(model).strip()


def _add_identity(meta: Dict[str, str], version: str, commit: str,
                  describe: str, hostname: str, started_at: str) -> None:
    _put(meta, "app.version", version)
    _put(meta, "app.commit", commit)
    _put(meta, "app.describe", describe)
    _put(meta, "host.name", hostname)
    _put(meta, "session.started_at", started_at)


def _add_transcription(meta: Dict[str, str], config: Mapping[str, Any]) -> None:
    model_cfg = _section(config, "model")
    _put(meta, "asr.type", model_cfg.get("type") or "whisper")
    _put(meta, "asr.backend", model_cfg.get("backend"))
    _put(meta, "asr.implementation", resolve_asr_implementation(model_cfg))
    _put(meta, "asr.model", resolve_asr_model(model_cfg))
    _put(meta, "asr.use_flash_attention",
         _section(model_cfg, "huggingface").get("use_flash_attention"))
    if (model_cfg.get("type") or "whisper") == "custom":
        _put(meta, "asr.custom.model_type", _section(model_cfg, "custom").get("model_type"))
    audio = _section(config, "audio")
    _put(meta, "asr.language", audio.get("language"))
    _put(meta, "asr.use_gpu", _section(config, "performance").get("use_gpu"))

    decode = _section(config, "whisper_decoding", "live_transcription")
    for key in _DECODE_KEYS:
        if key in decode:
            _put(meta, f"asr.decode.{key}", decode[key])

    vad = _section(config, "vad")
    _put(meta, "asr.vad.enabled", vad.get("enabled"))
    _put(meta, "asr.vad.threshold", vad.get("threshold"))

    _add_audio_shaping(meta, audio)
    _add_segmentation(meta, audio)
    _add_file_transcription(meta, config, model_cfg)


def _add_audio_shaping(meta: Dict[str, str], audio: Mapping[str, Any]) -> None:
    """What reached the model besides the microphone signal itself.

    ``context_prompt`` is decode-affecting in the same way beam_size is — the
    previous captions are passed to Whisper as initial_prompt, so the same audio
    decodes differently depending on what preceded it. ``loudness_normalization``
    changes the samples before the model ever sees them.
    """
    ctx = _section(audio, "context_prompt")
    _put(meta, "asr.context_prompt.enabled", ctx.get("enabled"))
    _put(meta, "asr.context_prompt.max_chars", ctx.get("max_chars"))

    norm = _section(audio, "loudness_normalization")
    _put(meta, "asr.normalize.enabled", norm.get("enabled"))
    _put(meta, "asr.normalize.target_rms_dbfs", norm.get("target_rms_dbfs"))
    _put(meta, "asr.normalize.max_gain", norm.get("max_gain"))


def _add_segmentation(meta: Dict[str, str], audio: Mapping[str, Any]) -> None:
    """Where one caption ends and the next begins.

    These do not change the words, they change how the words are cut into rows —
    which is what a reader comparing two transcripts of the same service is
    actually looking at when the line breaks fall in different places.
    """
    _put(meta, "asr.segment.energy_threshold", audio.get("energy_threshold"))
    _put(meta, "asr.segment.phrase_timeout", audio.get("phrase_timeout"))
    _put(meta, "asr.segment.same_output_threshold", audio.get("same_output_threshold"))
    _put(meta, "asr.segment.stabilize_live_text", audio.get("stabilize_live_text"))

    pending = _section(audio, "pending_buffer")
    _put(meta, "asr.segment.pending_buffer.enabled", pending.get("enabled"))
    _put(meta, "asr.segment.pending_buffer.max_words", pending.get("max_words"))
    _put(meta, "asr.segment.pending_buffer.max_age_seconds", pending.get("max_age_seconds"))


def _add_file_transcription(meta: Dict[str, str], config: Mapping[str, Any],
                            live_model_cfg: Mapping[str, Any]) -> None:
    """File-transcription settings under asr.file.*.

    ``file_transcription.model.backend`` empty means "same as the main model",
    so it is resolved here — recording the empty string would leave a reader
    unable to tell which implementation ran.
    """
    ft = _section(config, "file_transcription")
    if not ft:
        return
    ft_model = _section(ft, "model")
    merged = dict(ft_model)
    if not merged.get("backend"):
        merged["backend"] = live_model_cfg.get("backend")
    if not merged.get("type"):
        merged["type"] = live_model_cfg.get("type")

    _put(meta, "asr.file.type", merged.get("type") or "whisper")
    _put(meta, "asr.file.backend", merged.get("backend"))
    _put(meta, "asr.file.implementation", resolve_asr_implementation(merged))
    _put(meta, "asr.file.model", resolve_asr_model(merged) or resolve_asr_model(live_model_cfg))
    _put(meta, "asr.file.language", ft.get("language"))

    decode = _section(config, "whisper_decoding", "file_transcription")
    for key in _DECODE_KEYS:
        if key in decode:
            _put(meta, f"asr.file.decode.{key}", decode[key])


def is_offloaded(lt_cfg: Mapping[str, Any]) -> bool:
    """Whether translation is handed to a paired remote rather than run locally."""
    remote = _section(lt_cfg, "remote")
    return bool(remote.get("enabled") and remote.get("endpoint"))


def uses_llm(lt_cfg: Mapping[str, Any]) -> bool:
    """Whether captions are translated by an LLM rather than an NMT model."""
    return lt_cfg.get("translation_method") == "llm"


def llm_provider(llm_cfg: Mapping[str, Any]) -> str:
    """'local' (in-process GGUF) or 'endpoint' (OpenAI-compatible server)."""
    return str(llm_cfg.get("provider") or "endpoint").strip().lower()


def llm_model_id(llm_cfg: Mapping[str, Any]) -> str:
    """The LLM that translates, named the way its provider identifies it.

    For provider=local that is the GGUF: an explicit gguf_path wins over
    gguf_repo/gguf_file, mirroring the load order in get_local_llm() — recording
    the repo when a path overrode it would name a file that never loaded.
    """
    if llm_provider(llm_cfg) == "local":
        path = _stringify(llm_cfg.get("gguf_path")).strip()
        if path:
            return path
        repo = _stringify(llm_cfg.get("gguf_repo")).strip()
        file = _stringify(llm_cfg.get("gguf_file")).strip()
        return f"{repo}/{file}" if repo and file else (file or repo)
    return _stringify(llm_cfg.get("model")).strip()


def _effective_local_model(lt: Mapping[str, Any], madlad_default: str) -> str:
    """The model that will actually translate, as far as this box can know.

    When offloading with a blank remote model ("use Machine B's own model"), the
    local model id is not what translates — recording it as mt.model would assert
    the wrong model, which is the exact failure this table exists to prevent. It
    stays empty until the remote is probed (see remote_provenance); the local
    config value remains available as mt.model_configured.

    An LLM session is the same failure in a second form: translation_model still
    holds the NMT id, which on that session is only the fallback and is not what
    captioned the service. The LLM is named instead, and the NMT id stays
    readable as mt.model_configured.
    """
    if is_offloaded(lt) and not _section(lt, "remote").get("model"):
        return ""
    if uses_llm(lt) and not is_offloaded(lt):
        return llm_model_id(_section(lt, "llm"))
    return resolve_translation_model_id(lt, madlad_default)


def _add_llm(meta: Dict[str, str], lt: Mapping[str, Any]) -> None:
    """mt.llm.* — the settings that decide what an LLM session produces.

    Only written for an LLM session: on an NMT one these values are inert, and a
    table that records them anyway invites a reader to explain the captions with
    a prompt that was never sent.

    The prompt is recorded *effective* rather than as configured. A blank
    system_prompt means the shipped template was used, and build_system_prompt()
    then substitutes the target language and appends the target directive — so
    the configured value alone (usually "") says nothing about what the model was
    told, which is the one input an operator tunes captions with.

    api_key is deliberately never recorded: session databases are delivered to a
    NAS and handed around. Whether one was sent is recorded instead, which is
    what a later "why did it 401" needs.
    """
    if not uses_llm(lt):
        return
    llm = _section(lt, "llm")
    provider = llm_provider(llm)
    _put(meta, "mt.llm.provider", provider)
    _put(meta, "mt.llm.model", llm_model_id(llm))
    for key in (_LLM_LOCAL_KEYS if provider == "local" else _LLM_ENDPOINT_KEYS):
        _put(meta, f"mt.llm.{key}", llm.get(key))
    for key in _LLM_KEYS:
        _put(meta, f"mt.llm.{key}", llm.get(key))
    custom = _stringify(llm.get("system_prompt")).strip()
    _put(meta, "mt.llm.system_prompt_custom", bool(custom))
    _put(meta, "mt.llm.system_prompt",
         build_system_prompt(custom or DEFAULT_SYSTEM_PROMPT_TEMPLATE,
                             lt.get("target_language"), TRANSLATION_LANGUAGES))
    _put(meta, "mt.llm.api_key_set", bool(_stringify(llm.get("api_key")).strip()))


def remote_provenance(status: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """mt.remote.effective.* from a paired remote's /api/translation/status.

    On an offloaded session the remote's model is the one doing the work, so
    without this the database records nothing about what actually translated.
    Returns {} for a missing, failed, or unrecognisable response — an unreachable
    remote must leave no claim behind rather than a misleading one.
    """
    if not isinstance(status, Mapping) or not status.get("success", True):
        return {}
    fields = (
        ("model", "translation_model"),
        ("method", "translation_method"),
        ("device", "model_device"),
        ("dtype", "model_dtype"),
        ("ct2", "is_ctranslate2"),
        ("ct2_compute_type", "ct2_compute_type"),
        # Present only when the remote translates with an LLM; an NMT remote omits
        # them and they are simply absent from the transcript.
        ("llm_provider", "llm_provider"),
        ("llm_endpoint", "llm_endpoint"),
        # How the remote runs that model. On an offloaded session these live only on
        # the other box, so a transcript without them names the translator but not
        # the configuration that shaped every caption — and a service recorded that
        # way cannot be replayed against a changed setting, because nobody can say
        # what the setting used to be. The prompt matters most: it is the surface an
        # operator tunes captions with, and it is invisible in the output.
        ("llm_max_tokens", "llm_max_tokens"),
        ("llm_n_ctx", "llm_n_ctx"),
        ("llm_retry_on_reject", "llm_retry_on_reject"),
        ("llm_fallback", "llm_fallback"),
        ("llm_context_window", "llm_context_window"),
        ("llm_system_prompt", "llm_system_prompt"),
    )
    meta: Dict[str, str] = {}
    for suffix, source in fields:
        if status.get(source) is not None:
            _put(meta, f"mt.remote.effective.{suffix}", status[source])
    return meta


def _add_translation(meta: Dict[str, str], config: Mapping[str, Any],
                     madlad_default: str) -> None:
    lt = _section(config, "live_translation")
    _put(meta, "mt.enabled", lt.get("enabled"))
    _put(meta, "mt.method", lt.get("translation_method"))
    _put(meta, "mt.offloaded", is_offloaded(lt))
    _put(meta, "mt.model", _effective_local_model(lt, madlad_default))
    _put(meta, "mt.model_configured", lt.get("translation_model"))
    _put(meta, "mt.source_language", lt.get("source_language"))
    _put(meta, "mt.target_language", lt.get("target_language"))
    _put(meta, "mt.context_window", lt.get("context_window"))
    _put(meta, "mt.translate_in_progress", lt.get("translate_in_progress"))
    _put(meta, "mt.display_mode", lt.get("display_mode"))
    _put(meta, "mt.warmup", lt.get("warmup"))
    _put(meta, "mt.srt_enabled", lt.get("srt_enabled"))
    _put(meta, "mt.html_enabled", lt.get("html_enabled"))
    _put(meta, "mt.max_entries_to_send", lt.get("max_entries_to_send"))

    # NMT engine settings. Skipped for a local LLM session, where nothing loads a
    # torch/CT2 model and these describe a model that never ran — the same reason
    # /api/translation/status reports them as null for an LLM session.
    if not (uses_llm(lt) and not is_offloaded(lt)):
        _put(meta, "mt.use_ctranslate2", lt.get("use_ctranslate2"))
        _put(meta, "mt.ct2_compute_type", lt.get("ct2_compute_type"))
        _put(meta, "mt.ct2_inter_threads", lt.get("ct2_inter_threads"))
        _put(meta, "mt.ct2_intra_threads", lt.get("ct2_intra_threads"))
        _put(meta, "mt.use_gpu", lt.get("use_gpu"))
        _put(meta, "mt.use_fp16", lt.get("use_fp16"))

        gen = _section(lt, "generation_params")
        for key in _GEN_KEYS:
            if key in gen:
                _put(meta, f"mt.gen.{key}", gen[key])

    _add_llm(meta, lt)
    _add_tts(meta, lt)
    _add_file_translation(meta, config)

    remote = _section(lt, "remote")
    if remote:
        _put(meta, "mt.remote.enabled", remote.get("enabled"))
        _put(meta, "mt.remote.endpoint", remote.get("endpoint"))
        _put(meta, "mt.remote.model", remote.get("model"))
        _put(meta, "mt.remote.fallback", remote.get("fallback"))
        # A cache hit returns text translated under whatever prompt, model or
        # glossary was live when it was first stored — so a session can contain
        # captions this table's other keys do not explain.
        _put(meta, "mt.remote.server_cache_enabled", remote.get("server_cache_enabled"))
        _put(meta, "mt.remote.server_cache_size", remote.get("server_cache_size"))
        _put(meta, "mt.remote.sync_dictionary_on_edit", remote.get("sync_dictionary_on_edit"))


def _add_tts(meta: Dict[str, str], lt: Mapping[str, Any]) -> None:
    """mt.tts.* — the voice the translation was spoken in, when it was spoken.

    Only for an enabled TTS session: on a caption-only service these name a voice
    nobody heard.
    """
    tts = _section(lt, "tts")
    if not tts.get("enabled"):
        _put(meta, "mt.tts.enabled", bool(tts.get("enabled")))
        return
    backend = tts.get("backend") or "edge"
    _put(meta, "mt.tts.enabled", True)
    _put(meta, "mt.tts.backend", backend)
    _put(meta, "mt.tts.voice",
         tts.get("piper_model") if backend == "piper" else tts.get("edge_voice"))
    _put(meta, "mt.tts.speed", tts.get("speed"))


def _add_file_translation(meta: Dict[str, str], config: Mapping[str, Any]) -> None:
    """mt.file.* — the translation half of the file-transcription pipeline.

    asr.file.* records how a dropped-in file was transcribed but said nothing
    about how it was then translated, which for a batch-produced transcript is
    half the provenance missing.
    """
    ft = _section(config, "file_transcription")
    if not ft:
        return
    _put(meta, "mt.file.enabled", ft.get("translate_enabled"))
    _put(meta, "mt.file.target_language", ft.get("translate_to"))
    _put(meta, "mt.file.model", ft.get("translation_model"))


def _add_filters(meta: Dict[str, str], config: Mapping[str, Any]) -> None:
    """filter.* — everything that can remove or rewrite a caption after decoding.

    This is the group a reader needs most and had least of. A caption dropped by
    min_words, by the fuzzy-duplicate check, by the Music verdict or by a
    hallucination phrase leaves nothing behind in the transcript: the row is
    simply absent, and absence reads as "the model did not hear it". Recording
    the thresholds makes the difference recoverable.
    """
    hf = _section(config, "hallucination_filter")
    _put(meta, "filter.hallucination_enabled", hf.get("enabled"))
    phrases = hf.get("phrases")
    _put(meta, "filter.hallucination_phrase_count",
         len(phrases) if isinstance(phrases, (list, tuple)) else 0)
    # The count says how many filters ran; only the digest says which. Phrases are
    # edited between services, so two sessions with equal counts are not
    # necessarily comparable — and the list is far too long to store in full.
    _put(meta, "filter.hallucination_phrase_digest", content_digest(phrases))
    _put(meta, "filter.cjk_enabled", hf.get("cjk_filter_enabled"))

    audio = _section(config, "audio")
    _put(meta, "filter.min_words", audio.get("min_words"))
    _put(meta, "filter.fuzzy_duplicate_threshold", audio.get("fuzzy_duplicate_threshold"))

    pf = _section(config, "profanity_filter")
    _put(meta, "filter.profanity_enabled", pf.get("enabled"))
    _put(meta, "filter.profanity_replacement", pf.get("replacement"))
    words = pf.get("words")
    _put(meta, "filter.profanity_word_count",
         len(words) if isinstance(words, (list, tuple)) else 0)

    # A Music segment is discarded outright unless transcribe_detected_music is
    # set, so this is the one audio-type setting that decides whether a row exists
    # at all. Stated as the consequence rather than the toggle, because that is
    # the question being asked of it.
    std = _section(config, "speech_type_detection")
    _put(meta, "filter.music_dropped",
         bool(std.get("enabled")) and not bool(std.get("transcribe_detected_music")))


def glossary_provenance(dictionary: Optional[Mapping[str, Any]],
                        source: str = "local") -> Dict[str, str]:
    """glossary.* facts that live in the dictionary file, not in config.

    The terms are what rewrite the captions, and config records only the file
    name — so a glossary edited between two services leaves the two transcripts
    looking identically configured. Recorded as a digest plus counts rather than
    the table itself, which can run to hundreds of entries.

    ``source`` distinguishes this machine's own custom_dictionary.json from a
    glossary pushed by a paired Machine A for the session: on an offloaded
    session the client's table is the one that applies, and naming the local file
    would point a reader at terms that were never used.

    Called by the monolith at session start (the file read belongs there);
    returns {} for a missing or malformed dictionary so a session still starts.
    """
    glossary = (dictionary or {}).get("glossary")
    if not isinstance(glossary, Mapping):
        return {}
    pairs = {k: v for k, v in glossary.items() if isinstance(v, Mapping)}
    meta: Dict[str, str] = {}
    _put(meta, "glossary.source", source)
    _put(meta, "glossary.pairs", ",".join(sorted(pairs)))
    _put(meta, "glossary.term_count", sum(len(v) for v in pairs.values()))
    _put(meta, "glossary.digest", content_digest(glossary))
    return meta


def _add_audio_type(meta: Dict[str, str], config: Mapping[str, Any]) -> None:
    """audio_type.* — the Speaking/Music/Quiet classifier behind that drop.

    Recorded whether or not it drops anything: the verdict is stored per segment
    in the transcript, and a reader comparing those tags across sessions needs
    the thresholds that produced them.
    """
    std = _section(config, "speech_type_detection")
    if not std:
        return
    _put(meta, "audio_type.enabled", std.get("enabled"))
    _put(meta, "audio_type.method", std.get("method"))
    _put(meta, "audio_type.device", std.get("device"))
    _put(meta, "audio_type.music_threshold", std.get("music_threshold"))
    _put(meta, "audio_type.music_prob_threshold", std.get("music_prob_threshold"))
    _put(meta, "audio_type.quiet_db_threshold", std.get("quiet_db_threshold"))
    _put(meta, "audio_type.smoothing_window", std.get("smoothing_window"))
    _put(meta, "audio_type.transcribe_music", std.get("transcribe_detected_music"))


def _add_corrections(meta: Dict[str, str], config: Mapping[str, Any]) -> None:
    """corrections.* — whether a caption could be edited before it was published.

    output_delay holds a caption back for review; with auto_publish off a caption
    that was never approved never reached the screen, which is another way for a
    transcript row and what the congregation saw to disagree.
    """
    corr = _section(config, "corrections")
    if not corr:
        return
    _put(meta, "corrections.enabled", corr.get("enabled"))
    _put(meta, "corrections.confidence_threshold", corr.get("confidence_threshold"))
    delay = _section(corr, "output_delay")
    _put(meta, "corrections.output_delay.enabled", delay.get("enabled"))
    _put(meta, "corrections.output_delay.seconds", delay.get("delay_seconds"))
    _put(meta, "corrections.output_delay.auto_publish", delay.get("auto_publish"))


def _add_glossary(meta: Dict[str, str], config: Mapping[str, Any]) -> None:
    """glossary.* — the dictionary that rewrites translated text.

    Omitted for a local LLM session: apply_glossary() is called only from the NMT
    decode paths (_translate_text_ct2 and translate_text), so on an LLM session
    these settings had no effect and recording them invites a reader to explain a
    caption with a substitution that never ran.

    Only the file name is knowable from config; the terms themselves live in that
    file and are added by glossary_provenance() at session start.
    """
    lt = _section(config, "live_translation")
    if uses_llm(lt) and not is_offloaded(lt):
        return
    cd = _section(config, "custom_dictionary")
    _put(meta, "glossary.enabled", cd.get("nllb_glossary_enabled"))
    _put(meta, "glossary.file", cd.get("file"))


def build_session_meta(config: Optional[Mapping[str, Any]], version: str, commit: str,
                       describe: str, hostname: str, madlad_default: str,
                       started_at: Optional[str] = None) -> Dict[str, str]:
    """Build the session_meta mapping for a session starting now.

    ``started_at`` defaults to the local wall clock in ISO 8601 seconds
    precision, matching the ``timestamp`` column convention; callers pass it
    explicitly to keep tests deterministic.

    Best-effort by contract: a partial or malformed config yields whatever could
    be read rather than raising, because a session must still start.
    """
    meta: Dict[str, str] = {}
    if started_at is None:
        started_at = datetime.now().isoformat(timespec="seconds")
    try:
        _add_identity(meta, version, commit, describe, hostname, started_at)
        _add_transcription(meta, config or {})
        _add_translation(meta, config or {}, madlad_default)
        _add_filters(meta, config or {})
        _add_audio_type(meta, config or {})
        _add_corrections(meta, config or {})
        _add_glossary(meta, config or {})
    except Exception as e:  # pragma: no cover - defensive; sections already guard
        print(f"[SESSION-META] WARNING: could not build full provenance: {e}")
    return meta


def latest_values(meta: Mapping[str, str]) -> Dict[str, str]:
    """Every key's value as of now, with appended changes applied.

    A base key holds the session-start value and a "<key>@<timestamp>" row
    supersedes it from that moment. Comparing live config against this — rather
    than against the base keys — is what stops a setting that already changed
    from being re-appended on every subsequent config write.
    """
    latest = {k: v for k, v in meta.items() if CHANGE_SUFFIX not in k}
    for key in sorted(meta):  # sorted -> chronological per key, so last wins
        root, separator, _ = key.partition(CHANGE_SUFFIX)
        if separator:
            latest[root] = meta[key]
    return latest


def changed_keys(previous: Mapping[str, str], current: Mapping[str, str]) -> Dict[str, str]:
    """Keys whose value differs, for appending a mid-session change.

    Only keys present in ``current`` are considered: a setting disappearing from
    config is not a value change worth recording, and inventing an empty-string
    row for it would read as though it had been cleared.
    """
    return {k: v for k, v in current.items() if previous.get(k) != v}


def _connect(db_path: str) -> sqlite3.Connection:
    # busy_timeout so a write landing during heavy transcription waits its turn
    # instead of raising "database is locked".
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def ensure_table(cursor: sqlite3.Cursor) -> None:
    """Create the session_meta table if absent. Caller owns the transaction."""
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS session_meta (key TEXT PRIMARY KEY, value TEXT)"
    )


def write_session_meta(db_path: str, meta: Mapping[str, str]) -> bool:
    """Create the table and store ``meta``. Returns True on success.

    Used once at session start, so a plain upsert is correct here: the base keys
    are being established, not changed.
    """
    if not meta:
        return False
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            ensure_table(cursor)
            cursor.executemany(
                "INSERT OR REPLACE INTO session_meta (key, value) VALUES (?, ?)",
                sorted(meta.items()),
            )
            conn.commit()
        return True
    except Exception as e:
        print(f"[SESSION-META] WARNING: could not record session provenance: {e}")
        return False


def write_missing(db_path: str, meta: Mapping[str, str]) -> bool:
    """Store only keys not already present. Returns True if anything was added.

    For session-start facts that can only be learned late — the paired remote's
    model has to be fetched over the network, so it can't be part of the initial
    write. Establishing them as base keys (rather than appending a timestamped
    row) keeps the timeline honest: nothing changed, we just found out. A later
    genuine change still goes through append_changes.
    """
    if not meta:
        return False
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            ensure_table(cursor)
            cursor.executemany(
                "INSERT OR IGNORE INTO session_meta (key, value) VALUES (?, ?)",
                sorted(meta.items()),
            )
            added = cursor.rowcount > 0
            conn.commit()
        return added
    except Exception as e:
        print(f"[SESSION-META] WARNING: could not record late provenance: {e}")
        return False


def append_changes(db_path: str, changes: Mapping[str, str],
                   changed_at: Optional[str] = None) -> bool:
    """Append mid-session changes as "<key>@<ISO timestamp>" rows.

    Never touches the base key: it must keep meaning "at session start", so a
    reader can tell a session that began in one configuration and changed from
    one that started in the final configuration. Reading a key's history is
    ``read_history(db_path, key)``.
    """
    if not changes:
        return False
    if changed_at is None:
        # Milliseconds, not seconds: the row key is "<key>@<timestamp>", so two
        # changes to one key inside the same second collided on that key and the
        # second INSERT OR REPLACEd the first — a real change disappearing rather
        # than being recorded. A shorter second-precision stamp from an older
        # session still sorts before a millisecond one in the same second, so
        # both readers that treat lexicographic order as chronological
        # (latest_values here, the preview panel in file-manager.html) are
        # unaffected.
        changed_at = datetime.now().isoformat(timespec="milliseconds")
    try:
        with _connect(db_path) as conn:
            cursor = conn.cursor()
            ensure_table(cursor)
            cursor.executemany(
                "INSERT OR REPLACE INTO session_meta (key, value) VALUES (?, ?)",
                [(f"{key}{CHANGE_SUFFIX}{changed_at}", value)
                 for key, value in sorted(changes.items())],
            )
            conn.commit()
        return True
    except Exception as e:
        print(f"[SESSION-META] WARNING: could not record settings change: {e}")
        return False


def append_new_changes(db_path: str, values: Mapping[str, str],
                       changed_at: Optional[str] = None) -> Dict[str, str]:
    """Append only the values that actually differ from what is already recorded.

    ``append_changes`` writes whatever it is handed, and its "the caller has
    diffed already" invariant held for exactly one of its three callers. The
    other two re-appended values identical to the recorded ones — a probe of an
    unchanged remote on every offloaded save, and an unload/reload of the same
    translation model — filling the timeline with rows that carry no
    information. The guard belongs here, where no caller can skip it.

    Compares against ``latest_values``, not against the base keys: a setting that
    goes A -> B -> A has genuinely returned to A, and that return is a change
    worth a row. Diffing against the base key is precisely the bug this replaces.

    Returns the mapping that was written ({} when nothing differed), so a caller
    can log what really happened rather than announcing a write that did not
    occur. Never raises: an unreadable database yields {}.
    """
    if not values:
        return {}
    stored, read_error = load_session_meta(db_path)
    if read_error:
        print(f"[SESSION-META] WARNING: could not read before appending: {read_error}")
        return {}
    fresh = changed_keys(latest_values(stored), values)
    if not fresh:
        return {}
    return fresh if append_changes(db_path, fresh, changed_at) else {}


def _read_all(uri: str) -> Tuple[Dict[str, str], Optional[str]]:
    """Read the table through one read-only URI. Raises on an unusable database."""
    with sqlite3.connect(uri, uri=True, timeout=30) as conn:
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_meta'"
        ).fetchone()
        if not present:
            return {}, None  # pre-provenance session: absence is the answer
        rows: Iterable[Any] = conn.execute(
            "SELECT key, value FROM session_meta ORDER BY key"
        ).fetchall()
    return {str(key): "" if value is None else str(value) for key, value in rows}, None


def load_session_meta(db_path: str) -> Tuple[Dict[str, str], Optional[str]]:
    """Read the whole table as (meta, error).

    Always read-only, so reading provenance can never lock or mutate a session
    that is still recording, nor leave sidecar files beside an archived one.

    A session recorded before this feature existed has no such table, which is
    normal and yields ({}, None). A database that could not be read yields
    ({}, "<reason>") — the two must stay distinguishable, because collapsing them
    into a bare {} makes a healthy session that merely failed to open look like
    one that was never recorded, and sends the reader after the wrong problem.
    """
    # Which URI to try first is decided by the -wal sidecar, not by catching an
    # error, because the two platforms fail differently and only one of them
    # fails at all:
    #
    #   macOS  plain read-only on a detached WAL database raises, because SQLite
    #          wants to create the -shm and read-only forbids it.
    #   Linux  the same open SUCCEEDS and creates -shm and -wal beside the file.
    #
    # So an error-driven fallback silently wrote two files into whatever
    # directory a session was delivered to — a NAS share we were asked only to
    # read. Choosing by the sidecar makes both platforms take the same path:
    # a database still being written has its -wal present and is read normally,
    # while a detached one (delivered, archived, downloaded as a lone .db) is
    # read with immutable=1, which needs no sidecar and creates none.
    attached = os.path.exists(db_path + "-wal")
    uris = ([f"file:{db_path}?mode=ro", f"file:{db_path}?mode=ro&immutable=1"] if attached
            else [f"file:{db_path}?mode=ro&immutable=1", f"file:{db_path}?mode=ro"])
    first_error = None
    for uri in uris:
        try:
            return _read_all(uri)
        except sqlite3.Error as e:
            if first_error is None:
                first_error = e
        except Exception as e:
            return {}, f"{type(e).__name__}: {e}"
    return {}, f"{type(first_error).__name__}: {first_error}"




def read_session_meta(db_path: str) -> Dict[str, str]:
    """The mapping alone, {} if unreadable — for callers that diff it.

    Use load_session_meta() when the distinction between "nothing recorded" and
    "could not read" matters, which is anywhere a human sees the result.
    """
    meta, _ = load_session_meta(db_path)
    return meta


def read_history(meta: Mapping[str, str], key: str) -> list:
    """Timeline for one key as [(changed_at, value)], oldest first.

    The first entry is the session-start value with an empty timestamp; each
    later entry is an appended change. A single entry means it never changed.
    """
    timeline = []
    if key in meta:
        timeline.append(("", meta[key]))
    prefix = f"{key}{CHANGE_SUFFIX}"
    for k in sorted(meta):
        if k.startswith(prefix):
            timeline.append((k[len(prefix):], meta[k]))
    return timeline
