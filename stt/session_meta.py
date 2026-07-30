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
    _put(meta, "asr.language", _section(config, "audio").get("language"))

    decode = _section(config, "whisper_decoding", "live_transcription")
    for key in _DECODE_KEYS:
        if key in decode:
            _put(meta, f"asr.decode.{key}", decode[key])

    vad = _section(config, "vad")
    _put(meta, "asr.vad.enabled", vad.get("enabled"))
    _put(meta, "asr.vad.threshold", vad.get("threshold"))

    _add_file_transcription(meta, config, model_cfg)


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

    remote = _section(lt, "remote")
    if remote:
        _put(meta, "mt.remote.enabled", remote.get("enabled"))
        _put(meta, "mt.remote.endpoint", remote.get("endpoint"))
        _put(meta, "mt.remote.model", remote.get("model"))
        _put(meta, "mt.remote.fallback", remote.get("fallback"))


def _add_filters(meta: Dict[str, str], config: Mapping[str, Any]) -> None:
    hf = _section(config, "hallucination_filter")
    _put(meta, "filter.hallucination_enabled", hf.get("enabled"))
    phrases = hf.get("phrases")
    _put(meta, "filter.hallucination_phrase_count",
         len(phrases) if isinstance(phrases, (list, tuple)) else 0)
    _put(meta, "filter.cjk_enabled", hf.get("cjk_filter_enabled"))

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
        changed_at = datetime.now().isoformat(timespec="seconds")
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
