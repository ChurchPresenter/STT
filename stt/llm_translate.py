"""LLM-backed translation: request building and — mainly — response validation.

An NMT model can only ever return a translation. An LLM can return a refusal, its own
reasoning, the prompt echoed back, a commentary on the text, or the source language
untouched. Every one of those was observed while measuring candidate models against
real captions from a service:

* a reasoning model ignored every switch for disabling thinking and returned
  "Okay, let's tackle this translation request. The user wants me to..." for each
  caption, consuming the whole token budget
* a 1B model returned "Мы preparing for the meeting" — source language leaking
  into the output — and once left a caption almost entirely in Russian
* the same model echoed the prompt structure back ("Russian: ... Draft translation: ...")
* and once refused: "I can't assist with providing translations that may be
  mistranslated"

So a caption is only usable if it survives validation, and unusable output must be
*rejected* (returning None so the caller falls back to the NMT model) rather than
displayed. A wrong caption in front of a congregation is worse than a late one.

Stdlib-only and free of provider SDKs so it can be unit-tested without a model: the
caller performs the HTTP request or the in-process call, and passes the raw string here.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Mapping, Optional

# Cyrillic blocks. Reused for the "did it actually translate?" check, mirroring the
# CJK screen stt/text_utils.py:filter_hallucinated_text already applies to Whisper.
_CYRILLIC = re.compile(r"[Ѐ-ӿԀ-ԯ]")

# Scripts that indicate the target language was NOT produced, keyed by target code.
# Only languages whose script differs from the target's are checkable this way; for
# same-script pairs (en->de) this test cannot help and is skipped.
_WRONG_SCRIPT_FOR_TARGET = {
    "en": _CYRILLIC,
    "es": _CYRILLIC,
    "fr": _CYRILLIC,
    "de": _CYRILLIC,
    "pt": _CYRILLIC,
    "it": _CYRILLIC,
    "nl": _CYRILLIC,
    "pl": _CYRILLIC,
}

# Openers a model uses when it starts reasoning or narrating instead of translating.
_REASONING_OPENERS = (
    "okay, let's", "okay, so", "ok, let's", "let's tackle", "let me tackle",
    "we are given", "first, the user", "the user wants", "the user has given",
    "i need to translate", "sure, here", "sure! here", "here is the translation",
    "here's the translation", "translation:", "english:", "russian:",
)

# Refusals. A refusal is not a translation and must never reach a caption.
_REFUSAL_MARKERS = (
    "i can't assist", "i cannot assist", "i can't help", "i cannot help",
    "i'm unable to", "i am unable to", "i can't provide", "i cannot provide",
    "as an ai", "i'm sorry, but i", "i am sorry, but i",
)

# A wrapper the model added around an otherwise fine translation.
_STRIPPABLE_PREFIXES = (
    "translation:", "english:", "english translation:", "translated:",
    "corrected:", "corrected caption:", "caption:", "output:",
)


# The shipped prompt, as a template. "{language}" is filled with the configured
# target language's English name, because the wording that makes this feature work
# — render terminology the way that language's Bibles and churches do — is
# necessarily language-specific. An earlier version hard-coded English here, and a
# session configured for Spanish silently produced English captions: the target
# language reached only the validator, whose wrong-script screen looks for Cyrillic
# and so waved English through.
DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You translate live captions for a church service. Output ONLY the translation — "
    "no notes, no explanation, no reasoning, no quotation marks. Render biblical and "
    "liturgical terminology the way {language} Bibles and church usage do, and use the "
    "standard {language} spelling of biblical names. Keep the speaker's register: plain "
    "spoken language, not archaic {language}, except inside direct scripture quotations. "
    "Preserve meaning exactly — never add, omit, explain, or answer the text. If the "
    "input is a fragment, translate it as a fragment. If the input is a scripture "
    "reference, translate only the reference; do not quote the passage."
)


def language_name(code: Optional[str], names: Optional[Mapping[str, str]] = None) -> str:
    """English name for a UI language code ("es" -> "Spanish").

    Unknown codes return the code itself rather than a guess or an empty string, so
    a prompt built from one still names *something* specific and the operator can
    see what was sent. ``names`` is the catalog to look in, passed by the caller so
    this module stays free of import-time dependencies.
    """
    text = (code or "").strip()
    if not text:
        return ""
    if names:
        found = names.get(text) or names.get(text.lower())
        if found:
            return str(found)
    return text


def build_system_prompt(base_prompt: str, target_lang: Optional[str],
                        names: Optional[Mapping[str, str]] = None) -> str:
    """The system prompt for a given target language.

    Fills "{language}" in a template, and then states the target explicitly at the
    end regardless. The trailing directive is not redundant belt-and-braces: an
    operator's custom prompt is written for whatever language they had configured
    when they wrote it, and without it a later target-language switch would leave
    the model still being told to produce the old one — the silent failure this
    function exists to prevent. Stating the target twice costs a few tokens; a
    service captioned into the wrong language costs rather more.
    """
    name = language_name(target_lang, names)
    text = (base_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT_TEMPLATE
    if "{language}" in text:
        text = text.replace("{language}", name or "the target language")
    if not name:
        return text
    return f"{text}\n\nTranslate into {name}. Output only {name}."


def build_chat_messages(text: str, system_prompt: str,
                        draft: Optional[str] = None,
                        source_name: str = "Source") -> List[Dict[str, str]]:
    """OpenAI-style messages for one caption.

    ``draft`` switches to post-editing (source plus an NMT draft). Measurement favours
    leaving it None — translating from the source caught meaning errors that
    post-editing anchored on and missed — but the shape is kept because post-editing
    was better on terminology and may be wanted when both models can be resident.

    ``source_name`` labels the source line in that post-editing prompt; it defaults
    to a neutral word rather than naming a language, since the source is often
    "auto".
    """
    user = f"{source_name}: {text}\nDraft translation: {draft}" if draft else text
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user}]


def build_chat_payload(model: str, text: str, system_prompt: str, *,
                       draft: Optional[str] = None, max_tokens: int = 120,
                       temperature: float = 0.0,
                       keep_alive: Optional[Any] = -1,
                       extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Body for an OpenAI-compatible /chat/completions (or Ollama /api/chat) request.

    ``temperature`` defaults to 0: captions should be reproducible, and a session
    replayed for review must produce what the congregation saw.

    ``keep_alive=-1`` pins the model in memory. This is not a tuning nicety — an
    unpinned model measured p90 4.89s against p50 0.29s purely because the runtime
    unloaded it between captions; pinning collapsed p90 to 0.40s. Providers that do
    not understand the field ignore it.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": build_chat_messages(text, system_prompt, draft),
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if extra:
        payload.update(dict(extra))
    return payload


def uses_local_llm(translation_config: Optional[Mapping[str, Any]]) -> bool:
    """True when captions are translated by the in-process GGUF on this machine.

    The two translation engines have separate weights and separate releasers, so
    every caller that loads, unloads or describes "the model" has to know which one
    is in play. Deriving that inline let the copies drift: the remote-unload route
    tested the NMT flag while an LLM session was running, decided nothing was
    loaded, and left the GGUF resident for the life of the process. One definition,
    so there is nothing to drift.

    "endpoint" is the default provider — an absent provider means the model belongs
    to another machine, and unloading it is not this machine's business.
    """
    config = translation_config or {}
    if (config.get("translation_method") or "nllb") != "llm":
        return False
    llm = config.get("llm") or {}
    return str(llm.get("provider") or "endpoint").strip().lower() == "local"


def local_model_path(models_dir: str, gguf_repo: str, gguf_file: str) -> str:
    """Where a downloaded GGUF lives, mirroring how other models are stored.

    The Model Manager keeps a model under a directory named for its repo with "/"
    replaced by "--" (google/madlad400-3b-mt -> google--madlad400-3b-mt), so a GGUF
    follows the same convention and is found by the same browse/delete tooling.
    """
    return os.path.join(models_dir, gguf_repo.replace("/", "--"), gguf_file)


# Weight files that mark a directory as a transformers model rather than a
# standalone GGUF release.
_TRANSFORMERS_WEIGHTS = (".safetensors", ".bin", ".msgpack", ".h5")


def scan_gguf_models(models_dir: str) -> List[Dict[str, Any]]:
    """Downloaded GGUF models as [{repo, files: [{name, size_bytes}]}], repo-sorted.

    The inverse of local_model_path: a directory named for the repo with "/"
    replaced by "--", holding one or more quantisations. Only directories that
    actually contain a .gguf are reported, so a half-deleted or unrelated model
    directory does not appear as an empty choice in the picker.

    A directory that also holds transformers weights is skipped even when it does
    contain .gguf files. google/madlad400-3b-mt is the case that forced this: the
    HuggingFace repo ships model-q2k/q3k/q4k.gguf beside model.safetensors, so
    downloading MADLAD as a translation model put "madlad400-3b-mt" in the LLM
    picker. It is an NMT model with no chat template — selecting it would send
    chat-completion requests to something that cannot answer them. A genuine GGUF
    release (bartowski, ggml-org, TheBloke) ships .gguf and metadata, no weights
    in any other format.

    A missing or unreadable models directory yields [] rather than raising — the
    caller is a settings page that must still render.
    """
    out: List[Dict[str, Any]] = []
    try:
        entries = sorted(os.listdir(models_dir))
    except OSError:
        return out
    for entry in entries:
        path = os.path.join(models_dir, entry)
        if not os.path.isdir(path):
            continue
        files: List[Dict[str, Any]] = []
        try:
            names = sorted(os.listdir(path))
        except OSError:
            continue
        if any(n.lower().endswith(_TRANSFORMERS_WEIGHTS) for n in names):
            continue
        for name in names:
            if not name.lower().endswith(".gguf"):
                continue
            try:
                size = os.path.getsize(os.path.join(path, name))
            except OSError:
                size = 0
            files.append({"name": name, "size_bytes": size})
        if files:
            out.append({"repo": entry.replace("--", "/"), "dir": entry, "files": files})
    return out


def resolve_gpu_layers(n_gpu_layers: Any, has_gpu: bool) -> int:
    """Layers to offload to the GPU: -1 for all, 0 for none.

    "auto" (the default) means all layers when an accelerator is present and none
    otherwise. CPU-only is the safe fallback rather than an error, because a caption
    is short — measured p50 19 output tokens — so CPU inference is slow but viable,
    and refusing to run at all would be worse than running slowly.
    """
    if isinstance(n_gpu_layers, int):
        return n_gpu_layers
    text = str(n_gpu_layers or "auto").strip().lower()
    if text in ("auto", ""):
        return -1 if has_gpu else 0
    try:
        return int(text)
    except ValueError:
        return -1 if has_gpu else 0


def extract_chat_text(response: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Pull the assistant text out of either response shape, or None.

    Handles OpenAI (``choices[0].message.content``) and Ollama
    (``message.content``), plus the plain-completion ``response`` field.
    """
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            msg = first.get("message")
            if isinstance(msg, Mapping) and msg.get("content"):
                return str(msg["content"])
            if first.get("text"):
                return str(first["text"])
    msg = response.get("message")
    if isinstance(msg, Mapping) and msg.get("content"):
        return str(msg["content"])
    if response.get("response"):
        return str(response["response"])
    return None


def _strip_wrappers(text: str) -> str:
    """Remove quote wrappers, labels, and an echoed prompt block."""
    out = text.strip()

    # An echoed prompt: keep only what follows the last "Draft translation:" or the
    # trailing line, since the model repeated our own framing back at us.
    if "draft translation:" in out.lower():
        out = re.split(r"(?i)draft translation:\s*", out)[-1].strip()

    # Reasoning models emit <think>...</think>; drop it if the block is closed.
    out = re.sub(r"(?is)<think>.*?</think>", "", out).strip()

    for _ in range(3):  # labels can nest: 'Translation: "English: ..."'
        low = out.lower()
        for prefix in _STRIPPABLE_PREFIXES:
            if low.startswith(prefix):
                out = out[len(prefix):].strip()
                break
        else:
            if len(out) >= 2 and out[0] in "\"'“«" and out[-1] in "\"'”»":
                out = out[1:-1].strip()
                continue
            break
    return out


def _word_count(text: str) -> int:
    return len(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))


def validate_translation(raw: Optional[str], source: str, target_lang: str, *,
                         max_expansion: float = 3.0, min_word_budget: int = 8) -> Optional[str]:
    """The cleaned caption, or None if the output is not a usable translation.

    None means "fall back to the NMT model" — never "show this". Rejects, in order:
    empty output; a refusal; a reasoning or narration opener; wrong-script output
    (the source language leaking through); multi-paragraph output; and output far
    longer than the source, which is how commentary and recitation present.

    ``max_expansion`` is deliberately loose, because Russian to English legitimately
    expands. But a pure ratio is not enough on its own: a short source has a tiny
    budget, so an earlier version exempted sources under three words — and that hole
    let through the worst failure seen over a full service. Asked to translate
    "1 Фессалоникийцам 5 глава." (two words after digits are discounted), the model
    replied "1 Corinthians 11:1-24" followed by the recited text of the passage.
    ``min_word_budget`` closes it: short sources get a small absolute allowance
    instead of an unlimited one, so "Щит веры." -> "The shield of faith."
    still passes while a recitation does not.
    """
    if raw is None:
        return None
    text = _strip_wrappers(raw)
    if not text:
        return None

    low = text.lower()
    if any(marker in low for marker in _REFUSAL_MARKERS):
        return None
    if low.startswith(_REASONING_OPENERS):
        return None

    wrong_script = _WRONG_SCRIPT_FOR_TARGET.get((target_lang or "").lower())
    if wrong_script is not None and wrong_script.search(text):
        # The model returned (some of) the source language instead of translating.
        return None

    # A caption is one utterance. Blank-line-separated output means the model produced
    # a document — the scripture-recitation failure mode.
    if "\n\n" in text.strip():
        return None

    # Absolute floor as well as a ratio, so a short source still has a bounded budget.
    allowed = max(min_word_budget, int(max_expansion * _word_count(source)))
    if _word_count(text) > allowed:
        return None

    return text


def looks_like_reasoning_model(response: Mapping[str, Any]) -> bool:
    """Whether a probe response shows the model reasoning rather than answering.

    Worth checking once at startup instead of discovering it mid-service. A reasoning
    model may expose its thinking in a separate field, or — the case actually
    observed — inline it in the content where it cannot be separated out.
    """
    if not isinstance(response, Mapping):
        return False
    msg = response.get("message")
    if isinstance(msg, Mapping) and str(msg.get("thinking") or "").strip():
        return True
    if str(response.get("thinking") or "").strip():
        return True
    text = (extract_chat_text(response) or "").strip().lower()
    return text.startswith(_REASONING_OPENERS) or "<think>" in text
