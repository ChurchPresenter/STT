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


def build_chat_messages(text: str, system_prompt: str,
                        draft: Optional[str] = None) -> List[Dict[str, str]]:
    """OpenAI-style messages for one caption.

    ``draft`` switches to post-editing (source plus an NMT draft). Measurement favours
    leaving it None — translating from the source caught meaning errors that
    post-editing anchored on and missed — but the shape is kept because post-editing
    was better on terminology and may be wanted when both models can be resident.
    """
    user = f"Russian: {text}\nDraft translation: {draft}" if draft else text
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
                         max_expansion: float = 3.0) -> Optional[str]:
    """The cleaned caption, or None if the output is not a usable translation.

    None means "fall back to the NMT model" — never "show this". Rejects, in order:
    empty output; a refusal; a reasoning or narration opener; wrong-script output
    (the source language leaking through); and output far longer than the source,
    which is how commentary and repetition present.

    ``max_expansion`` is deliberately loose. Russian to English legitimately expands,
    and short captions expand most (a two-word source can triple), so this is a
    backstop against runaway generation rather than a style rule.
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

    src_words = _word_count(source)
    if src_words >= 3 and _word_count(text) > max_expansion * src_words:
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
