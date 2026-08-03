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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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

# The rule names check_translation reports. Stable strings: they are counted and
# compared across replay runs, so renaming one silently breaks a comparison against
# an earlier report.
REJECT_EMPTY = "empty"
REJECT_REFUSAL = "refusal"
REJECT_REASONING = "reasoning"
REJECT_WRONG_SCRIPT = "wrong_script"
REJECT_PARAGRAPHS = "paragraphs"
REJECT_LIST = "list"
REJECT_NUMBERS = "numbers"
REJECT_TOO_LONG = "too_long"
REJECT_TOO_SHORT = "too_short"


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
    "reference, translate only the reference; do not quote the passage. Keep chapter "
    "and verse numbers exactly as spoken: never replace a reference with the text it "
    "points to, and never expand a range such as 'verses 2-9' into a list of numbers. "
    # The sentences below were measured, not reasoned about. They exist because the
    # input is not clean text — it is speech transcribed by Whisper, and the model
    # treated a misheard word as a cue to supply something it recognised instead:
    # a bare "3 глава, 14 стих" came back as a recitation of a different verse
    # entirely, and a mis-transcribed Russian patronymic came back half-translated,
    # an English given name in front of untouched Cyrillic.
    #
    # An earlier wording said "translate what is written, as written". It scored 8
    # fixed and 0 broken on the service it was written against — and then on a
    # held-out service the model read "as written" as leave it as it is, echoing
    # short announcements and whole Russian sentences back untranslated. That is the
    # reason for the explicit "must come back in {language}" clause, and the reason
    # both services are now replayed before any wording here is believed.
    #
    # Measured over both services, ~1210 captions, against the prompt without these
    # sentences: 10 captions fixed, 5 broken. That is the honest size of it — a net
    # five or so in twelve hundred. The variance between the two services is larger
    # than the gap between this wording and the one it replaced, so neither is
    # "better" by measurement; this one is kept because the fault it avoids is the
    # worse one to have on screen. Do not tune this against a single service.
    "The speech was transcribed automatically and may contain misheard words, "
    "especially names of Bible books. A garbled word is still to be translated: render "
    "it into {language} as best you can, and never replace it with a passage or a "
    "phrase you recognise instead. Every caption must come back in {language} — never "
    "repeat the input unchanged. Translate every clause: if the input has three "
    "sentences, the output has three sentences."
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


# What to tell the model about the answer it just gave, keyed by the rule that
# rejected it. A rejection is otherwise a downgrade: the caption falls to the NMT
# model, which for most of these is not the better answer — the LLM had the caption
# nearly right and broke one rule. Naming the broken rule is the cheapest way to ask
# for the caption it should have produced.
#
# Absent from this table, deliberately: "empty" and "wrong_script". An empty answer
# usually means the call itself failed, and a retry spends a second caption's budget
# to find that out again; wrong-script output means the model is not translating at
# all, which a nudge does not fix. Both go straight to NMT as before.
_RETRY_NOTES = {
    # This note was measured and rewritten twice, because both obvious phrasings push
    # the model off one edge or the other. Telling it not to recite the passage made it
    # answer with a bare "John 3:16" and drop the sentence the speaker built around the
    # reference — a numbers rejection traded for a too_short one. Telling it to
    # translate the words around the reference broke the opposite case, where the
    # caption really is nothing but a bare "9 глава, 22 стих". So the note now states
    # the rule both cases share: translate the words that are there, whichever they are.
    REJECT_NUMBERS: (
        "Your previous answer dropped or changed the chapter and verse numbers. Keep "
        "every figure exactly as the input has it, and translate exactly the words the "
        "input contains — no more and no fewer. If the input is only a reference, give "
        "only that reference. If the speaker said more around it, or quoted the verse "
        "aloud, translate that too. Never replace a reference with the text it points to."
    ),
    REJECT_TOO_SHORT: (
        "Your previous answer left out part of the input. Translate every clause of "
        "it, including the words around any quotation — not only the quotation."
    ),
    REJECT_TOO_LONG: (
        "Your previous answer added material the input does not contain. Translate "
        "only the words that are there and stop when they stop. If the input merely "
        "names a passage or a subject, translate that name — do not supply the text "
        "it refers to."
    ),
    REJECT_LIST: (
        "Your previous answer was a list. Return one caption, on one line, as running "
        "text; never count a range of verses out into separate numbers."
    ),
    REJECT_PARAGRAPHS: (
        "Your previous answer was several paragraphs. Return one caption, on one line."
    ),
    REJECT_REASONING: (
        "Your previous answer explained or introduced the translation. Return only the "
        "translated text itself, with no preamble and no label."
    ),
    REJECT_REFUSAL: (
        "Your previous answer declined the request. This is a live church caption and "
        "the text is ordinary speech; translate it as given."
    ),
}


def retry_system_prompt(system_prompt: str, reason: Optional[str]) -> Optional[str]:
    """The prompt for one corrective second attempt, or None if a retry won't help.

    Returning None is the important half: a retry is only worth a caption's latency
    where the model plausibly had the answer and broke a stated rule. The caller must
    treat None as "fall back to the NMT model now", and must never retry a call that
    failed or timed out — that budget is already spent, and a second attempt would
    push a slow caption past the point where it is worth showing at all.
    """
    note = _RETRY_NOTES.get(reason or "")
    if not note:
        return None
    return "%s\n\n%s" % (system_prompt.strip(), note)


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


_NUMBER = re.compile(r"\d+")

# Spelled-out forms, so "3 глава" -> "chapter three" is not read as a dropped number.
#
# These were a hand-written table stopping at twelve, and a replay found the cliff: a
# correct caption reading "the first thirteen verses", for a source that wrote the
# figure 13, was rejected as a lost number because thirteen was one past the end of
# the table. Chapter and verse numbers do not stop at twelve, so the forms are
# generated instead, through the range where a translation still writes words rather
# than digits.
#
# Latin-script targets only. A language whose numerals are not covered here falls back
# to comparing digits, which is the safe direction: it can reject a good caption but
# never accepts a lost figure.
_EN_UNITS = ("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
             "seventeen", "eighteen", "nineteen")
_EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_EN_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}
_ES_UNITS = ("", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
             "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis",
             "diecisiete", "dieciocho", "diecinueve")
_ES_TENS = ("", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta",
            "ochenta", "noventa")
_ES_ORDINALS = {"uno": ("primero", "primera"), "dos": ("segundo", "segunda"),
                "tres": ("tercero", "tercera")}


def _en_ordinal(word: str) -> str:
    """The ordinal of an English number word ("four" -> "fourth")."""
    if word in _EN_ORDINALS:
        return _EN_ORDINALS[word]
    if word.endswith("y"):          # twenty -> twentieth
        return word[:-1] + "ieth"
    return word + "th"


def _spelled_forms(number: str, lang: str) -> Tuple[str, ...]:
    """Every way a translation might write ``number`` as words, or ()."""
    try:
        value = int(number)
    except ValueError:
        return ()
    if not 1 <= value <= 99:
        # Past a hundred a translation writes the digits, and the compound forms
        # ("one hundred and nineteen") stop being worth enumerating.
        return ()

    if lang == "en":
        units, tens = _EN_UNITS, _EN_TENS
    elif lang == "es":
        units, tens = _ES_UNITS, _ES_TENS
    else:
        return ()

    ten, unit = divmod(value, 10)
    tens_word: str
    unit_word: str
    joiners: Tuple[str, ...]
    if value < 20:
        tens_word, unit_word, joiners = "", units[value], ("",)
    elif unit == 0:
        tens_word, unit_word, joiners = tens[ten], "", ("",)
    elif lang == "en":
        # Both spellings are seen; "twenty three" is not wrong, just unhyphenated.
        tens_word, unit_word, joiners = tens[ten], units[unit], ("-", " ")
    elif ten == 2:
        # Spanish writes 21-29 as one word, and 31+ as "treinta y uno".
        tens_word, unit_word, joiners = "veinti", units[unit], ("",)
    else:
        tens_word, unit_word, joiners = tens[ten], units[unit], (" y ",)

    forms = []
    for joiner in joiners:
        cardinal = tens_word + joiner + unit_word
        forms.append(cardinal)
        if lang != "en":
            continue
        # The ordinal inflects only the last word: twenty-three -> twenty-third.
        if unit_word:
            forms.append(tens_word + joiner + _en_ordinal(unit_word))
        else:
            forms.append(_en_ordinal(cardinal))
    if lang == "es":
        forms.extend(_ES_ORDINALS.get(forms[0], ()))
    return tuple(dict.fromkeys(f for f in forms if f))


def numbers_survived(source: str, text: str, target_lang: str = "",
                     *, max_invented: int = 3) -> bool:
    """Do the source's figures still appear, without a crowd of invented ones?

    Numbers are the tell for the two ways this model fails on scripture, and they fail in
    opposite directions. Asked to translate a reference — an epistle, a chapter, a verse —
    it answers with a remembered passage instead, and the chapter and verse simply vanish.
    Asked to translate a range, "verses 2 through 9", it counts the range out as the
    numbers 1 to 13, one per line, which are figures the source never contained.

    Measured over two full services, 678 translated captions: this rejects 7, each one a
    genuine fault, and leaves the rest alone.

    Both are fluent, correctly-scripted English of plausible length, so every other check
    here waves them through. Comparing figures is the cheapest thing that does not.

    ``max_invented`` is a crowd, not a stray: a translation may legitimately introduce one
    number the source wrote in words, and rejecting on that would be noisier than the fault.
    """
    src = _NUMBER.findall(source or "")
    out = _NUMBER.findall(text or "")
    if not src and not out:
        return True

    lang = (target_lang or "").lower()
    low = (text or "").lower()
    for number in set(src):
        if number in out:
            continue
        if any(re.search(r"(?<!\w)%s(?!\w)" % re.escape(word), low)
               for word in _spelled_forms(number, lang)):
            continue
        return False

    return len(set(out) - set(src)) <= max_invented


def _looks_like_a_list(text: str, *, min_lines: int = 3) -> bool:
    """Several short lines is a document, not a caption.

    The blank-line check below catches a recited passage, which arrives as paragraphs. A
    counted-out verse range arrives as bare single-newline lines and slips straight past it.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return len(lines) >= min_lines and all(_word_count(line) <= 2 for line in lines)


def validate_translation(raw: Optional[str], source: str, target_lang: str, *,
                         max_expansion: float = 3.0, min_word_budget: int = 8,
                         max_slack_words: int = 24, min_coverage: float = 0.45,
                         min_coverage_source_words: int = 8) -> Optional[str]:
    """The cleaned caption, or None if the output is not a usable translation.

    Thin wrapper over :func:`check_translation`, which carries the rules and also
    names the rule that fired. Callers on the caption path only need the text.

    None means "fall back to the NMT model" — never "show this". Rejects, in order:
    empty output; a refusal; a reasoning or narration opener; wrong-script output
    (the source language leaking through); multi-paragraph output; output far
    longer than the source, which is how commentary and recitation present; and
    output far *shorter* than the source, which is how dropped clauses present.

    ``max_expansion`` is deliberately loose, because Russian to English legitimately
    expands. But a pure ratio is not enough on its own: a short source has a tiny
    budget, so an earlier version exempted sources under three words — and that hole
    let through the worst failure seen over a full service. Asked to translate
    "1 Фессалоникийцам 5 глава." (two words after digits are discounted), the model
    replied "1 Corinthians 11:1-24" followed by the recited text of the passage.
    ``min_word_budget`` closes it: short sources get a small absolute allowance
    instead of an unlimited one, so "Щит веры." -> "The shield of faith."
    still passes while a recitation does not.

    ``max_slack_words`` closes the opposite end. A pure ratio grants slack in
    proportion to length, so a long source hands back a large absolute allowance —
    and with a context window enabled the source *is* long, because the prefix is
    prepended before translation. A 30-word combined source bought a 90-word budget,
    leaving room for a recitation to ride along behind a correct caption.

    Measured over 36,264 aligned source/translation pairs from real services, the
    expansion ratio converges as the source grows: p99 is 5.00 at 1-3 words but 1.52
    at 26+, where the largest expansion seen at all was 1.73x and the largest absolute
    growth 22 words. So the ratio governs short sources, where it must, and a flat
    slack ceiling governs long ones, where it is the ratio that stops meaning anything.
    At 24 words the two rules together reject 0.044% of those real translations
    against the ratio alone's 0.039% — two captions in 36,264, each of which falls
    back to the NMT model rather than being lost.

    ``min_coverage`` is the floor under all of that, and it catches the opposite
    failure: not a model that says too much, but one that quietly says less. See
    :func:`check_translation` for the measurement behind the threshold.
    """
    text, _reason = check_translation(
        raw, source, target_lang, max_expansion=max_expansion,
        min_word_budget=min_word_budget, max_slack_words=max_slack_words,
        min_coverage=min_coverage, min_coverage_source_words=min_coverage_source_words)
    return text


def check_translation(raw: Optional[str], source: str, target_lang: str, *,
                      max_expansion: float = 3.0, min_word_budget: int = 8,
                      max_slack_words: int = 24, min_coverage: float = 0.45,
                      min_coverage_source_words: int = 8) -> Tuple[Optional[str], Optional[str]]:
    """``(caption, None)`` if the output is usable, else ``(None, rule_name)``.

    The rules and their order are documented on :func:`validate_translation`, which is
    the caption path's entry point. This variant additionally names the rule that
    fired, which is what the offline replay harness reports: "this prompt traded four
    number-losses for two refusals" is a decision an operator can act on, where "six
    rejections" is not. Keeping both behind one implementation is the point — a
    harness that scored captions by its own copy of these rules would drift from the
    thing it is supposed to be measuring.

    ``min_coverage`` bounds how much *shorter* than its source a caption may be. Every
    other rule here bounds a model that says too much; this one catches the model that
    quietly says less, which is the failure the length rules were blind to. Replaying
    the 2026-08-02 services found six captions that passed every other check while
    dropping whole clauses: a sentence naming the second thing a speaker saw in a
    passage came back as "Thank you." — fluent, correctly scripted, number-clean,
    and wrong.

    The threshold is measured, not guessed: over the 36,307 aligned source/translation
    pairs in the session archive, translations below 0.45x their source's word count
    are 0.04-0.11% of every source-length band. So the rule almost never fires on a
    real translation, while catching six per service. It is deliberately blind to
    sources under ``min_coverage_source_words``, where a ratio on a handful of words
    says nothing — a source counting "one, two, maybe three" compresses to about half
    its length in English and is a perfectly good caption at 0.50.

    This rule in particular is why a rejection must not simply mean "use NMT": the
    borderline cases it catches are captions the LLM got mostly right, and the NMT
    model is not a better answer for them. It is meant to be paired with a retry.
    """
    if raw is None:
        return None, REJECT_EMPTY
    text = _strip_wrappers(raw)
    if not text:
        return None, REJECT_EMPTY

    low = text.lower()
    if any(marker in low for marker in _REFUSAL_MARKERS):
        return None, REJECT_REFUSAL
    if low.startswith(_REASONING_OPENERS):
        return None, REJECT_REASONING

    wrong_script = _WRONG_SCRIPT_FOR_TARGET.get((target_lang or "").lower())
    if wrong_script is not None and wrong_script.search(text):
        # The model returned (some of) the source language instead of translating.
        return None, REJECT_WRONG_SCRIPT

    # A caption is one utterance. Blank-line-separated output means the model produced
    # a document — the scripture-recitation failure mode.
    if "\n\n" in text.strip():
        return None, REJECT_PARAGRAPHS
    if _looks_like_a_list(text):
        return None, REJECT_LIST

    # The figures the speaker gave are the one part of a reference that can be checked
    # mechanically, and both scripture failure modes move them.
    if not numbers_survived(source, text, target_lang):
        return None, REJECT_NUMBERS

    # Absolute floor as well as a ratio, so a short source still has a bounded budget —
    # and an absolute ceiling, so a long one does not buy unlimited room to hide in.
    src_words = _word_count(source)
    out_words = _word_count(text)
    allowed = max(min_word_budget, int(max_expansion * src_words))
    allowed = min(allowed, src_words + max_slack_words)
    if out_words > allowed:
        return None, REJECT_TOO_LONG

    # And the floor: a caption far shorter than its source has dropped clauses.
    if src_words >= min_coverage_source_words and out_words < min_coverage * src_words:
        return None, REJECT_TOO_SHORT

    return text, None


def word_count(text: str) -> int:
    """Words, ignoring digits and punctuation — the unit the length rules count in.

    Public because the replay harness reports a source/output length distribution and
    must measure it the same way the rules do, or its thresholds would not transfer.
    """
    return _word_count(text)


# --- Input budgeting -------------------------------------------------------------
#
# The output side above rejects a bad caption. This side keeps one from being asked
# for at all: llama.cpp raises once the prompt exceeds n_ctx, and that surfaces as a
# generic exception the caller can only turn into a fallback. Measuring the input
# first lets context be shed instead, which keeps the LLM on captions that would
# otherwise be handed to the NMT model.
#
# Characters per token, by script. A BPE vocabulary trained mostly on English splits
# Cyrillic far harder than Latin — often to two or three tokens per word — so the two
# are counted separately. Both figures are deliberately pessimistic: over-estimating
# costs a little context, under-estimating costs the exception this exists to avoid.
_CHARS_PER_TOKEN_CYRILLIC = 2.0
_CHARS_PER_TOKEN_OTHER = 3.7

# Chat-template scaffolding — role markers, BOS/EOS, the turn structure that
# create_chat_completion adds around our two messages. We never see it, so it is
# reserved rather than measured.
_CHAT_TEMPLATE_MARGIN = 64

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Approximate token count for ``text``, erring high.

    Used when no real tokenizer is available — the HTTP endpoint provider, where the
    model runs on another machine and its vocabulary is unknown. The local GGUF path
    passes the model's own tokenizer as a ``counter`` instead and does not use this.
    """
    if not text:
        return 0
    cyrillic = len(_CYRILLIC.findall(text))
    other = len(text) - cyrillic
    est = cyrillic / _CHARS_PER_TOKEN_CYRILLIC + other / _CHARS_PER_TOKEN_OTHER
    # Never report zero for non-empty input: a caller comparing against a budget of 0
    # would otherwise conclude that anything fits.
    return max(1, int(est) + 1)


def _count(text: str, counter: Optional[TokenCounter]) -> int:
    """Token count via ``counter``, falling back to the heuristic if it fails.

    A tokenizer call must never break a caption, so a raising counter degrades to the
    estimate rather than propagating.
    """
    if counter is not None:
        try:
            return int(counter(text))
        except Exception:
            pass
    return estimate_tokens(text)


def input_token_budget(n_ctx: int, max_tokens: int, system_prompt: str, *,
                       counter: Optional[TokenCounter] = None,
                       margin: int = _CHAT_TEMPLATE_MARGIN) -> int:
    """How many tokens of user text fit, given the context window and reply reservation.

    ``max_tokens`` is subtracted because llama.cpp needs room for the answer inside the
    same window; a prompt that fits exactly would leave nothing to generate into.
    """
    budget = int(n_ctx) - int(max_tokens) - _count(system_prompt or "", counter) - int(margin)
    return max(0, budget)


def input_fits(text: str, budget: int, *, counter: Optional[TokenCounter] = None) -> bool:
    """Whether ``text`` fits the user-text budget from :func:`input_token_budget`."""
    if budget <= 0:
        return False
    return _count(text or "", counter) <= budget


def fit_context_prefix(context_texts: Sequence[str], target_text: str, budget: int, *,
                       counter: Optional[TokenCounter] = None) -> List[str]:
    """The longest suffix of ``context_texts`` that still fits alongside ``target_text``.

    Oldest entries are shed first, so a context window degrades toward 1 rather than the
    whole caption being declined. Returns ``[]`` when not even one context entry fits —
    the target alone may still fit, and that is the caller's check to make, because the
    target is the caption and can never be dropped.
    """
    kept = [t for t in context_texts if t]
    while kept:
        combined = " ".join(kept) + " " + target_text
        if input_fits(combined, budget, counter=counter):
            return kept
        kept = kept[1:]
    return []


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
