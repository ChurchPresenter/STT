"""CTranslate2 int8 translation backend helpers.

CTranslate2 runs NLLB / MADLAD seq2seq models at int8 on CPU (or int8_float16
on CUDA) — far smaller/faster than the transformers backend and, on Apple
Silicon, the only way to run MADLAD-3B in ~16 GB (CT2 has no Metal, so it runs
CPU-int8 there). This module holds the pure/glue-free bits so they can be
unit-tested without ctranslate2 or a real model: the on-disk cache path,
compute-type resolution, and the NLLB/MADLAD token plumbing that differs from
`model.generate`. The Translator lifecycle + calls live in the monolith.

CTranslate2's API works on token *strings*: you encode with the HF tokenizer,
`translate_batch` on the token list, then decode. NLLB forces the target
language via a `target_prefix` (which comes back as the first output token and
must be stripped); MADLAD carries the target as a `<2xx>` prefix in the source
text (no target_prefix). Stdlib-only.
"""

import os
from typing import Any, List, Optional

# Import from the sibling catalog so the MADLAD input format stays in one place.
from stt.nllb_catalog import build_madlad_input


def resolve_compute_type(configured: str, device: str) -> str:
    """Concrete CTranslate2 compute type. "auto" -> int8_float16 on CUDA,
    int8 on CPU (the low-RAM win, and the only option on Apple Silicon since
    CT2 has no Metal backend)."""
    if configured and configured != "auto":
        return configured
    return "int8_float16" if device == "cuda" else "int8"


def ct2_model_dir(hf_model_dir: str, compute_type: str) -> str:
    """Cache directory for the converted CT2 model, beside the HF model dir.

    e.g. ".../google--madlad400-3b-mt" -> ".../google--madlad400-3b-mt-ct2-int8".
    Keyed by compute type so int8 and int8_float16 don't collide.
    """
    # Strip either separator: Windows accepts "/" as well as os.sep, so a path
    # configured as "C:/models/" would otherwise yield "C:/models/-ct2-int8".
    return f'{hf_model_dir.rstrip("/" + os.sep)}-ct2-{compute_type}'


def nllb_ct2_target_prefix(tgt_code: str) -> List[List[str]]:
    """CTranslate2 target_prefix forcing the NLLB target-language token."""
    return [[tgt_code]]


def strip_target_prefix(tokens: List[str], tgt_code: str) -> List[str]:
    """Drop NLLB's leading forced target-language token from a hypothesis."""
    if tokens and tokens[0] == tgt_code:
        return tokens[1:]
    return tokens


def nllb_source_tokens(tokenizer: Any, text: str, src_code: str) -> List[str]:
    """Encode NLLB source text to CT2 token strings (sets the source language)."""
    tokenizer.src_lang = src_code
    return tokenizer.convert_ids_to_tokens(tokenizer.encode(text))


def madlad_source_tokens(tokenizer: Any, text: str, target_lang: str) -> List[str]:
    """Encode MADLAD source (a '<2xx>' target prefix on the text) to CT2 tokens."""
    return tokenizer.convert_ids_to_tokens(tokenizer.encode(build_madlad_input(text, target_lang)))


def decode_ct2_tokens(tokenizer: Any, tokens: List[str]) -> str:
    """Decode CT2 output token strings back to text via the HF tokenizer."""
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)


def score_to_confidence(score: Optional[float]) -> Optional[float]:
    """Map a CTranslate2 log-prob sequence score to a 0-1 confidence, or None.

    CT2 scores are average token log-probabilities (<= 0); exp() puts them on a
    0-1 scale comparable to the transformers backend's sequence confidence.
    """
    if score is None:
        return None
    try:
        import math
        return math.exp(float(score))
    except (ValueError, OverflowError):
        return None
