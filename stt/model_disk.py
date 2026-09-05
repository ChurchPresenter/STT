"""On-disk model-weight detection shared by the model-manager endpoints.

A model directory can survive a partial deletion — the large weight files are
removed to reclaim space while config/tokenizer files stay behind — and an
interrupted download leaves the same shape. Directory existence therefore does
not prove a usable model is present. Every type-specific status endpoint
(nllb-status, nllb-list, faster-whisper/list) already keys "downloaded" off an
actual weight file; this module is the single predicate they share so the
"Downloaded Models" listing stays consistent with them instead of reporting a
leftover directory as still downloaded.

Extracted from speech_to_text.py so it can be imported and unit-tested without
the monolith's import-time side effects. Stdlib-only; paths are passed in.
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, List, NamedTuple

# Single-file weight layouts across the model families this app downloads:
#   HF Transformers -> model.safetensors / pytorch_model.bin
#   CTranslate2     -> model.bin              (faster-whisper)
#   llama.cpp       -> *.gguf                 (LLM translation)
_WEIGHT_FILENAMES = frozenset({"model.safetensors", "pytorch_model.bin", "model.bin"})


def is_weight_file(name: str) -> bool:
    """True when a single filename is a recognized model weight file.

    Covers the single-file names above, the sharded HuggingFace layouts
    (``pytorch_model-00001-of-000NN.bin`` and ``model-00001-of-000NN.safetensors``),
    OpenAI Whisper checkpoints (``*.pt``), and llama.cpp ``*.gguf`` files.
    Deliberately narrow: a bare ``*.bin`` is not enough (HF ships non-weight blobs
    like ``training_args.bin``), so a config/tokenizer-only leftover is never
    mistaken for a downloaded model.

    A GGUF directory holds nothing else — no config.json, no tokenizer — so
    without the ``.gguf`` case a downloaded LLM was invisible to every caller
    here: absent from the downloaded-models list, and so impossible to delete
    from the Model Manager despite being the largest file on disk.
    """
    if name in _WEIGHT_FILENAMES:
        return True
    if name.startswith("pytorch_model-") and name.endswith(".bin"):
        return True
    if name.startswith("model-") and name.endswith(".safetensors"):
        return True
    # .onnx is Piper and Supertonic; .pth is the PANNs checkpoint. Both were
    # missing, so those directories classified as "not a model at all" and were
    # invisible to every caller here — including the diagnostic report, which
    # silently skipped the two families an operator is most likely to have
    # half-downloaded over a bad link.
    return name.endswith((".pt", ".pth", ".gguf", ".onnx"))


def has_weight_file(filenames: Iterable[str]) -> bool:
    """True when any name in ``filenames`` is a recognized weight file."""
    return any(is_weight_file(n) for n in filenames)


def dir_has_weights(path: str) -> bool:
    """True when directory ``path`` exists and holds a recognized weight file.

    Non-recursive: HF / CTranslate2 / Whisper model directories keep their
    weights at the top level. A missing path or an unreadable directory is
    treated as "no usable model present" (returns False).
    """
    try:
        return has_weight_file(os.listdir(path))
    except OSError:
        return False


#: Marker separating a model directory from its CTranslate2 conversion, as
#: produced by stt.ct2_translate.ct2_model_dir(): "<hf_dir>-ct2-<compute_type>".
_CT2_MARKER = "-ct2-"


def ct2_variant_names(entries: Iterable[str], dir_name: str) -> List[str]:
    """Compute types of the CTranslate2 conversions of ``dir_name`` in ``entries``.

    e.g. ``["google--madlad400-3b-mt-ct2-int8_float16"]`` for
    ``"google--madlad400-3b-mt"`` -> ``["int8_float16"]``.

    Converting a model leaves the weights in this sibling directory, and the
    original HF weights are often deleted afterwards to reclaim the space — so
    the conversion, not the HF directory, is what proves the model is usable.

    Deliberately exact about the stem: a bare ``<dir_name>-ct2-`` with no compute
    type is not a conversion, and ``madlad400-10b-mt`` must never be reported as
    a conversion of ``madlad400-3b-mt``.
    """
    prefix = f"{dir_name}{_CT2_MARKER}"
    variants = []
    for entry in entries:
        if entry.startswith(prefix):
            compute_type = entry[len(prefix):]
            if compute_type:
                variants.append(compute_type)
    return sorted(variants)


class ModelPresence(NamedTuple):
    """What of a model is actually on disk.

    Kept structured rather than a bool because the difference matters to the
    caller: a box holding only a conversion runs fine through the CTranslate2
    path but has no weights for anything else, and an operator seeing an
    engine fail there deserves to know why.
    """

    has_weights: bool
    ct2_variants: List[str]

    @property
    def downloaded(self) -> bool:
        """Whether a usable copy of the model is present, in any form."""
        return self.has_weights or bool(self.ct2_variants)


def model_presence(models_dir: str, dir_name: str) -> ModelPresence:
    """Inspect ``models_dir`` for ``dir_name``'s weights and CT2 conversions.

    A missing or unreadable models directory reports nothing present rather
    than raising — the caller is a status endpoint, not a downloader.
    """
    try:
        entries = os.listdir(models_dir)
    except OSError:
        return ModelPresence(False, [])
    return ModelPresence(
        has_weights=dir_has_weights(os.path.join(models_dir, dir_name)),
        ct2_variants=ct2_variant_names(entries, dir_name),
    )


def dir_is_writable(path: str) -> bool:
    """True when the current process can create a file inside ``path``.

    Probes by actually creating and deleting a temp file rather than calling
    ``os.access(path, os.W_OK)``, which is unreliable under root and POSIX ACLs
    (it consults the permission bits, not whether a write would really succeed).
    A missing directory, or one owned by another user, returns False.
    """
    try:
        with tempfile.NamedTemporaryFile(dir=path):
            return True
    except OSError:
        return False
