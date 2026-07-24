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
import shutil
import tempfile
from typing import Callable, Iterable, List, Tuple

# Single-file weight layouts across the model families this app downloads:
#   HF Transformers -> model.safetensors / pytorch_model.bin
#   CTranslate2     -> model.bin              (faster-whisper)
_WEIGHT_FILENAMES = frozenset({"model.safetensors", "pytorch_model.bin", "model.bin"})


def is_weight_file(name: str) -> bool:
    """True when a single filename is a recognized model weight file.

    Covers the single-file names above, the sharded HuggingFace layouts
    (``pytorch_model-00001-of-000NN.bin`` and ``model-00001-of-000NN.safetensors``),
    and OpenAI Whisper checkpoints (``*.pt``). Deliberately narrow: a bare
    ``*.bin`` is not enough (HF ships non-weight blobs like ``training_args.bin``),
    so a config/tokenizer-only leftover is never mistaken for a downloaded model.
    """
    if name in _WEIGHT_FILENAMES:
        return True
    if name.startswith("pytorch_model-") and name.endswith(".bin"):
        return True
    if name.startswith("model-") and name.endswith(".safetensors"):
        return True
    return name.endswith(".pt")


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


def resolve_writable_models_dir(preferred: str, fallback: str) -> Tuple[str, bool]:
    """Pick a models directory the running process can actually write to.

    Tries ``preferred`` first (creating it if absent). If it exists but is not
    writable — the classic case being a ``models/`` created by root while the
    server now runs as a normal user — falls back to ``fallback`` under the
    user's home, which every OS lets a process write. Returns
    ``(chosen_dir, used_fallback)``; ``used_fallback`` is True when the caller
    should warn that downloads are going somewhere other than ``preferred``.

    If neither is writable, returns ``(preferred, False)`` so the caller still
    has a stable path to surface a clear "cannot write" error against.
    """
    for candidate in (preferred, fallback):
        try:
            os.makedirs(candidate, exist_ok=True)
        except OSError:
            continue
        if dir_is_writable(candidate):
            return candidate, candidate != preferred
    return preferred, False


def stranded_model_dirs(preferred: str, fallback: str) -> List[str]:
    """Names of model directories that a fallback to ``fallback`` would orphan.

    A directory counts as stranded when it lives in ``preferred``, holds a
    weight file, and is not already mirrored (with weights) in ``fallback``.
    These are exactly the previously-downloaded models the app would stop
    seeing once it switches to the fallback location. Sorted for stable output;
    an unreadable ``preferred`` yields an empty list.
    """
    stranded: List[str] = []
    try:
        entries = sorted(os.listdir(preferred))
    except OSError:
        return stranded
    for name in entries:
        src = os.path.join(preferred, name)
        if not os.path.isdir(src) or not dir_has_weights(src):
            continue
        if dir_has_weights(os.path.join(fallback, name)):
            continue  # already present in the fallback — not stranded
        stranded.append(name)
    return stranded


def migrate_model_dirs(
    names: Iterable[str],
    preferred: str,
    fallback: str,
    log: Callable[[str], None] = print,
) -> List[str]:
    """Copy each named model directory from ``preferred`` into ``fallback``.

    Copies rather than moves: the source may be read-only (owned by another
    user), which is the whole reason the fallback was needed. A target that
    already has weights is treated as done and left untouched. Returns the
    names that ended up present in ``fallback`` (copied now or already there);
    directories that fail to copy are logged and omitted.
    """
    migrated: List[str] = []
    for name in names:
        src = os.path.join(preferred, name)
        dst = os.path.join(fallback, name)
        if dir_has_weights(dst):
            migrated.append(name)
            continue
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            migrated.append(name)
        except OSError as e:
            log(f"could not migrate {name}: {e}")
    return migrated
