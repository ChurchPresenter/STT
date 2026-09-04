"""What "downloaded" has to mean before a model file is handed to a loader.

Three separate failures in the field share one cause: a download that stopped
part-way left a truncated file under its **final** name, and every check we had
said the model was present.

* A truncated ``pytorch_model.bin`` reached ``torch.load`` and threw
  ``PytorchStreamReader failed reading zip archive: failed finding central
  directory`` — from the settings-save thread, so not even the transcription
  worker's crash path was involved.
* A faster-whisper directory missing ``tokenizer.json`` sent ``WhisperModel`` into
  ``tokenizers.Tokenizer.from_pretrained("openai/whisper-tiny")``, which is a Rust
  builtin with its own HTTP client: it honours neither ``HF_HUB_OFFLINE`` nor any
  timeout we can pass, so on a connection that stalls rather than refusing it
  blocks for ever. ``local_files_only=True`` does **not** help — faster-whisper
  only forwards that flag to ``download_model()``, which is skipped entirely when
  the path is a local directory. Refusing to call the loader is the only fix, so
  ``tokenizer.json`` is a *required* file here rather than a nice-to-have.
* A dropped piper download left a truncated ``.onnx``.

So this module answers two questions, and deliberately keeps them apart:

**"Are these the bytes we were promised?"** — the Hub reports a size for every
file and a sha256 for LFS ones, which the app already fetched and used only to
draw a progress bar. :func:`write_manifest` records them next to the weights so
the answer survives the download that learned it.

**"Is this directory loadable?"** — :func:`faster_whisper_status`, a
family-specific required-file list. ``stt.model_disk.dir_has_weights`` stays as
it is: it answers a broader question ("is any weight file here at all") for the
NLLB/GGUF/Whisper listings, and tightening it would change all of them.

Stdlib only, paths passed in, no logging of its own — the callers decide what to
print and what to raise.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Union

#: Sidecar written into a model directory once every file has been verified.
#: Dot-prefixed so the model-manager listings, which already skip dotfiles, do
#: not mistake it for a model of its own.
MANIFEST_NAME = ".stt-download.json"

#: Files ``faster_whisper.WhisperModel`` cannot start without. A tuple entry means
#: "any one of these", which the vocabulary needs: Systran ships
#: ``vocabulary.txt`` on the small models and ``vocabulary.json`` on the large
#: ones, so requiring a fixed name would call half the catalogue broken.
#:
#: Measured rather than guessed, against faster-whisper 1.2.1 and a real
#: ``Systran/faster-whisper-tiny`` directory:
#:
#: * no vocabulary file -> ``ctranslate2.models.Whisper`` raises "Cannot load the
#:   vocabulary from the model directory". A hard failure, but the model still
#:   listed as downloaded, so it was only ever discovered by pressing Start.
#: * no ``tokenizer.json`` -> the model loads *fine*, because faster-whisper
#:   quietly fetches ``openai/whisper-tiny`` over HTTP. That is the failure this
#:   list exists for: on a connection that stalls rather than refusing, it never
#:   returns and the UI reports STARTING for ever.
#:
#: ``preprocessor_config.json`` is deliberately absent — it ships only with the
#: large models, and faster-whisper falls back to built-in feature-extractor
#: defaults without it.
REQUIRED_FASTER_WHISPER = (
    "model.bin",
    "config.json",
    "tokenizer.json",
    ("vocabulary.txt", "vocabulary.json"),
)

#: Read size for streaming hashes. Matches the OpenAI-Whisper ``.pt`` download
#: path in the monolith, which has verified its checksum this way all along.
_HASH_CHUNK = 1024 * 1024


class FileExpectation(NamedTuple):
    """What one file in a repo should look like once it has landed.

    ``sha256`` is ``None`` for the small non-LFS files (config, tokenizer) — the
    Hub publishes a git blob id for those, not a content hash of the file we
    receive. Size alone still catches truncation, which is the whole failure.
    """

    size: Optional[int]
    sha256: Optional[str] = None


class DirStatus(NamedTuple):
    """Whether a model directory can be handed to its loader.

    Three states rather than a bool because the middle one is the interesting
    one: a directory that exists, occupies disk, and gets loaded anyway must be
    visible to the operator and removable, not hidden.
    """

    state: str  #: "absent" | "incomplete" | "complete"
    missing: List[str]  #: required files that are absent or the wrong size

    @property
    def complete(self) -> bool:
        return self.state == "complete"


def part_path(dest_path: str) -> str:
    """Where a transfer is staged before it earns ``dest_path``.

    A sibling rather than a temp directory so ``os.replace`` stays on one
    filesystem (it is only atomic within one), and so ``wget -c`` / ``curl -C -``
    can resume a leftover across *calls* — today they can only resume within a
    single call's retry loop, because the retries all point at the final name.
    """
    return dest_path + ".part"


def file_sha256(path: str) -> str:
    """Streaming sha256 of ``path``. Raises ``OSError`` if it cannot be read."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: str, expected_size: Optional[int] = None, expected_sha256: Optional[str] = None) -> bool:
    """True when ``path`` exists and matches whatever we know about it.

    With neither expectation supplied this degrades to the old rule — a non-empty
    file passes — because for some sources that genuinely is all we know. An
    empty file never passes: a zero-byte weight file is always wreckage.

    The hash is checked only when one is given, and callers should give one only
    at download time. Re-hashing a 3 GB ``model.bin`` on every status poll would
    cost more than the corruption it is looking for.
    """
    try:
        actual = os.path.getsize(path)
    except OSError:
        return False
    if actual == 0:
        return False
    if expected_size is not None and actual != expected_size:
        return False
    if expected_sha256:
        try:
            return file_sha256(path) == expected_sha256.lower()
        except OSError:
            return False
    return True


def files_needing_download(model_dir: str, expected: Mapping[str, FileExpectation]) -> List[str]:
    """The subset of ``expected`` that is missing or the wrong size on disk.

    This is the repairing replacement for the ``os.path.getsize(dest) > 0`` skip
    the download loops used: a truncated file was "already downloaded", so a
    re-download could never fix one and the corruption was permanent until
    somebody deleted the folder by hand.

    Hashes are deliberately not checked here — this runs before a download, over
    files that may total gigabytes, and size already separates "truncated" from
    "fine". The hash is verified once, on the bytes as they arrive.
    """
    return [
        name
        for name, want in expected.items()
        if not verify_file(os.path.join(model_dir, name), want.size)
    ]


def manifest_path(model_dir: str) -> str:
    """Location of the sidecar for ``model_dir``."""
    return os.path.join(model_dir, MANIFEST_NAME)


def write_manifest(model_dir: str, repo_id: str, expected: Mapping[str, FileExpectation]) -> None:
    """Record what a completed download was supposed to contain.

    Written last, after every file has verified, so the presence of a manifest is
    itself the claim "this directory finished downloading". Best-effort: a model
    that downloaded fine must not be reported as failed because its sidecar could
    not be written, so ``OSError`` is swallowed. The cost of losing it is falling
    back to the size-only checks, which is where we were before.
    """
    payload = {
        "repo": repo_id,
        "files": {name: {"size": want.size, "sha256": want.sha256} for name, want in expected.items()},
    }
    try:
        with open(manifest_path(model_dir), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except OSError:
        pass


def read_manifest(model_dir: str) -> Dict[str, FileExpectation]:
    """Expectations recorded for ``model_dir``, or ``{}`` when there are none.

    Empty is the honest answer for every pre-existing install: models downloaded
    before this sidecar existed have no manifest and must keep working, so no
    caller may treat "no manifest" as "not downloaded". A corrupt or truncated
    manifest reads as empty for the same reason.
    """
    try:
        with open(manifest_path(model_dir), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    files = data.get("files")
    if not isinstance(files, dict):
        return {}
    out: Dict[str, FileExpectation] = {}
    for name, entry in files.items():
        if not isinstance(entry, dict):
            continue
        size = entry.get("size")
        sha = entry.get("sha256")
        out[str(name)] = FileExpectation(
            size=size if isinstance(size, int) else None,
            sha256=sha if isinstance(sha, str) else None,
        )
    return out


def manifest_mismatches(model_dir: str) -> List[str]:
    """Files in ``model_dir`` that no longer match its manifest.

    Size-only, for the same reason as :func:`files_needing_download`. Returns
    nothing for a directory with no manifest — see :func:`read_manifest`.
    """
    return files_needing_download(model_dir, read_manifest(model_dir))


def missing_required(model_dir: str, required: Sequence[Union[str, Sequence[str]]]) -> List[str]:
    """Which of ``required`` are absent, empty, or contradict the manifest.

    An entry that is itself a sequence is a set of alternatives, satisfied by any
    one of them and reported as "a or b" when none is present.

    The manifest, when there is one, is what turns "the file is there" into "the
    file is whole" — the truncated-download case, where every filename check
    passes and the loader still dies.
    """
    expected = read_manifest(model_dir)
    missing = []
    for entry in required:
        names = [entry] if isinstance(entry, str) else list(entry)
        if not any(
            verify_file(os.path.join(model_dir, name),
                        expected.get(name, FileExpectation(None, None)).size)
            for name in names
        ):
            missing.append(" or ".join(names))
    return missing


def faster_whisper_status(model_dir: str) -> DirStatus:
    """Whether ``model_dir`` can be handed to ``faster_whisper.WhisperModel``.

    ``absent`` means there is nothing there — a model that was never downloaded,
    which the UI already shows as "Available". ``incomplete`` means a directory
    exists but a required file does not, and is the state the app had no word
    for: it listed as downloaded (or, worse, was hidden entirely) while the
    loader still picked it up and hung or crashed on it.
    """
    if not os.path.isdir(model_dir):
        return DirStatus("absent", [])
    missing = missing_required(model_dir, REQUIRED_FASTER_WHISPER)
    if missing:
        return DirStatus("incomplete", missing)
    return DirStatus("complete", [])


def describe_missing(missing: Iterable[str]) -> str:
    """Human phrasing for a missing-file list, for error text and UI badges."""
    names = list(missing)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]
