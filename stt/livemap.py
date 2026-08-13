"""The anonymous live-map ping: what is sent, and when there is anything to send.

STT reports two things to the project's live map — that an install is running, and that
one is actually captioning a service. They are separate signals: a machine deployed in a
building and left on all week is invisible to a ping that only fires when someone
presses Start, and "installs online" and "services being captioned" are different
numbers that were previously collapsed into one.

The two are distinguished by ``event``. Every parameter the collector has always received
keeps its name, value and relative position; the fields added since are appended after
``commit``, with ``event`` last.

Alongside those the ping describes the install itself — OS release, CPU architecture, GPU
model, and the configured speech and translation models — because "macos, 26.2.165" is
not enough to answer a support question, and on an installer build there is no commit hash
to fall back on. What is never sent: any endpoint, any API key, and any filesystem path.
A custom model reports its architecture and a GGUF its bare filename, since a model path
is on the operator's disk and routinely contains their name.

Nothing here performs IO. The caller supplies the config, the version strings and the
platform, and makes the HTTP call itself on a daemon thread — a ping must never delay a
service start or a server boot, and a collector that is down or slow must cost nothing
but a printed line.
"""

from __future__ import annotations

import urllib.parse
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple

# Why the install is pinging. Sent as the last query parameter, so a collector reading
# the older single-ping shape sees exactly what it always did plus one unknown field.
EVENT_APP_START = "app_start"
EVENT_TRANSCRIPTION_START = "transcription_start"

# Platform names the collector groups by. Anything unrecognised is reported as linux
# rather than dropped: the map counts installs, and an unknown platform is still an
# install — this mirrors the mapping the ping has always used.
_OS_NAMES = {"darwin": "macos", "win32": "windows", "linux": "linux"}

_UNKNOWN_VERSION = "unknown"

# Ceilings for the free-text descriptive fields. A GPU name or a distro string is short
# in every sane case; the cap is there so a machine reporting something absurd sends a
# truncated label rather than a kilobyte of URL.
_LABEL_MAX = 40
_MODEL_LABEL_MAX = 60


def os_name_for_platform(sys_platform: str) -> str:
    """The collector's name for ``sys.platform``."""
    return _OS_NAMES.get((sys_platform or "").strip().lower(), "linux")


def _clean(value: Any, limit: int) -> str:
    """One line of trimmed, whitespace-collapsed, length-capped text ("" when empty).

    Every descriptive field goes through this. Collapsing matters because these come
    from ``/etc/os-release`` and vendor tools, which pad and occasionally embed
    newlines, and a blank result is what makes ``build_ping_url`` omit the field
    instead of sending an empty one.
    """
    return " ".join(str(value or "").split())[:limit]


def os_version_label(sys_platform: str, *, mac_ver: str = "", win_ver: str = "",
                     distro: str = "", kernel: str = "") -> str:
    """The specific OS release, as distinct from the ``os`` family the map groups by.

    "macos" cannot tell a Sonoma box from a Sequoia one, and "linux" says nothing at
    all; support questions are almost always about the specific release. The caller
    supplies each platform's raw string because reading them is IO — this only chooses.

    An unrecognised platform falls back to distro-then-kernel, matching
    ``os_name_for_platform`` treating anything unknown as linux.
    """
    platform_name = (sys_platform or "").strip().lower()
    if platform_name == "darwin":
        return _clean(mac_ver, _LABEL_MAX)
    if platform_name == "win32":
        return _clean(win_ver, _LABEL_MAX)
    return _clean(distro, _LABEL_MAX) or _clean(kernel, _LABEL_MAX)


def arch_label(machine: str) -> str:
    """CPU architecture, verbatim from the platform.

    Deliberately not normalised: macOS says "arm64" where Linux says "aarch64" for the
    same silicon (see the note in stt/tunnel.py), and which name an install reports is
    itself the signal — collapsing them would throw that away.
    """
    return _clean(machine, _LABEL_MAX).lower()


def gpu_label(name: str) -> str:
    """The accelerator's model name, or "" on a CPU-only box.

    Blank is a real answer here rather than a missing one: it says the install runs
    Whisper on the CPU, which is the single most useful thing to know about a machine
    reporting that transcription is slow.
    """
    return _clean(name, _LABEL_MAX)


def transcription_model_label(config: Mapping[str, Any]) -> str:
    """Which speech model this install is configured to run, "" when none is chosen.

    Blank is the shipped state since a fresh install carries no model at all, so an
    absent field distinguishes "never set one up" from "running large-v3".

    A custom model reports its architecture and never its path: the path is on the
    operator's disk and routinely contains their name.
    """
    model = config.get("model") or {}
    kind = str(model.get("type") or "").strip().lower()
    if kind == "huggingface":
        model_id = _clean((model.get("huggingface") or {}).get("model_id"), _MODEL_LABEL_MAX)
        return "hf:" + model_id if model_id else ""
    if kind == "custom":
        custom = _clean((model.get("custom") or {}).get("model_type"), _MODEL_LABEL_MAX)
        return "custom:" + custom if custom else "custom"
    return _clean((model.get("whisper") or {}).get("model"), _MODEL_LABEL_MAX)


def translation_model_label(config: Mapping[str, Any]) -> str:
    """Which translator this install is configured to run, "" when translation is off.

    Prefixed by method because the engines are not comparable: an NLLB checkpoint, a
    MADLAD one and a GGUF chat model produce different translations at different
    speeds, and the map would otherwise show three unrelated things in one column. The
    NMT methods name their checkpoint in ``translation_model``; only "llm" reads the
    llm section, and an unknown method still reports itself rather than nothing.

    A GGUF reports its filename only, and the endpoint and API key are never reported
    at all — an endpoint URL identifies the operator's own infrastructure.
    """
    translation = config.get("live_translation") or {}
    if not translation.get("enabled"):
        return ""
    method = str(translation.get("translation_method") or "nllb").strip().lower() or "nllb"
    if method != "llm":
        model = _clean(translation.get("translation_model"), _MODEL_LABEL_MAX)
        return (method + ":" + model) if model else method
    llm = translation.get("llm") or {}
    name = _clean(llm.get("model"), _MODEL_LABEL_MAX)
    if not name:
        # A locally-run GGUF has no model name, only a file. Basename, never the path.
        name = _clean(str(llm.get("gguf_file") or "").replace("\\", "/").rsplit("/", 1)[-1],
                      _MODEL_LABEL_MAX)
    return ("llm:" + name) if name else "llm"


def numeric_version(display_version: Optional[str],
                    fallback: Optional[str] = None) -> str:
    """The dotted-numeric part of a display version ("26.1.22-gc588d29" -> "26.1.22").

    The collector rejects anything beyond dotted numerics, and the commit hash travels
    in its own parameter. A blank display version falls back to the plain version, and
    a blank fallback to "unknown" — the ping is worth sending with an unknown version,
    since the map is counting installs rather than releases.
    """
    text = (display_version or "").split("-", 1)[0].strip()
    if text:
        return text
    return (fallback or "").strip() or _UNKNOWN_VERSION


def build_ping_url(endpoint: Optional[str], *, event: str, os_name: str, version: str,
                   transcribe_lang: str = "", translate_lang: str = "",
                   commit: str = "", os_version: str = "", arch: str = "", gpu: str = "",
                   stt_model: str = "", mt_model: str = "",
                   offloaded: bool = False) -> Optional[str]:
    """The URL for one ping, or None when pinging is switched off.

    None is the kill switch: a blank ``analytics.endpoint`` means the operator (or a
    self-hosting fork) has opted out, and the caller must then make no request at all
    rather than send one somewhere else.

    Empty optional fields are omitted rather than sent blank, which is what keeps the
    app-start ping to what exists at boot — no session has started, so there are no
    languages to report and claiming "auto"/"none" would be inventing a session that is
    not running. It is also what makes an absent ``stt_model`` mean "this install has
    never chosen one" rather than "the field was not sent".

    Parameter order is the historical one, with the descriptive fields inserted after
    ``commit`` and ``event`` still last: every parameter the collector has always
    received keeps its name, value and relative position. Values are percent-encoded;
    they were previously interpolated raw, and a language code or version containing a
    space or an ampersand produced a malformed URL — which now matters far more, since
    an OS release, a GPU name and a model id all routinely contain spaces and slashes.
    """
    base = (endpoint or "").strip()
    if not base:
        return None
    params = [("os", os_name), ("version", version)]
    if transcribe_lang:
        params.append(("transcribe_lang", transcribe_lang))
    if translate_lang:
        params.append(("translate_lang", translate_lang))
    if commit:
        params.append(("commit", commit))
    for name, value in (("os_version", os_version), ("arch", arch), ("gpu", gpu),
                        ("stt_model", stt_model), ("mt_model", mt_model)):
        if value:
            params.append((name, value))
    if offloaded:
        params.append(("offloaded", "1"))
    params.append(("event", event))
    return base + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def ping_fields_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The session-shaped fields of a transcription-start ping, read from live config.

    The fallbacks are the ones the ping has always used and they are not
    interchangeable: "auto" means the language will be detected, "none" means
    translation is switched off, and "unknown" means it is on but no target has been
    chosen — three states the map would otherwise have to guess apart.

    A config missing whole sections yields the same defaults rather than raising: this
    runs on the way to a fire-and-forget ping, and a half-written config must not be
    able to break a transcription start.
    """
    translation = config.get("live_translation") or {}
    remote = translation.get("remote") or {}
    target = str(translation.get("target_language") or "").strip()
    return {
        "transcribe_lang": str((config.get("audio") or {}).get("language")
                               or "auto").strip() or "auto",
        "translate_lang": (target or "unknown") if translation.get("enabled") else "none",
        "offloaded": bool(remote.get("enabled") and remote.get("endpoint")),
    }


def install_fields_from_config(config: Mapping[str, Any]) -> Dict[str, str]:
    """The configured-model fields, which both events carry.

    Unlike ``ping_fields_from_config`` these describe how the install is set up rather
    than what a session is doing, and they are equally true at boot — so app_start
    reports them too. That is the point: a machine left running all week with no model
    configured is a support case, and it never reaches a transcription start to say so.
    """
    return {
        "stt_model": transcription_model_label(config),
        "mt_model": translation_model_label(config),
    }


def ensure_install_id(analytics: MutableMapping[str, Any],
                      new_id: Callable[[], str]) -> Tuple[str, bool]:
    """``(install_id, changed)`` — the anonymous per-install id, generating one once.

    ``changed`` tells the caller whether the config now needs saving, so a boot that
    already has an id does not rewrite the file. The generator is a parameter so a test
    can assert on a fixed id rather than on a uuid.

    The mapping is mutated in place rather than replaced: it is the live config's own
    ``analytics`` section, and rebuilding it would drop the endpoint and the comment
    keys alongside it.
    """
    existing = str(analytics.get("install_id") or "").strip()
    if existing:
        return existing, False
    generated = new_id()
    analytics["install_id"] = generated
    return generated, True
