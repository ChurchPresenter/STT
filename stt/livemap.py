"""The anonymous live-map ping: what is sent, and when there is anything to send.

STT reports two things to the project's live map — that an install is running, and that
one is actually captioning a service. They are separate signals: a machine deployed in a
building and left on all week is invisible to a ping that only fires when someone
presses Start, and "installs online" and "services being captioned" are different
numbers that were previously collapsed into one.

The two are distinguished by ``event``. Everything else about the request is unchanged
from the single ping that preceded them, deliberately: the transcription-start URL is
byte-identical to the one the collector has always received, with ``&event=`` appended.

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


def os_name_for_platform(sys_platform: str) -> str:
    """The collector's name for ``sys.platform``."""
    return _OS_NAMES.get((sys_platform or "").strip().lower(), "linux")


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
                   commit: str = "", offloaded: bool = False) -> Optional[str]:
    """The URL for one ping, or None when pinging is switched off.

    None is the kill switch: a blank ``analytics.endpoint`` means the operator (or a
    self-hosting fork) has opted out, and the caller must then make no request at all
    rather than send one somewhere else.

    Empty optional fields are omitted rather than sent blank, which is what keeps the
    app-start ping to the four things that exist at boot — no session has started, so
    there are no languages to report and claiming "auto"/"none" would be inventing a
    session that is not running.

    Parameter order matches the ping that preceded this module, with ``event`` last, so
    a transcription-start URL is the historical string plus one field. Values are
    percent-encoded; they were previously interpolated raw, and a language code or
    version containing a space or an ampersand produced a malformed URL.
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
