"""Strip a recording machine's identity out of captured API responses.

The demo's canned responses are recorded from a real server, because guessing the
shape of 20-odd endpoints by hand drifts silently and gets details wrong. But a real
server's responses carry things that must not ship in a public download: the operator's
home directory, the machine's hostname, the paired machine's address, SMB credentials,
the install id that identifies it on the live map.

So every captured payload passes through here first. The rule is deny-by-default on
key *names* rather than on values: a value that looks harmless today can become
identifying tomorrow, but a key called ``smb_password`` is never safe.

Stdlib-only, pure, and side-effect free.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Substring match against a lowercased key name. Anything hit is replaced with a
# type-preserving blank ("" for a string, [] for a list, and so on) rather than
# removed, so the page still finds the field it expects.
SENSITIVE_KEY_PARTS: Tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "credential",
    "username",
    "user_name",
    "install_id",
    "machine_id",
    "device_uuid",
    "serial",
    "endpoint",
    "hostname",
    "host_name",
    "smb_domain",
    "email",
    "dsn",
)

# Key names that hold a filesystem path. Kept (a blank path breaks pages that render
# it) but rewritten to a neutral location.
PATH_KEY_PARTS: Tuple[str, ...] = (
    "path",
    "dir",
    "directory",
    "folder",
    "location",
    "file",
)

PLACEHOLDER_HOME = "/Users/demo"
PLACEHOLDER_HOST = "demo-machine"

_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# A Windows or POSIX home directory, whoever it belongs to.
_HOME_POSIX = re.compile(r"/(?:Users|home)/[^/\s\"']+")
_HOME_WINDOWS = re.compile(r"[A-Za-z]:\\\\?Users\\\\?[^\\\s\"']+")
# UNC share, which is where an SMB destination names a real server.
_UNC = re.compile(r"//[^/\s\"']+/[^\s\"']*|\\\\[^\\\s\"']+\\[^\s\"']*")

_LOOPBACK = {"127.0.0.1", "0.0.0.0", "255.255.255.255", "1.1.1.1", "8.8.8.8"}


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _is_pathish(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in PATH_KEY_PARTS)


def _blank_like(value: Any) -> Any:
    """A blank of the same type, so consumers keep finding the shape they expect."""
    if isinstance(value, str):
        return ""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return 0
    if isinstance(value, list):
        return []
    if isinstance(value, dict):
        return {}
    return None


def redact_text(text: str, hostname: Optional[str] = None) -> str:
    """Rewrite identifying strings inside a free-text value."""
    if not text:
        return text
    result = _HOME_WINDOWS.sub(r"C:\\Users\\demo", text)
    result = _HOME_POSIX.sub(PLACEHOLDER_HOME, result)
    result = _UNC.sub("//demo-share/services", result)
    result = _EMAIL.sub("demo@example.com", result)
    result = _IPV4.sub(lambda m: m.group(0) if m.group(0) in _LOOPBACK else "203.0.113.10",
                       result)
    if hostname:
        result = re.sub(re.escape(hostname), PLACEHOLDER_HOST, result, flags=re.IGNORECASE)
    return result


def redact(value: Any, hostname: Optional[str] = None,
           extra_keys: Iterable[str] = ()) -> Any:
    """A captured payload with the recording machine's identity removed.

    Recursive and type-preserving: the result is the same JSON shape, so a page that
    renders it cannot tell it came from a different machine.
    """
    sensitive_extra = tuple(k.lower() for k in extra_keys)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            cleaned: Dict[str, Any] = {}
            for key, item in node.items():
                lowered = str(key).lower()
                # Keys identify too: a map keyed by peer address or hostname carries
                # the addresses in its keys, where value-only redaction never looks.
                safe_key = redact_text(str(key), hostname) if isinstance(key, str) else key
                if safe_key != key:
                    # Two distinct originals can redact to the same text; keep both
                    # rather than silently dropping one.
                    suffix = 2
                    unique = safe_key
                    while unique in cleaned:
                        unique = f"{safe_key}-{suffix}"
                        suffix += 1
                    safe_key = unique
                if _is_sensitive(str(key)) or lowered in sensitive_extra:
                    cleaned[safe_key] = _blank_like(item)
                elif isinstance(item, str) and _is_pathish(str(key)):
                    cleaned[safe_key] = redact_text(item, hostname)
                else:
                    cleaned[safe_key] = _walk(item)
            return cleaned
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if isinstance(node, str):
            return redact_text(node, hostname)
        return node

    return _walk(value)


def residual_flags(value: Any, hostname: Optional[str] = None) -> List[str]:
    """Anything still identifying after a redaction pass, as human-readable notes.

    Read before a recording is committed: the deny-list catches what it knows about,
    and this is what says whether it was enough.
    """
    flags: List[str] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                child = f"{path}.{key}" if path else str(key)
                if isinstance(key, str):
                    _walk_leaf(key, child + " (key)")
                _walk(item, child)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                _walk(item, f"{path}[{index}]")
        elif isinstance(node, str):
            _walk_leaf(node, path)

    def _walk_leaf(text: str, path: str) -> None:
        if _HOME_POSIX.search(text) and PLACEHOLDER_HOME not in text:
            flags.append(f"{path}: home directory")
        if _HOME_WINDOWS.search(text):
            flags.append(f"{path}: Windows home directory")
        if _EMAIL.search(text):
            flags.append(f"{path}: email address")
        for match in _IPV4.finditer(text):
            if match.group(0) not in _LOOPBACK and match.group(0) != "203.0.113.10":
                flags.append(f"{path}: IP address {match.group(0)}")
        if hostname and hostname.lower() in text.lower():
            flags.append(f"{path}: hostname")

    _walk(value, "")
    return flags
