"""Failures that come from the machine, not from the code.

A Windows box can refuse to load a native library (missing VC++ runtime, a
corrupt wheel) or refuse to run a file outright (Application Control / WDAC).
The operator fixes those on their machine; reporting them as crashes only
fills Sentry with issues nobody can act on in the code.
"""

from __future__ import annotations

from typing import Optional

#: ERROR_MOD_NOT_FOUND — a DLL or one of its dependencies could not be loaded.
WIN_MODULE_NOT_FOUND = 126
#: ERROR_APPHOST_BLOCKED — Application Control refused to run the file.
WIN_APP_CONTROL_BLOCKED = 4551
#: ERROR_BAD_EXE_FORMAT — a DLL for the wrong architecture, or a truncated file.
WIN_BAD_EXE_FORMAT = 193

_ENV_WINERRORS = frozenset({WIN_MODULE_NOT_FOUND, WIN_APP_CONTROL_BLOCKED, WIN_BAD_EXE_FORMAT})


def environment_problem(exc: BaseException) -> Optional[str]:
    """A one-line operator-facing explanation if ``exc`` is a machine problem, else None."""
    winerror = getattr(exc, "winerror", None)
    text = str(exc)
    if winerror == WIN_APP_CONTROL_BLOCKED or "[WinError 4551]" in text or "os error 4551" in text:
        return (
            "Windows Application Control blocked a file STT needs. "
            "Ask whoever manages this PC to allow the STT folder (~/.stt)."
        )
    if winerror in _ENV_WINERRORS or any(f"[WinError {n}]" in text for n in _ENV_WINERRORS):
        if ".dll" in text.lower():
            return (
                "A native library failed to load (usually a missing Microsoft Visual C++ "
                "Redistributable). Install the latest x64 VC++ Redistributable and restart STT."
            )
    return None
