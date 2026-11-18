from stt.env_errors import environment_problem


def _os_error(winerror: int, text: str) -> OSError:
    e = OSError(f"[WinError {winerror}] {text}")
    e.winerror = winerror  # type: ignore[attr-defined]
    return e


def test_torch_dll_load_failure_is_environmental():
    exc = _os_error(126, 'Error loading "x\\torch\\lib\\c10.dll" or one of its dependencies.')
    assert "Visual C++" in (environment_problem(exc) or "")


def test_localised_message_is_still_recognised():
    exc = OSError('[WinError 126] Impossibile trovare il modulo specificato. Error loading "c10.dll"')
    assert environment_problem(exc) is not None


def test_application_control_block():
    assert "Application Control" in (environment_problem(OSError("[WinError 4551] An Application Control policy has blocked this file")) or "")


def test_ordinary_errors_are_not_swallowed():
    assert environment_problem(RuntimeError("boom")) is None
    assert environment_problem(OSError("[WinError 126] no dll named here")) is None
