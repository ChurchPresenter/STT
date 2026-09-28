@echo off
REM Start the STT Watchdog in headless mode (Windows).
REM Works with both the compiled binary and Python source installs.

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
REM The watchdog logs to the data dir in both modes, never the checkout.
set "LOG_DIR=%USERPROFILE%\.stt\logs"

if exist "%SCRIPT_DIR%STT-Watchdog.exe" (
    REM ── Compiled binary ──────────────────────────────────────────────────
    start /min "STT Watchdog" "%SCRIPT_DIR%STT-Watchdog.exe" --headless
    echo [OK] Watchdog started (binary mode).
    echo      Logs: !LOG_DIR!\watchdog.log
) else (
    REM ── Python source fallback ────────────────────────────────────────────
    REM !var! not %var%: inside this block %PYTHON_BIN% is expanded when the whole
    REM block is parsed, before the set runs, so start was handed an empty program.
    set "PYTHON_BIN=%SCRIPT_DIR%.venv\Scripts\python.exe"
    if not exist "!PYTHON_BIN!" set "PYTHON_BIN=python"
    start /min "STT Watchdog" "!PYTHON_BIN!" "%SCRIPT_DIR%stt\watchdog.py" --headless
    echo [OK] Watchdog started (Python source mode).
    echo      Logs: !LOG_DIR!\watchdog.log
)
