@echo off
setlocal EnableDelayedExpansion
REM Speech-to-Text Restart Script (Windows)
REM Called from server settings page via /api/server/restart

cd /d "%~dp0"

REM Show current version + update status (git) — like update_server
git rev-parse --git-dir >nul 2>&1 && (
    echo [GIT] Current version:
    git log --oneline -1
    git fetch --quiet >nul 2>&1
    for /f %%b in ('git rev-list --count HEAD..@{u} 2^>nul') do if not "%%b"=="0" echo [GIT] Update available: %%b commit^(s^) behind -- applied on startup
)

echo [RESTART] Stopping server...

REM ─── Kill python processes running speech_to_text.py ───────────────
REM PowerShell rather than wmic: wmic is removed from Windows 11 24H2, where the old
REM command-line lookup silently matched nothing and the server was never stopped.
for /f %%c in ('powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Process ^| Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*speech_to_text*' }); $p ^| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; $p.Count" 2^>nul') do echo Stopped %%c server process^(es^).

REM Also kill by window title (start_server.bat / below title the window "STT Server")
taskkill /F /FI "WINDOWTITLE eq STT Server*" >nul 2>&1

REM ─── Kill orphaned ffmpeg processes ────────────────────────────────
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq ffmpeg.exe" /FO CSV /NH 2^>nul') do (
    powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter ('ProcessId=' + %%~a)).CommandLine" 2>nul | findstr /I "dshow wasapi pipe" >nul 2>&1
    if !errorlevel! equ 0 taskkill /F /PID %%~a >nul 2>&1
)

REM Wait for processes to die
timeout /t 2 /nobreak >nul

REM ─── Verify stopped ───────────────────────────────────────────────
set "RETRIES=0"
:wait_loop
set "STILL_RUNNING=0"
for /f %%c in ('powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Process ^| Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*speech_to_text*' }); $p ^| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; $p.Count" 2^>nul') do if not "%%c"=="0" set "STILL_RUNNING=1"
if "!STILL_RUNNING!"=="1" (
    set /a RETRIES+=1
    if !RETRIES! lss 10 (
        timeout /t 1 /nobreak >nul
        goto wait_loop
    ) else (
        echo [WARNING] Could not stop all processes after 10 attempts
    )
)

echo [RESTART] All server processes stopped.
timeout /t 2 /nobreak >nul

REM ─── Start server ─────────────────────────────────────────────────
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo [RESTART] Starting server...
start "STT Server" "!PYTHON_BIN!" speech_to_text.py

REM ─── Read port from the live config in the data dir ─────────────────
REM STT_DATA_DIR, else ~/.stt — the checkout's config/ holds only the template.
set "PORT="
for /f "delims=" %%p in ('"!PYTHON_BIN!" -c "import os,json;d=os.environ.get('STT_DATA_DIR') or os.path.join(os.path.expanduser('~'),'.stt');print(json.load(open(os.path.join(d,'config','config.json'))).get('web_server',{}).get('port',8080))" 2^>nul') do set "PORT=%%p"
if not defined PORT set "PORT=8080"

REM ─── Verify started ───────────────────────────────────────────────
timeout /t 3 /nobreak >nul
tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>nul | findstr /I "python" >nul 2>&1
if %errorlevel% equ 0 (
    echo [RESTART] Server started successfully on port !PORT!.
) else (
    echo [RESTART] WARNING: Server may not have started. Check for errors.
)
