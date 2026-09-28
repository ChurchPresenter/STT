@echo off
setlocal EnableDelayedExpansion
REM Speech-to-Text Restart Script (Windows)
REM Called from server settings page via /api/server/restart

cd /d "%~dp0"

REM Show current version + update status (git) -- like update_server
git rev-parse --git-dir >nul 2>&1 && (
    echo [GIT] Current version:
    git log --oneline -1
    git fetch --quiet >nul 2>&1
    for /f %%b in ('git rev-list --count HEAD..@{u} 2^>nul') do if not "%%b"=="0" echo [GIT] Update available: %%b commit^(s^) behind -- applied on startup
)

echo [RESTART] Stopping server...

REM --- Kill python processes running speech_to_text.py ---------------
REM PowerShell rather than wmic: wmic is removed from Windows 11 24H2, where the old
REM command-line lookup silently matched nothing and the server was never stopped.
for /f %%c in ('powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*speech_to_text*' }); $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; $p.Count" 2^>nul') do echo Stopped %%c server process^(es^).

REM Also kill by window title (start_server.bat / below title the window "STT Server")
taskkill /F /FI "WINDOWTITLE eq STT Server*" >nul 2>&1

REM --- Kill orphaned ffmpeg processes --------------------------------
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq ffmpeg.exe" /FO CSV /NH 2^>nul') do (
    powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter ('ProcessId=' + %%~a)).CommandLine" 2>nul | findstr /I "dshow wasapi pipe" >nul 2>&1
    if !errorlevel! equ 0 taskkill /F /PID %%~a >nul 2>&1
)

REM Wait for processes to die
REM ping, not timeout: timeout quits at once when stdin is redirected, which it is when
REM the server's own restart button runs this script, so every wait was skipped.
ping -n 3 127.0.0.1 >nul

REM --- Verify stopped -----------------------------------------------
set "RETRIES=0"
:wait_loop
set "STILL_RUNNING=0"
for /f %%c in ('powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*speech_to_text*' }); $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; $p.Count" 2^>nul') do if not "%%c"=="0" set "STILL_RUNNING=1"
if "!STILL_RUNNING!"=="1" (
    set /a RETRIES+=1
    if !RETRIES! lss 10 (
        ping -n 2 127.0.0.1 >nul
        goto wait_loop
    ) else (
        echo [WARNING] Could not stop all processes after 10 attempts
    )
)

echo [RESTART] All server processes stopped.
ping -n 3 127.0.0.1 >nul

REM --- Start server -------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo [RESTART] Starting server...
start "STT Server" "!PYTHON_BIN!" speech_to_text.py

REM --- Read port from the live config in the data dir -----------------
REM STT_DATA_DIR, else ~/.stt -- the checkout's config/ holds only the template.
set "PORT="
REM stt\server_port.py is the one answer every launcher and the watchdog share; run from
REM the checkout (cd above), so -m finds the package without PYTHONPATH.
for /f "delims=" %%p in ('"!PYTHON_BIN!" -m stt.server_port 2^>nul') do set "PORT=%%p"
if not defined PORT set "PORT=8080"

REM --- Verify started -----------------------------------------------
REM Success means something answers on the port within 30s. The old check passed
REM whenever any python.exe was running, so it reported success for a server that had
REM died. The 30s is a clock, not a count of tries: a refused connect takes ~2s on
REM Windows, so 30 tries took ~95s.
set "ANSWERED=0"
for /f %%r in ('powershell -NoProfile -Command "$sw = [Diagnostics.Stopwatch]::StartNew(); while ($sw.Elapsed.TotalSeconds -lt 30) { $c = New-Object Net.Sockets.TcpClient; try { if ($c.ConnectAsync('127.0.0.1', !PORT!).Wait(1000)) { 'yes'; exit } } catch {} finally { $c.Close() }; Start-Sleep -Milliseconds 500 }; 'no'" 2^>nul') do if "%%r"=="yes" set "ANSWERED=1"
if "!ANSWERED!"=="1" (
    echo [RESTART] Server started successfully on port !PORT!.
) else (
    echo [RESTART] WARNING: nothing answered on port !PORT! after 30s. Check for errors.
)
