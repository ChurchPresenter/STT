@echo off
setlocal EnableDelayedExpansion
REM Speech-to-Text Stop Script (Windows)
REM
REM Two things were wrong here and each one alone made the script a no-op:
REM   * !errorlevel! inside a for loop needs EnableDelayedExpansion (above), or cmd
REM     compares the literal text "!errorlevel!" and the taskkill branch never runs.
REM     The script then printed "[OK] Server stopped." while every process kept going.
REM   * wmic is removed from Windows 11 24H2, so the command-line lookup it used to
REM     identify our python.exe found nothing on a current install.
REM Both loops are now a single PowerShell call that matches on the command line and
REM kills what it matched, which also avoids the batch quoting the old form needed.

cd /d "%~dp0"

echo Stopping Speech-to-Text server...

set "KILLED=0"
for /f %%c in ('powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Process ^| Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*speech_to_text*' }); $p ^| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; $p.Count" 2^>nul') do set "KILLED=%%c"

REM Fallback for a server started from a console window by start_server.bat, which
REM titles the window "STT Server" for exactly this purpose.
taskkill /F /FI "WINDOWTITLE eq STT Server*" >nul 2>&1

REM Orphaned ffmpeg from audio capture. One session spawns one ffmpeg, but a stream
REM that stalls is respawned, so a torn-down session can leave one behind.
set "FFKILLED=0"
for /f %%c in ('powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Process ^| Where-Object { $_.Name -eq 'ffmpeg.exe' -and ($_.CommandLine -like '*dshow*' -or $_.CommandLine -like '*wasapi*' -or $_.CommandLine -like '*pipe:*') }); $p ^| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; $p.Count" 2^>nul') do set "FFKILLED=%%c"

if "!KILLED!"=="0" (
    echo [OK] No server process was running.
) else (
    echo [OK] Server stopped ^(!KILLED! process^(es^)^).
)
if not "!FFKILLED!"=="0" echo [OK] Stopped !FFKILLED! orphaned ffmpeg process^(es^).
endlocal
