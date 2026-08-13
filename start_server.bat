@echo off
REM Speech-to-Text Start Script (Windows)

cd /d "%~dp0"

REM Show current version + update status (git) — like update_server
git rev-parse --git-dir >nul 2>&1 && (
    echo [GIT] Current version:
    git log --oneline -1
    git fetch --quiet >nul 2>&1
    for /f %%b in ('git rev-list --count HEAD..@{u} 2^>nul') do if not "%%b"=="0" echo [GIT] Update available: %%b commit^(s^) behind -- applied on startup
)

REM Check if already running
tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>nul | findstr /I "speech_to_text" >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARNING] Server is already running.
    echo Use restart_server.bat to restart or stop_server.bat to stop.
    pause
    exit /b 1
)

REM Determine Python binary
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo Starting Speech-to-Text server...
REM The window gets a real title so stop_server.bat's WINDOWTITLE fallback can find
REM it. With start "" the title becomes the exe path and that fallback matched nothing.
start "STT Server" "%PYTHON_BIN%" speech_to_text.py
echo [OK] Server starting...

REM Read the port from the live config, which lives in the data dir (STT_DATA_DIR, or
REM ~/.stt) — not in the checkout. config/ here holds only the shipped template, so the
REM old lookup always threw and always printed 8080, whatever the server was bound to.
set "PORT="
for /f "delims=" %%p in ('"%PYTHON_BIN%" -c "import os,json;d=os.environ.get('STT_DATA_DIR') or os.path.join(os.path.expanduser('~'),'.stt');print(json.load(open(os.path.join(d,'config','config.json'))).get('web_server',{}).get('port',8080))" 2^>nul') do set "PORT=%%p"
if not defined PORT set "PORT=8080"

echo Open your browser to http://localhost:%PORT%
timeout /t 3 >nul
