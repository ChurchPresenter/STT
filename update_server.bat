@echo off
setlocal
REM Speech-to-Text Update Script (Windows)
REM Pull the latest code and restart the server to apply it now, instead of
REM waiting for the nightly auto-update. Restart is delegated to
REM restart_server.bat (no duplicated logic).

cd /d "%~dp0"

echo [UPDATE] Pulling latest code (git pull --ff-only)...
git pull --ff-only
if errorlevel 1 (
    echo [ERROR] git pull --ff-only failed.
    echo   The working tree is probably dirty or the branch has diverged/unpushed
    echo   commits. Commit/stash your changes ^(or push^) and try again -- nothing
    echo   was changed and the server was NOT restarted.
    exit /b 1
)

echo [UPDATE] Restarting to apply...
call "%~dp0restart_server.bat"
