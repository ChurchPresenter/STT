@echo off
setlocal
REM Speech-to-Text Update Script (Windows)
REM Pull the latest code and restart the server to apply it now, instead of
REM waiting for the nightly auto-update. Restart is delegated to
REM restart_server.bat (no duplicated logic).

REM cmd reads a batch file by byte offset as it goes, so when git pull rewrites this
REM file mid-run cmd resumes at the old offset in the NEW text and runs whatever lands
REM there. Run from a copy in %TEMP% instead, and pass the checkout along.
if /i not "%~1"=="--from-copy" (
    copy /y "%~f0" "%TEMP%\stt_update_server.bat" >nul
    "%TEMP%\stt_update_server.bat" --from-copy "%~dp0"
)
set "CHECKOUT=%~2"
cd /d "%CHECKOUT%"

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
call "%CHECKOUT%restart_server.bat"
