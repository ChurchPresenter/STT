# Start the STT Watchdog in headless mode (PowerShell / Windows).
# Works with both the compiled binary and Python source installs.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Binary    = Join-Path $ScriptDir "STT-Watchdog.exe"
# The watchdog logs to the data dir in both modes, never the checkout.
$LogDir    = Join-Path $env:USERPROFILE ".stt\logs"

if (Test-Path $Binary) {
    # -- Compiled binary ------------------------------------------------------
    $proc = Start-Process -FilePath $Binary -ArgumentList "--headless" `
        -WindowStyle Minimized -PassThru
    Write-Host "[OK] Watchdog started (PID $($proc.Id)) -- binary mode"
    Write-Host "     Logs: $LogDir\watchdog.log"
} else {
    # -- Python source fallback ------------------------------------------------
    $PythonBin = Join-Path $ScriptDir ".venv\Scripts\python.exe"
    if (-not (Test-Path $PythonBin)) { $PythonBin = "python" }
    $proc = Start-Process -FilePath $PythonBin `
        -ArgumentList "`"$(Join-Path $ScriptDir 'stt/watchdog.py')`" --headless" `
        -WorkingDirectory $ScriptDir -WindowStyle Minimized -PassThru
    Write-Host "[OK] Watchdog started (PID $($proc.Id)) -- Python source mode"
    Write-Host "     Logs: $LogDir\watchdog.log"
}
