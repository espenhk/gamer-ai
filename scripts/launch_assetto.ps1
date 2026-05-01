# Launch Assetto Corsa via Steam if it isn't already running.
# Per issue #79: small launcher that checks the AC executable is running and starts it if not.

$ErrorActionPreference = "Stop"

$proc = Get-Process -Name "acs" -ErrorAction SilentlyContinue
if ($null -ne $proc) {
    Write-Host "Assetto Corsa already running (PID $($proc.Id))"
    exit 0
}

Write-Host "Assetto Corsa not running — launching via Steam..."
Start-Process "steam://rungameid/244210"

# Wait up to 60s for the process to appear.
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    $proc = Get-Process -Name "acs" -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        Write-Host "Assetto Corsa started (PID $($proc.Id))"
        exit 0
    }
}

Write-Error "Timed out waiting for Assetto Corsa to start"
exit 1
