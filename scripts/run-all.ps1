$root = Split-Path -Parent $PSScriptRoot
Start-Process powershell -WindowStyle Hidden -WorkingDirectory $root -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$PSScriptRoot\run-backend.ps1"
Start-Process powershell -WindowStyle Hidden -WorkingDirectory $root -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$PSScriptRoot\run-dashboard.ps1"
Write-Host "Backend:   http://localhost:8000"
Write-Host "Dashboard: http://localhost:5173"
