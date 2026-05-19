$ffmpegBin = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
$env:PATH = "$ffmpegBin;$env:PATH"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
& $python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
