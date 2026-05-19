uv sync --system-certs
uv run playwright install chromium
Push-Location dashboard
npm.cmd install --strict-ssl=false
Pop-Location
