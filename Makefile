.PHONY: setup run-backend run-dashboard run-all

setup:
	uv sync --system-certs
	uv run playwright install chromium
	cd dashboard && npm.cmd install --strict-ssl=false

run-backend:
	uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --loop asyncio

run-dashboard:
	cd dashboard && npm.cmd run dev -- --host 0.0.0.0 --port 5173

run-all:
	$(MAKE) -j2 run-backend run-dashboard
