# cua-bench

Computer-use agent benchmark harness for comparing a strawman screenshot-loop agent with a four-layer optimized agent.

## Stack

- Python 3.11, FastAPI, uvicorn
- Async Playwright wrapper
- Stubbed Sarvam, Claude, and router clients
- Supabase recipe storage stub
- Vite, React, Tailwind dashboard
- Server-sent events at `GET /stream/{run_id}`

## Setup

Install prerequisites:

- Python 3.11
- Node.js/npm
- ffmpeg on PATH for server-side voice audio conversion. On Windows, verify with
  `ffmpeg -version`; install with `winget install Gyan.FFmpeg --source winget` or
  `choco install ffmpeg` if it is missing.

```bash
make setup
cp .env.example .env
```

On Windows PowerShell, use:

```powershell
.\scripts\setup.ps1
Copy-Item .env.example .env
```

## Run

```bash
make run-all
```

On Windows PowerShell, use:

```powershell
.\scripts\run-all.ps1
```

Backend: `http://localhost:8000`

Dashboard: `http://localhost:5173`

## API

- `POST /run` starts baseline and optimized agents in parallel.
- `GET /status/{run_id}` returns current run state.
- `GET /trace/{run_id}` returns all meter events.
- `GET /stream/{run_id}` streams token, latency, step, model, and cost events.
