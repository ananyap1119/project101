# cua-bench

Computer-use agent benchmark harness for comparing a strawman screenshot-loop agent with a four-layer optimized agent.

## Demo
1. computer-use-agent : https://x.com/AnanNo_11/status/2072218934389113170?s=20
2. benchmark comparison : https://x.com/AnanNo_11/status/2073026420700500406?s=20


## Stack

- Python 3.11, FastAPI, uvicorn
- Async Playwright wrapper
- Stubbed Sarvam, Claude, and router clients
- Supabase recipe storage stub
- Vite, React, Tailwind dashboard
- Server-sent events at `GET /stream/{run_id}`

# Description

Project101 is a multilingual AI web agent built to navigate complex Indian government, enterprise, and consumer websites through natural voice commands.

Instead of requiring users to search through multiple websites and forms, Project101 understands spoken instructions, plans the required actions, and completes tasks directly in the browser.

The project focuses on making AI systems practical for Indian users while remaining fast and cost-efficient. It uses a model cascade architecture to reduce unnecessary large-model calls, improving both latency and inference cost without sacrificing task completion.

### Highlights

- 🌏 Multilingual voice interaction
- 🖥️ Browser automation for real-world websites
- 🧠 Model cascade for efficient inference
- ⚡ 4.15× fewer tokens
- 💰 3.05× lower cost
- 🚀 1.65× faster execution

Project101 was built as an exploration into practical AI agents that can reliably interact with complex web interfaces rather than functioning only as conversational assistants.

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
