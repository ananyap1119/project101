from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Playwright spawns Chromium via create_subprocess_exec which requires ProactorEventLoop on Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

# Load .env before any other backend import so API keys are available.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ModuleNotFoundError:
    _env = Path(__file__).resolve().parents[1] / ".env"
    if _env.exists():
        for _line in _env.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                if _k.strip() and not os.environ.get(_k.strip()):
                    os.environ[_k.strip()] = _v.strip()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from backend.agents.baseline import BaselineAgent
from backend.agents.optimized import Flags, OptimizedAgent
from backend.instrumentation.meter import BudgetExceededError, MeterEvent, RunMeter
from backend.models.sarvam import SarvamClient
from backend.tasks.ntes import NTESTask


_sarvam: SarvamClient | None = None

_SAARIKA_NATIVE_EXTS = {".wav"}
_SAARIKA_NATIVE_MIMES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
}


def _ffmpeg_executable() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found

    winget_path = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Microsoft"
        / "WinGet"
        / "Packages"
        / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        / "ffmpeg-8.1.1-full_build"
        / "bin"
        / "ffmpeg.exe"
    )
    if winget_path.exists():
        return str(winget_path)

    return "ffmpeg"


def _uploaded_audio_format(audio: UploadFile) -> tuple[str, str, str]:
    content_type = (audio.content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(audio.filename or "").suffix.lower()
    if not suffix and content_type:
        mime_exts = {
            "audio/aac": ".aac",
            "audio/flac": ".flac",
            "audio/m4a": ".m4a",
            "audio/mp4": ".mp4",
            "audio/mpeg": ".mp3",
            "audio/ogg": ".ogg",
            "audio/opus": ".opus",
            "audio/wav": ".wav",
            "audio/webm": ".webm",
            "audio/x-wav": ".wav",
            "audio/wave": ".wav",
            "video/webm": ".webm",
        }
        suffix = mime_exts.get(content_type, "")
    if not suffix or len(suffix) > 10 or not suffix[1:].replace("-", "").isalnum():
        suffix = ".audio"
    label = suffix.lstrip(".") or content_type or "unknown"
    return content_type, suffix, label


def _needs_wav_transcode(content_type: str, suffix: str) -> bool:
    if suffix in _SAARIKA_NATIVE_EXTS or content_type in _SAARIKA_NATIVE_MIMES:
        return False
    return True


def _transcode_audio_to_wav(audio_bytes: bytes, suffix: str, input_label: str) -> bytes:
    in_path = ""
    out_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
            src.write(audio_bytes)
            in_path = src.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst:
            out_path = dst.name

        print(f"[voice] transcoding {input_label} -> wav (16kHz mono)", flush=True)
        result = subprocess.run(
            [
                _ffmpeg_executable(),
                "-y",
                "-i",
                in_path,
                "-t",
                "15",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                out_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            print(f"[voice] ffmpeg stderr: {result.stderr.strip()}", flush=True)
            raise HTTPException(status_code=400, detail="audio format conversion failed")

        return Path(out_path).read_bytes()
    except FileNotFoundError:
        print("[voice] ffmpeg not found on PATH", flush=True)
        raise HTTPException(status_code=400, detail="audio format conversion failed")
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        print(f"[voice] ffmpeg timed out: {stderr}", flush=True)
        raise HTTPException(status_code=400, detail="audio format conversion failed")
    finally:
        for path in (in_path, out_path):
            if path:
                with suppress(OSError):
                    os.unlink(path)


async def _prepare_audio_for_saarika(audio: UploadFile, audio_bytes: bytes) -> bytes:
    content_type, suffix, label = _uploaded_audio_format(audio)
    if not _needs_wav_transcode(content_type, suffix):
        return audio_bytes
    return await asyncio.to_thread(_transcode_audio_to_wav, audio_bytes, suffix, label)


def _get_sarvam() -> SarvamClient:
    global _sarvam
    if _sarvam is None:
        _sarvam = SarvamClient()
    return _sarvam


class RunRequest(BaseModel):
    task: str = Field(default="ntes")
    headed: bool = Field(default=False)
    # Optimised agent layers: 1=DOM, 2=summarisation, 4=speculation (cascade always on).
    # Default [1,2,4] = all layers.  Set e.g. [1,2] to disable speculation.
    layers: list[int] = Field(default=[1, 2, 4])


app = FastAPI(title="cua-bench", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS: dict[str, dict[str, Any]] = {}
METERS: dict[str, RunMeter] = {}
BENCHMARK_RESULTS: list[dict[str, Any]] = []


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/benchmark-results")
async def benchmark_results() -> dict[str, Any]:
    return {"results": BENCHMARK_RESULTS}


@app.post("/benchmark-results")
async def add_benchmark_result(result: dict[str, Any]) -> dict[str, str]:
    BENCHMARK_RESULTS.append(result)
    return {"status": "ok"}


@app.delete("/benchmark-results")
async def clear_benchmark_results() -> dict[str, str]:
    BENCHMARK_RESULTS.clear()
    return {"status": "ok"}


@app.post("/run")
async def run_both(request: RunRequest) -> dict[str, str]:
    if request.task != "ntes":
        raise HTTPException(status_code=404, detail=f"Unknown task: {request.task}")

    run_id = str(uuid.uuid4())
    meter = RunMeter(run_id)
    task = NTESTask(headed=request.headed, config_name="full_optimized")
    METERS[run_id] = meter
    RUNS[run_id] = {
        "run_id": run_id,
        "task": request.task,
        "status": "running",
        "trace": [],
    }

    async def execute() -> None:
        task.reset()
        agents = [
            BaselineAgent(meter),
            OptimizedAgent(meter, flags=Flags.from_layers(request.layers)),
        ]
        try:
            results = await asyncio.gather(*(agent.run(task) for agent in agents))
            RUNS[run_id]["status"] = "completed"
            RUNS[run_id]["results"] = [result.model_dump() for result in results]
        except Exception as exc:  # pragma: no cover - defensive boundary for background task
            RUNS[run_id]["status"] = "failed"
            RUNS[run_id]["error"] = str(exc)
            await meter.publish(
                MeterEvent(
                    run_id=run_id,
                    agent="system",
                    event="error",
                    current_step="failed",
                    detail=str(exc),
                )
            )
        finally:
            RUNS[run_id]["trace"] = [event.model_dump() for event in meter.trace]
            await meter.close()

    asyncio.create_task(execute())
    return {"run_id": run_id, "status": "running"}


@app.get("/status/{run_id}")
async def status(run_id: str) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/trace/{run_id}")
async def trace(run_id: str) -> dict[str, Any]:
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    meter = METERS[run_id]
    return {"run_id": run_id, "events": [event.model_dump() for event in meter.trace]}


class VoiceReplyRequest(BaseModel):
    run_id: str
    success: bool
    train_number: str
    reply_language: str = "hi-IN"


@app.post("/voice")
async def voice_run(
    audio: UploadFile = File(...),
    language_hint: str = Form(default="hi-IN"),
) -> dict:
    """Transcribe audio → extract NTES intent → kick off benchmark if intent is clear."""
    sarvam = _get_sarvam()
    audio_bytes = await audio.read()
    print(
        f"[voice] received {len(audio_bytes)} bytes "
        f"content_type={audio.content_type!r} filename={audio.filename!r}",
        flush=True,
    )
    saarika_audio_bytes = await _prepare_audio_for_saarika(audio, audio_bytes)

    try:
        transcription = await sarvam.saarika_transcribe(saarika_audio_bytes, language_hint)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Saarika transcription failed: {exc}")

    transcript = transcription["transcript"]
    if not transcript:
        return {
            "run_id": None,
            "transcript": "",
            "intent": {"task": "unknown", "train_number": None,
                       "reply_language": language_hint, "confidence": 0.0},
            "action": "clarification_needed",
            "clarification_message": "Sorry, I couldn't hear that. Please try again.",
        }

    try:
        intent = await sarvam.extract_intent_for_ntes(transcript)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Intent extraction failed: {exc}")

    conf = float(intent.get("confidence", 0.0))
    task = intent.get("task", "unknown")
    train_number_raw = intent.get("train_number") or ""
    train_number = re.sub(r"\D", "", train_number_raw)   # strip non-digits

    if task == "train_status" and conf >= 0.6 and not train_number:
        return {
            "run_id": None,
            "transcript": transcript,
            "intent": intent,
            "action": "clarification_needed",
            "clarification_message": (
                "Train number not detected. Please include the train number, "
                "e.g. 'Train 22691'. Train number include karein."
            ),
        }

    if task == "train_status" and conf >= 0.6 and not (4 <= len(train_number) <= 5):
        return {
            "run_id": None,
            "transcript": transcript,
            "intent": intent,
            "action": "clarification_needed",
            "clarification_message": (
                f"'{train_number_raw}' doesn't look like a valid train number. "
                "Please say a 4-5 digit number, e.g. 'Train 22691'."
            ),
        }

    if task == "train_status" and train_number and conf >= 0.6:
        run_id = str(uuid.uuid4())
        meter  = RunMeter(run_id)
        ntes_task = NTESTask(headed=False, config_name="full_optimized",
                             train_number=train_number)
        METERS[run_id] = meter
        RUNS[run_id] = {"run_id": run_id, "task": "ntes", "status": "running", "trace": []}

        async def _execute_voice() -> None:
            ntes_task.reset()
            agents = [BaselineAgent(meter), OptimizedAgent(meter)]
            try:
                results = await asyncio.gather(*(a.run(ntes_task) for a in agents))
                RUNS[run_id]["status"] = "completed"
                RUNS[run_id]["results"] = [r.model_dump() for r in results]
            except Exception as exc:
                RUNS[run_id]["status"] = "failed"
                RUNS[run_id]["error"] = str(exc)
            finally:
                RUNS[run_id]["trace"] = [e.model_dump() for e in meter.trace]
                await meter.close()

        asyncio.create_task(_execute_voice())
        return {
            "run_id": run_id,
            "transcript": transcript,
            "intent": intent,
            "action": "started",
            "clarification_message": None,
        }

    # Not enough confidence or unknown task — ask for clarification
    if task != "train_status":
        msg = ("Please say a train status query, e.g. 'Train 22691 kahan hai'. "
               "Kripya train query bolein.")
    elif not train_number:
        msg = ("Train number not detected. Please include the train number, "
               "e.g. 'Train 22691'. Train number include karein.")
    else:
        msg = "Could not understand the query with sufficient confidence. Please try again."

    return {
        "run_id": None,
        "transcript": transcript,
        "intent": intent,
        "action": "clarification_needed",
        "clarification_message": msg,
    }


@app.post("/voice/reply")
async def voice_reply(request: VoiceReplyRequest) -> Response:
    """Generate and synthesise a spoken reply for the end of a voice-triggered run."""
    sarvam = _get_sarvam()
    lang   = request.reply_language
    num    = request.train_number

    if request.success:
        text = {
            "hi-IN": f"Train {num} ka status check ho gaya. Dashboard mein result dekh sakte hain.",
            "kn-IN": f"Train {num} ra status check aaagide. Dashboard nodi.",
        }.get(lang, f"Train {num} status has been checked. See the dashboard for results.")
    else:
        text = {
            "hi-IN": f"Train {num} ka status abhi nahi mila. Phir se try karein.",
            "kn-IN": f"Train {num} ra status sigalyilla. Matte try maadi.",
        }.get(lang, f"Could not fetch status for train {num}. Please try again.")

    try:
        audio_bytes = await sarvam.bulbul_synthesize(text, lang)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Bulbul synthesis failed: {exc}")

    if not audio_bytes:
        raise HTTPException(status_code=204, detail="TTS not available in stub mode")

    return Response(content=audio_bytes, media_type="audio/wav")


@app.get("/stream/{run_id}")
async def stream(run_id: str) -> StreamingResponse:
    meter = METERS.get(run_id)
    if not meter:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_source():
        subscriber = meter.subscribe()
        try:
            for event in meter.trace:
                yield event.to_sse()
                if event.event == "done":
                    return
            while True:
                event = await subscriber.get()
                yield event.to_sse()
                if event.event == "done":
                    break
        except asyncio.CancelledError:
            raise
        finally:
            with suppress(ValueError):
                meter.unsubscribe(subscriber)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
