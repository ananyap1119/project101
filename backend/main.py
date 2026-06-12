from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

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
from backend.agents.optimized import Flags, OptimizedAgent, SarvamActionRouter
from backend.instrumentation.meter import BudgetExceededError, MeterEvent, RunMeter
from backend.models.sarvam import SarvamClient
from backend.tasks.blinkit import BlinkitTask
from backend.tasks.cricket import CricketTask
from backend.tasks.grocery import BlinkitPlannerTask, GroceryItem
from backend.tasks.ntes import NTESTask
from backend.tasks.orders import ShoppingOrdersTask
from backend.tasks.railway import RailwayJourneyTask, normalize_travel_date
from backend.tasks.weather import WeatherTask


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
    product: str = Field(default="milk")
    location: str = Field(default="560001")
    city: str = Field(default="Bangalore")
    question: str = Field(default="")
    origin: str = Field(default="Bangalore")
    destination: str = Field(default="Chennai")
    travel_date: str = Field(default="")
    platform: str = Field(default="all")
    shopping_action: str = Field(default="orders")
    grocery_mode: str = Field(default="meal_plan")
    grocery_items: list[dict[str, Any]] = Field(default_factory=list)
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
    if request.task not in {"ntes", "railway_journey", "shopping_orders", "blinkit", "blinkit_planner", "weather"}:
        raise HTTPException(status_code=404, detail=f"Unknown task: {request.task}")

    run_id = str(uuid.uuid4())
    meter = RunMeter(run_id)
    if request.task == "blinkit":
        task = BlinkitTask(
            headed=request.headed,
            config_name="full_optimized",
            product=request.product,
            location=request.location,
        )
    elif request.task == "blinkit_planner":
        mode = request.grocery_mode if request.grocery_mode in {
            "meal_plan", "missing_ingredients", "household_restock"
        } else "meal_plan"
        task = BlinkitPlannerTask(
            request=request.question,
            mode=mode,  # type: ignore[arg-type]
            items=[GroceryItem.model_validate(item) for item in request.grocery_items],
            location=request.location,
        )
    elif request.task == "weather":
        task = WeatherTask(
            headed=request.headed,
            config_name="full_optimized",
            city=request.city,
            question=request.question,
        )
    elif request.task == "railway_journey":
        task = RailwayJourneyTask(
            origin=request.origin,
            destination=request.destination,
            travel_date=normalize_travel_date(request.travel_date),
        )
    elif request.task == "shopping_orders":
        platform = request.platform if request.platform in {"amazon", "flipkart", "all"} else "all"
        action = request.shopping_action if request.shopping_action in {
            "orders", "cart", "wishlist", "saved_items", "buy_again", "browsing_history", "invoices"
        } else "orders"
        task = ShoppingOrdersTask(platform=platform, action=action, query=request.question)  # type: ignore[arg-type]
    else:
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
        if request.task in {"blinkit", "blinkit_planner", "weather", "railway_journey", "shopping_orders"}:
            flags = Flags.from_layers(request.layers)
            router = None
            if not os.environ.get("OPENROUTER_API_KEY"):
                flags = Flags(
                    dom=flags.dom,
                    summarization=flags.summarization,
                    cascade=False,
                    speculation=flags.speculation,
                )
                router = SarvamActionRouter()
            agents = [OptimizedAgent(meter, flags=flags, router=router)]
        else:
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
    reply_language: str = "hi-IN"
    task_type: str = "train_status"
    extracted_answer: str = ""
    user_question: str = ""


def _build_task_from_intent(intent: dict) -> Any | None:
    """Construct the right task object from a parsed intent dict."""
    task_type = intent.get("task", "unknown")
    if task_type == "train_status":
        raw = intent.get("train_number") or ""
        num = re.sub(r"\D", "", raw)
        if not (4 <= len(num) <= 5):
            return None
        return NTESTask(config_name="full_optimized", train_number=num)
    if task_type == "train_search":
        origin = (intent.get("origin") or "").strip()
        destination = (intent.get("destination") or "").strip()
        travel_date = normalize_travel_date(intent.get("travel_date"))
        if not origin or not destination or not travel_date:
            return None
        return RailwayJourneyTask(
            origin=origin,
            destination=destination,
            travel_date=travel_date,
            preferences=(intent.get("travel_preferences") or "").strip(),
        )
    if task_type == "shopping_orders":
        platform = (intent.get("platform") or "all").lower()
        if platform not in {"amazon", "flipkart", "all"}:
            platform = "all"
        query = (intent.get("shopping_query") or intent.get("order_query") or "").strip()
        action = (intent.get("shopping_action") or "").lower()
        if action not in {
            "orders", "cart", "wishlist", "saved_items", "buy_again", "browsing_history", "invoices"
        }:
            lowered = query.lower()
            if "cart" in lowered:
                action = "cart"
            elif "wish" in lowered or "list" in lowered:
                action = "wishlist"
            elif "saved" in lowered or "later" in lowered:
                action = "saved_items"
            elif "buy again" in lowered or "repurchase" in lowered:
                action = "buy_again"
            elif "history" in lowered or "viewed" in lowered or "brows" in lowered:
                action = "browsing_history"
            elif "invoice" in lowered or "bill" in lowered:
                action = "invoices"
            else:
                action = "orders"
        return ShoppingOrdersTask(
            platform=platform,
            action=action,  # type: ignore[arg-type]
            query=query or "show my shopping activity",
        )
    if task_type == "cricket":
        q = intent.get("cricket_query") or intent.get("team") or ""
        return CricketTask(config_name="full_optimized", query=q)
    if task_type == "weather":
        return WeatherTask(
            config_name="full_optimized",
            city=intent.get("city") or "Bangalore",
            question=intent.get("weather_question") or "",
        )
    if task_type == "blinkit":
        product = intent.get("product") or "milk"
        return BlinkitTask(config_name="full_optimized", product=product)
    if task_type == "blinkit_planner":
        mode = intent.get("grocery_mode") or "meal_plan"
        if mode not in {"meal_plan", "missing_ingredients", "household_restock"}:
            mode = "meal_plan"
        raw_items = intent.get("grocery_items") or []
        items: list[GroceryItem] = []
        for raw_item in raw_items[:12]:
            if not isinstance(raw_item, dict):
                continue
            normalized_item = dict(raw_item)
            try:
                quantity = int(normalized_item.get("quantity", 1))
            except (TypeError, ValueError):
                quantity = 1
            normalized_item["quantity"] = quantity if 1 <= quantity <= 6 else 1
            try:
                item = GroceryItem.model_validate(normalized_item)
            except Exception:
                continue
            if item.name and item.search_query:
                items.append(item)
        if not items:
            return None
        return BlinkitPlannerTask(
            request=(intent.get("grocery_request") or "prepare my grocery cart").strip(),
            mode=mode,
            items=items,
            pantry_items=[str(item) for item in (intent.get("pantry_items") or [])],
        )
    return None


def _clarification_for(task_type: str, intent: dict) -> str:
    if task_type == "train_status":
        raw = intent.get("train_number") or ""
        if not raw:
            return "Train number not detected. Please say a 4-5 digit train number."
        return f"'{raw}' doesn't look like a valid train number. Please say a 4-5 digit number."
    if task_type == "blinkit":
        return "Which product are you looking for on Blinkit? Please say the product name."
    if task_type == "blinkit_planner":
        return "Please tell me the meal, recipe, or household restock you want to prepare."
    if task_type == "train_search":
        return "Please include the origin, destination, and travel date."
    if task_type == "unknown":
        return (
            "I can help with train journeys, order tracking, train status, cricket, weather, or Blinkit. "
            "Please say what you need."
        )
    return "Could not understand the query with sufficient confidence. Please try again."


@app.post("/voice")
async def voice_run(
    audio: UploadFile = File(...),
    language_hint: str = Form(default="auto"),
) -> dict:
    """Transcribe → unified intent extraction → kick off appropriate benchmark."""
    sarvam = _get_sarvam()
    audio_bytes = await audio.read()
    print(
        f"[voice] received {len(audio_bytes)} bytes "
        f"content_type={audio.content_type!r} filename={audio.filename!r}",
        flush=True,
    )
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio was recorded. Please hold the mic while speaking.")

    saarika_audio_bytes = await _prepare_audio_for_saarika(audio, audio_bytes)

    try:
        stt_lang = language_hint if language_hint not in ("auto", "") else "hi-IN"
        transcription = await sarvam.saarika_transcribe(saarika_audio_bytes, stt_lang)
    except Exception as exc:
        print(f"[voice] saarika EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(status_code=502, detail=f"Saarika transcription failed: {exc}")

    transcript = transcription["transcript"]
    detected_lang = transcription.get("language_detected", language_hint)

    if not transcript:
        return {
            "run_id": None, "transcript": "",
            "intent": {"task": "unknown", "reply_language": detected_lang, "confidence": 0.0},
            "action": "clarification_needed",
            "clarification_message": "Sorry, I couldn't hear that. Please try again.",
        }

    try:
        intent = await sarvam.extract_intent(transcript)
    except Exception as exc:
        print(f"[voice] intent EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(status_code=502, detail=f"Intent extraction failed: {exc}")

    # Honor the user's explicit language pill. Romanized Hindi/Kannada is often
    # misclassified as English by text-only intent models.
    if language_hint not in ("auto", ""):
        intent["reply_language"] = language_hint
    elif detected_lang:
        intent["reply_language"] = detected_lang

    task_type = intent.get("task", "unknown")
    conf = float(intent.get("confidence", 0.0))
    reply_lang = intent.get("reply_language", detected_lang)

    task_obj = _build_task_from_intent(intent) if conf >= 0.5 else None

    if task_obj is None:
        return {
            "run_id": None,
            "transcript": transcript,
            "intent": intent,
            "action": "clarification_needed",
            "clarification_message": _clarification_for(task_type, intent),
        }

    run_id = str(uuid.uuid4())
    meter = RunMeter(run_id)
    METERS[run_id] = meter
    RUNS[run_id] = {
        "run_id": run_id,
        "task_type": task_type,
        "task_name": task_obj.name,
        "status": "running",
        "trace": [],
    }

    async def _execute_voice() -> None:
        try:
            await meter.publish(
                MeterEvent(
                    run_id=run_id,
                    agent="optimized",
                    event="usage",
                    elapsed_seconds=0.0,
                    current_step="starting",
                    current_model="initializing",
                    detail=f"Starting {task_type} task",
                )
            )
            task_obj.reset()
            if os.environ.get("OPENROUTER_API_KEY"):
                flags = Flags(dom=True, summarization=True, cascade=True, speculation=False)
                router = None
            else:
                print("[voice] OPENROUTER_API_KEY missing; using Sarvam action router", flush=True)
                flags = Flags(dom=True, summarization=True, cascade=False, speculation=False)
                router = SarvamActionRouter()
            # Voice runs use optimized agent only; baseline requires a separate model key.
            agents: list = [OptimizedAgent(meter, flags=flags, router=router)]
            results = await asyncio.gather(*(a.run(task_obj) for a in agents))
            RUNS[run_id]["status"] = "completed"
            RUNS[run_id]["results"] = [r.model_dump() for r in results]
        except Exception as exc:
            print(f"[voice] execute EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
            RUNS[run_id]["status"] = "failed"
            RUNS[run_id]["error"] = str(exc)
            await meter.publish(
                MeterEvent(
                    run_id=run_id,
                    agent="optimized",
                    event="error",
                    elapsed_seconds=round(time.perf_counter() - meter.started_at, 2),
                    current_step="failed_to_start",
                    current_model="none",
                    detail=str(exc),
                )
            )
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


@app.post("/voice/reply")
async def voice_reply(request: VoiceReplyRequest) -> Response:
    """Use Sarvam-M to generate a natural reply, then synthesise with Bulbul."""
    sarvam = _get_sarvam()

    try:
        text = await sarvam.generate_voice_reply(
            task_type=request.task_type,
            result_snippet=request.extracted_answer,
            reply_language=request.reply_language,
            success=request.success,
            user_question=request.user_question,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Reply generation failed: {exc}")

    try:
        audio_bytes = await sarvam.bulbul_synthesize(text, request.reply_language)
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
