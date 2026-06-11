"""Sarvam AI client.

Calls https://api.sarvam.ai/v1/chat/completions (OpenAI-compatible).
Falls back to stub responses when SARVAM_API_KEY is absent so the optimized
agent can be exercised in mock mode without any credentials.
"""
from __future__ import annotations

import asyncio
import json as _json_mod
import os
import re
import time
import uuid
import warnings
from dataclasses import dataclass

import httpx
import urllib3

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SARVAM_BASE = "https://api.sarvam.ai/v1"

_JSON_CT = {"Content-Type": "application/json; charset=utf-8"}


def _json_bytes(obj: object) -> bytes:
    """Encode object as UTF-8 JSON bytes — avoids Windows charmap issues with httpx json=."""
    return _json_mod.dumps(obj, ensure_ascii=False).encode("utf-8")


def _dedup_reply(text: str) -> str:
    """Remove repeated sentences — guards against Sarvam-M repetition loops."""
    import re as _re
    parts = _re.split(r"(?<=[।.!?])\s+", text.strip())
    seen: list[str] = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.append(p)
        if len(seen) >= 2:   # hard cap: 2 sentences max
            break
    return " ".join(seen) if seen else text[:120]


_TRAIN_NUMBER_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "won": "1",
    "two": "2", "too": "2", "to": "2",
    "three": "3", "tree": "3",
    "four": "4", "for": "4",
    "five": "5",
    "six": "6", "sex": "6",
    "seven": "7",
    "eight": "8", "ate": "8",
    "nine": "9",
    "twenty": "2", "thirty": "3", "forty": "4", "fifty": "5",
    "sixty": "6", "seventy": "7", "eighty": "8", "ninety": "9",
    "baais": "22", "bais": "22", "teis": "23", "chaubees": "24",
    "pachis": "25", "chabbis": "26", "sattais": "27", "athais": "28",
    "untees": "29",
    "shunya": "0", "sunya": "0", "ek": "1", "do": "2", "teen": "3",
    "char": "4", "chaar": "4", "panch": "5", "paanch": "5", "che": "6",
    "chhe": "6", "chhah": "6", "saat": "7", "aath": "8", "nau": "9",
}


def _train_number_from_transcript(transcript: str) -> str | None:
    """Extract a 4-5 digit train number without an LLM when possible."""
    digit_match = re.search(r"\b\d(?:[\s-]*\d){3,4}\b", transcript)
    if digit_match:
        digits = re.sub(r"\D", "", digit_match.group(0))
        if 4 <= len(digits) <= 5:
            return digits

    tokens = re.findall(r"[a-zA-Z]+", transcript.lower())
    digit_runs: list[str] = []
    current = ""
    for token in tokens:
        digit = _TRAIN_NUMBER_WORDS.get(token)
        if digit is None:
            if current:
                digit_runs.append(current)
                current = ""
            continue
        current += digit
    if current:
        digit_runs.append(current)

    for digits in digit_runs:
        if 4 <= len(digits) <= 5:
            return digits
    return None


def _build_multipart(fields: dict[str, str], file_name: str, file_bytes: bytes, file_mime: str) -> tuple[bytes, str]:
    """Build a multipart/form-data body as pure bytes — avoids httpx charmap issues on Windows."""
    boundary = uuid.uuid4().hex.encode()
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="' + key.encode() + b'"\r\n'
            b"\r\n" + value.encode("utf-8") + b"\r\n"
        )
    parts.append(
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="' + file_name.encode() + b'"\r\n'
        b"Content-Type: " + file_mime.encode() + b"\r\n"
        b"\r\n" + file_bytes + b"\r\n"
    )
    parts.append(b"--" + boundary + b"--\r\n")
    return b"".join(parts), f"multipart/form-data; boundary={boundary.decode()}"


def _sniff_audio(data: bytes) -> tuple[str, str]:
    """Return (filename, mime_type) by inspecting magic bytes."""
    if data[:4] == b"RIFF":
        return "recording.wav", "audio/wav"
    if data[:4] == b"OggS":
        return "recording.ogg", "audio/ogg"
    if data[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "recording.mp3", "audio/mpeg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "recording.mp4", "audio/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "recording.webm", "audio/webm"
    return "recording.mp4", "audio/mp4"   # safe default for unknown
SARVAM_API_BASE = "https://api.sarvam.ai"    # base for Saarika/Bulbul (no /v1/ prefix)
_DEPRECATED_CHAT_MODELS = {"sarvam-m"}


def _chat_model_name() -> str:
    model = os.environ.get("SARVAM_CHAT_MODEL", "sarvam-30b").strip() or "sarvam-30b"
    if model in _DEPRECATED_CHAT_MODELS:
        print(
            f"[sarvam] {model} is deprecated; using sarvam-30b instead",
            flush=True,
        )
        return "sarvam-30b"
    return model


SARVAM_MODEL = _chat_model_name()

# Known-good Bulbul speaker IDs per language.  Female voices first.
_BULBUL_SPEAKERS: dict[str, list[str]] = {
    "hi-IN": ["diya", "anushka", "arvind", "amol"],
    "en-IN": ["meera", "anushka", "arvind"],
    "kn-IN": ["anushka", "arvind"],
    "ta-IN": ["anushka"],
    "te-IN": ["anushka"],
    "ml-IN": ["anushka"],
    "bn-IN": ["anushka"],
    "gu-IN": ["anushka"],
    "mr-IN": ["anushka"],
}
_BULBUL_DEFAULT_SPEAKERS = ["anushka", "arvind"]

# Sarvam pricing not publicly listed; using estimate comparable to other 24B models.
# Update when Sarvam publishes a pricing page.
_IN_USD_PER_TOK = 0.15 / 1_000_000
_OUT_USD_PER_TOK = 0.60 / 1_000_000
_USD_TO_INR = 85.0


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_inr: float
    confidence: float = 1.0   # 0–1 derived from finish_reason / logprobs


# ── backward-compat dataclass kept for ModelRouter.complete() stub path ──────
@dataclass(frozen=True, slots=True)
class StubModelResponse:
    text: str
    model: str
    vision_tokens: int
    context_tokens: int
    latency_ms: int
    cost_inr: float


class SarvamClient:
    def __init__(self) -> None:
        api_key = os.environ.get("SARVAM_API_KEY", "")
        self._live = bool(api_key)
        if self._live:
            self._http = httpx.AsyncClient(
                base_url=SARVAM_BASE,
                headers={
                    "api-subscription-key": api_key,
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                verify=False,   # corporate TLS proxy workaround (same as Claude client)
                timeout=60.0,
            )
            self._api_http = httpx.AsyncClient(
                base_url=SARVAM_API_BASE,
                headers={"api-subscription-key": api_key},
                verify=False,
                timeout=90.0,
            )

    # ── primary interface used by optimized agent ─────────────────────────────

    async def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> ModelResponse:
        if not self._live:
            return self._stub_chat(user)

        t0 = time.perf_counter()
        payload = {
            "model": SARVAM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": None,
            "response_format": {"type": "json_object"},   # suppress think blocks
        }
        r = await self._http.post("/chat/completions", content=_json_bytes(payload), headers=_JSON_CT)
        if r.status_code == 400 and "response_format" in r.text:
            payload.pop("response_format", None)
            r = await self._http.post("/chat/completions", content=_json_bytes(payload), headers=_JSON_CT)
        if not r.is_success:
            print(f"[sarvam-chat] {r.status_code} error: {r.text[:500]}", flush=True)
        r.raise_for_status()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        data = r.json()

        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason", "stop")
        usage = data.get("usage", {})
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        cost = (inp * _IN_USD_PER_TOK + out * _OUT_USD_PER_TOK) * _USD_TO_INR
        # confidence proxy: stop→1.0, length→0.5 (ran out of budget), else→0.3
        conf = {"stop": 1.0, "length": 0.5}.get(finish_reason, 0.3)

        return ModelResponse(
            text=text,
            model=SARVAM_MODEL,
            input_tokens=inp,
            output_tokens=out,
            latency_ms=latency_ms,
            cost_inr=round(cost, 6),
            confidence=conf,
        )

    def _stub_chat(self, user: str) -> ModelResponse:
        return ModelResponse(
            text='{"action":"wait","thought":"stub","confidence":0.9}',
            model=f"{SARVAM_MODEL}-stub",
            input_tokens=180,
            output_tokens=20,
            latency_ms=80,
            cost_inr=0.003,
            confidence=0.9,
        )

    # ── legacy stub methods kept for ModelRouter.complete() ───────────────────

    async def sarvam_m(self, prompt: str) -> StubModelResponse:
        await asyncio.sleep(0.08)
        return StubModelResponse(
            text=f"Stub Sarvam-M plan for: {prompt[:80]}",
            model="sarvam-m-stub",
            vision_tokens=180,
            context_tokens=520,
            latency_ms=260,
            cost_inr=0.42,
        )

    async def saarika(self, audio_url: str) -> StubModelResponse:
        await asyncio.sleep(0.05)
        return StubModelResponse(
            text=f"Stub Saarika transcript for {audio_url}",
            model="saarika-stub",
            vision_tokens=0,
            context_tokens=240,
            latency_ms=180,
            cost_inr=0.16,
        )

    # ── Phase 4: voice intent extraction (not wired into action cascade) ─────

    async def extract_intent(self, transcript: str) -> dict:
        """Extract structured intent from a voice transcript using Sarvam-M.

        Returns a dict with keys: action, target, train_number, confidence.
        Reasoning tokens are an asset here — they improve structured extraction.
        Called only in Phase 4 (voice → intent); never in the action cascade.
        """
        system = (
            "Extract the user's intent from the voice transcript. "
            "Return ONLY valid JSON with keys: "
            '{"action": str, "target": str, "train_number": str|null, "confidence": float}. '
            "Strip any <think>...</think> blocks before the JSON."
        )
        resp = await self.chat(system=system, user=transcript, max_tokens=512)
        text = resp.text.strip()
        # Strip reasoning blocks if present
        import re as _re
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
        if not text:
            text = _re.sub(r"<think>.*$", "", resp.text, flags=_re.DOTALL).strip()
        try:
            import json as _json
            return _json.loads(text)
        except Exception:
            return {"action": "unknown", "target": "", "train_number": None, "confidence": 0.0}

    async def bulbul(self, text: str) -> StubModelResponse:
        await asyncio.sleep(0.05)
        return StubModelResponse(
            text=f"Stub Bulbul speech for: {text[:80]}",
            model="bulbul-stub",
            vision_tokens=0,
            context_tokens=180,
            latency_ms=190,
            cost_inr=0.18,
        )

    # ── Phase 4: real Saarika / Bulbul / NTES intent implementations ──────────

    async def saarika_transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "hi-IN",
    ) -> dict:
        """Transcribe audio via Saarika v2.5.

        Returns {"transcript": str, "language_detected": str, "duration_seconds": float}.
        """
        if not self._live:
            return {
                "transcript": "Train 22691 abhi kahan hai",
                "language_detected": language_code,
                "duration_seconds": 2.5,
            }
        fname, mime = _sniff_audio(audio_bytes)
        print(f"[saarika] sending {len(audio_bytes)} bytes as {fname} ({mime})", flush=True)
        body, ct = _build_multipart(
            {"model": "saarika:v2.5", "language_code": language_code, "mode": "transcribe"},
            fname, audio_bytes, mime,
        )
        t0 = time.perf_counter()
        try:
            r = await self._api_http.post(
                "/speech-to-text",
                content=body,
                headers={"Content-Type": ct},
            )
        except Exception as net_exc:
            print(f"[saarika] network exception: {type(net_exc).__name__}: {net_exc}", flush=True)
            raise
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if not r.is_success:
            print(f"[saarika] {r.status_code} error: {r.text[:500]}", flush=True)
            r.raise_for_status()
        resp = r.json()
        transcript = " ".join((resp.get("transcript") or "").split())  # normalise whitespace
        lang_detected = resp.get("language_code", language_code)
        print(f"  [saarika] {transcript!r} lang={lang_detected} {latency_ms}ms", flush=True)
        return {
            "transcript": transcript,
            "language_detected": lang_detected,
            "duration_seconds": float(resp.get("duration_seconds") or 0.0),
        }

    async def bulbul_synthesize(
        self,
        text: str,
        language_code: str = "hi-IN",
        speaker: str = "",   # empty = use first (female) speaker for the language
        pace: float = 1.0,
    ) -> bytes:
        """Synthesise speech via Bulbul v2.  Returns raw WAV bytes."""
        if not self._live:
            return b""
        speakers = _BULBUL_SPEAKERS.get(language_code, _BULBUL_DEFAULT_SPEAKERS)
        if not speaker or speaker not in speakers:
            speaker = speakers[0]   # always female — female voices listed first
        payload = {
            "text": text[:2500],
            "target_language_code": language_code,
            "speaker": speaker,
            "model": "bulbul:v2",
            "pace": pace,
            "output_audio_codec": "wav",
        }
        t0 = time.perf_counter()
        r = await self._api_http.post("/text-to-speech", content=_json_bytes(payload), headers=_JSON_CT)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if not r.is_success:
            print(f"[bulbul] {r.status_code} error: {r.text[:500]}", flush=True)
            # Retry with next-best speaker
            if len(speakers) > 1:
                payload["speaker"] = speakers[1]
                r = await self._api_http.post("/text-to-speech", content=_json_bytes(payload), headers=_JSON_CT)
            if not r.is_success:
                r.raise_for_status()
        import base64 as _b64
        audios = r.json().get("audios") or []
        if not audios:
            print(f"[bulbul] empty audios in response", flush=True)
            return b""
        audio_bytes = _b64.b64decode(audios[0])
        print(f"  [bulbul] {len(audio_bytes)} bytes in {latency_ms}ms", flush=True)
        return audio_bytes

    async def extract_intent(self, transcript: str) -> dict:
        """Unified intent extraction for all supported tasks from any Indian language.

        Returns a dict with keys:
          task, reply_language, confidence,
          train_number (train_status), city (weather),
          product (blinkit), team (cricket).
        """
        system = """\
You are a voice command parser for an Indian AI assistant that handles:
  - Train live running status (NTES)
  - Cricket match scores (Cricbuzz)
  - Weather lookup (wttr.in)
  - Grocery prices on Blinkit

The transcript may be in Hindi, English, Kannada, Telugu, Tamil, Malayalam, or code-mixed.
Return ONLY valid JSON — no markdown, no prose, no <think> tags:
{
  "task": "train_status" | "cricket" | "weather" | "blinkit" | "unknown",
  "reply_language": "hi-IN" | "en-IN" | "kn-IN" | "te-IN" | "ta-IN" | "ml-IN",
  "confidence": 0.0–1.0,
  "train_number": "NNNNN" or null,
  "city": "city name" or null,
  "weather_question": "specific weather aspect asked" or null,
  "product": "product name" or null,
  "cricket_query": "team name, tournament name, or match description" or null
}

Rules:
- train_status: mentions of train/rail/railu/gaadi/train numbers, "kahan hai", "late", "running status",
  "ಯಾವ station", "எங்கே", spoken numbers ("baais chhe nau ek" = 22691)
- cricket: "cricket", "match", "score", "IPL", "test", "wicket", "run", team/player names,
  "ODI", "T20", "finals", "ಕ್ರಿಕೆಟ್", "क्रिकेट"
- weather: "weather", "mausam", "barish", "rain", "garmi", "temperature", city + "mein kaisa",
  "ಹವಾಮಾನ", "வானிலை", "వాతావరణం"
- blinkit: "Blinkit", product names for instant delivery (milk/doodh/haalu, bread, eggs/anda,
  vegetables, fruit, groceries), "kitna ka hai" for grocery items
- train numbers: detect written digits ("22691") and spoken forms in any language
- city for weather: extract city name IN ENGLISH (e.g. "Bangalore", "Mumbai"). null if not mentioned.
- weather_question: IN ENGLISH, the specific aspect asked — "will it rain", "temperature", "humidity", etc.
  null if just general weather. Examples: "will it rain today", "is it hot", "rain chance".
- product for blinkit: extract the product name IN ENGLISH (e.g. "milk", "eggs", "bread").
- cricket_query: extract the FULL subject IN ENGLISH — team name (India, CSK), tournament (IPL, World Cup),
  or specific match description (IPL Finals 2026, India vs Australia). Include year if mentioned.
  ALWAYS write this field in English, even if the transcript is in Hindi/Kannada/Tamil.
  Examples: "IPL Finals 2026", "India vs Australia", "CSK", "World Cup final". null if general query.
- Language detection: Kannada script/words → "kn-IN", Hindi → "hi-IN",
  Telugu → "te-IN", Tamil → "ta-IN", Malayalam → "ml-IN", else "en-IN"
- confidence = 1.0 if task + params are clear, 0.7 if task clear but params inferred, 0.3 if uncertain"""

        resp = await self.chat(system=system, user=f"Transcript: {transcript}", max_tokens=512)
        raw = resp.text.strip()
        import re as _re, json as _json

        text = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        if not text or text.startswith("<think>"):
            text = _re.sub(r"<think>.*$", "", raw, flags=_re.DOTALL).strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()

        try:
            result = _json.loads(text)
            print(f"  [intent] {result}", flush=True)
            return result
        except _json.JSONDecodeError:
            pass

        for candidate in (text + "}", text + '", "confidence": 0.5}'):
            try:
                result = _json.loads(candidate)
                if "task" in result:
                    print(f"  [intent] rescued {result}", flush=True)
                    return result
            except _json.JSONDecodeError:
                pass

        task_m  = _re.search(r'"task"\s*:\s*"([^"]+)"', raw)
        lang_m  = _re.search(r'"reply_language"\s*:\s*"([^"]+)"', raw)
        conf_m  = _re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
        if task_m:
            return {
                "task":           task_m.group(1),
                "reply_language": lang_m.group(1) if lang_m else "hi-IN",
                "confidence":     float(conf_m.group(1)) if conf_m else 0.5,
                "train_number":     None,
                "city":             None,
                "weather_question": None,
                "product":          None,
                "cricket_query":    None,
            }

        print(f"  [intent] parse error: {raw[:200]!r}", flush=True)
        return {"task": "unknown", "reply_language": "hi-IN", "confidence": 0.0,
                "train_number": None, "city": None, "product": None, "cricket_query": None}

    async def generate_voice_reply(
        self,
        task_type: str,
        result_snippet: str,
        reply_language: str,
        success: bool,
        user_question: str = "",
    ) -> str:
        """Use Sarvam-M to directly answer the user's question in their language."""
        if not self._live:
            return self._stub_reply(task_type, reply_language, success)

        if not success or not result_snippet.strip():
            return {
                "hi-IN": "Kaam nahi ho paya. Thodi der baad dobara try karein.",
                "kn-IN": "Kaarya aagalyilla. Swalpa samayadanantara matte try maadi.",
                "ta-IN": "Seyal nadakkavillai. Thayavu seithu meedum muyarchi seyyungal.",
                "te-IN": "Pani jarigindi ledu. Daya cheshu malli prayatninchandi.",
                "ml-IN": "Pravshanam nadannilla. Daya vech veeendum shramikkuka.",
            }.get(reply_language, "Sorry, could not complete the task. Please try again.")

        system = (
            f"You are a voice assistant. Reply in {reply_language}. "
            "Write EXACTLY ONE sentence — no more. "
            "For yes/no questions start with haan/nahi (Hindi), haudu/illa (Kannada), or yes/no. "
            "Never repeat yourself. Never restate the question."
        )
        question_part = f'Question: "{user_question}"\n' if user_question else ""
        # Keep snippet short — large input causes repetition loops
        snippet = result_snippet[:250]
        try:
            resp = await self.chat(
                system=system,
                user=f"{question_part}Facts: {snippet}",
                max_tokens=45,
                temperature=0.0,
            )
            text = _dedup_reply(resp.text.strip())
            text = re.split(r"(?<=[।.!?])\s+", text, maxsplit=1)[0].strip()
            if text:
                print(f"  [voice_reply] {text!r}", flush=True)
                return text
        except Exception as exc:
            print(f"  [voice_reply] generation failed: {exc}", flush=True)

        return {
            "hi-IN": "Kaam ho gaya. Result mil gaya.",
            "kn-IN": "Kaarya aagide. Result sigiide.",
            "en-IN": "Done. Result found.",
        }.get(reply_language, "Task completed.")

    def _stub_reply(self, task_type: str, reply_language: str, success: bool) -> str:
        if not success:
            return "Could not complete the task. Please try again."
        stubs = {
            "train_status": {"hi-IN": "Train ka status check ho gaya.", "kn-IN": "Train status check aaagide.", "en-IN": "Train status checked."},
            "cricket":      {"hi-IN": "Cricket score mil gaya.", "kn-IN": "Cricket score sigiide.", "en-IN": "Cricket score found."},
            "weather":      {"hi-IN": "Mausam ki jaankari mil gayi.", "kn-IN": "Havamana mahiti sigiide.", "en-IN": "Weather info found."},
            "blinkit":      {"hi-IN": "Blinkit pe daam mil gaya.", "kn-IN": "Blinkit price sigiide.", "en-IN": "Blinkit price found."},
        }
        return stubs.get(task_type, {}).get(reply_language, "Task completed.")

    async def extract_intent_for_ntes(self, transcript: str) -> dict:
        """Extract structured intent for NTES train-status queries.

        Handles Hindi, English, and code-mixed transcripts.
        Returns {"task": str, "train_number": str|null,
                 "reply_language": str, "confidence": float}.
        """
        system = """\
You are a voice command parser for Indian Railways NTES queries.
Extract intent from the transcript (may be Hindi, English, or code-mixed).
Return ONLY valid JSON — no markdown, no prose, no <think> tags:
{
  "task": "train_status" or "unknown",
  "train_number": "NNNNN" (4-5 digit train number) or null,
  "reply_language": "hi-IN" or "en-IN" or "kn-IN",
  "confidence": 0.0–1.0
}
Rules:
- Detect train numbers as digits ("22691"), spoken digits ("twenty two six nine one"),
  or Hindi spoken form ("baais chhe nau ek").
- If the query is about any train status/location/delay, task = "train_status".
- Detect language: Hindi → "hi-IN", Kannada → "kn-IN", otherwise "en-IN".
- confidence = 1.0 if train number present and clear, 0.5 if inferred, 0.2 if missing."""
        # 1024 tokens: Sarvam-M emits a <think> block (~400 tok) before the JSON (~100 tok).
        resp = await self.chat(system=system, user=f"Transcript: {transcript}", max_tokens=1024)
        raw  = resp.text.strip()
        import re as _re, json as _json

        # Strip complete <think>...</think> blocks.
        text = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        # Strip partial (unclosed) think block — model ran out of tokens mid-think.
        if not text or text.startswith("<think>"):
            text = _re.sub(r"<think>.*$", "", raw, flags=_re.DOTALL).strip()
        # Strip markdown fences.
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()

        # Try full JSON parse first.
        fallback_train_number = _train_number_from_transcript(transcript)
        try:
            result = _json.loads(text)
            if fallback_train_number:
                result["train_number"] = fallback_train_number
            print(f"  [intent_ntes] {result}", flush=True)
            return result
        except _json.JSONDecodeError:
            pass

        # Rescue: try appending "}" in case the JSON was truncated mid-object.
        for candidate in (text + "}", text + '", "confidence": 0.8}'):
            try:
                result = _json.loads(candidate)
                if "task" in result:
                    if fallback_train_number:
                        result["train_number"] = fallback_train_number
                    print(f"  [intent_ntes] rescued {result}", flush=True)
                    return result
            except _json.JSONDecodeError:
                pass

        # Rescue: extract key fields from the raw text with regex.
        task_m   = _re.search(r'"task"\s*:\s*"([^"]+)"', raw)
        num_m    = _re.search(r'"train_number"\s*:\s*"?(\d{4,5})"?', raw)
        lang_m   = _re.search(r'"reply_language"\s*:\s*"([^"]+)"', raw)
        conf_m   = _re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
        if task_m:
            result = {
                "task":           task_m.group(1),
                "train_number":   fallback_train_number or (num_m.group(1) if num_m else None),
                "reply_language": lang_m.group(1) if lang_m else "hi-IN",
                "confidence":     float(conf_m.group(1)) if conf_m else 0.7,
            }
            print(f"  [intent_ntes] regex-rescued {result}", flush=True)
            return result

        if fallback_train_number:
            result = {
                "task": "train_status",
                "train_number": fallback_train_number,
                "reply_language": "hi-IN",
                "confidence": 0.8,
            }
            print(f"  [intent_ntes] transcript-fallback {result}", flush=True)
            return result

        print(f"  [intent_ntes] parse error: {raw[:200]!r}", flush=True)
        return {"task": "unknown", "train_number": None, "reply_language": "hi-IN", "confidence": 0.0}
