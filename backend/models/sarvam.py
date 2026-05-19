"""Sarvam AI client.

Calls https://api.sarvam.ai/v1/chat/completions (OpenAI-compatible).
Falls back to stub responses when SARVAM_API_KEY is absent so the optimized
agent can be exercised in mock mode without any credentials.
"""
from __future__ import annotations

import asyncio
import os
import time
import warnings
from dataclasses import dataclass

import httpx
import urllib3

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SARVAM_BASE = "https://api.sarvam.ai/v1"


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
SARVAM_MODEL = "sarvam-m"   # legacy 24-B; kept for backward compat with codebase refs

# Known-good Bulbul speaker IDs per language.  Primary first; rest are fallbacks.
_BULBUL_SPEAKERS: dict[str, list[str]] = {
    "hi-IN": ["anushka", "arvind", "amol", "diya"],
    "en-IN": ["anushka", "arvind", "meera"],
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
            "response_format": {"type": "json_object"},   # suppress think blocks
        }
        r = await self._http.post("/chat/completions", json=payload)
        if r.status_code == 400 and "response_format" in r.text:
            # API doesn't support response_format — retry without it
            payload.pop("response_format", None)
            r = await self._http.post("/chat/completions", json=payload)
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
        import io as _io
        fname, mime = _sniff_audio(audio_bytes)
        print(f"[saarika] sending {len(audio_bytes)} bytes as {fname} ({mime})", flush=True)
        files = {"file": (fname, _io.BytesIO(audio_bytes), mime)}
        data = {"model": "saarika:v2.5", "language_code": language_code, "mode": "transcribe"}
        t0 = time.perf_counter()
        r = await self._api_http.post("/speech-to-text", files=files, data=data)
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
        speaker: str = "anushka",
        pace: float = 1.0,
    ) -> bytes:
        """Synthesise speech via Bulbul v2.  Returns raw WAV bytes."""
        if not self._live:
            return b""
        speakers = _BULBUL_SPEAKERS.get(language_code, _BULBUL_DEFAULT_SPEAKERS)
        if speaker not in speakers:
            speaker = speakers[0]
        payload = {
            "text": text[:2500],
            "target_language_code": language_code,
            "speaker": speaker,
            "model": "bulbul:v2",
            "pace": pace,
            "output_audio_codec": "wav",
        }
        t0 = time.perf_counter()
        r = await self._api_http.post("/text-to-speech", json=payload)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if not r.is_success:
            print(f"[bulbul] {r.status_code} error: {r.text[:500]}", flush=True)
            # Retry with next-best speaker
            if len(speakers) > 1:
                payload["speaker"] = speakers[1]
                r = await self._api_http.post("/text-to-speech", json=payload)
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
        try:
            result = _json.loads(text)
            print(f"  [intent_ntes] {result}", flush=True)
            return result
        except _json.JSONDecodeError:
            pass

        # Rescue: try appending "}" in case the JSON was truncated mid-object.
        for candidate in (text + "}", text + '", "confidence": 0.8}'):
            try:
                result = _json.loads(candidate)
                if "task" in result:
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
                "train_number":   num_m.group(1) if num_m else None,
                "reply_language": lang_m.group(1) if lang_m else "hi-IN",
                "confidence":     float(conf_m.group(1)) if conf_m else 0.7,
            }
            print(f"  [intent_ntes] regex-rescued {result}", flush=True)
            return result

        print(f"  [intent_ntes] parse error: {raw[:200]!r}", flush=True)
        return {"task": "unknown", "train_number": None, "reply_language": "hi-IN", "confidence": 0.0}
