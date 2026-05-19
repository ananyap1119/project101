"""OpenRouter client — multi-model, per-model pricing.

Accepts Anthropic-format messages (with base64 image blocks) and converts them
to the OpenAI-compatible format that OpenRouter expects.

Provider pinning:
  - anthropic/* models  → pinned to Anthropic native API (required for base64 images)
  - all other models    → default OpenRouter routing (no pinning)
"""
from __future__ import annotations

import os
import time
import warnings

import httpx
import urllib3

from backend.models.sarvam import ModelResponse

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Default model (used by the baseline agent)
DEFAULT_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"

# Per-model pricing: (input $/MTok, output $/MTok)
_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4-5":           (3.000,  15.000),
    "qwen/qwen3-vl-235b-a22b-instruct":      (0.200,   0.880),
    "qwen/qwen3-vl-235b-a22b-thinking":      (0.260,   2.600),
    "qwen/qwen3-vl-32b-instruct":            (0.104,   0.416),
    "deepseek/deepseek-v3.2":                (0.252,   0.378),
    "deepseek/deepseek-v3.2-exp":            (0.270,   0.410),
    "qwen/qwen3-next-80b-a3b-thinking":      (0.098,   0.780),
    "qwen/qwen3-235b-a22b":                  (0.455,   1.820),
}
_DEFAULT_PRICING = (0.500, 2.000)   # conservative fallback for unknown models

_USD_TO_INR = 85.0


def _cost_inr(model: str, inp: int, out: int) -> float:
    price_in, price_out = _PRICING.get(model, _DEFAULT_PRICING)
    cost_usd = (inp * price_in + out * price_out) / 1_000_000
    return round(cost_usd * _USD_TO_INR, 6)


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Convert Anthropic-format (system + messages) → OpenAI messages list."""
    result: list[dict] = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            result.append({"role": role, "content": str(content)})
            continue

        parts: list[dict] = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif btype == "image":
                src = block.get("source", {})
                if src.get("type") == "base64":
                    media = src.get("media_type", "image/png")
                    data  = src.get("data", "")
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{data}"},
                    })
                elif src.get("type") == "url":
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": src.get("url", "")},
                    })

        if parts:
            result.append({"role": role, "content": parts})

    return result


class OpenRouterClient:
    def __init__(self) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._http = httpx.AsyncClient(
            base_url=OPENROUTER_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/northstarz/cua-bench",
                "X-Title": "cua-bench",
            },
            verify=False,   # corporate TLS-inspection proxy workaround
            timeout=120.0,
        )

    async def chat(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        model: str = DEFAULT_MODEL,
        json_mode: bool = False,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> ModelResponse:
        openai_msgs = _to_openai_messages(system, messages)
        payload: dict = {
            "model": model,
            "messages": openai_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # Anthropic's API requires provider pinning to accept base64 data-URL images.
        # Other providers (Qwen, DeepSeek) route correctly without pinning.
        if model.startswith("anthropic/"):
            payload["provider"] = {"order": ["Anthropic"], "allow_fallbacks": False}

        t0 = time.perf_counter()
        r = await self._http.post("/chat/completions", json=payload)
        if not r.is_success:
            body = r.text[:500]
            print(f"[openrouter:{model}] {r.status_code} error: {body}", flush=True)
            r.raise_for_status()
        latency_ms = int((time.perf_counter() - t0) * 1000)

        data = r.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        usage = data.get("usage", {})
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)

        return ModelResponse(
            text=text,
            model=model,
            input_tokens=inp,
            output_tokens=out,
            latency_ms=latency_ms,
            cost_inr=_cost_inr(model, inp, out),
            confidence=1.0,
        )
