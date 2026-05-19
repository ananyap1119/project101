"""Claude client — shared between baseline and optimised agents.

Priority order:
  1. OPENROUTER_API_KEY set  → calls api.openrouter.ai (OpenAI-compat)
  2. ANTHROPIC_API_KEY set   → calls api.anthropic.com directly
  3. Neither set             → stub responses (mock / test mode)
"""
from __future__ import annotations

import asyncio
import os
import warnings

import urllib3

from backend.models.sarvam import ModelResponse, StubModelResponse

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SONNET_4_5 = "claude-sonnet-4-5-20250514"


class ClaudeClient:
    def __init__(self) -> None:
        if os.environ.get("OPENROUTER_API_KEY"):
            from backend.models.openrouter import OpenRouterClient
            self._backend = OpenRouterClient()
            self._mode = "openrouter"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self._backend = _AnthropicBackend()
            self._mode = "anthropic"
        else:
            self._backend = None
            self._mode = "stub"

    async def chat(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        model: str = SONNET_4_5,
    ) -> ModelResponse:
        if self._mode == "stub":
            return self._stub_chat()
        return await self._backend.chat(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            model=model,
        )

    def _stub_chat(self) -> ModelResponse:
        return ModelResponse(
            text='{"action":"wait","thought":"stub","confidence":1.0}',
            model=f"{SONNET_4_5}-stub",
            input_tokens=320,
            output_tokens=30,
            latency_ms=100,
            cost_inr=0.085,
            confidence=1.0,
        )

    # ── legacy stub kept for ModelRouter.complete() ───────────────────────────
    async def complete(self, prompt: str) -> StubModelResponse:
        await asyncio.sleep(0.1)
        return StubModelResponse(
            text=f"Stub Claude action for: {prompt[:80]}",
            model="claude-sonnet-stub",
            vision_tokens=320,
            context_tokens=760,
            latency_ms=340,
            cost_inr=0.73,
        )


# ── Thin Anthropic-SDK backend (used only when no OpenRouter key) ─────────────

class _AnthropicBackend:
    def __init__(self) -> None:
        import httpx
        import anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            http_client=httpx.AsyncClient(verify=False),
        )

    async def chat(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int,
        model: str,
    ) -> ModelResponse:
        import time
        _IN  = 3.0  / 1_000_000
        _OUT = 15.0 / 1_000_000
        _INR = 85.0
        t0 = time.perf_counter()
        resp = await self._client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        inp, out = resp.usage.input_tokens, resp.usage.output_tokens
        return ModelResponse(
            text=resp.content[0].text,
            model=model,
            input_tokens=inp,
            output_tokens=out,
            latency_ms=latency_ms,
            cost_inr=round((inp * _IN + out * _OUT) * _INR, 6),
            confidence=1.0,
        )
