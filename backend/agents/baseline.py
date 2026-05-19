"""Baseline NTES agent - deliberately unoptimised strawman.

Every step:
  - full 1920×1080 PNG screenshot (no downsample)
  - entire conversation history sent to the model (no trimming)
  - always uses Qwen3-VL-235B via OpenRouter (no cascade)
  - freeform CoT reasoning, coordinate-based actions (no SoM marks)
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from pathlib import Path

from pydantic import BaseModel

from backend.instrumentation.meter import BudgetExceededError, MeterEvent, RunMeter
from backend.models.openrouter import DEFAULT_MODEL as _BASELINE_OR_MODEL, OpenRouterClient
from backend.tasks.ntes import DOWNLOADS_DIR, NTESTask, extract_status_string

# Each agent saves to its own sub-directory so concurrent runs don't
# cross-contaminate success checks.
_AGENT_DL = DOWNLOADS_DIR / "baseline"


MODEL_LABEL = "qwen3-vl-235b via openrouter"
NAIVE_MAX_STEPS = 25
COMPETENT_MAX_STEPS = 12

NAIVE_SYSTEM_PROMPT = """\
You are a web browser automation agent. Navigate the NTES live train running \
status website for train 22691.

You will receive a 1920x1080 screenshot. Output ONE JSON action on the final line.

Action format:
  {"action":"click",  "x":760, "y":340,   "thought":"...", "confidence":0.9}
  {"action":"type",   "text":"22691",      "thought":"...", "confidence":0.9}
  {"action":"press",  "key":"Enter",       "thought":"...", "confidence":0.95}
  {"action":"scroll", "direction":"down",  "thought":"...", "confidence":0.7}
  {"action":"goto",   "url":"https://...", "thought":"...", "confidence":0.8}
  {"action":"wait",                        "thought":"...", "confidence":0.6}

Find the train status using the screenshot and choose the next browser action.
"""

COMPETENT_SYSTEM_PROMPT = """\
You are a web browser automation agent. Your sole job is to navigate the NTES \
live train running status website and answer the user's train status goal.

You will receive a 1920×1080 screenshot. Output ONE JSON action on the final line.

Action format (x and y are PLAIN integers — never use a list):
  {"action":"click",  "x":760, "y":340,   "thought":"...", "confidence":0.9}
  {"action":"type",   "text":"22691",      "thought":"...", "confidence":0.9}
  {"action":"press",  "key":"Enter",       "thought":"...", "confidence":0.95}
  {"action":"scroll", "direction":"down",  "thought":"...", "confidence":0.7}
  {"action":"goto",   "url":"https://...", "thought":"...", "confidence":0.8}
  {"action":"wait",                        "thought":"...", "confidence":0.6}
  {"action":"done",                        "thought":"...", "confidence":1.0}
  {"action":"done", "extracted_answer":"<one sentence describing current station and delay>"}

Navigation path:
  1. Start from the NTES live train status page.
  2. Find the train search field, type train number 22691, and submit.
  3. Open the live running status result for train 22691.
  4. Read the current station and delay or on-time status.
  5. Signal "done" only after the visible page contains the requested answer.

Termination: The task is complete when the page shows the train's current \
location AND on-time/delay status. As soon as you see this information on \
screen, output:
{"action": "done", "extracted_answer": "<one sentence describing current station and delay>"}
Do not continue clicking after the answer is visible.
"""

# ── mock action cycle (no API calls needed) ───────────────────────────────────
_MOCK_CYCLE: list[dict] = [
    {"action": "goto",   "url": "https://enquiry.indianrail.gov.in/mntes/",
     "thought": "navigate to live train status", "confidence": 0.9},
    {"action": "wait",   "thought": "page loading",     "confidence": 0.9},
    {"action": "type",   "text": "22691", "thought": "enter train number", "confidence": 0.85},
    {"action": "press",  "key": "Enter", "thought": "submit train query", "confidence": 0.8},
    {"action": "scroll", "direction": "down", "thought": "look for running status", "confidence": 0.7},
    {"action": "wait",   "thought": "settling",         "confidence": 0.6},
]


class AgentResult(BaseModel):
    agent: str
    success: bool
    elapsed_seconds: float
    final_step: str
    total_input_tokens: int  = 0
    total_output_tokens: int = 0
    total_cost_inr: float    = 0.0
    steps: int               = 0
    extracted_status_string: str = ""


class BaselineAgent:
    name = "baseline"

    def __init__(self, meter: RunMeter, mock: bool = False, mode: str = "naive") -> None:
        self.meter = meter
        self.mock  = mock
        self.mode = mode
        self.name = f"baseline_{mode}" if mode in {"naive", "competent"} else "baseline"
        if not mock:
            self._openrouter = OpenRouterClient()

    async def run(self, task: NTESTask) -> AgentResult:
        from playwright.async_api import async_playwright

        _AGENT_DL.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        total_inp = total_out = 0
        total_cost: float = 0.0
        step = 0
        success = False
        final_step_desc = "max_steps_reached"
        final_page_text = ""

        # Full conversation history — never trimmed (deliberate baseline cost)
        conversation: list[dict] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=not task.headed)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                accept_downloads=True,
            )
            page = await context.new_page()

            async def _on_download(dl) -> None:
                dest = _AGENT_DL / dl.suggested_filename
                await dl.save_as(str(dest))

            page.on("download", _on_download)

            # When a link opens a new tab, follow it and treat it as the current page.
            async def _on_popup(popup) -> None:
                nonlocal page
                await popup.wait_for_load_state("domcontentloaded")
                popup.on("download", _on_download)
                page = popup

            context.on("page", _on_popup)

            try:
                await page.goto(task.start_url, wait_until="domcontentloaded", timeout=30_000)

                goal_prefix = (
                    f"Goal: {task.objective}\n"
                    f"Train number: {task.train_number}\n"
                )

                max_steps = COMPETENT_MAX_STEPS if self.mode == "competent" else NAIVE_MAX_STEPS
                prompt = (
                    COMPETENT_SYSTEM_PROMPT if self.mode == "competent" else NAIVE_SYSTEM_PROMPT
                ).replace("22691", task.train_number)
                recent_signatures: list[str] = []

                while step < max_steps:
                    step += 1

                    page_body = await page.inner_text("body")
                    final_page_text = page_body
                    if task.success(page_body):
                        success = True
                        final_step_desc = f"status_found_before_step_{step}"
                        break

                    # --- DELIBERATE BASELINE COST: full 1920×1080 PNG every step ---
                    screenshot_bytes = await page.screenshot(full_page=False)
                    screenshot_b64   = base64.standard_b64encode(screenshot_bytes).decode()
                    current_url = page.url
                    print(
                        f"[baseline step {step:02d}/{max_steps}]  "
                        f"url={current_url}  img={len(screenshot_bytes)//1024}KB",
                        flush=True,
                    )

                    if self.mock:
                        action = _MOCK_CYCLE[(step - 1) % len(_MOCK_CYCLE)]
                        inp_tok = out_tok = 0
                        step_cost = 0.0
                        latency_ms = 0
                        print(f"  [mock] {action}", flush=True)
                    else:
                        # --- DELIBERATE BASELINE COST: full history every call ---
                        user_msg: dict = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"{goal_prefix}"
                                        f"Step {step}/{max_steps}. URL: {current_url}\n"
                                        "Analyse the screenshot and output your action JSON."
                                    ),
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": screenshot_b64,
                                    },
                                },
                            ],
                        }
                        conversation.append(user_msg)

                        call_start = time.perf_counter()
                        resp = await self._openrouter.chat(
                            system=prompt,
                            messages=conversation,   # no trimming
                            max_tokens=1024,
                            model=_BASELINE_OR_MODEL,
                            temperature=0.0,
                        )
                        latency_ms = int((time.perf_counter() - call_start) * 1000)

                        inp_tok, out_tok = resp.input_tokens, resp.output_tokens
                        step_cost = resp.cost_inr
                        action = _parse_action(resp.text)
                        conversation.append({"role": "assistant", "content": resp.text})

                        thought = action.get("thought", resp.text[:80])
                        print(f"  Qwen: {thought}", flush=True)
                        print(
                            f"  Action: {action}  "
                            f"[{inp_tok}in/{out_tok}out tok, {latency_ms}ms, Rs.{step_cost:.4f}]",
                            flush=True,
                        )

                    total_inp  += inp_tok
                    total_out  += out_tok
                    total_cost += step_cost

                    try:
                        await self.meter.publish_model_usage(
                            agent=self.name,
                            current_step=f"step_{step}",
                            model=MODEL_LABEL,
                            vision_tokens=inp_tok,
                            context_tokens=out_tok,
                            latency_ms=latency_ms,
                            cost_inr=step_cost,
                        )
                    except BudgetExceededError as e:
                        final_step_desc = f"budget_exceeded_step_{step}"
                        print(f"  [budget] {e}", flush=True)
                        break

                    if self.mode == "competent":
                        sig = _action_signature(action)
                        if len(recent_signatures) >= 2 and recent_signatures[-1] == sig and recent_signatures[-2] == sig:
                            forced = _force_different_action(action)
                            print(
                                f"  [baseline-competent] duplicate action suppressed: {sig} -> {forced}",
                                flush=True,
                            )
                            action = forced
                            sig = _action_signature(action)
                        recent_signatures.append(sig)
                        recent_signatures = recent_signatures[-2:]

                    if self.mode == "competent" and action.get("action") == "type":
                        intended = str(action.get("text", ""))
                        if intended and await _page_has_input_value(page, intended):
                            print(
                                f"  [baseline] skip type; field already contains {intended}, pressing Enter",
                                flush=True,
                            )
                            action = {"action": "press", "key": "Enter"}

                    done = await _execute_action(page, action)
                    if done:
                        await asyncio.sleep(2)
                        page_body = await page.inner_text("body")
                        final_page_text = page_body
                        success = task.success(page_body)
                        final_step_desc = f"done_signal_step_{step}_status={'yes' if success else 'no'}"
                        break

                    await asyncio.sleep(1)

            except BudgetExceededError as e:
                final_step_desc = f"budget_exceeded_step_{step}"
                print(f"  [budget] {e}", flush=True)
            except Exception as exc:
                final_step_desc = f"error_step_{step}: {exc}"
                await self.meter.publish(
                    MeterEvent(
                        run_id=self.meter.run_id,
                        agent=self.name,
                        event="error",
                        current_step=f"step_{step}",
                        detail=str(exc),
                    )
                )
            finally:
                await context.close()
                await browser.close()

        elapsed = time.perf_counter() - started
        await self.meter.publish(
            MeterEvent(
                run_id=self.meter.run_id,
                agent=self.name,
                event="agent_complete",
                elapsed_seconds=round(elapsed, 2),
                current_step=final_step_desc,
                current_model=MODEL_LABEL,
            )
        )
        return AgentResult(
            agent=self.name,
            success=success,
            elapsed_seconds=round(elapsed, 2),
            final_step=final_step_desc,
            total_input_tokens=total_inp,
            total_output_tokens=total_out,
            total_cost_inr=round(total_cost, 4),
            steps=step,
            extracted_status_string=extract_status_string(final_page_text),
        )


# ── action parsing ─────────────────────────────────────────────────────────────

def _parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    for m in reversed(list(re.finditer(r"\{[^{}]+\}", text))):
        try:
            obj = json.loads(m.group())
            if "action" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return {"action": "wait", "thought": "parse_error", "confidence": 0.5}


# ── action execution ───────────────────────────────────────────────────────────

def _coord(action: dict) -> tuple[int, int]:
    """Extract x,y from action — handles Qwen returning coords as a list."""
    x, y = action.get("x", 0), action.get("y", 0)
    if isinstance(x, list) and len(x) >= 2:
        return int(x[0]), int(x[1])
    if isinstance(x, list):
        x = x[0] if x else 0
    if isinstance(y, list):
        y = y[0] if y else 0
    return int(x or 0), int(y or 0)


def _action_signature(action: dict) -> str:
    kind = str(action.get("action", ""))
    if kind == "click":
        return f"click:{_coord(action)}"
    if kind == "type":
        return f"type:{action.get('text', '')}"
    if kind == "press":
        return f"press:{action.get('key', '')}"
    if kind == "scroll":
        return f"scroll:{action.get('direction', '')}"
    if kind == "goto":
        return f"goto:{action.get('url', '')}"
    return kind


def _force_different_action(action: dict) -> dict:
    kind = action.get("action", "")
    if kind == "type":
        return {"action": "press", "key": "Enter", "thought": "duplicate type suppressed"}
    if kind == "click":
        return {"action": "press", "key": "Enter", "thought": "duplicate click suppressed"}
    if kind == "wait":
        return {"action": "scroll", "direction": "down", "thought": "duplicate wait suppressed"}
    if kind == "press":
        return {"action": "wait", "thought": "duplicate keypress suppressed"}
    return {"action": "wait", "thought": "duplicate action suppressed"}


async def _page_has_input_value(page, value: str) -> bool:
    try:
        return bool(
            await page.evaluate(
                """wanted => Array.from(document.querySelectorAll('input, textarea'))
                    .some(el => String(el.value || '').trim().includes(wanted))""",
                value,
            )
        )
    except Exception:
        return False


async def _execute_action(page, action: dict) -> bool:
    """Execute action; return True on DONE signal. Swallows exceptions."""
    kind = action.get("action", "wait")
    try:
        if kind == "click":
            cx, cy = _coord(action)
            await page.mouse.click(cx, cy)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=8_000)
            except Exception:
                pass
        elif kind == "type":
            await page.keyboard.type(str(action.get("text", "")), delay=25)
        elif kind == "press":
            await page.keyboard.press(str(action.get("key", "Enter")))
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=8_000)
            except Exception:
                pass
        elif kind == "scroll":
            d = action.get("direction", "down")
            delta = 400
            dx = delta if d == "right" else (-delta if d == "left" else 0)
            dy = delta if d == "down"  else (-delta if d == "up"   else 0)
            await page.mouse.wheel(dx, dy)
        elif kind == "goto":
            await page.goto(str(action.get("url", "")), wait_until="domcontentloaded", timeout=30_000)
        elif kind == "wait":
            await asyncio.sleep(2)
        elif kind == "done":
            return True
    except Exception as exc:
        print(f"[baseline] action '{kind}' raised: {exc}", flush=True)
    return False
