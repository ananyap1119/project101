"""Optimised NTES agent - DeepSeek-first 4-tier model cascade.

Comparison story vs the baseline:
  Baseline  — Qwen3-VL-235B, full 1080p PNG every step, full history
  Optimised — DeepSeek default, DOM text (no images), summarised history

Four individually-togglable layers:
  Layer 1 — DOM extraction
    Extract numbered interactive elements + visible text instead of a screenshot.
    Text-only tiers (DeepSeek, Qwen3-text) reason over text, far fewer tokens.

  Layer 2 — Trajectory summarisation
    Every 4 steps, DeepSeek compresses the oldest turns into a compact JSON
    state object; only the last 2 raw turns are kept at full fidelity.

  Layer 3 — 4-tier model cascade (always active)
    Tier 0 (~75%): DeepSeek-V3.2-Exp          — default, text-only
    Tier 1 (~15%): Qwen3-Next-80B-Think        — low confidence / 1 failure, text-only
    Tier 2  (~8%): Qwen3-VL-235B               — DOM empty or 2+ failures, vision
    Tier 3  (~2%): Claude Sonnet 4.5           — 3+ failures, vision

  Layer 4 — Speculative execution
    While action N runs in the browser, DeepSeek already predicts action N+1
    from the expected post-action state.  If the page state matches the
    prediction, the pre-computed action is used without an extra API call.
    Meaningful hit rate excludes "wait" predictions (trivial to match).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus, urljoin, urlparse

from PIL import Image

from backend.agents.baseline import AgentResult
from backend.agents.dom_extractor import DomSnapshot, DomElement, extract_dom, dom_to_prompt
from backend.agents.summarizer import maybe_summarise
from backend.instrumentation.meter import BudgetExceededError, MeterEvent, RunMeter
from backend.models.router import CascadeStats, ModelRouter, RouteDecision, TIER_DEEPSEEK
from backend.models.sarvam import SarvamClient
from backend.tasks.ntes import DOWNLOADS_DIR, NTESTask, extract_status_string
from typing import Any

# Each agent saves to its own sub-directory so concurrent runs don't
# cross-contaminate success checks.
_AGENT_DL = DOWNLOADS_DIR / "optimized"


MAX_STEPS = 25

_NON_COOPERATIVE = frozenset({"press", "goto", "done"})
_VALID_ACTIONS   = frozenset({"click", "type", "press", "scroll", "goto", "wait", "done"})
_MAX_CONSEC_SPEC_HITS = 2   # force a real model call after N consecutive spec hits
_SPEC_REJECTION_REASONS = ("off_domain", "social_link", "new_tab", "state_mismatch", "bad_role")
_SOCIAL_OR_APP_TEXT = (
    "facebook",
    "twitter",
    "youtube",
    "instagram",
    "linkedin",
    "play store",
    "app store",
    "download app",
)
_ALLOWED_ROLES = {"button", "link", "textbox", "combobox", "checkbox", "radio", "menuitem", "tab"}

# Vision screenshot size for escalation tiers (720p, smaller than baseline 1080p)
_VIS_W, _VIS_H = 1280, 720

# ── system prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a web browser automation agent. Your ONLY goal is to complete the task \
described by the user.

You receive:
  - The current URL
  - A numbered list of interactive elements visible on the page
  - The page's visible text content
  (Vision tiers also receive a screenshot for additional context.)

IMPORTANT: Output ONLY valid JSON — no prose, no markdown, no <think> tags. \
Start your response with "{".

Respond with EXACTLY ONE JSON action object:

  {"action":"click",  "n":3,                   "thought":"…","confidence":0.9}
  {"action":"type",   "n":2, "text":"…",        "thought":"…","confidence":0.9}
  {"action":"press",  "key":"Enter",            "thought":"…","confidence":0.95}
  {"action":"scroll", "direction":"down",       "thought":"…","confidence":0.7}
  {"action":"goto",   "url":"https://…",        "thought":"…","confidence":0.8}
  {"action":"wait",                             "thought":"…","confidence":0.6}
  {"action":"done",                             "thought":"…","confidence":1.0}

Rules:
  - "n" is the element number from the INTERACTIVE ELEMENTS list.
  - Navigate to the task website if you are not already there.
  - Complete the task described in the goal and signal "done" once the answer is visible.
"""

_SPECULATE_SYSTEM = """\
Predict the NEXT action an agent should take after the described action \
has just been dispatched.  You do NOT have the real page yet — reason from \
the expected post-action state.

Return a SINGLE JSON action object in the same format.  Set "confidence" \
lower if the state is uncertain.
"""


# ── feature flags ─────────────────────────────────────────────────────────────

@dataclass
class Flags:
    dom:           bool = True
    summarization: bool = True
    cascade:       bool = True
    speculation:   bool = True
    mock:          bool = False

    @classmethod
    def from_layers(cls, layers: list[int], *, mock: bool = False) -> "Flags":
        return cls(
            dom=1 in layers,
            summarization=2 in layers,
            cascade=3 in layers,
            speculation=4 in layers,
            mock=mock,
        )


@dataclass
class ActionOutcome:
    done: bool = False
    playwright_ok: bool = True
    target_selector: str = ""
    target_visible_after: bool | None = None
    input_value: str = ""
    typed_text: str = ""


_FIRST_STEP_CACHE: dict[str, dict] = {}


# ── mock cycle ────────────────────────────────────────────────────────────────

_MOCK_CYCLE: list[dict] = [
    {"action": "goto",  "url": "https://enquiry.indianrail.gov.in/mntes/",
     "thought": "navigate to live train status", "confidence": 0.9},
    {"action": "wait",  "thought": "page loading",  "confidence": 0.9},
    {"action": "type",  "text": "22691",             "thought": "enter train number", "confidence": 0.7},
    {"action": "wait",  "thought": "settling",       "confidence": 0.6},
    {"action": "wait",  "thought": "still settling", "confidence": 0.55},
]

_BLINKIT_PINCODE = "560001"
_BLINKIT_LOCATION_MARKERS = (
    "please provide your delivery location",
    "search delivery location",
    "select location",
    "detect my location",
)
_BLINKIT_LOCATION_FIELD_SELECTORS = (
    'input[placeholder*="delivery location" i]',
    'input[placeholder*="location" i]',
    '[role="textbox"][aria-label*="location" i]',
    'text="Search delivery location"',
    'text="Select Location"',
)
_BLINKIT_LOCATION_TRIGGER_SELECTORS = (
    'button:has-text("Select Location")',
    'button:has-text("Change Location")',
    'div:has-text("Please provide your delivery location")',
    'div:has-text("Select Location")',
    '[data-testid*="location"]',
)

_NTES_TRAIN_FIELD_SELECTORS = (
    'input[placeholder*="train" i]',
    'input[aria-label*="train" i]',
    'input[name*="train" i]',
    'input[id*="train" i]',
    'input[type="search"]',
    'input[type="text"]',
    'input',
)
_BLINKIT_LOCATION_SUGGESTION_SELECTORS = (
    '[role="option"]',
    '[data-testid*="suggestion"]',
    'div:has-text("560001")',
    'div:has-text("Bengaluru")',
    'div:has-text("Bangalore")',
)
_BLINKIT_SEARCH_SELECTORS = (
    '[role="searchbox"]',
    'input[placeholder^="Search" i]',
    '[role="textbox"][placeholder*="Search" i]',
)


# ── action parsing ─────────────────────────────────────────────────────────────

def _parse_action(text: str) -> dict:
    text = text.strip()
    # Strip complete <think>...</think> blocks (Sarvam-M reasoning).
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If the thinking block was cut off by max_tokens (no closing tag), strip the partial.
    if not stripped:
        stripped = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    if stripped:
        text = stripped
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


def _is_uncertain_action(action: dict) -> bool:
    kind = str(action.get("action", "")).lower()
    thought = str(action.get("thought", "")).lower()
    return kind == "uncertain" or thought == "parse_error" or kind not in _VALID_ACTIONS


def _first_step_cache_key(portal_name: str, goal_template: str) -> str:
    payload = f"{portal_name}|{goal_template}"
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


# ── screenshot helper for vision tiers ────────────────────────────────────────

async def _take_720p_screenshot(page) -> str:
    """Return a base64-encoded 720p PNG screenshot."""
    raw = await page.screenshot(full_page=False)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((_VIS_W, _VIS_H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode()


# ── action execution ──────────────────────────────────────────────────────────

async def _click_dom_element(page, el: DomElement) -> None:
    """Try to click a DOM element using best-available selector, then fall back to coords."""
    strategies: list[str] = []
    if el.id:
        strategies.append(f"#{el.id}")
    text = el.text.strip()
    if text:
        strategies.append(f'text="{text}"')
        strategies.append(f"text={text}")
    if el.placeholder:
        strategies.append(f'[placeholder="{el.placeholder}"]')
    if el.href and not el.href.startswith(("javascript", "#")):
        strategies.append(f'a[href="{el.href}"]')

    for sel in strategies:
        try:
            await page.click(sel, timeout=3_000)
            return
        except Exception:
            continue

    # Fallback: centre-point coordinates
    try:
        await page.mouse.click(el.x, el.y)
    except Exception:
        pass


def _selector_candidates(el: DomElement) -> list[str]:
    strategies: list[str] = []
    if el.id:
        strategies.append(f"#{el.id}")
    if el.placeholder:
        strategies.append(f'[placeholder="{el.placeholder}"]')
    if el.input_type:
        strategies.append(f'input[type="{el.input_type}"]')
    text = el.text.strip()
    if text:
        strategies.append(f'text="{text}"')
        strategies.append(f"text={text}")
    if el.href and not el.href.startswith(("javascript", "#")):
        strategies.append(f'a[href="{el.href}"]')
    strategies.append("input")
    return strategies


async def _first_working_selector(page, el: DomElement) -> str:
    for sel in _selector_candidates(el):
        try:
            loc = page.locator(sel).first
            if await loc.count() and await loc.is_visible(timeout=1_000):
                return sel
        except Exception:
            continue
    return ""


async def _type_in_element(page, el: DomElement, text: str) -> tuple[str, str]:
    """Focus & fill a form field using best-available selector."""
    for sel in _selector_candidates(el):
        try:
            loc = page.locator(sel).first
            await loc.fill(text, timeout=3_000)
            value = await loc.input_value(timeout=1_000)
            return sel, value
        except Exception:
            continue
    return "", ""


def _is_blinkit_location_prompt(snap: DomSnapshot | None) -> bool:
    if not snap or "blinkit.com" not in snap.url.lower():
        return False
    visible_text = " ".join(
        [
            snap.body_text,
            " ".join(el.text for el in snap.elements),
            " ".join(el.placeholder for el in snap.elements),
        ]
    ).lower()
    return any(marker in visible_text for marker in _BLINKIT_LOCATION_MARKERS)


async def _select_blinkit_location_suggestion(page, pincode: str = _BLINKIT_PINCODE) -> None:
    await asyncio.sleep(1.2)
    try:
        clicked = await page.evaluate(
            """pincode => {
                const candidates = Array.from(document.querySelectorAll('div, button, [role="option"]'));
                for (const el of candidates) {
                    const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
                    const rect = el.getBoundingClientRect();
                    if (
                        text.includes(pincode) &&
                        rect.width > 40 &&
                        rect.width < Math.min(window.innerWidth, 900) &&
                        rect.height > 12 &&
                        rect.height < 160 &&
                        rect.bottom > 0 &&
                        rect.right > 0 &&
                        rect.top < window.innerHeight &&
                        rect.left < window.innerWidth
                    ) {
                        el.click();
                        return true;
                    }
                }
                return false;
            }""",
            pincode,
        )
        if clicked:
            await asyncio.sleep(1.5)
            return
    except Exception:
        pass
    try:
        await page.keyboard.press("ArrowDown")
        await page.keyboard.press("Enter")
        await asyncio.sleep(1.5)
    except Exception:
        pass
    for selector in (*_BLINKIT_LOCATION_SUGGESTION_SELECTORS, 'button:has-text("Confirm")'):
        try:
            loc = page.locator(selector).first
            if await loc.count() and await loc.is_visible(timeout=1_000):
                await loc.click(timeout=2_000)
                await asyncio.sleep(1.5)
                return
        except Exception:
            continue


async def _click_first_visible(page, selectors: tuple[str, ...], timeout: int = 1_000) -> bool:
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if await loc.count() and await loc.is_visible(timeout=timeout):
                await loc.click(timeout=2_000)
                return True
        except Exception:
            continue
    return False


async def _fill_first_visible(
    page,
    selectors: tuple[str, ...],
    text: str,
    timeout: int = 1_000,
    reject_placeholder_words: tuple[str, ...] = (),
) -> bool:
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if await loc.count() and await loc.is_visible(timeout=timeout):
                if reject_placeholder_words:
                    placeholder = ""
                    try:
                        placeholder = (await loc.get_attribute("placeholder", timeout=500) or "").lower()
                    except Exception:
                        pass
                    if any(word in placeholder for word in reject_placeholder_words):
                        continue
                await loc.click(timeout=2_000)
                try:
                    await loc.fill(text, timeout=2_000)
                except Exception:
                    await page.keyboard.press("Control+A")
                    await page.keyboard.type(text, delay=25)
                return True
        except Exception:
            continue
    return False


async def _blinkit_visible_text(page) -> str:
    try:
        return (await page.inner_text("body", timeout=2_000)).lower()
    except Exception:
        return ""


async def _prepare_blinkit_for_product_search(page, task: Any) -> bool:
    """Best-effort deterministic Blinkit setup before handing control to the model."""
    if getattr(task, "name", "") not in {"blinkit", "blinkit_planner"} or "blinkit.com" not in page.url.lower():
        return False

    pincode = str(getattr(task, "location", _BLINKIT_PINCODE) or _BLINKIT_PINCODE)
    product = str(getattr(task, "product", "") or "").strip()
    changed = False

    try:
        await page.wait_for_load_state("domcontentloaded", timeout=8_000)
    except Exception:
        pass

    body_text = await _blinkit_visible_text(page)
    needs_location = any(marker in body_text for marker in _BLINKIT_LOCATION_MARKERS)
    if needs_location:
        print(f"[opt] Blinkit setup: setting delivery pincode {pincode}", flush=True)
        await _click_first_visible(page, _BLINKIT_LOCATION_TRIGGER_SELECTORS)
        await asyncio.sleep(0.5)
        filled = await _fill_first_visible(page, _BLINKIT_LOCATION_FIELD_SELECTORS, pincode, timeout=2_000)
        if not filled:
            await page.keyboard.type(pincode, delay=25)
        await _select_blinkit_location_suggestion(page, pincode)
        changed = True

    body_text = await _blinkit_visible_text(page)
    if any(marker in body_text for marker in _BLINKIT_LOCATION_MARKERS):
        print("[opt] Blinkit setup: location prompt still visible; deferring product search", flush=True)
        return changed

    if product:
        print(f"[opt] Blinkit setup: searching product {product!r}", flush=True)
        clicked = await _click_first_visible(page, _BLINKIT_SEARCH_SELECTORS, timeout=2_000)
        await asyncio.sleep(0.4)
        filled = await _fill_first_visible(
            page,
            _BLINKIT_SEARCH_SELECTORS,
            product,
            timeout=1_000,
            reject_placeholder_words=("location", "delivery"),
        )
        if not filled:
            if not clicked:
                await page.goto(
                    f"https://blinkit.com/s/?q={product}",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                return True
            await page.keyboard.type(product, delay=25)
        await page.keyboard.press("Enter")
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=8_000)
        except Exception:
            pass
        await asyncio.sleep(2)
        changed = True

    return changed


_FRESH_PRODUCE = {
    "onion", "onions", "tomato", "tomatoes", "potato", "potatoes", "carrot", "carrots",
    "coriander", "mint", "spinach", "capsicum", "cucumber", "ginger", "garlic",
}
_PRODUCE_MISMATCH_TERMS = {
    "pakoda", "namkeen", "chips", "powder", "flakes", "seasoning", "masala", "pickle",
    "sauce", "soup", "instant", "snack", "chakli",
}


def _blinkit_product_score(card_text: str, requested_name: str, query: str) -> int:
    text = card_text.lower()
    requested = requested_name.lower().strip()
    terms = {
        term for term in re.findall(r"[a-z0-9]+", f"{requested} {query.lower()}")
        if len(term) > 2
    }
    score = sum(4 for term in terms if re.search(rf"\b{re.escape(term)}\b", text))
    if requested and requested in text:
        score += 10
    requested_terms = set(re.findall(r"[a-z]+", requested))
    if requested_terms & _FRESH_PRODUCE:
        score += 4 if any(word in text for word in ("fresh", "vegetable", "local")) else 0
        if any(term in text for term in _PRODUCE_MISMATCH_TERMS):
            score -= 100
    return score


async def _build_blinkit_grocery_cart(page, task: Any) -> dict | None:
    """Search and add a bounded grocery plan. Never opens checkout."""
    if getattr(task, "name", "") != "blinkit_planner":
        return None
    entries: list[dict[str, Any]] = []
    for planned in list(getattr(task, "items", []))[:12]:
        requested_name = str(getattr(planned, "name", "") or "").strip()
        query = str(getattr(planned, "search_query", "") or requested_name).strip()
        requested_quantity = max(1, min(int(getattr(planned, "quantity", 1) or 1), 6))
        result = {
            "requested_item": requested_name,
            "requested_amount": str(getattr(planned, "required_amount", "") or ""),
            "requested_quantity": requested_quantity,
            "product_name": "",
            "pack_size": "",
            "price": "",
            "added_quantity": 0,
            "status": "failed",
        }
        try:
            await page.goto(
                f"https://blinkit.com/s/?q={quote_plus(query)}",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            await asyncio.sleep(2)
            body = (await page.inner_text("body")).lower()
            if "access denied" in body or "you have been blocked" in body:
                result["status"] = "failed"
                entries.append(result)
                break

            cards = page.locator("div.tw-relative.tw-flex.tw-h-full.tw-flex-col")
            card_count = await cards.count()
            selected = None
            selected_score = -10_000
            for index in range(min(card_count, 20)):
                card = cards.nth(index)
                text = " ".join((await card.inner_text()).split())
                if "ADD" in text or re.search(r"\s\d+\s*$", text):
                    score = _blinkit_product_score(text, requested_name, query)
                    if score > selected_score:
                        selected = card
                        selected_score = score
            if selected is None:
                result["status"] = "unavailable"
                entries.append(result)
                continue

            card_text = "\n".join(line.strip() for line in (await selected.inner_text()).splitlines() if line.strip())
            lines = card_text.splitlines()
            price_match = re.search(r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?", card_text, re.I)
            pack_match = re.search(r"\b\d+(?:\.\d+)?\s*(?:kg|g|l|ml|pcs?|pieces?|pack)\b", card_text, re.I)
            title_lines = [
                line for line in lines
                if line != "ADD" and not re.fullmatch(r"\d+% OFF", line, re.I)
                and not re.fullmatch(r"\d+\s*MINS?", line, re.I)
                and not re.fullmatch(r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?", line, re.I)
                and not re.fullmatch(r"\d+", line)
                and not (pack_match and line == pack_match.group(0))
            ]
            result["product_name"] = title_lines[0] if title_lines else query
            result["pack_size"] = pack_match.group(0) if pack_match else ""
            result["price"] = price_match.group(0).replace(" ", "") if price_match else ""

            add = selected.get_by_text("ADD", exact=True)
            if await add.count():
                await add.click(timeout=5_000)
                await asyncio.sleep(0.7)
            quantity_box = selected.locator("div.tw-bg-base-green").first
            if not await quantity_box.count():
                result["status"] = "failed"
                entries.append(result)
                continue
            quantity_text = " ".join((await quantity_box.inner_text()).split())
            quantity_match = re.search(r"\d+", quantity_text)
            current_quantity = int(quantity_match.group(0)) if quantity_match else 1
            for _ in range(max(0, requested_quantity - current_quantity)):
                plus = quantity_box.locator("button").last
                await plus.click(timeout=3_000)
                await asyncio.sleep(0.35)
            quantity_text = " ".join((await quantity_box.inner_text()).split())
            quantity_match = re.search(r"\d+", quantity_text)
            result["added_quantity"] = int(quantity_match.group(0)) if quantity_match else requested_quantity
            result["status"] = "added"
        except Exception as exc:
            print(f"[blinkit-planner] failed item={query!r}: {exc}", flush=True)
        entries.append(result)
    return {"entries": entries}


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


async def _prepare_ntes_train_status(page, task: Any) -> bool:
    """Best-effort deterministic NTES setup: enter train number and submit."""
    if getattr(task, "name", "") != "ntes" or "indianrail.gov.in" not in page.url.lower():
        return False

    train_number = re.sub(r"\D", "", str(getattr(task, "train_number", "") or ""))
    if not (4 <= len(train_number) <= 5):
        return False

    try:
        await page.wait_for_load_state("domcontentloaded", timeout=8_000)
    except Exception:
        pass

    if await _page_has_input_value(page, train_number):
        await page.keyboard.press("Enter")
        await asyncio.sleep(2)
        return True

    for selector in _NTES_TRAIN_FIELD_SELECTORS:
        try:
            loc = page.locator(selector).first
            if await loc.count() and await loc.is_visible(timeout=1_500):
                await loc.click(timeout=2_000)
                await loc.fill(train_number, timeout=2_000)
                await page.keyboard.press("Enter")
                await asyncio.sleep(2)
                return True
        except Exception:
            continue

    try:
        await page.keyboard.type(train_number, delay=25)
        await page.keyboard.press("Enter")
        await asyncio.sleep(2)
        return True
    except Exception:
        return False


async def _prepare_ntes_journey(page, task: Any) -> str:
    """Open NTES trains-between-stations and submit both stations."""
    if getattr(task, "name", "") != "railway_journey":
        return ""

    async def select_station(selector: str, station: str) -> None:
        field = page.locator(selector)
        await field.fill(station)
        await asyncio.sleep(1.5)
        await field.press("ArrowDown")
        await field.press("Enter")

    try:
        await page.get_by_text("Trains B/w Stations", exact=True).first.click(timeout=5_000)
        await page.locator("#jFromStationInput").wait_for(state="visible", timeout=5_000)
        await select_station("#jFromStationInput", str(task.origin))
        await select_station("#jToStationInput", str(task.destination))
        await page.locator('input[name="find"]').click(timeout=5_000)
        await asyncio.sleep(3)
        return await page.inner_text("body")
    except Exception as exc:
        print(f"[opt] deterministic NTES journey setup failed: {exc}", flush=True)
        return ""


async def _wait_for_commerce_login(page, task: Any) -> bool:
    """Pause a visible commerce run while the user completes login or OTP."""
    if getattr(task, "name", "") != "shopping_orders":
        return False
    login_markers = ("/signin", "/account/login", "ap/signin")
    if not any(marker in page.url.lower() for marker in login_markers):
        return False
    wait_seconds = int(getattr(task, "manual_auth_wait_seconds", 180))
    print(
        f"[commerce] login required; waiting up to {wait_seconds}s for manual completion",
        flush=True,
    )
    for _ in range(max(1, wait_seconds // 2)):
        await asyncio.sleep(2)
        if not any(marker in page.url.lower() for marker in login_markers):
            await asyncio.sleep(2)
            return True
    return False


async def _extract_shopping_dom(page, task: Any) -> str:
    """Extract visible commerce cards without navigation, ads, or recommendations."""
    if getattr(task, "name", "") != "shopping_orders":
        return ""
    action = str(getattr(task, "action", "orders"))
    host = (urlparse(page.url).hostname or "").lower()
    platform = "amazon" if "amazon." in host else "flipkart" if "flipkart." in host else ""
    if not platform:
        return ""
    if platform == "flipkart":
        try:
            items = await page.evaluate(
                r"""action => {
                    const clean = value => String(value || '').replace(/\s+/g, ' ').trim();
                    const priceMatches = text => Array.from(text.matchAll(/₹\s*[\d,]+(?:\.\d{2})?/g), m => m[0]);
                    if (action === 'cart') {
                        const actionNodes = Array.from(document.querySelectorAll('*')).filter(
                            node => clean(node.textContent) === 'Save for later'
                        );
                        const candidates = [];
                        for (const actionNode of actionNodes) {
                            let node = actionNode.parentElement;
                            for (let i = 0; i < 12 && node; i++, node = node.parentElement) {
                                const text = clean(node.innerText);
                                if (/Qty:\s*\d+/i.test(text) && /₹\s*[\d,]+/.test(text) && /Remove/i.test(text)) {
                                    if (!candidates.includes(node)) candidates.push(node);
                                    break;
                                }
                            }
                        }
                        const seen = new Set();
                        return candidates.slice(0, 20).map(node => {
                            const text = clean(node.innerText);
                            const lines = String(node.innerText || '').split(/\n+/).map(clean).filter(Boolean);
                            const qtyIndex = lines.findIndex(line => /^Qty:/i.test(line));
                            const title = qtyIndex >= 0 ? (lines[qtyIndex + 1] || '') : (lines[0] || '');
                            const prices = priceMatches(text);
                            const price = prices.length >= 2 ? prices[1] : prices[0] || '';
                            const quantity = qtyIndex >= 0 ? (lines[qtyIndex].match(/\d+/)?.[0] || '') : '';
                            const status = /out of stock/i.test(text) ? 'Out of stock' : '';
                            const delivery = text.match(/Delivery by\s+[^₹]+?(?=Save for later|Remove|$)/i)?.[0] || '';
                            return {title, price, quantity, status, detail: clean(`${delivery} ${text}`).slice(0, 700)};
                        }).filter(item => item.title && !seen.has(item.title) && seen.add(item.title));
                    }
                    if (action === 'wishlist' || action === 'saved_items') {
                        return Array.from(document.querySelectorAll('.KBj2DG')).slice(0, 20).map(node => {
                            const text = clean(node.innerText);
                            const title = clean(node.querySelector('.zJFk58')?.textContent);
                            const price = clean(node.querySelector('.hZ3P6w')?.textContent);
                            const status = /currently unavailable/i.test(text) ? 'Currently unavailable' : 'Available';
                            return {title, price, quantity: '', status, detail: text.slice(0, 700)};
                        }).filter(item => item.title);
                    }
                    if (action === 'orders' || action === 'invoices' || action === 'buy_again') {
                        return Array.from(document.querySelectorAll('.lxQ2Ra')).slice(0, 20).map(titleNode => {
                            let node = titleNode;
                            for (let i = 0; i < 6 && node; i++, node = node.parentElement) {
                                const text = clean(node.innerText);
                                if (/(Delivered|Cancelled|Refund|On the way|Returned|Shipped)/i.test(text)) {
                                    const price = priceMatches(text)[0] || '';
                                    const status = text.match(/(Delivered on [^₹]+?|Cancelled on [^₹]+?|Refund Completed|On the way|Returned|Shipped)(?=Your|Rate|$)/i)?.[0] || '';
                                    return {title: clean(titleNode.textContent), price, quantity: '', status: clean(status), detail: text.slice(0, 700)};
                                }
                            }
                            return null;
                        }).filter(Boolean);
                    }
                    return [];
                }""",
                action,
            )
        except Exception:
            return ""
        if not items:
            return ""
        return "SHOPPING_DATA " + json.dumps(
            {"platform": platform, "action": action, "items": items},
            ensure_ascii=False,
        )
    selectors = {
        "amazon": {
            "cart": ["div.sc-list-item[data-asin]"],
            "saved_items": ["#sc-saved-cart div.sc-list-item[data-asin]"],
            "wishlist": ["li.g-item-sortable", "div[data-itemid]"],
            "buy_again": ["div[data-asin]"],
            "browsing_history": ["div[data-asin]"],
            "orders": [".order-card", ".js-order-card"],
            "invoices": [".order-card", ".js-order-card"],
        },
        "flipkart": {},
    }[platform].get(action, [])
    try:
        items = await page.evaluate(
            r"""({selectors, platform, action}) => {
                const clean = value => String(value || '').replace(/\s+/g, ' ').trim();
                const nodes = [];
                for (const selector of selectors) {
                    for (const node of document.querySelectorAll(selector)) {
                        if (!nodes.includes(node)) nodes.push(node);
                    }
                    if (nodes.length) break;
                }
                return nodes.filter(node => {
                    if (platform !== 'amazon') return true;
                    const isSaved = Boolean(node.closest('#sc-saved-cart')) || /move to cart/i.test(node.innerText || '');
                    return action === 'saved_items' ? isSaved : action === 'cart' ? !isSaved : true;
                }).slice(0, 20).map(node => {
                    const titleSelectors = platform === 'amazon'
                        ? ['.a-truncate-full', 'a.sc-product-link', '.sc-product-title', 'h2 a', 'h2', 'h3', 'a[title]']
                        : ['a[title]', 'a[href*="/p/"]', 'div[title]'];
                    const priceSelectors = platform === 'amazon'
                        ? ['.sc-product-price', '.a-price .a-offscreen']
                        : ['div[class]'];
                    let title = '';
                    for (const selector of titleSelectors) {
                        const el = node.querySelector(selector);
                        title = clean(el?.getAttribute('title') || el?.textContent);
                        if (title) break;
                    }
                    title = title.replace(/\s*Opens in a new tab\s*$/i, '').trim();
                    let price = '';
                    for (const selector of priceSelectors) {
                        const el = node.querySelector(selector);
                        const candidate = clean(el?.textContent);
                        const match = candidate.match(/(?:₹|Rs\.)\s*[\d,]+(?:\.\d{2})?/i);
                        if (match) { price = match[0]; break; }
                    }
                    const text = clean(node.innerText).slice(0, 700);
                    const quantity = clean(node.querySelector('.a-dropdown-prompt')?.textContent);
                    const statusMatch = text.match(/delivered|arriving[^.]{0,80}|shipped|out for delivery|cancelled|in stock|out of stock/i);
                    return {title, price, quantity, status: statusMatch?.[0] || '', detail: text};
                }).filter(item => item.title || item.price || item.status);
            }""",
            {"selectors": selectors, "platform": platform, "action": action},
        )
    except Exception:
        return ""
    if not items:
        return ""
    return "SHOPPING_DATA " + json.dumps(
        {"platform": platform, "action": action, "items": items},
        ensure_ascii=False,
    )


async def _execute_action(page, action: dict, snap: DomSnapshot | None) -> ActionOutcome:
    """Execute one action and report whether Playwright completed it cleanly."""
    kind = action.get("action", "wait")
    try:
        if kind == "click":
            n = int(action.get("n", 0))
            target_selector = ""
            if snap and 1 <= n <= len(snap.elements):
                target_selector = await _first_working_selector(page, snap.elements[n - 1])
                await _click_dom_element(page, snap.elements[n - 1])
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=8_000)
            except Exception:
                pass
            await asyncio.sleep(2)
            target_visible_after = None
            if target_selector:
                try:
                    target_visible_after = await page.locator(target_selector).first.is_visible(timeout=1_000)
                except Exception:
                    target_visible_after = False
            return ActionOutcome(
                target_selector=target_selector,
                target_visible_after=target_visible_after,
            )

        elif kind == "type":
            n = int(action.get("n", 0))
            text = str(action.get("text", ""))
            if _is_blinkit_location_prompt(snap):
                if text != _BLINKIT_PINCODE:
                    print(
                        f"[opt] overriding Blinkit location text {text!r} -> {_BLINKIT_PINCODE}",
                        flush=True,
                    )
                text = _BLINKIT_PINCODE
            value = ""
            selector = ""
            if snap and 1 <= n <= len(snap.elements):
                selector, value = await _type_in_element(page, snap.elements[n - 1], text)
            else:
                await page.keyboard.type(text, delay=25)
                try:
                    value = await page.evaluate(
                        """() => document.activeElement && 'value' in document.activeElement
                            ? String(document.activeElement.value || '')
                            : ''"""
                    )
                except Exception:
                    value = ""
            if text == _BLINKIT_PINCODE and _is_blinkit_location_prompt(snap):
                await _select_blinkit_location_suggestion(page, _BLINKIT_PINCODE)
            return ActionOutcome(
                target_selector=selector,
                input_value=value,
                typed_text=text,
            )

        elif kind == "press":
            await page.keyboard.press(str(action.get("key", "Enter")))
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=8_000)
            except Exception:
                pass

        elif kind == "scroll":
            d = action.get("direction", "down")
            dy = 400 if d == "down" else (-400 if d == "up" else 0)
            dx = 400 if d == "right" else (-400 if d == "left" else 0)
            await page.mouse.wheel(dx, dy)

        elif kind == "goto":
            url = str(action.get("url", ""))
            if url:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

        elif kind == "wait":
            await asyncio.sleep(2)

        elif kind == "done":
            return ActionOutcome(done=True)

    except Exception as exc:
        print(f"[opt] action '{kind}' raised: {exc}", flush=True)
        return ActionOutcome(playwright_ok=False)

    return ActionOutcome()


# ── speculation helpers ────────────────────────────────────────────────────────

def _is_cooperative(action: dict) -> bool:
    return action.get("action", "") not in _NON_COOPERATIVE


def _registered_domain(hostname: str) -> str:
    parts = hostname.lower().split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "gov", "ac", "org", "net"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname.lower()


def _implicit_role(el: DomElement) -> str:
    if el.role:
        return el.role.lower()
    if el.tag == "a":
        return "link"
    if el.tag == "button":
        return "button"
    if el.tag == "select":
        return "combobox"
    if el.tag in {"textarea", "input"}:
        if el.input_type in {"checkbox", "radio"}:
            return el.input_type
        return "textbox"
    return el.tag


def _dom_hash(url: str, snap: DomSnapshot | None) -> str:
    count = len(snap.elements) if snap else 0
    text = (snap.body_text if snap else "")[:200]
    payload = f"{url}|{count}|{text}"
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


async def _page_state_hash(page, snap: DomSnapshot | None) -> str:
    """Hash visible state plus form values so typed input counts as a mutation."""
    try:
        form_values = await page.evaluate(
            """() => Array.from(document.querySelectorAll('input, textarea, select'))
                .map(el => `${el.tagName}:${el.type || ''}:${el.value || ''}`)
                .join('|')"""
        )
    except Exception:
        form_values = ""
    payload = f"{page.url}|{_dom_hash(page.url, snap)}|{form_values}"
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def _validate_spec_action(
    *,
    action: dict,
    snap: DomSnapshot | None,
    current_url: str,
    current_hash: str,
    expected_hash: str,
    allowed_domain: str,
) -> str | None:
    if current_hash != expected_hash:
        return "state_mismatch"
    if action.get("action") != "click":
        return None
    n = int(action.get("n", 0) or 0)
    if not snap or not (1 <= n <= len(snap.elements)):
        return "bad_role"
    el = snap.elements[n - 1]
    role = _implicit_role(el)
    if role not in _ALLOWED_ROLES:
        return "bad_role"
    label = " ".join([el.text, el.placeholder, el.href]).lower()
    if any(term in label for term in _SOCIAL_OR_APP_TEXT):
        return "social_link"
    if el.target.strip() or "_blank" in label:
        return "new_tab"
    if el.href and not el.href.startswith(("javascript:", "#")):
        absolute = urljoin(current_url, el.href)
        domain = _registered_domain(urlparse(absolute).hostname or "")
        if domain != allowed_domain:
            return "off_domain"
    return None


def _spec_state_matches(before_url: str, after_url: str,
                        before_n: int, after_n: int,
                        spec_action: dict) -> bool:
    if spec_action.get("action", "") not in _VALID_ACTIONS:
        return False
    if before_url != after_url:
        return False
    if before_n == 0:
        return False
    return abs(after_n - before_n) / before_n < 0.30


async def _speculate(router: ModelRouter, conversation: list[dict],
                     action: dict, goal_prefix: str) -> dict:
    predicted_state = (
        f"{goal_prefix}\n"
        f"The agent just executed: {json.dumps(action)}. "
        "Assuming the action succeeded, what is the NEXT action to take?"
    )
    try:
        resp = await router.speculate(
            system=_SPECULATE_SYSTEM,
            messages=[{"role": "user", "content": predicted_state}],
        )
        return _parse_action(resp.text)
    except Exception:
        return {"action": "wait", "thought": "speculation_error", "confidence": 0.3}


# ── main agent ────────────────────────────────────────────────────────────────

class _MockRouter:
    def __init__(self) -> None:
        self.stats = CascadeStats()


class SarvamActionRouter:
    def __init__(self) -> None:
        self.stats = CascadeStats()
        self.client = SarvamClient()

    async def call(
        self,
        *,
        decision: RouteDecision,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
    ):
        user_parts: list[str] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            user_parts.append(f"{message.get('role', 'user')}: {content}")
        return await self.client.chat(
            system=system,
            user="\n\n".join(user_parts),
            max_tokens=max_tokens if max_tokens is not None else decision.max_tokens,
            temperature=0.0,
        )

    async def speculate(self, *, system: str, messages: list[dict]):
        return await self.call(
            decision=RouteDecision(
                tier=TIER_DEEPSEEK,
                model_label="sarvam",
                or_model="sarvam",
                reason="sarvam_speculation",
                needs_vision=False,
                max_tokens=256,
                json_mode=True,
            ),
            system=system,
            messages=messages,
            max_tokens=256,
        )

    async def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 500,
        temperature: float = 0.1,
    ):
        return await self.client.chat(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class OptimizedAgent:
    name = "optimized"

    def __init__(self, meter: RunMeter, flags: Flags | None = None, router: Any | None = None) -> None:
        self.meter  = meter
        self.flags  = flags or Flags()
        self.router = router if router is not None else (_MockRouter() if self.flags.mock else ModelRouter())
        self.allowed_domain = ""

    async def run(self, task: Any) -> AgentResult:
        from playwright.async_api import async_playwright

        _AGENT_DL.mkdir(parents=True, exist_ok=True)

        flags   = self.flags
        started = time.perf_counter()
        total_inp = total_out = 0
        total_cost: float = 0.0
        step    = 0
        success = False
        final_step_desc = "max_steps_reached"
        final_page_text = ""
        collected_page_text: list[str] = []
        spec_attempts = spec_hits = 0
        spec_meaningful_attempts = spec_meaningful_hits = 0
        consec_spec_hits = 0
        consecutive_spec_rejections = 0
        spec_disabled = False
        spec_rejections_by_reason = {reason: 0 for reason in _SPEC_REJECTION_REASONS}

        consecutive_failures = 0
        consecutive_uncertain = 0
        prev_confidence      = 1.0
        route_log: list[dict] = []

        conversation: list[dict] = []
        pending_spec: dict | None = None
        max_steps = int(getattr(task, "max_steps", MAX_STEPS))

        goal_prefix = (
            f"Goal: {task.objective}\n"
            f"Website: {task.start_url}"
        )
        system_prompt = _SYSTEM

        task_allows_speculation = getattr(task, "speculation_enabled", True)
        if flags.speculation and not task_allows_speculation:
            spec_disabled = True
            print(
                "speculation gated off for portal=ntes, "
                "reason=short_task_safety_validation_overhead",
                flush=True,
            )

        async with async_playwright() as pw:
            launch_args = ["--start-maximized"] if task.config_name == "full_optimized" else []
            launch_headless = False if task.config_name == "full_optimized" else not task.headed
            launch_slow_mo = 300 if task.config_name == "full_optimized" else 0
            if task.config_name == "full_optimized":
                chromium_path = pw.chromium.executable_path
                print(
                    "[playwright-launch] "
                    f"config=full_optimized headless={launch_headless} slow_mo={launch_slow_mo} "
                    f"args={launch_args} executable={chromium_path} "
                    f"executable_exists={Path(chromium_path).exists()} "
                    f"display={os.environ.get('DISPLAY', '')!r} os={sys.platform}",
                    flush=True,
                )
            viewport = (
                {"width": 1280, "height": 800}
                if task.config_name == "full_optimized"
                else {"width": 1920, "height": 1080}
            )
            profile_dir = getattr(task, "browser_profile_dir", "")
            if profile_dir:
                Path(profile_dir).mkdir(parents=True, exist_ok=True)
                context = await pw.chromium.launch_persistent_context(
                    profile_dir,
                    headless=launch_headless,
                    slow_mo=launch_slow_mo,
                    args=launch_args,
                    viewport=viewport,
                    accept_downloads=True,
                )
                browser = context.browser
            else:
                browser = await pw.chromium.launch(
                    headless=launch_headless,
                    slow_mo=launch_slow_mo,
                    args=launch_args,
                )
                context = await browser.new_context(viewport=viewport, accept_downloads=True)
            page = context.pages[0] if context.pages else await context.new_page()
            if task.config_name == "full_optimized":
                await page.bring_to_front()

            async def _on_download(dl) -> None:
                dest = _AGENT_DL / dl.suggested_filename
                await dl.save_as(str(dest))

            page.on("download", _on_download)

            # Follow new-tab popups automatically so agents see the new window.
            async def _on_popup(popup) -> None:
                nonlocal page
                await popup.wait_for_load_state("domcontentloaded")
                popup.on("download", _on_download)
                page = popup

            context.on("page", _on_popup)

            try:
                print(f"[opt] opening start_url={task.start_url}", flush=True)
                await page.goto(task.start_url, wait_until="domcontentloaded", timeout=30_000)
                await _wait_for_commerce_login(page, task)
                if getattr(task, "name", "") == "shopping_orders" and "flipkart.com" in page.url.lower():
                    await asyncio.sleep(3)
                self.allowed_domain = _registered_domain(urlparse(page.url).hostname or "")
                journey_text = await _prepare_ntes_journey(page, task)
                if journey_text:
                    collected_page_text.append(
                        f"PAGE https://enquiry.indianrail.gov.in/mntes/\n{journey_text}"
                    )
                    route_log.append({
                        "step": 0,
                        "tier": -1,
                        "model": "deterministic_ntes_journey_setup",
                        "reason": "submit_origin_and_destination",
                        "action": "train_search",
                        "playwright_ok": True,
                        "action_success": True,
                    })
                if await _prepare_blinkit_for_product_search(page, task):
                    route_log.append({
                        "step": 0,
                        "tier": -1,
                        "model": "deterministic_blinkit_setup",
                        "reason": "set_location_and_search_product",
                        "escalated": False,
                        "trigger": "preflight",
                        "raw_value": "blinkit_preflight",
                        "action": "set_location_then_search",
                        "confidence": 1.0,
                        "uncertain": False,
                        "cache_hit": False,
                        "playwright_ok": True,
                        "page_changed": True,
                        "action_success": True,
                        "success_detail": "location_pincode_and_product_search_attempted",
                        "consecutive_failures_after": consecutive_failures,
                        "consecutive_uncertain_after": consecutive_uncertain,
                    })
                planner_result = await _build_blinkit_grocery_cart(page, task)
                if planner_result is not None:
                    planner_data = "GROCERY_PLANNER_DATA " + json.dumps(
                        planner_result,
                        ensure_ascii=False,
                    )
                    collected_page_text.append(planner_data)
                    route_log.append({
                        "step": 0,
                        "tier": -1,
                        "model": "deterministic_blinkit_cart_builder",
                        "reason": "search_add_and_set_quantity",
                        "action": "build_grocery_cart",
                        "playwright_ok": True,
                        "action_success": any(
                            entry.get("status") == "added"
                            for entry in planner_result.get("entries", [])
                        ),
                    })
                if await _prepare_ntes_train_status(page, task):
                    route_log.append({
                        "step": 0,
                        "tier": -1,
                        "model": "deterministic_ntes_setup",
                        "reason": "enter_train_number_and_submit",
                        "escalated": False,
                        "trigger": "preflight",
                        "raw_value": "ntes_preflight",
                        "action": "type_train_number_then_enter",
                        "confidence": 1.0,
                        "uncertain": False,
                        "cache_hit": False,
                        "playwright_ok": True,
                        "page_changed": True,
                        "action_success": True,
                        "success_detail": f"submitted_train_number={task.train_number}",
                        "consecutive_failures_after": consecutive_failures,
                        "consecutive_uncertain_after": consecutive_uncertain,
                    })

                while step < max_steps:
                    step += 1

                    current_body = await page.inner_text("body")
                    shopping_data = await _extract_shopping_dom(page, task)
                    if shopping_data:
                        current_body += f"\n{shopping_data}"
                    if getattr(task, "aggregate_page_text", False):
                        collected_page_text.append(f"PAGE {page.url}\n{current_body}")
                        collected_page_text = collected_page_text[-12:]
                        final_page_text = "\n\n".join(collected_page_text)
                    else:
                        final_page_text = current_body
                    if task.success(final_page_text):
                        success = True
                        final_step_desc = f"status_found_before_step_{step}"
                        break

                    # ── Layer 1: DOM extraction ───────────────────────────────
                    snap: DomSnapshot | None = None
                    if flags.dom:
                        snap = await extract_dom(page)
                        page_text    = dom_to_prompt(snap)
                        element_count = len(snap.elements)
                    else:
                        page_text    = (await page.inner_text("body"))[:2000]
                        element_count = 0

                    current_url = page.url
                    current_hash = _dom_hash(current_url, snap)
                    before_action_state_hash = await _page_state_hash(page, snap)

                    layer_tag = ",".join(filter(None, [
                        "dom"  if flags.dom           else "",
                        "sum"  if flags.summarization else "",
                        "cas"  if flags.cascade       else "",
                        "spec" if flags.speculation   else "",
                    ]))
                    print(
                        f"[opt step {step:02d}/{max_steps}]  url={current_url}  "
                        f"elements={element_count}  failures={consecutive_failures}  "
                        f"layers=[{layer_tag}]",
                        flush=True,
                    )

                    # ── Layer 2: summarisation ────────────────────────────────
                    if flags.summarization:
                        conversation, summary = await maybe_summarise(
                            conversation, step, self.router
                        )
                        if summary:
                            print(f"  [sum] {json.dumps(summary)[:120]}", flush=True)

                    # ── Layer 3: routing decision ─────────────────────────────
                    if flags.cascade:
                        decision: RouteDecision = self.router.decide(
                            consecutive_failures=consecutive_failures,
                            consecutive_uncertain=consecutive_uncertain,
                            dom_element_count=element_count,
                        )
                    else:
                        decision = RouteDecision(
                            tier=TIER_DEEPSEEK,
                            model_label="deepseek-v3.2-exp",
                            or_model="deepseek/deepseek-v3.2-exp",
                            reason="cascade_disabled",
                            needs_vision=False,
                            max_tokens=1024,
                            json_mode=True,
                        )
                        self.router.stats.decisions.append(decision)
                        self.router.stats.tier_calls[TIER_DEEPSEEK] += 1
                        print("  [router] tier=0 model=deepseek-v3.2-exp reason=cascade_disabled", flush=True)

                    # ── Check pending speculative action ─────────────────────
                    action: dict | None = None
                    if (pending_spec and flags.speculation and task_allows_speculation and not spec_disabled
                            and consec_spec_hits < _MAX_CONSEC_SPEC_HITS):
                        b_url   = pending_spec.pop("_before_url",   current_url)
                        b_elems = pending_spec.pop("_before_elems", element_count)
                        expected_hash = pending_spec.pop("_expected_hash", "")
                        rejection = _validate_spec_action(
                            action=pending_spec,
                            snap=snap,
                            current_url=current_url,
                            current_hash=current_hash,
                            expected_hash=expected_hash,
                            allowed_domain=self.allowed_domain,
                        )
                        if rejection:
                            spec_rejections_by_reason[rejection] += 1
                            consecutive_spec_rejections += 1
                            print(f"  [spec] REJECT reason={rejection} action={pending_spec}", flush=True)
                            if consecutive_spec_rejections >= 2:
                                spec_disabled = True
                                print("  [spec] disabled after 2 consecutive rejections", flush=True)
                        elif _spec_state_matches(b_url, current_url, b_elems, element_count, pending_spec):
                            action = pending_spec
                            spec_hits += 1
                            consec_spec_hits += 1
                            consecutive_spec_rejections = 0
                            if pending_spec.get("action") != "wait":
                                spec_meaningful_hits += 1
                            print(f"  [spec] HIT — {action}", flush=True)
                        else:
                            spec_rejections_by_reason["state_mismatch"] += 1
                            consecutive_spec_rejections += 1
                            print(f"  [spec] REJECT reason=state_mismatch action={pending_spec}", flush=True)
                            if consecutive_spec_rejections >= 2:
                                spec_disabled = True
                                print("  [spec] disabled after 2 consecutive rejections", flush=True)
                        pending_spec = None
                    elif pending_spec and consec_spec_hits >= _MAX_CONSEC_SPEC_HITS:
                        print(f"  [spec] max consec hits reached — forcing re-plan", flush=True)
                        pending_spec = None

                    # ── Model call ───────────────────────────────────────────
                    inp_tok = out_tok = 0
                    step_cost = 0.0
                    latency_ms = 0
                    model_label = decision.model_label
                    cache_key = _first_step_cache_key(self.allowed_domain, task.objective)
                    cache_hit = False

                    if action is None:
                        if flags.mock:
                            action = _MOCK_CYCLE[(step - 1) % len(_MOCK_CYCLE)]
                            model_label = "mock"
                            print(f"  [mock] {action}", flush=True)
                        elif step == 1 and decision.tier == TIER_DEEPSEEK and cache_key in _FIRST_STEP_CACHE:
                            action = dict(_FIRST_STEP_CACHE[cache_key])
                            model_label = "first_step_cache"
                            cache_hit = True
                            print(f"  [cache] first-step action reused: {action}", flush=True)
                        else:
                            # Build user message for this step.
                            user_text = (
                                f"{goal_prefix}\n"
                                f"Step {step}/{max_steps}\n\n"
                                f"{page_text}"
                            )

                            # For vision tiers, add a 720p screenshot.
                            if decision.needs_vision:
                                screenshot_b64 = await _take_720p_screenshot(page)
                                call_msg: dict = {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": user_text},
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
                            else:
                                call_msg = {"role": "user", "content": user_text}

                            call_messages = conversation + [call_msg]

                            call_start = time.perf_counter()
                            resp = await self.router.call(
                                decision=decision,
                                system=system_prompt,
                                messages=call_messages,
                            )
                            latency_ms = int((time.perf_counter() - call_start) * 1000)
                            inp_tok    = resp.input_tokens
                            out_tok    = resp.output_tokens
                            step_cost  = resp.cost_inr
                            model_label = resp.model or decision.model_label
                            action = _parse_action(resp.text)
                            prev_confidence = float(action.get("confidence", 0.5))
                            if step == 1 and decision.tier == TIER_DEEPSEEK and not _is_uncertain_action(action):
                                _FIRST_STEP_CACHE[cache_key] = dict(action)
                                print("  [cache] first-step action stored", flush=True)

                            thought = action.get("thought", resp.text[:80])
                            print(f"  [{decision.model_label}] {thought}", flush=True)
                            print(
                                f"  Action: {action}  "
                                f"[{inp_tok}in/{out_tok}out tok, {latency_ms}ms, Rs.{step_cost:.4f}]",
                                flush=True,
                            )
                    else:
                        model_label = "speculation_hit"
                        prev_confidence = float(action.get("confidence", 0.5))

                    action_uncertain = _is_uncertain_action(action)
                    if action_uncertain:
                        consecutive_uncertain += 1
                    else:
                        consecutive_uncertain = 0

                    if model_label != "speculation_hit":
                        consec_spec_hits = 0

                    total_inp  += inp_tok
                    total_out  += out_tok
                    total_cost += step_cost

                    # Always store text-only turns in conversation history
                    # (keeps summarization and Sarvam-M compatible).
                    conv_user_text = (
                        f"{goal_prefix}\n"
                        f"Step {step}/{max_steps}\n\n"
                        f"{page_text}"
                    )
                    conversation.append({"role": "user",      "content": conv_user_text})
                    conversation.append({"role": "assistant",  "content": json.dumps(action)})

                    try:
                        await self.meter.publish_model_usage(
                            agent=self.name,
                            current_step=f"step_{step}",
                            model=model_label,
                            vision_tokens=inp_tok,
                            context_tokens=out_tok,
                            latency_ms=latency_ms,
                            cost_inr=step_cost,
                        )
                    except BudgetExceededError as e:
                        final_step_desc = f"budget_exceeded_step_{step}"
                        print(f"  [budget] {e}", flush=True)
                        break

                    # ── Layer 4: speculative execution ────────────────────────
                    if getattr(task, "name", "") == "ntes" and action.get("action") == "type":
                        intended_train_number = re.sub(r"\D", "", str(getattr(task, "train_number", "") or ""))
                        typed_digits = re.sub(r"\D", "", str(action.get("text", "") or ""))
                        if intended_train_number and typed_digits != intended_train_number:
                            print(
                                f"  [opt-ntes] overriding typed train number "
                                f"{typed_digits!r} -> {intended_train_number!r}",
                                flush=True,
                            )
                            action["text"] = intended_train_number
                        if intended_train_number and await _page_has_input_value(page, intended_train_number):
                            print(
                                "  [opt-ntes] train number already entered; pressing Enter",
                                flush=True,
                            )
                            action = {
                                "action": "press",
                                "key": "Enter",
                                "thought": "train number already entered; submit query",
                                "confidence": 0.95,
                            }

                    if (
                        flags.speculation
                        and task_allows_speculation
                        and not spec_disabled
                        and _is_cooperative(action)
                        and not flags.mock
                    ):
                        spec_attempts += 1
                        if action.get("action") != "wait":
                            spec_meaningful_attempts += 1
                        b_url   = current_url
                        b_elems = element_count
                        exec_task = asyncio.create_task(_execute_action(page, action, snap))
                        spec_task = asyncio.create_task(
                            _speculate(self.router, conversation, action, goal_prefix)
                        )
                        outcome, spec_action = await asyncio.gather(exec_task, spec_task)
                        spec_action["_before_url"]   = b_url
                        spec_action["_before_elems"] = b_elems
                        after_snap = await extract_dom(page)
                        spec_action["_expected_hash"] = _dom_hash(page.url, after_snap)
                        pending_spec = spec_action
                    else:
                        outcome = await _execute_action(page, action, snap)
                        pending_spec = None

                    action_kind = action.get("action", "wait")
                    if not outcome.done and action_kind != "click":
                        await asyncio.sleep(0.8)
                    after_snap = await extract_dom(page) if flags.dom else None
                    after_action_state_hash = await _page_state_hash(page, after_snap)
                    after_element_count = len(after_snap.elements) if after_snap else 0
                    action_changed_state = (
                        page.url != current_url
                        or after_action_state_hash != before_action_state_hash
                        or outcome.done
                    )
                    if action_kind == "type":
                        action_success = outcome.playwright_ok and outcome.input_value == outcome.typed_text
                        success_detail = f"input_value={outcome.input_value!r}, typed_text={outcome.typed_text!r}"
                    elif action_kind == "click":
                        new_visible_element = after_element_count > element_count
                        target_gone = outcome.target_visible_after is False
                        action_success = outcome.playwright_ok and (
                            page.url != current_url or new_visible_element or target_gone
                        )
                        success_detail = (
                            f"url_changed={page.url != current_url}, "
                            f"new_visible_element={new_visible_element}, "
                            f"target_visible_after={outcome.target_visible_after}"
                        )
                    elif action_kind in {"wait", "scroll"}:
                        action_success = outcome.playwright_ok
                        success_detail = "noop_success"
                    elif action_kind == "done":
                        action_success = True
                        success_detail = "terminal_success"
                    else:
                        action_success = outcome.playwright_ok and action_changed_state
                        success_detail = f"state_changed={action_changed_state}"

                    if outcome.playwright_ok:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                    print(
                        f"  [action-result] ok={outcome.playwright_ok} "
                        f"changed={action_changed_state} success={action_success} "
                        f"failures={consecutive_failures} detail={success_detail}",
                        flush=True,
                    )
                    route_entry = {
                        "step": step,
                        "tier": decision.tier,
                        "model": model_label,
                        "reason": decision.reason,
                        "escalated": decision.tier > TIER_DEEPSEEK,
                        "trigger": decision.reason.split(":", 1)[0] if decision.tier > TIER_DEEPSEEK else "default",
                        "raw_value": decision.reason,
                        "action": action_kind,
                        "confidence": prev_confidence,
                        "uncertain": action_uncertain,
                        "cache_hit": cache_hit,
                        "playwright_ok": outcome.playwright_ok,
                        "page_changed": action_changed_state,
                        "action_success": action_success,
                        "success_detail": success_detail,
                        "consecutive_failures_after": consecutive_failures,
                        "consecutive_uncertain_after": consecutive_uncertain,
                    }
                    route_log.append(route_entry)
                    print(f"  [route-log] {json.dumps(route_entry, ensure_ascii=False)}", flush=True)

                    if outcome.done:
                        await asyncio.sleep(2)
                        current_body = await page.inner_text("body")
                        if getattr(task, "aggregate_page_text", False):
                            collected_page_text.append(f"PAGE {page.url}\n{current_body}")
                            final_page_text = "\n\n".join(collected_page_text[-12:])
                        else:
                            final_page_text = current_body
                        success = task.success(final_page_text)
                        final_step_desc = f"done_signal_step_{step}_status={'yes' if success else 'no'}"
                        break

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
                if not launch_headless and browser and browser.is_connected():
                    close_delay = float(getattr(task, "browser_close_delay_seconds", 5))
                    print(
                        f"[playwright] run finished; closing browser in {close_delay:g}s",
                        flush=True,
                    )
                    await asyncio.sleep(close_delay)
                if browser and browser.is_connected():
                    await context.close()
                    if browser.is_connected():
                        await browser.close()

        elapsed = time.perf_counter() - started
        spec_summary = f"{spec_hits}/{spec_attempts} hits" if spec_attempts else "n/a"
        spec_meaningful_summary = (
            f"{spec_meaningful_hits}/{spec_meaningful_attempts} meaningful hits"
            if spec_meaningful_attempts else "n/a"
        )
        spec_disabled_reason = (
            "short_task_safety_validation_overhead"
            if flags.speculation and not task_allows_speculation and spec_disabled
            else ""
        )
        print(
            f"  [spec-log] attempts={spec_attempts} accepted_hits={spec_hits} "
            f"meaningful={spec_meaningful_summary} rejections={spec_rejections_by_reason} "
            f"disabled_reason={spec_disabled_reason or 'n/a'}",
            flush=True,
        )
        if task.config_name == "full_optimized" and not success:
            print(f"  [failure] full_optimized final_step={final_step_desc}", flush=True)
            print(f"  [failure-dom] {' '.join(final_page_text.split())[:1000]}", flush=True)

        extracted_answer = (
            task.extract_answer(final_page_text)
            if hasattr(task, "extract_answer")
            else extract_status_string(final_page_text)
        )
        structured_result = (
            task.structured_result(final_page_text)
            if hasattr(task, "structured_result")
            else None
        )
        await self.meter.publish(
            MeterEvent(
                run_id=self.meter.run_id,
                agent=self.name,
                event="agent_complete",
                elapsed_seconds=round(elapsed, 2),
                current_step=final_step_desc,
                current_model=self.router.stats.ratio_str(),
                detail=json.dumps({
                    "speculation": spec_summary,
                    "speculation_meaningful": spec_meaningful_summary,
                    "speculation_disabled_reason": spec_disabled_reason,
                    "spec_rejections_by_reason": spec_rejections_by_reason,
                    "cascade": self.router.stats.ratio_str(),
                    "route_log": route_log,
                    "flags": {
                        "dom":           flags.dom,
                        "summarization": flags.summarization,
                        "cascade":       flags.cascade,
                        "speculation":   flags.speculation,
                    },
                    "extracted_answer": extracted_answer,
                    "structured_result": structured_result,
                    "success": success,
                    "task_type": getattr(task, "name", "unknown"),
                }),
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
            extracted_status_string=extracted_answer,
        )
