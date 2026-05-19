"""Set-of-Marks (SoM) grounding — Layer 1.

Extracts interactive DOM elements, annotates a downscaled screenshot with
numbered red marks, and provides a resolver that maps mark IDs back to
viewport coordinates for Playwright execution.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Page

TARGET_W, TARGET_H = 1280, 720  # SoM works at 720p — no need for full 1080p

_JS_GET_ELEMENTS = """
() => {
    const sel = [
        'button', 'a[href]', 'input:not([type="hidden"])',
        'select', 'textarea',
        '[role="button"]', '[role="link"]', '[role="checkbox"]',
        '[role="radio"]', '[role="menuitem"]', '[role="tab"]',
    ].join(', ');
    const seen = new Set();
    const results = [];
    document.querySelectorAll(sel).forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.width < 4 || rect.height < 4) return;
        if (rect.top < 0 || rect.left < 0) return;
        // deduplicate by top-left pixel to avoid stacked elements
        const key = `${Math.round(rect.left)},${Math.round(rect.top)}`;
        if (seen.has(key)) return;
        seen.add(key);
        const label = (
            el.innerText ||
            el.getAttribute('value') ||
            el.getAttribute('placeholder') ||
            el.getAttribute('aria-label') ||
            el.getAttribute('title') ||
            el.getAttribute('alt') ||
            el.getAttribute('href') ||
            ''
        ).trim().slice(0, 60);
        results.push({
            tag:  el.tagName.toLowerCase(),
            type: el.getAttribute('type') || '',
            text: label,
            href: el.getAttribute('href') || '',
            x: Math.round(rect.left),
            y: Math.round(rect.top),
            w: Math.round(rect.width),
            h: Math.round(rect.height),
        });
    });
    // cap to 60 to keep the marks table compact
    return results.slice(0, 60);
}
"""


@dataclass
class Mark:
    id: int
    tag: str
    type: str
    text: str
    href: str
    x: int   # original viewport coords
    y: int
    w: int
    h: int


@dataclass
class SoMResult:
    screenshot_png: bytes   # annotated 720p PNG
    marks: list[Mark]
    orig_w: int             # original viewport width
    orig_h: int             # original viewport height


async def build_som(page: Page) -> SoMResult:
    """Return annotated 720p screenshot + mark table for the current page."""
    raw_elements: list[dict] = await page.evaluate(_JS_GET_ELEMENTS)

    marks = [
        Mark(
            id=i + 1,
            tag=e["tag"],
            type=e["type"],
            text=e["text"],
            href=e["href"],
            x=e["x"],
            y=e["y"],
            w=e["w"],
            h=e["h"],
        )
        for i, e in enumerate(raw_elements)
    ]

    raw_png = await page.screenshot(full_page=False)
    img = Image.open(io.BytesIO(raw_png)).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    sx = TARGET_W / orig_w
    sy = TARGET_H / orig_h

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    for m in marks:
        mx = int(m.x * sx)
        my = int(m.y * sy)
        mw = max(int(m.w * sx), 10)
        mh = max(int(m.h * sy), 10)
        draw.rectangle([mx, my, mx + mw, my + mh], outline="#FF2222", width=1)
        label = str(m.id)
        bw = max(len(label) * 6 + 4, 14)
        draw.rectangle([mx, my, mx + bw, my + 13], fill="#FF2222")
        draw.text((mx + 2, my + 1), label, fill="#FFFFFF", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return SoMResult(screenshot_png=buf.getvalue(), marks=marks, orig_w=orig_w, orig_h=orig_h)


def marks_to_table(marks: list[Mark]) -> str:
    if not marks:
        return "(no interactive elements found)"
    rows = ["Mark | Tag    | Text / Label"]
    rows.append("-----|--------|" + "-" * 40)
    for m in marks:
        label = m.text or m.href or f"[{m.type or m.tag}]"
        rows.append(f"  {m.id:2d} | {m.tag:<6} | {label[:45]}")
    return "\n".join(rows)


def resolve_mark(mark_id: int, marks: list[Mark]) -> Mark | None:
    for m in marks:
        if m.id == mark_id:
            return m
    return None
