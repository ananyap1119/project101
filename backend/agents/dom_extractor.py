"""DOM text extractor — Layer 1 for the Sarvam-M CUA agent.

Extracts interactive elements and visible text from the current page and
formats them as a numbered, plain-text snapshot that Sarvam-M can reason
over without needing to process any image data.
"""
from __future__ import annotations

from dataclasses import dataclass

from playwright.async_api import Page

# Same selector set as SoM, but we only need metadata — no rendering.
_JS_EXTRACT = """
() => {
    const sel = [
        'button', 'a[href]', 'input:not([type="hidden"])',
        'select', 'textarea',
        '[role="button"]', '[role="link"]',
        '[role="checkbox"]', '[role="radio"]',
        '[role="menuitem"]', '[role="tab"]',
    ].join(', ');
    const seen  = new Set();
    const items = [];
    document.querySelectorAll(sel).forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.width < 4 || rect.height < 4) return;
        if (rect.top < 0 || rect.top > window.innerHeight + 200) return;
        const key = Math.round(rect.left) + ',' + Math.round(rect.top);
        if (seen.has(key)) return;
        seen.add(key);
        const text = (
            el.innerText ||
            el.getAttribute('value') ||
            el.getAttribute('placeholder') ||
            el.getAttribute('aria-label') ||
            el.getAttribute('title') ||
            el.getAttribute('alt') ||
            el.getAttribute('href') ||
            ''
        ).replace(/\\s+/g, ' ').trim().slice(0, 80);
        items.push({
            tag:         el.tagName.toLowerCase(),
            input_type:  el.getAttribute('type') || '',
            id:          el.id || '',
            text,
            placeholder: el.getAttribute('placeholder') || '',
            href:        (el.getAttribute('href') || '').slice(0, 120),
            target:      el.getAttribute('target') || '',
            role:        el.getAttribute('role') || '',
            x: Math.round(rect.left  + rect.width  / 2),
            y: Math.round(rect.top   + rect.height / 2),
        });
        if (items.length >= 60) return;
    });
    // Page body text (capped to avoid token bloat)
    const body = (document.body.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 1200);
    return { items, body };
}
"""


@dataclass
class DomElement:
    n: int           # 1-based index — used in action JSON
    tag: str
    input_type: str
    id: str
    text: str
    placeholder: str
    href: str
    target: str
    role: str
    x: int           # centre-point viewport coords (fallback for pixel-click)
    y: int


@dataclass
class DomSnapshot:
    elements: list[DomElement]
    body_text: str
    url: str


async def extract_dom(page: Page) -> DomSnapshot:
    """Return a structured DOM snapshot; returns empty snapshot on error (e.g. mid-navigation)."""
    try:
        raw = await page.evaluate(_JS_EXTRACT)
        elements = [
            DomElement(
                n=i + 1,
                tag=e["tag"],
                input_type=e["input_type"],
                id=e["id"],
                text=e["text"],
                placeholder=e["placeholder"],
                href=e["href"],
                target=e["target"],
                role=e["role"],
                x=e["x"],
                y=e["y"],
            )
            for i, e in enumerate(raw["items"])
        ]
        return DomSnapshot(elements=elements, body_text=raw["body"], url=page.url)
    except Exception:
        return DomSnapshot(elements=[], body_text="", url=page.url)


def dom_to_prompt(snap: DomSnapshot) -> str:
    """Format a DomSnapshot as numbered plain text for Sarvam-M."""
    lines: list[str] = [f"URL: {snap.url}", ""]

    if snap.elements:
        lines.append("INTERACTIVE ELEMENTS (use the number in your action):")
        for el in snap.elements[:40]:   # cap at 40 to keep prompt size bounded
            label = el.text or el.placeholder or el.href or f"[{el.input_type or el.tag}]"
            tag_str = el.tag + (f"[{el.input_type}]" if el.input_type else "")
            id_str = f'  id="{el.id}"' if el.id else ""
            ph_str = f'  placeholder="{el.placeholder}"' if el.placeholder and not el.text else ""
            lines.append(f"  {el.n:2d}. {tag_str}: {label}{id_str}{ph_str}")
    else:
        lines.append("(no interactive elements found on page)")

    if snap.body_text:
        lines.extend(["", "VISIBLE PAGE TEXT:", snap.body_text[:600]])

    return "\n".join(lines)
