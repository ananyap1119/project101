from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright


@dataclass(slots=True)
class BrowserController:
    headed: bool = False
    browser: Browser | None = None
    context: BrowserContext | None = None
    page: Page | None = None
    _playwright: Any | None = None

    async def __aenter__(self) -> "BrowserController":
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(headless=not self.headed)
        self.context = await self.browser.new_context(viewport={"width": 1440, "height": 1000})
        self.page = await self.context.new_page()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def goto(self, url: str) -> Page:
        if not self.page:
            raise RuntimeError("BrowserController must be entered before use")
        await self.page.goto(url, wait_until="domcontentloaded")
        return self.page

    async def screenshot(self) -> bytes:
        if not self.page:
            raise RuntimeError("BrowserController must be entered before use")
        return await self.page.screenshot(full_page=True)
