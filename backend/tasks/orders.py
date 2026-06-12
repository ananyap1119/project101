from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from backend.tasks.ntes import ConfigName

Platform = Literal["amazon", "flipkart", "all"]
ShoppingAction = Literal[
    "orders",
    "cart",
    "wishlist",
    "saved_items",
    "buy_again",
    "browsing_history",
    "invoices",
]
PROFILE_ROOT = Path(".runtime/browser-profiles")

PLATFORM_URLS: dict[str, dict[str, str]] = {
    "amazon": {
        "orders": "https://www.amazon.in/gp/your-account/order-history",
        "cart": "https://www.amazon.in/gp/cart/view.html",
        "wishlist": "https://www.amazon.in/hz/wishlist/ls",
        "saved_items": "https://www.amazon.in/gp/cart/view.html",
        "buy_again": "https://www.amazon.in/gp/buyagain",
        "browsing_history": "https://www.amazon.in/gp/history",
        "invoices": "https://www.amazon.in/gp/your-account/order-history",
    },
    "flipkart": {
        "orders": "https://www.flipkart.com/account/orders",
        "cart": "https://www.flipkart.com/viewcart",
        "wishlist": "https://www.flipkart.com/wishlist",
        "saved_items": "https://www.flipkart.com/wishlist",
        "buy_again": "https://www.flipkart.com/account/orders",
        "browsing_history": "https://www.flipkart.com/",
        "invoices": "https://www.flipkart.com/account/orders",
    },
}

ACTION_LABELS = {
    "orders": "recent orders and delivery status",
    "cart": "shopping cart",
    "wishlist": "wishlist",
    "saved_items": "saved items",
    "buy_again": "previous purchases available to buy again",
    "browsing_history": "recently viewed or browsing history",
    "invoices": "orders with available invoices",
}


class ShoppingEntry(BaseModel):
    platform: Literal["amazon", "flipkart"]
    title: str = ""
    status: str = ""
    price: str = ""
    detail: str = ""


class ShoppingActivity(BaseModel):
    action: ShoppingAction
    entries: list[ShoppingEntry]
    platforms_checked: list[str]
    message: str = ""


class ShoppingOrdersTask(BaseModel):
    name: str = "shopping_orders"
    config_name: ConfigName = "full_optimized"
    platform: Platform = "all"
    action: ShoppingAction = "orders"
    query: str = "show my recent orders and delivery status"
    headed: bool = True
    speculation_enabled: bool = False
    aggregate_page_text: bool = True
    auto_close_browser: bool = True
    max_steps: int = 36
    manual_auth_wait_seconds: int = 180

    @property
    def start_url(self) -> str:
        first_platform = "flipkart" if self.platform == "flipkart" else "amazon"
        return PLATFORM_URLS[first_platform][self.action]

    @property
    def browser_profile_dir(self) -> str:
        return str(PROFILE_ROOT / "commerce")

    @property
    def objective(self) -> str:
        label = ACTION_LABELS[self.action]
        targets = "Amazon and Flipkart" if self.platform == "all" else self.platform.title()
        second_site = ""
        if self.platform == "all":
            second_site = (
                f" After reading Amazon, navigate to {PLATFORM_URLS['flipkart'][self.action]} "
                "and read the same information on Flipkart."
            )
        invoice_note = (
            " Report which orders have an invoice link, but do not download anything."
            if self.action == "invoices"
            else ""
        )
        return (
            f"Open the {label} page on {targets} using the signed-in browser session. "
            f"The user's request is: {self.query}. Read and report the visible item names, prices, "
            "quantities, statuses, and relevant dates. This is strictly read-only: never add or remove "
            "items, change quantity, move items, purchase, cancel, return, review, or alter the account. "
            "If login, OTP, or CAPTCHA is shown, wait for the user to complete it."
            f"{invoice_note}{second_site}"
        )

    def success(self, page_text: str) -> bool:
        text = page_text.lower()
        signals = {
            "orders": ("your orders", "my orders", "ordered on", "delivered", "shipped"),
            "cart": ("shopping cart", "your cart", "cart is empty", "subtotal", "place order"),
            "wishlist": ("wish list", "wishlist", "my lists"),
            "saved_items": ("saved for later", "saved items", "wishlist"),
            "buy_again": ("buy again", "buy it again", "purchase history"),
            "browsing_history": ("browsing history", "recently viewed", "viewed items"),
            "invoices": ("your orders", "invoice", "order details"),
        }[self.action]
        found = any(signal in text for signal in signals)
        if self.platform == "all":
            return found and "amazon" in text and "flipkart" in text
        return found

    @staticmethod
    def _platform_sections(page_text: str) -> list[tuple[str, str]]:
        compact = " ".join(page_text.split())
        markers = list(re.finditer(r"PAGE https?://[^ ]+", compact, re.I))
        sections: list[tuple[str, str]] = []
        for index, marker in enumerate(markers):
            url = marker.group(0).lower()
            platform = "amazon" if "amazon." in url else "flipkart" if "flipkart." in url else ""
            if not platform:
                continue
            end = markers[index + 1].start() if index + 1 < len(markers) else len(compact)
            sections.append((platform, compact[marker.end() : end]))
        return sections

    def _empty_message(self, platform: str, section: str) -> str | None:
        text = section.lower()
        empty_phrases = {
            "orders": ("haven't placed an order", "no orders found", "you have no orders"),
            "cart": ("cart is empty", "your amazon cart is empty", "missing cart items"),
            "wishlist": ("your wish list is empty", "wishlist is empty", "no items in your wishlist"),
            "saved_items": ("no items saved for later", "saved for later (0 items)"),
            "buy_again": ("no items to buy again",),
            "browsing_history": ("no browsing history", "haven't viewed any items"),
            "invoices": ("haven't placed an order", "no orders found"),
        }[self.action]
        if any(phrase in text for phrase in empty_phrases):
            label = ACTION_LABELS[self.action]
            return f"No {label} were found on {platform.title()}."
        return None

    def structured_result(self, page_text: str) -> dict:
        entries: list[ShoppingEntry] = []
        checked: list[str] = []
        empty_messages: list[str] = []
        dom_payloads = re.findall(r"SHOPPING_DATA\s+([^\r\n]+)", page_text)
        for raw_payload in dom_payloads:
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                continue
            platform = payload.get("platform")
            if platform not in {"amazon", "flipkart"}:
                continue
            if platform not in checked:
                checked.append(platform)
            for item in payload.get("items", [])[:20]:
                if not isinstance(item, dict):
                    continue
                detail = str(item.get("detail") or "").strip()
                quantity = str(item.get("quantity") or "").strip()
                if quantity:
                    detail = f"Quantity: {quantity}. {detail}"
                entries.append(
                    ShoppingEntry(
                        platform=platform,
                        title=str(item.get("title") or "").strip(),
                        status=str(item.get("status") or "").strip(),
                        price=str(item.get("price") or "").strip(),
                        detail=detail[:700],
                    )
                )
        for platform, section in self._platform_sections(page_text):
            if platform not in checked:
                checked.append(platform)
            empty = self._empty_message(platform, section)
            if empty:
                empty_messages.append(empty)
                continue

            if any(entry.platform == platform for entry in entries):
                continue
            if self.action in {"orders", "invoices"}:
                pattern = re.compile(
                    r"(delivered|arriving[^.]{0,80}|shipped|out for delivery|cancelled|invoice)", re.I
                )
            else:
                pattern = re.compile(r"(?:₹|\bRs\.|\bINR)\s*[\d,]+(?:\.\d{2})?", re.I)
            for match in list(pattern.finditer(section))[:8]:
                start = max(0, match.start() - 180)
                end = min(len(section), match.end() + 100)
                context = section[start:end].strip()
                price = match.group(0) if self.action not in {"orders", "invoices"} else ""
                status = match.group(0) if self.action in {"orders", "invoices"} else ""
                entries.append(
                    ShoppingEntry(
                        platform=platform,
                        status=status,
                        price=price,
                        detail=context,
                    )
                )

        label = ACTION_LABELS[self.action]
        if entries:
            message = f"Found {len(entries)} visible entries in {label}."
        elif empty_messages:
            message = " ".join(dict.fromkeys(empty_messages))
        elif checked:
            message = f"The {label} page was checked, but no item details were visible."
        else:
            message = "No shopping platform results were available."
        return ShoppingActivity(
            action=self.action,
            entries=entries,
            platforms_checked=checked,
            message=message,
        ).model_dump()

    def extract_answer(self, page_text: str) -> str:
        result = ShoppingActivity.model_validate(self.structured_result(page_text))
        if not result.entries:
            return result.message
        platforms = " and ".join(p.title() for p in result.platforms_checked)
        lines = [f"{result.message} Checked {platforms}."]
        for entry in result.entries[:5]:
            summary = entry.title or entry.detail[:180]
            suffix = entry.status or entry.price
            lines.append(f"{entry.platform.title()}: {summary} {suffix}".strip())
        return "\n".join(lines)

    def reset(self) -> None:
        PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
