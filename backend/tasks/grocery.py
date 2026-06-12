from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from backend.tasks.ntes import ConfigName


PROFILE_ROOT = Path(".runtime/browser-profiles")


class GroceryItem(BaseModel):
    name: str
    search_query: str
    quantity: int = Field(default=1, ge=1, le=6)
    required_amount: str = ""
    reason: str = ""


class GroceryCartEntry(BaseModel):
    requested_item: str
    requested_amount: str = ""
    requested_quantity: int = 1
    product_name: str = ""
    pack_size: str = ""
    price: str = ""
    added_quantity: int = 0
    status: Literal["added", "unavailable", "failed"]


class GroceryPlannerResult(BaseModel):
    request: str
    mode: Literal["meal_plan", "missing_ingredients", "household_restock"]
    entries: list[GroceryCartEntry]
    added_count: int
    unavailable_count: int
    message: str


class BlinkitPlannerTask(BaseModel):
    name: str = "blinkit_planner"
    config_name: ConfigName = "full_optimized"
    start_url: str = "https://blinkit.com/"
    request: str
    mode: Literal["meal_plan", "missing_ingredients", "household_restock"] = "meal_plan"
    items: list[GroceryItem]
    pantry_items: list[str] = Field(default_factory=list)
    location: str = "560001"
    headed: bool = True
    speculation_enabled: bool = False
    aggregate_page_text: bool = True
    auto_close_browser: bool = True
    max_steps: int = 1

    @property
    def browser_profile_dir(self) -> str:
        return str(PROFILE_ROOT / "blinkit-commerce")

    @property
    def objective(self) -> str:
        item_text = ", ".join(
            f"{item.name} ({item.required_amount or f'{item.quantity} pack(s)'})" for item in self.items
        )
        return (
            f"Prepare a Blinkit cart for: {self.request}. Add only these planned items: {item_text}. "
            "Never proceed to checkout, select an address for checkout, choose payment, or place an order."
        )

    def success(self, page_text: str) -> bool:
        return "GROCERY_PLANNER_DATA " in page_text

    def structured_result(self, page_text: str) -> dict:
        import json
        import re

        matches = re.findall(r"GROCERY_PLANNER_DATA\s+([^\r\n]+)", page_text)
        if not matches:
            return GroceryPlannerResult(
                request=self.request,
                mode=self.mode,
                entries=[],
                added_count=0,
                unavailable_count=len(self.items),
                message="The grocery plan was created, but Blinkit cart changes could not be verified.",
            ).model_dump()
        try:
            payload = json.loads(matches[-1])
        except json.JSONDecodeError:
            payload = {"entries": []}
        entries = [GroceryCartEntry.model_validate(item) for item in payload.get("entries", [])]
        added_count = sum(1 for item in entries if item.status == "added")
        unavailable_count = len(entries) - added_count
        message = f"Added {added_count} of {len(entries)} planned ingredients to the Blinkit cart."
        if unavailable_count:
            message += f" {unavailable_count} could not be added."
        message += " Checkout was not opened."
        return GroceryPlannerResult(
            request=self.request,
            mode=self.mode,
            entries=entries,
            added_count=added_count,
            unavailable_count=unavailable_count,
            message=message,
        ).model_dump()

    def extract_answer(self, page_text: str) -> str:
        result = GroceryPlannerResult.model_validate(self.structured_result(page_text))
        lines = [result.message]
        for entry in result.entries:
            if entry.status == "added":
                lines.append(
                    f"{entry.requested_item}: {entry.product_name}, {entry.pack_size}, "
                    f"{entry.price}, quantity {entry.added_quantity}."
                )
            else:
                lines.append(f"{entry.requested_item}: could not be added.")
        return "\n".join(lines)

    def reset(self) -> None:
        pass
