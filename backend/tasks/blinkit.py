from __future__ import annotations

import re

from pydantic import BaseModel

from backend.tasks.ntes import ConfigName

BLINKIT_PINCODE = "560001"


class BlinkitTask(BaseModel):
    name: str = "blinkit"
    config_name: ConfigName = "full_optimized"
    start_url: str = "https://blinkit.com/"
    product: str = "milk"
    location: str = BLINKIT_PINCODE
    headed: bool = False
    speculation_enabled: bool = False

    @property
    def objective(self) -> str:
        return (
            f"Go to Blinkit and search for '{self.product}'. "
            f"If a location or delivery address modal appears, set the delivery pincode first: "
            f"click the 'search delivery location' field, type '{self.location}', wait for suggestions, "
            "and click the first suggestion. Do not type the product name into the location field. "
            f"Then search for '{self.product}' and find the price of the top result. "
            "Report the exact product name and price in rupees."
        )

    def success(self, page_text: str) -> bool:
        has_price = bool(re.search(r"[₹₨]\s*[\d,]+", page_text)) or bool(
            re.search(r"Rs\.?\s*\d+", page_text)
        )
        return has_price

    def extract_answer(self, page_text: str) -> str:
        compact = " ".join(page_text.split())
        m = re.search(r"[₹₨]\s*[\d,]+", compact)
        if m:
            start = max(0, m.start() - 150)
            end = min(len(compact), m.end() + 150)
            return compact[start:end].strip()
        return compact[:400]

    def reset(self) -> None:
        pass
