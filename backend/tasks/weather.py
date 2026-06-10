from __future__ import annotations

import re

from pydantic import BaseModel

from backend.tasks.ntes import ConfigName


class WeatherTask(BaseModel):
    name: str = "weather"
    config_name: ConfigName = "full_optimized"
    city: str = "Bangalore"
    question: str = ""
    headed: bool = False
    speculation_enabled: bool = False

    @property
    def start_url(self) -> str:
        city_slug = self.city.strip().replace(" ", "+") or "Bangalore"
        return f"https://wttr.in/{city_slug}?m"

    @property
    def objective(self) -> str:
        focus = f" Pay special attention to: {self.question}." if self.question else ""
        return (
            f"Open wttr.in for {self.city} and read the current weather directly. "
            "Do not use Bing, Google, or any search engine. "
            "Report only the temperature in Celsius, weather condition, rain/precipitation chance, "
            f"and humidity if visible.{focus} Signal done as soon as the weather text is visible."
        )

    def success(self, page_text: str) -> bool:
        text = page_text.lower()
        has_temp = bool(re.search(r"\d+\s*(?:°|deg|degrees)?\s*c\b", text)) or "celsius" in text
        has_cond = any(
            k in text
            for k in [
                "sunny",
                "cloudy",
                "rain",
                "clear",
                "overcast",
                "partly",
                "drizzle",
                "storm",
                "fog",
                "haze",
                "mist",
                "thunder",
                "humidity",
                "wind",
                "precipitation",
            ]
        )
        return has_temp or has_cond

    def extract_answer(self, page_text: str) -> str:
        compact = " ".join(page_text.split())
        for kw in ["°C", "Humidity", "Precipitation", "Rain", "Wind", "Feels like"]:
            idx = compact.find(kw)
            if idx >= 0:
                start = max(0, idx - 120)
                end = min(len(compact), idx + 220)
                return compact[start:end].strip()
        return compact[:280]

    def reset(self) -> None:
        pass
