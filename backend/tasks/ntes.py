from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

DOWNLOADS_DIR = Path("./downloads")
TRAIN_NUMBER = "22691"

SIMPLE_GOAL = (
    "Check live running status of train {train_number}. "
    "Find current station and whether it is on time or delayed."
)

RICH_GOAL = (
    "Check live running status of train {train_number}. "
    "Find current station, delay in minutes, expected arrival "
    "time at the next 3 scheduled stops, and whether the train "
    "departed on time from its origin."
)

STATUS_KEYWORDS = ("delay", "on time", "late", "running", "departed", "arrived")
STATION_PATTERN = re.compile(r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4}\b")
TIME_PATTERN = re.compile(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b")


def _has_station(text: str) -> bool:
    return bool(STATION_PATTERN.search(text))


def _has_status(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in STATUS_KEYWORDS)


def simple_success(text: str) -> bool:
    return _has_station(text) and _has_status(text)


def rich_success(text: str) -> bool:
    return _has_station(text) and bool(TIME_PATTERN.search(text)) and _has_status(text)


def extract_status_string(text: str) -> str:
    compact = " ".join(text.split())
    for keyword in STATUS_KEYWORDS:
        idx = compact.lower().find(keyword)
        if idx >= 0:
            start = max(0, idx - 100)
            end = min(len(compact), idx + 180)
            return compact[start:end].strip()
    return compact[:280]


SIMPLE_SUCCESS = simple_success
RICH_SUCCESS = rich_success

SuccessMode = Literal["simple", "rich"]
ConfigName = Literal[
    "baseline",
    "baseline_naive",
    "baseline_competent",
    "som_only",
    "som_summarization",
    "som_summarization_cascade",
    "full_optimized",
]

CONFIG_SUCCESS_MODE: dict[ConfigName, SuccessMode] = {
    "baseline": "simple",
    "baseline_naive": "simple",
    "baseline_competent": "simple",
    "som_only": "simple",
    "som_summarization": "simple",
    "som_summarization_cascade": "simple",
    "full_optimized": "simple",
}

CONFIG_GOALS: dict[ConfigName, str] = {
    "baseline": SIMPLE_GOAL,
    "baseline_naive": SIMPLE_GOAL,
    "baseline_competent": SIMPLE_GOAL,
    "som_only": SIMPLE_GOAL,
    "som_summarization": SIMPLE_GOAL,
    "som_summarization_cascade": SIMPLE_GOAL,
    "full_optimized": SIMPLE_GOAL,
}


class NTESTask(BaseModel):
    name: str = "ntes"
    config_name: ConfigName = "baseline"
    start_url: str = "https://enquiry.indianrail.gov.in/mntes/"
    train_number: str = TRAIN_NUMBER
    headed: bool = False
    speculation_enabled: bool = False

    @property
    def objective(self) -> str:
        return CONFIG_GOALS[self.config_name].format(train_number=self.train_number)

    @property
    def success_mode(self) -> SuccessMode:
        return CONFIG_SUCCESS_MODE[self.config_name]

    def success(self, page_text: str) -> bool:
        if self.success_mode == "rich":
            return RICH_SUCCESS(page_text)
        return SIMPLE_SUCCESS(page_text)

    def extract_answer(self, page_text: str) -> str:
        return extract_status_string(page_text)

    def reset(self) -> None:
        if DOWNLOADS_DIR.exists():
            shutil.rmtree(DOWNLOADS_DIR)
        DOWNLOADS_DIR.mkdir(exist_ok=True)
