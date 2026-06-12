from __future__ import annotations

import re
from datetime import date, datetime

from pydantic import BaseModel

from backend.tasks.ntes import ConfigName


class TrainOption(BaseModel):
    number: str
    name: str
    days: str
    category: str
    departure_time: str
    departure_station: str
    arrival_time: str
    arrival_station: str
    duration: str


class RailwayResult(BaseModel):
    origin: str
    destination: str
    travel_date: str
    total_found: int
    trains: list[TrainOption]


_TRAIN_PATTERN = re.compile(
    r"\b(?P<number>\d{5})\s+"
    r"(?P<name>.+?)\s+"
    r"(?P<days>Daily|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:,(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun))*)\s*\|\s*"
    r"(?P<category>.+?)\s+See Train Status\s*>>\s*"
    r"(?P<departure>\d{2}:\d{2})\s+"
    r"(?P<departure_station>.+?)\s+(?P<departure_code>[A-Z]{2,5})\s+"
    r"--(?P<duration>\d{2}:\d{2})\s+Hrs\.--\s+"
    r"(?P<arrival>\d{2}:\d{2})\s+"
    r"(?P<arrival_station>.+?)\s+(?P<arrival_code>[A-Z]{2,5})"
    r"(?=\s+\d{5}\s|\s+RailOne\b|\s+Property of Indian Railways\b|$)",
    re.I,
)


class RailwayJourneyTask(BaseModel):
    name: str = "railway_journey"
    config_name: ConfigName = "full_optimized"
    origin: str
    destination: str
    travel_date: str
    preferences: str = ""
    headed: bool = True
    speculation_enabled: bool = False
    aggregate_page_text: bool = True
    auto_close_browser: bool = True
    max_steps: int = 10

    @property
    def start_url(self) -> str:
        return "https://enquiry.indianrail.gov.in/mntes/"

    @property
    def objective(self) -> str:
        preference = f" Preferences: {self.preferences}." if self.preferences else ""
        return (
            f"Use NTES to find trains from {self.origin} to {self.destination} "
            f"for {self.travel_date}. Report suitable trains with train number/name, operating days, "
            f"departure, arrival, and journey duration.{preference} Do not book a ticket."
        )

    def success(self, page_text: str) -> bool:
        return bool(self._parse_trains(page_text))

    def _parse_trains(self, page_text: str) -> list[TrainOption]:
        compact = " ".join(page_text.split())
        trains: list[TrainOption] = []
        seen: set[str] = set()
        for match in _TRAIN_PATTERN.finditer(compact):
            number = match.group("number")
            if number in seen:
                continue
            seen.add(number)
            trains.append(
                TrainOption(
                    number=number,
                    name=match.group("name").strip(),
                    days=match.group("days").strip(),
                    category=match.group("category").strip(),
                    departure_time=match.group("departure"),
                    departure_station=f"{match.group('departure_station').strip()} ({match.group('departure_code').upper()})",
                    arrival_time=match.group("arrival"),
                    arrival_station=f"{match.group('arrival_station').strip()} ({match.group('arrival_code').upper()})",
                    duration=match.group("duration"),
                )
            )
        return trains

    def _trains_for_date(self, page_text: str) -> list[TrainOption]:
        trains = self._parse_trains(page_text)
        try:
            travel_day = datetime.fromisoformat(self.travel_date).strftime("%a")
        except ValueError:
            return trains
        return [
            train
            for train in trains
            if train.days.lower() == "daily" or travel_day in train.days.split(",")
        ]

    def structured_result(self, page_text: str) -> dict:
        compact = " ".join(page_text.split())
        count_match = re.search(r"(\d+)\s+Trains found", compact, re.I)
        trains = self._trains_for_date(page_text)
        result = RailwayResult(
            origin=self.origin,
            destination=self.destination,
            travel_date=self.travel_date,
            total_found=len(trains) if trains else int(count_match.group(1)) if count_match else 0,
            trains=trains[:8],
        )
        return result.model_dump()

    def extract_answer(self, page_text: str) -> str:
        result = RailwayResult.model_validate(self.structured_result(page_text))
        lines = [
            f"{result.total_found} trains found from {result.origin} to {result.destination} "
            f"for {result.travel_date}."
        ]
        for train in result.trains[:3]:
            lines.append(
                f"{train.number} {train.name}: {train.departure_time} from "
                f"{train.departure_station}, arrives {train.arrival_time} at "
                f"{train.arrival_station}, duration {train.duration}, runs {train.days}."
            )
        return "\n".join(lines)

    def reset(self) -> None:
        pass


def normalize_travel_date(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return value
