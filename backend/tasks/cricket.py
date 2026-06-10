from __future__ import annotations

import re
from urllib.parse import quote_plus

from pydantic import BaseModel

from backend.tasks.ntes import ConfigName


class CricketTask(BaseModel):
    name: str = "cricket"
    config_name: ConfigName = "full_optimized"
    query: str = ""   # team name, tournament name, or match description
    headed: bool = False
    speculation_enabled: bool = False

    @property
    def start_url(self) -> str:
        q = f"{self.query} cricket score" if self.query else "live cricket score today"
        return f"https://www.bing.com/search?q={quote_plus(q)}"

    @property
    def objective(self) -> str:
        subject = self.query or "the live match"
        return (
            f"Find the cricket score for {subject} from the Bing search results page. "
            "A score card appears near the top of the page — read it directly without clicking. "
            "Report the two teams, current score (e.g. 287/4 in 45.2 overs), and match status."
        )

    def success(self, page_text: str) -> bool:
        has_score = bool(re.search(r"\d+/\d+", page_text)) or bool(re.search(r"\d+ runs?", page_text.lower()))
        has_cricket = any(
            k in page_text.lower()
            for k in ["cricket", "innings", "over", "wicket", "run", "batting", "bowling", "vs"]
        )
        return has_score and has_cricket

    def extract_answer(self, page_text: str) -> str:
        compact = " ".join(page_text.split())
        m = re.search(r"\d+/\d+", compact)
        if m:
            start = max(0, m.start() - 200)
            end = min(len(compact), m.end() + 200)
            return compact[start:end].strip()
        # Fallback: find "vs" context
        m2 = re.search(r"\w+\s+vs\s+\w+", compact, re.IGNORECASE)
        if m2:
            start = max(0, m2.start() - 50)
            end = min(len(compact), m2.end() + 300)
            return compact[start:end].strip()
        return compact[:400]

    def reset(self) -> None:
        pass
