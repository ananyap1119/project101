from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

from pydantic import BaseModel, Field

# ₹425 = $5 × ₹85/USD  — hard cap shared across all agents in a run
_DEFAULT_BUDGET_CAP_INR: float = 425.0


class BudgetExceededError(RuntimeError):
    """Raised when cumulative spend exceeds the run's budget cap."""


class MeterEvent(BaseModel):
    run_id: str
    agent: str
    event: Literal["usage", "agent_complete", "error", "done"] = "usage"
    vision_tokens: int = 0
    context_tokens: int = 0
    cumulative_cost_inr: float = 0.0
    elapsed_seconds: float = 0.0
    current_step: str = "queued"
    current_model: str = "none"
    latency_ms: int = 0
    detail: str | None = None
    created_at: float = Field(default_factory=time.time)

    def to_sse(self) -> str:
        return f"event: {self.event}\ndata: {json.dumps(self.model_dump())}\n\n"


class RunMeter:
    def __init__(self, run_id: str, budget_cap_inr: float = _DEFAULT_BUDGET_CAP_INR) -> None:
        self.run_id = run_id
        self.started_at = time.perf_counter()
        self.trace: list[MeterEvent] = []
        self._subscribers: list[asyncio.Queue[MeterEvent]] = []
        self._totals: dict[str, dict[str, float]] = {}
        self._budget_cap_inr = budget_cap_inr

    # ── subscription ──────────────────────────────────────────────────────────

    def subscribe(self) -> asyncio.Queue[MeterEvent]:
        queue: asyncio.Queue[MeterEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[MeterEvent]) -> None:
        self._subscribers.remove(queue)

    # ── budget ────────────────────────────────────────────────────────────────

    @property
    def total_cost_inr(self) -> float:
        return sum(v["cumulative_cost_inr"] for v in self._totals.values())

    # ── publishing ────────────────────────────────────────────────────────────

    async def publish_model_usage(
        self,
        *,
        agent: str,
        current_step: str,
        model: str,
        vision_tokens: int,
        context_tokens: int,
        latency_ms: int,
        cost_inr: float,
    ) -> MeterEvent:
        totals = self._totals.setdefault(
            agent,
            {"vision_tokens": 0.0, "context_tokens": 0.0, "cumulative_cost_inr": 0.0},
        )
        totals["vision_tokens"]      += vision_tokens
        totals["context_tokens"]     += context_tokens
        totals["cumulative_cost_inr"] += cost_inr

        # Check budget after accumulating so the event is still published
        over_budget = (
            self._budget_cap_inr is not None
            and self.total_cost_inr > self._budget_cap_inr
        )

        event = await self.publish(
            MeterEvent(
                run_id=self.run_id,
                agent=agent,
                event="usage",
                vision_tokens=int(totals["vision_tokens"]),
                context_tokens=int(totals["context_tokens"]),
                cumulative_cost_inr=round(totals["cumulative_cost_inr"], 2),
                elapsed_seconds=round(time.perf_counter() - self.started_at, 2),
                current_step=current_step,
                current_model=model,
                latency_ms=latency_ms,
            )
        )

        if over_budget:
            raise BudgetExceededError(
                f"Budget cap ₹{self._budget_cap_inr:.0f} exceeded "
                f"(spent ₹{self.total_cost_inr:.2f})"
            )

        return event

    async def publish(self, event: MeterEvent) -> MeterEvent:
        self.trace.append(event)
        for subscriber in list(self._subscribers):
            await subscriber.put(event)
        return event

    async def close(self) -> None:
        await self.publish(
            MeterEvent(
                run_id=self.run_id,
                agent="system",
                event="done",
                elapsed_seconds=round(time.perf_counter() - self.started_at, 2),
                current_step="done",
                current_model="none",
            )
        )
