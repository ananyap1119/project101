"""Model cascade router — 4-tier, DeepSeek-first.

Tier 0 — DeepSeek-V3.2-Exp   (~75%): default, text-only, json_mode
Tier 1 — Qwen3-Next-80B-Think (~15%): low confidence or 1 failure, text-only
Tier 2 — Qwen3-VL-235B         (~8%): DOM empty or 2+ failures, vision
Tier 3 — Claude Sonnet 4.5     (~2%): last resort, vision

Sarvam-M is NOT part of the action cascade. It moves to Phase 4
(voice intent extraction) where reasoning tokens are an asset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backend.models.openrouter import OpenRouterClient
from backend.models.sarvam import ModelResponse

TIER_DEEPSEEK  = 0
TIER_QWEN_TEXT = 1
TIER_QWEN_VL   = 2
TIER_CLAUDE    = 3

_TIER_LABELS: dict[int, str] = {
    TIER_DEEPSEEK:  "deepseek-v3.2-exp",
    TIER_QWEN_TEXT: "qwen3-80b-thinking",
    TIER_QWEN_VL:   "qwen3-vl-235b",
    TIER_CLAUDE:    "claude-sonnet-4-5",
}

_TIER_OR_MODEL: dict[int, str] = {
    TIER_DEEPSEEK:  "deepseek/deepseek-v3.2-exp",
    TIER_QWEN_TEXT: "qwen/qwen3-next-80b-a3b-thinking",
    TIER_QWEN_VL:   "qwen/qwen3-vl-235b-a22b-instruct",
    TIER_CLAUDE:    "anthropic/claude-sonnet-4-5",
}

_TIER_MAX_TOKENS: dict[int, int] = {
    TIER_DEEPSEEK:  1024,
    TIER_QWEN_TEXT: 1024,
    TIER_QWEN_VL:   1024,
    TIER_CLAUDE:    1024,
}

# DeepSeek reliably honours json_mode; Qwen-text may not, but gets a prompt hint.
_TIER_JSON_MODE: dict[int, bool] = {
    TIER_DEEPSEEK:  True,
    TIER_QWEN_TEXT: True,
    TIER_QWEN_VL:   False,  # vision models don't need it (prompt is clear)
    TIER_CLAUDE:    False,
}


@dataclass
class RouteDecision:
    tier:         int
    model_label:  str
    or_model:     str
    reason:       str
    needs_vision: bool
    max_tokens:   int
    json_mode:    bool


@dataclass
class CascadeStats:
    tier_calls: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    decisions:  list[RouteDecision] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.tier_calls)

    def ratio_str(self) -> str:
        total = self.total or 1
        names = ["DeepSeek", "Qwen3-Text", "Qwen3-VL", "Claude"]
        parts = [
            f"{self.tier_calls[i]}/{total} {names[i]}"
            for i in range(4)
            if self.tier_calls[i] > 0
        ]
        return ", ".join(parts) if parts else "no calls yet"


class ModelRouter:
    def __init__(self) -> None:
        self.openrouter = OpenRouterClient()
        self.stats      = CascadeStats()

    def decide(
        self,
        *,
        consecutive_failures: int   = 0,
        consecutive_uncertain: int   = 0,
        dom_element_count:    int   = -1,
    ) -> RouteDecision:
        trigger = "default"
        if consecutive_failures >= 3:
            tier   = TIER_CLAUDE
            reason = f"consecutive_failures:{consecutive_failures}"
            trigger = "consecutive_failures"
        elif consecutive_failures >= 2:
            tier   = TIER_QWEN_VL
            reason = f"consecutive_failures:{consecutive_failures}"
            trigger = "consecutive_failures"
        elif consecutive_failures == 1:
            tier   = TIER_QWEN_TEXT
            reason = "consecutive_failures:1"
            trigger = "consecutive_failures"
        elif consecutive_uncertain >= 2:
            tier = TIER_QWEN_TEXT
            reason = f"consecutive_uncertain:{consecutive_uncertain}"
            trigger = "consecutive_uncertain"
        elif dom_element_count == 0 and self.stats.tier_calls[TIER_DEEPSEEK] == 0:
            tier = TIER_QWEN_VL
            reason = "dom_empty"
            trigger = "empty_dom"
        else:
            tier   = TIER_DEEPSEEK
            reason = "default"

        d = RouteDecision(
            tier=tier,
            model_label=_TIER_LABELS[tier],
            or_model=_TIER_OR_MODEL[tier],
            reason=reason,
            needs_vision=tier >= TIER_QWEN_VL,
            max_tokens=_TIER_MAX_TOKENS[tier],
            json_mode=_TIER_JSON_MODE[tier],
        )
        self.stats.decisions.append(d)
        self.stats.tier_calls[tier] += 1
        print(f"  [router] tier={tier} model={d.model_label} reason={reason}", flush=True)
        if tier > TIER_DEEPSEEK:
            print(
                f"  [cascade-escalation] tier={tier} model={d.model_label} "
                f"trigger={trigger} detail={reason}",
                flush=True,
            )
        return d

    async def call(
        self,
        *,
        decision:   RouteDecision,
        system:     str,
        messages:   list[dict],
        max_tokens: int | None = None,
    ) -> ModelResponse:
        mt = max_tokens if max_tokens is not None else decision.max_tokens
        return await self.openrouter.chat(
            system=system,
            messages=messages,
            max_tokens=mt,
            model=decision.or_model,
            json_mode=decision.json_mode,
            temperature=0.0,
        )

    async def speculate(self, *, system: str, messages: list[dict]) -> ModelResponse:
        """Always use tier-0 (DeepSeek) for speculation — cheapest real model."""
        return await self.openrouter.chat(
            system=system,
            messages=messages,
            max_tokens=256,
            model=_TIER_OR_MODEL[TIER_DEEPSEEK],
            json_mode=True,
            temperature=0.0,
        )

    async def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> ModelResponse:
        """Sarvam-compatible shim used by the trajectory summariser (Layer 2).
        Routes through tier-0 DeepSeek — text-only, json_mode."""
        return await self.openrouter.chat(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            model=_TIER_OR_MODEL[TIER_DEEPSEEK],
            json_mode=True,
            temperature=0.0,
        )
