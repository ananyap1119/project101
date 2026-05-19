"""Trajectory summariser — Layer 2.

Every SUMMARIZE_EVERY steps, the last WINDOW raw turns are compressed into a
structured JSON state object via Sarvam-M.  The raw turns are then replaced
with a single synthetic "summary" turn, keeping the context window bounded.
The two most-recent turns are always kept at full fidelity.
"""
from __future__ import annotations

import json
import re
from typing import Any

SUMMARIZE_EVERY = 4   # compress after every N steps
KEEP_RAW = 2          # always keep this many recent turns uncompressed

_SYSTEM = """\
You are a web-navigation trajectory compressor.
Given a list of recent agent steps, produce a compact JSON summary.
Return ONLY valid JSON — no markdown fences, no prose.

Schema:
{
  "visited_pages": ["list of URLs or page names"],
  "key_extracted_data": {"field": "value"},
  "current_goal_progress": "one concise sentence",
  "last_action_result": "what the last action produced",
  "blockers": ["list any obstacles; empty array if none"]
}
"""

_USER_TMPL = """\
Summarise the following {n} navigation steps into the JSON schema.

STEPS:
{history}

Return only JSON.
"""


def _turns_to_text(turns: list[dict]) -> str:
    lines: list[str] = []
    for t in turns:
        role = t.get("role", "?").upper()
        content = t.get("content", "")
        if isinstance(content, list):
            parts = [
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            content = " ".join(parts)
        lines.append(f"[{role}] {str(content)[:400]}")
    return "\n".join(lines)


async def maybe_summarise(
    conversation: list[dict],
    step: int,
    sarvam,          # SarvamClient — imported lazily to avoid circular deps
) -> tuple[list[dict], dict[str, Any] | None]:
    """
    If it is time to summarise, compress the oldest turns and return
    (updated_conversation, summary_dict).  Otherwise return the conversation
    unchanged and None.
    """
    if step % SUMMARIZE_EVERY != 0 or len(conversation) <= KEEP_RAW * 2:
        return conversation, None

    # Split: turns to compress vs. turns to keep raw
    keep_raw_count = KEEP_RAW * 2          # each step = 1 user + 1 assistant turn
    compress = conversation[:-keep_raw_count]
    keep = conversation[-keep_raw_count:]

    if not compress:
        return conversation, None

    history_text = _turns_to_text(compress)
    prompt = _USER_TMPL.format(n=len(compress), history=history_text)

    try:
        resp = await sarvam.chat(
            system=_SYSTEM,
            user=prompt,
            max_tokens=500,    # keep short so thinking doesn't eat the full budget
            temperature=0.1,
        )
        text = resp.text.strip()
        # Strip Sarvam-M <think>...</think> reasoning blocks.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Strip markdown code fences.
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        summary: dict[str, Any] = json.loads(text)
    except Exception as exc:
        summary = {
            "visited_pages": [],
            "key_extracted_data": {},
            "current_goal_progress": f"(summarisation error: {exc})",
            "last_action_result": "",
            "blockers": [],
        }

    # Replace compressed turns with a single assistant summary turn
    summary_turn = {
        "role": "assistant",
        "content": f"[TRAJECTORY SUMMARY]\n{json.dumps(summary, indent=2)}",
    }
    return [summary_turn] + keep, summary
