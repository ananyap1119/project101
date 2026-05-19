from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RecipeStore:
    supabase_url: str | None = None
    supabase_key: str | None = None

    async def get_recipe(self, task_name: str) -> dict[str, Any] | None:
        return None

    async def save_recipe(self, task_name: str, recipe: dict[str, Any]) -> dict[str, Any]:
        return {"task_name": task_name, "recipe": recipe, "stored": False}
