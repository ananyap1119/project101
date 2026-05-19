from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ModuleNotFoundError:
    pass


@dataclass(frozen=True)
class RunConfig:
    name: str
    label: str
    goal_type: str
    use_som: bool
    use_summarization: bool
    use_cascade: bool
    use_speculation: bool
    headed: bool


CONFIGS = [
    RunConfig("baseline_naive", "Baseline naive", "simple", False, False, False, False, False),
    RunConfig("baseline_competent", "Baseline competent", "simple", False, False, False, False, False),
    RunConfig("som_only", "SoM only", "simple", True, False, False, False, False),
    RunConfig("som_summarization", "SoM + summarization", "simple", True, True, False, False, False),
    RunConfig("som_summarization_cascade", "SoM + summarization + cascade", "simple", True, True, True, False, False),
    RunConfig("full_optimized", "Full optimized", "simple", True, True, True, True, True),
]


def _layers(cfg: RunConfig) -> list[int]:
    layers: list[int] = []
    if cfg.use_som:
        layers.append(1)
    if cfg.use_summarization:
        layers.append(2)
    if cfg.use_cascade:
        layers.append(3)
    if cfg.use_speculation:
        layers.append(4)
    return layers


async def _post_result(result: dict[str, Any]) -> None:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post("http://127.0.0.1:8000/benchmark-results", json=result)
    except Exception as exc:
        print(f"[dashboard] result post failed: {exc}", flush=True)


async def _run_config(index: int, cfg: RunConfig) -> dict[str, Any]:
    from backend.agents.baseline import BaselineAgent
    from backend.agents.optimized import Flags, OptimizedAgent
    from backend.instrumentation.meter import RunMeter
    from backend.tasks.ntes import NTESTask

    print(f"\n=== Running config {index}/6: {cfg.label} ===", flush=True)
    print(f"RunConfig: {json.dumps(asdict(cfg))}", flush=True)

    meter = RunMeter(f"ntes-{cfg.name}", budget_cap_inr=5.0)
    task = NTESTask(headed=cfg.headed, config_name=cfg.name)  # type: ignore[arg-type]
    task.reset()

    if cfg.name == "baseline_naive":
        agent = BaselineAgent(meter, mode="naive")
    elif cfg.name == "baseline_competent":
        agent = BaselineAgent(meter, mode="competent")
    else:
        agent = OptimizedAgent(
            meter,
            flags=Flags.from_layers(_layers(cfg)),
        )

    started = time.perf_counter()
    status = "failed"
    last_event_count = 0
    last_event_at = time.perf_counter()

    async def monitor() -> None:
        nonlocal last_event_count, last_event_at
        while True:
            await asyncio.sleep(1)
            if len(meter.trace) != last_event_count:
                last_event_count = len(meter.trace)
                last_event_at = time.perf_counter()
                event = meter.trace[-1]
                tokens = event.vision_tokens + event.context_tokens
                print(
                    f"[step-event] tier=? model={event.current_model} "
                    f"action={event.current_step} tokens_so_far={tokens} "
                    f"cost_so_far=₹{event.cumulative_cost_inr:.2f} "
                    f"elapsed={event.elapsed_seconds:.1f}s",
                    flush=True,
                )
            elif time.perf_counter() - last_event_at > 10:
                print("[waiting] waiting for browser navigation or model response", flush=True)
                last_event_at = time.perf_counter()

    monitor_task = asyncio.create_task(monitor())
    timeout_seconds = (
        60 if cfg.name == "baseline_naive"
        else 45 if cfg.name == "baseline_competent"
        else 90
    )
    try:
        result = await asyncio.wait_for(agent.run(task), timeout=timeout_seconds)
        if result.total_cost_inr >= 5:
            status = "cost_runaway"
        elif result.success:
            status = "completed"
        else:
            status = "failed"
    except asyncio.TimeoutError:
        status = "timeout"
        result = None
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    wall = time.perf_counter() - started
    if result is None:
        usage_events = [event for event in meter.trace if event.event == "usage"]
        last_usage = usage_events[-1] if usage_events else None
        partial_vision = last_usage.vision_tokens if last_usage else 0
        partial_context = last_usage.context_tokens if last_usage else 0
        partial_cost = last_usage.cumulative_cost_inr if last_usage else 0.0
        summary = {
            "name": cfg.name,
            "label": cfg.label,
            "status": status,
            "steps": len(usage_events),
            "vision_tokens": partial_vision,
            "context_tokens": partial_context,
            "total_tokens": partial_vision + partial_context,
            "cost_inr": partial_cost,
            "wall_seconds": round(wall, 1),
            "cascade_distribution": "n/a",
            "meaningful_speculation_hit_rate": "n/a",
            "speculation_disabled_reason": "",
            "spec_rejections_by_reason": {},
            "failure_reason": "timeout",
            "route_log": [],
            "extracted_status_string": "",
        }
    else:
        detail: dict[str, Any] = {}
        for event in reversed(meter.trace):
            if event.detail:
                try:
                    detail = json.loads(event.detail)
                    break
                except json.JSONDecodeError:
                    pass
        summary = {
            "name": cfg.name,
            "label": cfg.label,
            "status": status,
            "steps": result.steps,
            "vision_tokens": result.total_input_tokens,
            "context_tokens": result.total_output_tokens,
            "total_tokens": result.total_input_tokens + result.total_output_tokens,
            "cost_inr": result.total_cost_inr,
            "wall_seconds": round(wall, 1),
            "cascade_distribution": detail.get("cascade", "n/a"),
            "meaningful_speculation_hit_rate": detail.get("speculation_meaningful", "n/a"),
            "speculation_disabled_reason": detail.get("speculation_disabled_reason", ""),
            "spec_rejections_by_reason": detail.get("spec_rejections_by_reason", {}),
            "failure_reason": result.final_step if status == "failed" else "",
            "route_log": detail.get("route_log", []),
            "extracted_status_string": result.extracted_status_string,
        }

    await _post_result(summary)
    print(
        f"[done] status={summary['status']} steps={summary['steps']} "
        f"tokens={summary['total_tokens']} cost=₹{summary['cost_inr']:.4f} "
        f"time={summary['wall_seconds']:.1f}s answer={summary['extracted_status_string']!r}",
        flush=True,
    )
    if cfg.name == "full_optimized":
        print(
            "[full-optimized-log] "
            f"meaningful_speculation_hit_rate={summary['meaningful_speculation_hit_rate']} "
            f"rejections={summary['spec_rejections_by_reason']} "
            f"disabled_reason={summary['speculation_disabled_reason'] or 'n/a'} "
            f"failure_reason={summary['failure_reason'] or 'n/a'}",
            flush=True,
        )
    if cfg.name in {"som_summarization_cascade", "full_optimized"}:
        print(f"[route-log-summary] {cfg.name}", flush=True)
        for entry in summary.get("route_log", []):
            print(
                "  "
                f"step={entry.get('step')} tier={entry.get('tier')} "
                f"trigger={entry.get('trigger')} raw={entry.get('raw_value')} "
                f"action={entry.get('action')} confidence={entry.get('confidence')} "
                f"page_changed={entry.get('page_changed')} "
                f"success={entry.get('action_success')} detail={entry.get('success_detail')} "
                f"cache_hit={entry.get('cache_hit')}",
                flush=True,
            )
    return summary


def _print_running_totals(index: int, results: list[dict[str, Any]]) -> None:
    baseline = next((r for r in results if r["name"] == "baseline_naive"), None)
    current = results[-1]
    if baseline and baseline["total_tokens"]:
        multiplier = baseline["total_tokens"] / max(current["total_tokens"], 1)
        print(
            f"After config {index}/6: baseline_naive={baseline['total_tokens']} tokens, "
            f"current={current['total_tokens']} tokens, multiplier so far = {multiplier:.2f}x",
            flush=True,
        )
    else:
        print(
            f"After config {index}/6: baseline_naive={current['total_tokens']} tokens, "
            f"current={current['total_tokens']} tokens, multiplier so far = 1.00x",
            flush=True,
        )


def _print_final(results: list[dict[str, Any]]) -> None:
    print("\n=== Final per-config results ===", flush=True)
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)

    by_name = {result["name"]: result for result in results}
    naive = by_name.get("baseline_naive")
    competent = by_name.get("baseline_competent")
    full = by_name.get("full_optimized")
    if full:
        print("\n=== Summary table ===", flush=True)
        for base, label in [
            (naive, "baseline_naive"),
            (competent, "baseline_competent"),
        ]:
            if base:
                print(
                    f"Full optimized vs {label}: "
                    f"tokens {base['total_tokens'] / max(full['total_tokens'], 1):.2f}x, "
                    f"cost {base['cost_inr'] / max(full['cost_inr'], 0.000001):.2f}x, "
                    f"latency {base['wall_seconds'] / max(full['wall_seconds'], 0.1):.2f}x",
                    flush=True,
                )

    print("Per-layer marginal contribution:", flush=True)
    for left, right, label in [
        ("baseline_naive", "baseline_competent", "naive->competent"),
        ("baseline_competent", "som_only", "competent->som"),
        ("som_only", "som_summarization", "som->summ"),
        ("som_summarization", "som_summarization_cascade", "summ->cascade"),
        ("som_summarization_cascade", "full_optimized", "cascade->full"),
    ]:
        a = by_name.get(left)
        b = by_name.get(right)
        if a and b:
            print(
                f"  {label}: tokens {a['total_tokens']}->{b['total_tokens']}, "
                f"cost ₹{a['cost_inr']:.4f}->₹{b['cost_inr']:.4f}, "
                f"time {a['wall_seconds']:.1f}s->{b['wall_seconds']:.1f}s",
                flush=True,
            )


async def main() -> None:
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            await client.delete("http://127.0.0.1:8000/benchmark-results")
        except Exception:
            pass

    print("Six-config NTES benchmark starting. Paid model calls are capped at ₹5 per config.", flush=True)
    results: list[dict[str, Any]] = []
    for index, cfg in enumerate(CONFIGS, 1):
        result = await _run_config(index, cfg)
        results.append(result)
        _print_running_totals(index, results)
        lower_answer = result["extracted_status_string"].lower()
        if "captcha" in lower_answer or "bot" in lower_answer:
            print("[stop] Captcha or anti-bot signal detected. Stopping all runs.", flush=True)
            break
    _print_final(results)


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(main())
