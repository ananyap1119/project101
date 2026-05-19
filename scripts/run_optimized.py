"""
Layered benchmark runner for the optimized NTES agent.

Usage:
    # Run all four layers (full optimized):
    python scripts/run_optimized.py

    # Run with specific layers only (comma-separated; 1=SoM 2=sum 3=cascade 4=spec):
    python scripts/run_optimized.py --layers 1
    python scripts/run_optimized.py --layers 1,2
    python scripts/run_optimized.py --layers 1,2,3

    # Benchmark all four configurations sequentially and print a comparison table:
    python scripts/run_optimized.py --benchmark

    # Mock mode (no API keys needed — verifies browser plumbing only):
    python scripts/run_optimized.py --mock --headed
    python scripts/run_optimized.py --benchmark --mock --headed
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ModuleNotFoundError:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip()


# ─────────────────────────────────────────────────────────────────────────────

def _layer_label(layers: list[int]) -> str:
    mapping = {
        (1,): "som_only",
        (1, 2): "som_summarization",
        (1, 2, 3): "som_summarization_cascade",
        (1, 2, 3, 4): "full_optimized",
    }
    key = tuple(sorted(layers))
    if key in mapping:
        return mapping[key]
    names = {1: "som", 2: "summarization", 3: "cascade", 4: "speculation"}
    return "+".join(names[l] for l in sorted(layers)) if layers else "baseline_clone"


async def run_once(layers: list[int], *, headed: bool, mock: bool) -> dict:
    from backend.agents.optimized import Flags, OptimizedAgent
    from backend.instrumentation.meter import RunMeter
    from backend.tasks.ntes import NTESTask

    label = _layer_label(layers)
    known_configs = {
        "som_only",
        "som_summarization",
        "som_summarization_cascade",
        "full_optimized",
    }
    task = NTESTask(
        headed=headed,
        config_name=label if label in known_configs else "full_optimized",
    )
    task.reset()
    meter = RunMeter("opt-bench")
    flags = Flags.from_layers(layers, mock=mock)
    agent = OptimizedAgent(meter, flags=flags)

    print(f"\n{'='*60}")
    print(f"RUN: {label}  mock={mock}  headed={headed}")
    print(f"{'='*60}")

    wall_start = time.perf_counter()
    result = await agent.run(task)
    wall_s = time.perf_counter() - wall_start

    cascade = agent.router.stats
    return {
        "label": label,
        "layers": layers,
        "success": result.success,
        "steps": result.steps,
        "input_tokens": result.total_input_tokens,
        "output_tokens": result.total_output_tokens,
        "total_tokens": result.total_input_tokens + result.total_output_tokens,
        "cost_inr": result.total_cost_inr,
        "wall_s": round(wall_s, 1),
        "sarvam_calls": cascade.sarvam_calls,
        "claude_calls": cascade.claude_calls,
        "cascade_ratio": cascade.ratio_str() if cascade.total else "n/a",
        "final_step": result.final_step,
    }


def _print_result(r: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"RESULT: {r['label']}")
    print(f"{'─'*60}")
    print(f"  Success          : {r['success']}")
    print(f"  Steps            : {r['steps']}")
    print(f"  Input tokens     : {r['input_tokens']:,}")
    print(f"  Output tokens    : {r['output_tokens']:,}")
    print(f"  Total tokens     : {r['total_tokens']:,}")
    print(f"  Cost             : ₹{r['cost_inr']:.4f}")
    print(f"  Wall-clock       : {r['wall_s']}s")
    print(f"  Cascade          : {r['cascade_ratio']}")
    print(f"  Final step       : {r['final_step']}")


def _print_comparison(results: list[dict]) -> None:
    if not results:
        return
    base = results[0]
    print(f"\n{'='*72}")
    print(f"{'BENCHMARK COMPARISON':^72}")
    print(f"{'='*72}")
    hdr = f"{'Config':<22} {'Tokens':>10} {'Cost(Rs.)':>10} {'Wall(s)':>8} {'Steps':>6} {'Cascade'}"
    print(hdr)
    print("─" * 72)
    for r in results:
        tok_mult = (
            f"({r['total_tokens']/base['total_tokens']:.2f}x)"
            if base["total_tokens"] else ""
        )
        cost_mult = (
            f"({r['cost_inr']/base['cost_inr']:.2f}x)"
            if base["cost_inr"] else ""
        )
        print(
            f"  {r['label']:<20} "
            f"{r['total_tokens']:>8,} {tok_mult:<6} "
            f"{r['cost_inr']:>8.4f} {cost_mult:<6} "
            f"{r['wall_s']:>6.1f}  "
            f"{r['steps']:>4}   "
            f"{r['cascade_ratio']}"
        )
    print("─" * 72)
    print("(multipliers relative to first row)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the optimized NTES agent.")
    parser.add_argument(
        "--layers",
        default="1,2,3,4",
        help="Comma-separated layer numbers to enable: 1=SoM 2=sum 3=cascade 4=spec",
    )
    parser.add_argument("--headed",    action="store_true")
    parser.add_argument("--mock",      action="store_true",
                        help="Skip real API calls; verify browser plumbing only")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run all four layer combinations and print comparison")
    args = parser.parse_args()

    if args.benchmark:
        configs = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]
        results = []
        for layers in configs:
            r = asyncio.run(run_once(layers, headed=args.headed, mock=args.mock))
            _print_result(r)
            results.append(r)
        _print_comparison(results)
    else:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
        r = asyncio.run(run_once(layers, headed=args.headed, mock=args.mock))
        _print_result(r)

    print("\nKnown failure modes (documented, not fixed):")
    print("  - NTES may alter labels or search flow")
    print("  - Live train status may be temporarily unavailable")
    print("  - Sarvam-M pricing is estimated (not published); update when available")
    print("  - Speculation metadata uses dict attributes — replace with a dataclass")


if __name__ == "__main__":
    main()
