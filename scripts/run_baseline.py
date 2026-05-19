"""
Standalone runner for the baseline NTES agent.

Usage:
    python scripts/run_baseline.py [--headed]

Requires ANTHROPIC_API_KEY in the environment or in .env at the project root.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output so the rupee symbol prints on Windows consoles.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env if python-dotenv is available.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ModuleNotFoundError:
    # Fall back to reading .env manually for the keys we care about.
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline NTES agent once.")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip Claude API calls; use canned actions to verify browser plumbing only",
    )
    args = parser.parse_args()

    from backend.agents.baseline import MODEL, BaselineAgent
    from backend.instrumentation.meter import RunMeter
    from backend.tasks.ntes import NTESTask

    task = NTESTask(headed=args.headed, config_name="baseline")
    task.reset()

    run_id = "standalone-baseline"
    meter = RunMeter(run_id)
    agent = BaselineAgent(meter, mock=args.mock)

    print(f"Model      : {MODEL}{'  [MOCK — no API calls]' if args.mock else ''}")
    print(f"Max steps  : 25")
    print(f"Start URL  : {task.start_url}")
    print(f"Train no.  : {task.train_number}")
    print(f"Downloads  : {Path('./downloads').resolve()}")
    print("-" * 60)

    wall_start = time.perf_counter()
    result = asyncio.run(agent.run(task))
    wall_seconds = time.perf_counter() - wall_start

    print()
    print("=" * 60)
    print("BASELINE RUN COMPLETE")
    print("=" * 60)
    print(f"Success          : {result.success}")
    print(f"Steps taken      : {result.steps}")
    print(f"Final step       : {result.final_step}")
    print(f"Total input tok  : {result.total_input_tokens:,}")
    print(f"Total output tok : {result.total_output_tokens:,}")
    print(f"Total tokens     : {result.total_input_tokens + result.total_output_tokens:,}")
    print(f"Total cost       : ₹{result.total_cost_inr:.4f}")
    print(f"Wall-clock time  : {wall_seconds:.1f} s")
    print("=" * 60)

    if not result.success:
        print("\nKnown failure modes to document:")
        print("  - NTES may alter labels or search flow")
        print("  - Live train status may be temporarily unavailable")
        print("  - Headless Chromium fingerprint may be detected as automation")
        print("  - Train search suggestions may require an explicit selection")


if __name__ == "__main__":
    main()
