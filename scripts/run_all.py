"""Run every experiment in the paper, in priority order, retrying failed runs.

    python scripts/run_all.py cpu --jobs 5      # grey_wm venv (CPU-only torch)
    python scripts/run_all.py gpu               # a CUDA venv: the transformer runs

Lane ``cpu``:
  1. main lambda grid: grey-box and the recurrent baselines, both envs, 3 seeds
  2. ablations at lambda = 0.75, both envs, 3 seeds
  3. prior misspecification sweep on sepsis (grey-box with / without residual)
Lane ``gpu``:
  1. transformer on the main lambda grid (one run at a time on the GPU)

A run that dies (memory pressure on this machine) resumes from its last
checkpoint on the next pass; finished runs are skipped. Up to ``--passes``
passes per phase.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

LAMS = ["0", "0.25", "0.5", "0.75", "1"]
SEEDS = ["0", "1", "2"]
ENVS = ["sepsis", "glucose"]

PHASES = {
    "cpu": [
        ("main grid", ["--envs", *ENVS, "--lams", *LAMS, "--seeds", *SEEDS,
                       "--models", "greybox", "gru", "gru-ode", "crn"]),
        ("ablations", ["--envs", *ENVS, "--lams", "0.75", "--seeds", *SEEDS,
                       "--models", "greybox-latent", "greybox-actres", "greybox-nores", "greybox-leak"]),
        ("misspecification", ["--envs", "sepsis", "--seeds", *SEEDS, "--models", "greybox", "greybox-nores", "--misspec"]),
    ],
    "gpu": [
        ("transformer grid", ["--envs", *ENVS, "--lams", *LAMS, "--seeds", *SEEDS, "--models", "transformer",
                              "--device", "cuda", "--results", "results_transformer.jsonl"]),
    ],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("lane", choices=sorted(PHASES))
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--max-steps", default="3000")
    p.add_argument("--passes", type=int, default=3)
    args = p.parse_args()
    sweep = str(Path(__file__).with_name("sweep.py"))
    for name, phase_args in PHASES[args.lane]:
        cmd = [sys.executable, "-u", sweep, *phase_args, "--max-steps", args.max_steps]
        if args.lane == "cpu":
            cmd += ["--jobs", str(args.jobs)]
        for attempt in range(1, args.passes + 1):
            print(f"=== {args.lane}: {name}, pass {attempt} ({time.strftime('%Y-%m-%d %H:%M')})", flush=True)
            out = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            lines = [ln for ln in out.stdout.splitlines() if ln.startswith(("[done]", "[failed]")) or "runs to do" in ln]
            print("\n".join(lines), flush=True)
            if out.returncode != 0:
                print(out.stderr[-3000:], flush=True)
            if not any(ln.startswith("[failed]") for ln in lines) and out.returncode == 0:
                break
    print(f"=== {args.lane} lane finished ({time.strftime('%Y-%m-%d %H:%M')})", flush=True)


if __name__ == "__main__":
    main()
