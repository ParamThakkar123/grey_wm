"""Shifted randomised sets (post hoc, lambda = 1): effect skill and regret margin per half.

    python scripts/shift_summary.py            # all seeds present
    python scripts/shift_summary.py 0 1 2      # only these seeds
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

HALVES = ("first50", "mild50", "sick50")
MODELS = ("greybox", "gru", "gru-ode", "crn", "transformer")


def main(seeds: set[int] | None) -> None:
    g = collections.defaultdict(list)
    for r in map(json.loads, Path("results_shift.jsonl").read_text().splitlines()):
        g[(r["env"], r["model"], r["variant"])].append(r)
    for r in map(json.loads, Path("results_shift_ft.jsonl").read_text().splitlines()):
        base, half = r["model"].split("+ft_")
        g[(r["env"], base, half)].append(r)
    for env in ("sepsis", "glucose"):
        print(env)
        for m in MODELS:
            cells = []
            for h in HALVES:
                rs = [r for r in g.get((env, m, h), []) if seeds is None or r["seed"] in seeds]
                if not rs:
                    cells.append(f"{h}: -"); continue
                sk = [r["cal_skill_dose_up"] for r in rs]
                mar = [r["cal_regret_fixed"] - r["cal_regret"] for r in rs]
                beat = sum(x > 0 for x in mar)
                cells.append(f"{h}: skill {st.mean(sk):+.2f} margin {st.mean(mar):+.1f} "
                             f"[{min(mar):+.1f},{max(mar):+.1f}] beats fixed {beat}/{len(rs)}")
            print(f"  {m:12s} " + " | ".join(cells))


if __name__ == "__main__":
    main({int(a) for a in sys.argv[1:]} or None)
