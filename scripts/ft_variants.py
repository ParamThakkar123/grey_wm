"""Post hoc: plain fine-tuning against output-layer-only and L2-SP fine-tuning, and the grey-box model.

    python scripts/ft_variants.py            # seeds 0-2
    python scripts/ft_variants.py 0-9        # seeds 0-9 (lambda = 1 only has seeds 3-9)
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

from paper_tables import val

FILES = ("results_v2.jsonl", "results_ft.jsonl", "results_v2_transformer.jsonl", "results_ft_head.jsonl", "results_ft_l2sp.jsonl",
         "results_seeds.jsonl", "results_seeds_ft.jsonl", "results_confirm.jsonl", "results_confirm_ft.jsonl",
         "results_confirm_transformer.jsonl")
BASE = ("gru", "gru-ode", "crn", "transformer")


def main(seeds: range = range(3)) -> None:
    g = collections.defaultdict(list)
    for f in FILES:
        if Path(f).exists():
            for line in Path(f).read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["seed"] in seeds:
                        g[(r["env"], r["lam"], r["run"].split("/")[2])].append(r)
    for env in ("sepsis", "glucose"):
        for lam in (0.5, 1.0):
            grey = g[(env, lam, "greybox")]
            print(f"== {env} lambda={lam}  grey-box: skill {np.mean(val(grey, 'skill_dose_up')):.2f} "
                  f"margin {np.mean([r['cal_regret_fixed'] - r['cal_regret'] for r in grey]):.1f} "
                  f"cf {np.mean([r['cal_cf_rmse'] for r in grey]):.2f}")
            for b in BASE:
                cells = []
                for suf in ("+ft", "+fthead", "+ftl2sp"):
                    rs = g[(env, lam, b + suf)]
                    if not rs:
                        cells.append(f"{suf:8s} -")
                        continue
                    cells.append(f"{suf:8s} n={len(rs)} skill {np.mean(val(rs, 'skill_dose_up')):5.2f} "
                                 f"margin {np.mean([r['cal_regret_fixed'] - r['cal_regret'] for r in rs]):6.1f} "
                                 f"cf {np.mean([r['cal_cf_rmse'] for r in rs]):.2f}")
                alphas = [r.get("ft_alpha") for r in g[(env, lam, b + "+ftl2sp")]]
                print(f"  {b:11s} " + " | ".join(cells) + (f"  alphas {alphas}" if alphas else ""))


if __name__ == "__main__":
    lo, hi = map(int, (sys.argv[1] if sys.argv[1:] else "0-2").split("-"))
    main(range(lo, hi + 1))
