"""Compare the IPW variants (post hoc): effect skill and regret margin, mean over seeds.

    python scripts/ipw_compare.py
"""

from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

FILES = {"gru": "results_v2.jsonl", "gru-ipw": "results_ipw.jsonl", "gru-ipw-oracle": "results_ipw_oracle.jsonl",
         "gru-ipw-h": "results_ipw_h.jsonl", "gru-ipw-h-oracle": "results_ipw_h_oracle.jsonl"}


def main() -> None:
    g = collections.defaultdict(list)
    for model, f in FILES.items():
        if not Path(f).exists():
            continue
        for r in map(json.loads, Path(f).read_text().splitlines()):
            if r["model"] == model and r["seed"] in (0, 1, 2):
                g[(r["env"], r["lam"], model)].append(r)
    for env in ("sepsis", "glucose"):
        print(env)
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            cells = []
            for m in FILES:
                rs = g.get((env, lam, m), [])
                if not rs:
                    cells.append(f"{m}: -"); continue
                sk = st.mean(r["skill_dose_up"] for r in rs)
                mar = st.mean(r["cal_regret_fixed"] - r["cal_regret"] for r in rs)
                cells.append(f"{m}: {sk:+.2f}/{mar:+.1f} (n={len(rs)})")
            print(f"  lam={lam:.2f}  " + "  ".join(cells))
    print("cells: effect skill (dose up) / regret margin over the fixed dose (positive is better)")


if __name__ == "__main__":
    main()
