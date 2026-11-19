"""Print results.jsonl as a quick comparison table (mean over seeds).

    python scripts/table.py [--results results.jsonl] [--env sepsis]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ("rmse_h5", "rmse_h20", "coverage_h20", "cf_rmse", "norepinephrine_sign_agree", "norepinephrine_sensitivity",
        "basal_sign_agree", "basal_sensitivity", "bolus_sign_agree", "bolus_sensitivity", "residual_norm")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["results.jsonl", "results_transformer.jsonl"])
    p.add_argument("--env", default=None)
    args = p.parse_args()
    groups = defaultdict(list)
    lines = [line for path in args.results if Path(path).exists() for line in open(path)]
    for line in lines:
        r = json.loads(line)
        if args.env and r["env"] != args.env:
            continue
        tag = "".join(f" {k}={r[k]:g}" for k in ("ke_scale", "ec50_scale") if k in r)
        groups[(r["env"], r["lam"], r["model"] + tag)].append(r)
    keys = [k for k in KEYS if any(k in r for rs in groups.values() for r in rs)]
    print(f"{'env':8}{'lam':>6} {'model':28}{'n':>3}" + "".join(f"{k[:16]:>18}" for k in keys))
    for (env, lam, model), rs in sorted(groups.items()):
        cells = []
        for k in keys:
            v = [r[k] for r in rs if k in r]
            cells.append(f"{np.mean(v):9.3f}+-{np.std(v):<6.3f}" if v else f"{'-':>18}")
        print(f"{env:8}{lam:6.2f} {model:28}{len(rs):3}" + "".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
