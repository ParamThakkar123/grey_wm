"""Pass/fail report on trained world models, from a results file.

    python scripts/audit.py                         # after calibration
    python scripts/audit.py --stage raw             # as trained
    python scripts/audit.py --envs glucose --lams 1

Each gate is decided on the mean over seeds. The ``deployable`` gates use nothing
a real deployment would lack, so they answer the practical question: would these
checks have caught a broken model without a simulator to compare against?
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from grey_wm.audit import GATES, derived

ACTIONS = {"sepsis": ("norepinephrine", "fluid"), "glucose": ("basal", "bolus")}


def passed(value: float, op: str, limit: float) -> bool:
    if value != value:  # nan: the check could not be made
        return False
    return value >= limit if op == "ge" else value <= limit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results_v2.jsonl")
    p.add_argument("--stage", choices=("raw", "cal"), default="cal")
    p.add_argument("--envs", nargs="+", default=list(ACTIONS))
    p.add_argument("--lams", type=float, nargs="+", default=None)
    p.add_argument("--out", default=None, help="also write the report as JSON")
    args = p.parse_args()

    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["env"] in args.envs and (args.lams is None or r["lam"] in args.lams):
            groups[(r["env"], r["lam"], r["run"].split("/")[2])].append(r)  # run tag keeps misspec variants apart

    prefix = "cal_" if args.stage == "cal" else ""
    report = []
    for (env, lam, model), runs in sorted(groups.items()):
        vals = [derived(r, ACTIONS[env], prefix) for r in runs]
        mean = {k: st.mean(v[k] for v in vals) for k in vals[0]}
        fails = {tier: [] for tier in ("benchmark", "deployable")}
        for tier, key, op, limit in GATES:
            if not passed(mean[key], op, limit):
                fails[tier].append(f"{key}={mean[key]:.2f}")
        report.append({"env": env, "lam": lam, "model": model, "seeds": len(runs), **mean, "fails": fails})
        verdict = "PASS" if not fails["benchmark"] and not fails["deployable"] else "FAIL"
        print(f"{verdict} {env} lam={lam:g} {model} ({len(runs)} seeds)")
        for tier in fails:
            if fails[tier]:
                print(f"     {tier}: " + ", ".join(fails[tier]))
    caught = sum(1 for r in report if r["fails"]["benchmark"] and r["fails"]["deployable"])
    broken = sum(1 for r in report if r["fails"]["benchmark"])
    print(f"\n{broken} of {len(report)} models fail the benchmark gates; deployable gates flag {caught} of those {broken}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
