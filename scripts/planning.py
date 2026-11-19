"""Multi-step planning with each world model (post hoc, saved models, lambda = 1).

Each model picks, for each test patient, a schedule for the first drug over the
whole counterfactual horizon (5 h sepsis, 6 h glucose): three equal segments,
each at 0, 50% or 100% of the maximum dose (27 schedules; the second drug keeps
its recorded doses). It picks the schedule whose predicted outcome stays closest
to the target on average over the horizon. The simulator then scores the
schedule. Regret: true mean distance from target of the chosen schedule minus
that of the patient's best schedule. Bar to beat: the one schedule that is best
for all patients together.

    python scripts/planning.py     # writes results_planning.jsonl; prints the means
"""

from __future__ import annotations

import collections
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

from grey_wm.audit import TARGETS
from grey_wm.data import load_split
from grey_wm.metrics import _Predictor, _truth
from grey_wm.models import build_model

MODELS = ("greybox", "greybox_pkpd", "gru", "gru-ipw", "gru-ode", "crn", "transformer", "gru-ipw-oracle", "gru-ipw-h", "gru-ipw-h-oracle",
          "gru+ft", "gru-ode+ft", "crn+ft", "transformer+ft")
LEVELS = (0.0, 0.5, 1.0)
SEGMENTS = 3


def schedules(base: np.ndarray, high: float) -> list[np.ndarray]:
    h = base.shape[1]
    edges = np.linspace(0, h, SEGMENTS + 1).round().astype(int)
    out = []
    for combo in itertools.product(LEVELS, repeat=SEGMENTS):
        a = base.copy()
        for (lo, hi), lev in zip(zip(edges[:-1], edges[1:]), combo):
            a[:, lo:hi, 0] = lev * high
        out.append(a)
    return out


def main(lam: float = 1.0, out: str = "results_planning.jsonl") -> None:
    torch.set_num_threads(2)
    done = set()
    if Path(out).exists():
        done = {(r["env"], r["model"], r["seed"]) for r in map(json.loads, Path(out).read_text().splitlines())}
    for env in ("sepsis", "glucose"):
        split = load_split("data", env, lam)
        truth_cost = None
        for tag in MODELS:
            for d in sorted(Path(f"runs_v2/{env}/lam{lam:.2f}/{tag}").glob("s*/model")):
                seed = int(d.parent.name[1:])
                if (env, tag, seed) in done or not (d / "model.pt").exists():
                    continue
                cfg = json.loads((d / "model.json").read_text())
                m = dict(cfg["model"])
                kind = m.pop("kind")
                e = cfg["eval"]
                test = split["test"].subset(slice(0, min(e["test_trajectories"], len(split["test"]))))
                spec, k, o = test.spec, e["n_context"] - 1, test.spec.outcome_index
                h = e["cf_horizon"]
                base = test["actions"][:, k : k + h].copy()
                plans = schedules(base, spec.action_max[0])
                target = TARGETS[spec.name][0]
                if truth_cost is None:
                    truth_cost = np.stack([np.abs(_truth(test, e["n_context"], a)[..., o] - target).mean(1)
                                           for a in plans], 1)  # (patients, schedules)
                model = build_model(kind, spec, **m)
                model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
                model.eval()
                with torch.no_grad():
                    pred = _Predictor(model, test, e["n_context"], torch.device("cpu"))
                    cost = np.stack([np.abs(pred(h, a)[0][..., o] - target).mean(1) for a in plans], 1)
                rows = np.arange(len(cost))
                best = truth_cost.min(1)
                r = {"env": env, "lam": lam, "model": tag, "seed": seed,
                     "plan_regret": float(np.mean(truth_cost[rows, cost.argmin(1)] - best)),
                     "plan_regret_fixed": float((truth_cost - best[:, None]).mean(0).min())}
                with open(out, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(env, tag, seed, round(r["plan_regret"], 2), "fixed", round(r["plan_regret_fixed"], 2), flush=True)
    summary(out)


def summary(out: str) -> None:
    g = collections.defaultdict(list)
    for r in map(json.loads, Path(out).read_text().splitlines()):
        g[(r["env"], r["model"])].append(r)
    for (env, tag), rs in sorted(g.items()):
        reg = [r["plan_regret"] for r in rs]
        print(f"{env:7s} {tag:16s} n={len(rs)} regret={st.mean(reg):.2f} (min {min(reg):.2f}, max {max(reg):.2f}) "
              f"fixed={rs[0]['plan_regret_fixed']:.2f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--summary"]:
        summary(sys.argv[2])
    elif sys.argv[1:2] == ["--lam"]:  # e.g. --lam 0.5 results_planning_lam050.jsonl
        main(float(sys.argv[2]), sys.argv[3])
    else:
        main()
