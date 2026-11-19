"""Closed-loop planning: re-plan on new readings (post hoc, saved models, lambda = 1).

As ``planning.py`` (three segments of the first drug at 0, 50% or 100% of the
maximum), but the schedule is chosen one segment at a time. At the start of each
segment the model observes the readings the simulated patient produced so far
(clean simulator output plus measurement noise, at the trajectory's own
measurement times) and picks the best remaining segments; only the next one is
applied. The applied schedule is still one of the 27, so regret is measured
against the same per-patient best schedule as in open loop.

    python scripts/planning_closed_loop.py    # writes results_planning_cl.jsonl; prints the means
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
from grey_wm.data import load_split, model_inputs
from grey_wm.metrics import _truth
from grey_wm.models import build_model, filter_rollout
from planning import LEVELS, MODELS, SEGMENTS, schedules


def closed_loop(model, test, n_context: int, h: int, target: float, rng: np.random.Generator) -> np.ndarray:
    spec, o, k = test.spec, test.spec.outcome_index, n_context - 1
    high = spec.action_max[0]
    edges = np.linspace(0, h, SEGMENTS + 1).round().astype(int)
    inp = {key: v.copy() for key, v in model_inputs(test).items()}
    applied = test["actions"][:, k : k + h].copy()
    n = len(applied)
    std, mean = np.asarray(spec.obs_std, np.float32), np.asarray(spec.obs_mean, np.float32)
    for j in range(SEGMENTS):
        ctx = n_context + edges[j]
        batch = {key: torch.as_tensor(v) for key, v in inp.items()}
        costs, combos = [], list(itertools.product(LEVELS, repeat=SEGMENTS - j))
        for combo in combos:
            fut = applied.copy()
            for seg, lev in zip(range(j, SEGMENTS), combo):
                fut[:, edges[seg] : edges[seg + 1], 0] = lev * high
            with torch.no_grad():
                out = filter_rollout(model, batch, ctx, h - edges[j], torch.as_tensor(fut[:, edges[j] :]))
            pred = out.mean.float().numpy()[..., o] * std[o] + mean[o]
            costs.append(np.abs(pred - target).mean(1))
        pick = np.stack(costs, 1).argmin(1)
        applied[:, edges[j] : edges[j + 1], 0] = np.array([combos[i][0] for i in pick])[:, None] * high
        if j + 1 < SEGMENTS:  # the readings the patient produces under the doses applied so far
            clean = _truth(test, n_context, applied)[:, : edges[j + 1]]
            y = clean + rng.standard_normal(clean.shape) * np.asarray(spec.obs_noise)
            sl = slice(n_context, n_context + edges[j + 1])
            inp["x"][:, sl] = ((y - mean) / std * inp["mask"][:, sl]).astype(np.float32)
            inp["actions"][:, k : k + edges[j + 1]] = applied[:, : edges[j + 1]]
    return applied


def main(lam: float = 1.0, out: str = "results_planning_cl.jsonl") -> None:
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
                h, o, k = e["cf_horizon"], test.spec.outcome_index, e["n_context"] - 1
                target = TARGETS[test.spec.name][0]
                if truth_cost is None:
                    plans = schedules(test["actions"][:, k : k + h].copy(), test.spec.action_max[0])
                    truth_cost = np.stack([np.abs(_truth(test, e["n_context"], a)[..., o] - target).mean(1)
                                           for a in plans], 1)
                model = build_model(kind, test.spec, **m)
                model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
                model.eval()
                applied = closed_loop(model, test, e["n_context"], h, target, np.random.default_rng(seed))
                cost = np.abs(_truth(test, e["n_context"], applied)[..., o] - target).mean(1)
                best = truth_cost.min(1)
                r = {"env": env, "lam": lam, "model": tag, "seed": seed,
                     "cl_regret": float(np.mean(cost - best)),
                     "plan_regret_fixed": float((truth_cost - best[:, None]).mean(0).min())}
                with open(out, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(env, tag, seed, round(r["cl_regret"], 2), "fixed", round(r["plan_regret_fixed"], 2), flush=True)
    summary(out)


def summary(out: str) -> None:
    g = collections.defaultdict(list)
    for r in map(json.loads, Path(out).read_text().splitlines()):
        g[(r["env"], r["model"])].append(r)
    for (env, tag), rs in sorted(g.items()):
        reg = [r["cl_regret"] for r in rs]
        print(f"{env:7s} {tag:16s} n={len(rs)} regret={st.mean(reg):.2f} (min {min(reg):.2f}, max {max(reg):.2f}) "
              f"fixed={rs[0]['plan_regret_fixed']:.2f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--summary"]:
        summary(sys.argv[2])
    elif sys.argv[1:2] == ["--lam"]:  # e.g. --lam 0.5 results_planning_lam050.jsonl
        main(float(sys.argv[2]), sys.argv[3])
    else:
        main()
