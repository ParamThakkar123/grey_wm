"""Bolus effect under realistic dose changes, for every glucose model at lambda = 1.

Uses ``recalibrate.bolus_effects`` (one extra bolus; recorded boluses x1.3) on the
saved, calibrated models, and the paper's sustained-dosing sensitivity for comparison.

    python scripts/bolus_realistic.py     # writes results_bolus.jsonl; prints the means
"""

from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

import torch

from grey_wm.data import load_split
from grey_wm.models import build_model
from recalibrate import bolus_effects

MODELS = ("greybox", "greybox_pkpd", "gru", "gru-ipw", "gru-ode", "crn", "transformer", "gru+ft", "gru-ode+ft", "crn+ft", "transformer+ft")


def main(lam: float = 1.0, out: str = "results_bolus.jsonl") -> None:
    torch.set_num_threads(2)
    split = load_split("data", "glucose", lam)
    rows = []
    for tag in MODELS:
        for d in sorted(Path(f"runs_v2/glucose/lam{lam:.2f}/{tag}").glob("s[0-2]/model")):
            cfg = json.loads((d / "model.json").read_text())
            m = dict(cfg["model"])
            kind = m.pop("kind")
            e = cfg["eval"]
            model = build_model(kind, split["test"].spec, **m)
            model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
            model.eval()
            test = split["test"].subset(slice(0, min(e["test_trajectories"], len(split["test"]))))
            with torch.no_grad():
                r = bolus_effects(model, test, e["n_context"], e["effect_window"])
            r.update(model=tag, seed=int(d.parent.name[1:]), lam=lam)
            rows.append(r)
            print(tag, r["seed"], {k: round(v, 2) for k, v in r.items() if k.endswith("sensitivity")}, flush=True)
    Path(out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    g = collections.defaultdict(list)
    for r in rows:
        g[r["model"]].append(r)
    for tag, rs in g.items():
        print(f"{tag:15s} " + "  ".join(f"{k}={st.mean(r[k] for r in rs):.2f}"
                                       for k in ("bolus_once_sensitivity", "bolus_x1.3_sensitivity")))


if __name__ == "__main__":
    main()
