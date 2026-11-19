"""Audit v3 on saved models: both drugs' effect skill and the randomised effect test.

Plan, fixed before running: notes/audit_v3_preregistration.md.

For every saved model under ``runs_v2/<env>/lam*/<variant>/s*/model`` (fine-tuned
baselines included once ``scripts/finetune.py`` has saved them):

* effect skill for each drug on the first 300 test patients (needs the simulator);
  drug 0 repeats the v2 number and is checked against it in the analysis;
* the per-window data of the randomised effect test on the 72 held-out randomised
  patients, saved to ``audit_v3/<run>.npz``, and the test at all 72.

Saved grey-box weights already carry their calibrated multipliers (buffers). Only
predicted means are used, so the conformal sd multiplier does not matter here.

    python scripts/reaudit.py --jobs 2
"""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = Path("audit_v3")


def audit_one(model_dir: str, root: str, threads: int) -> dict:
    import numpy as np
    import torch

    from grey_wm import audit, calibration
    from grey_wm.data import load_split
    from grey_wm.models import build_model

    torch.set_num_threads(threads)
    d = Path(model_dir)
    env, lam_dir, tag, seed_dir = d.parts[-5:-1]
    lam = float(lam_dir[3:])
    cfg = json.loads((d / "model.json").read_text())
    mcfg = dict(cfg["model"])
    kind = mcfg.pop("kind")
    e = cfg["eval"]
    splits = load_split(root, env, lam)
    model = build_model(kind, splits["train"].spec, **mcfg)
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    model.eval()
    _, holdout = calibration.randomised_sets(root, env)
    test = splits["test"].subset(slice(0, min(e["test_trajectories"], len(splits["test"]))))

    run = f"{env}/{lam_dir}/{tag}/{seed_dir}"
    out: dict = {"run": run, "env": env, "lam": lam, "model": tag, "seed": int(seed_dir[1:])}
    for c, action in enumerate(test.spec.action_names):
        for k, v in audit.drug_skill(model, test, e["n_context"], e["cf_horizon"], c).items():
            out[f"{action}_{k}"] = v
    data = audit.effect_test_data(model, holdout, e["n_context"], e["cf_horizon"])
    path = OUT / f"{run}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    t = audit.effect_test(data)
    out.update({"eff_flag": t["flag"], "eff_zmax": float(t["z"].max()),
                **{f"{a}_eff_{k}": float(t[k][c]) for c, a in enumerate(test.spec.action_names)
                   for k in ("coef", "lo", "hi", "z", "ratio")}})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs_v2")
    p.add_argument("--root", default="data")
    p.add_argument("--results", default="results_audit_v3.jsonl")
    p.add_argument("--only", nargs="*", default=[], help="substrings a run must contain")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    out = Path(args.results)
    done = {json.loads(l)["run"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    dirs = sorted(str(m.parent) for m in Path(args.runs).glob("*/lam*/*/s*/model/model.pt"))
    todo = [d for d in dirs if "/".join(Path(d).parts[-5:-1]) not in done and all(s in d.replace("\\", "/") for s in args.only)]
    print(f"{len(todo)} models to audit, {len(done)} already done", flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(audit_one, d, args.root, args.threads): d for d in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
                with out.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(f"[done] {r['run']} eff_flag={r['eff_flag']} zmax={r['eff_zmax']:.1f}", flush=True)
            except Exception:
                print(f"[failed] {futs[f]}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
