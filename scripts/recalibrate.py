"""Post hoc fixes to the grey-box calibration, tested on saved models (no retraining).

Each saved grey-box model is reloaded, its calibration (multipliers, effect sd,
conformal sd) is reset, and it is calibrated again on the same randomised
trajectories under one variant, then scored exactly as in training:

* ``base``     the paper's calibration; reproduces the ``cal_`` numbers (check).
* ``perstep``  conformal sd multiplier per rollout step instead of one.
* ``ec50``     also one EC50 multiplier per drug (dose-response shape), same prior.
* ``loose``    multiplier prior sd 0.5 -> 1.5 (a factor of about 4.5 instead of 1.6).

Glucose models also get the bolus effect under in-distribution changes
(``bolus_once``: one extra bolus at the start of the window; ``bolus_x1.3``:
the recorded boluses raised by 30%), as the mean change over the window,
model over truth.

    python scripts/recalibrate.py --variants base perstep ec50 --only lam1.00/greybox/ --jobs 2
"""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

VARIANTS = {
    "base": {},
    "perstep": {"per_step": True},
    "ec50": {"fit_ec50": True},
    "loose": {"prior_sd": 1.5},
    "ec50_perstep": {"fit_ec50": True, "per_step": True},
    "fluid_leak": {},  # diagnostic: sepsis fluid elimination at the rate of a mean-severity patient
    # post hoc (2026-11-16): calibrate on half the randomised set, chosen by hidden severity
    "first50": {}, "mild50": {}, "sick50": {},
}
# The sepsis simulator's fluid leak is k_e + 0.5 s (s = severity, about 0.5 on average);
# the prior has k_e alone. ``fluid_leak`` raises the prior's fluid k_e to 0.6 + 0.25.
FLUID_LEAK_KE = 0.85


def bolus_effects(model, data, n_context: int, window: int) -> dict[str, float]:
    import numpy as np

    from grey_wm.metrics import _Predictor, _truth

    predict = _Predictor(model, data, n_context, next(model.parameters()).device)
    k, o = n_context - 1, data.spec.outcome_index
    c = data.spec.action_names.index("bolus")
    high = np.asarray(data.spec.action_max)
    base = data["actions"][:, k : k + window].copy()
    window = base.shape[1]
    m0, t0 = predict(window, base)[0][..., o], _truth(data, n_context, base)[..., o]
    once = base.copy()
    once[:, 0, c] = np.minimum(once[:, 0, c] + 0.2 * high[c], high[c])
    scaled = base.copy()
    scaled[..., c] = np.minimum(1.3 * scaled[..., c], high[c])
    out = {}
    for name, fut in (("bolus_once", once), ("bolus_x1.3", scaled)):
        dm = predict(window, fut)[0][..., o] - m0
        dt = _truth(data, n_context, fut)[..., o] - t0
        keep = np.abs(dt).mean(1) > 1e-3  # patients whose outcome the change moves at all
        out[f"{name}_sensitivity"] = float(dm[keep].mean() / dt[keep].mean()) if keep.any() else float("nan")
        out[f"{name}_n"] = int(keep.sum())
    return out


def one(model_dir: str, variant: str, root: str, threads: int) -> dict:
    import torch

    from grey_wm import audit, calibration, metrics
    from grey_wm.data import load_split
    from grey_wm.models import build_model

    torch.set_num_threads(threads)
    d = Path(model_dir)
    env, lam_dir, tag, seed_dir = d.parts[-5:-1]
    cfg = json.loads((d / "model.json").read_text())
    mcfg = dict(cfg["model"])
    kind = mcfg.pop("kind")
    e = cfg["eval"]
    splits = load_split(root, env, float(lam_dir[3:]))
    model = build_model(kind, splits["train"].spec, **mcfg)
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    model.eval()
    model.effect_scale.fill_(1.0)  # undo the saved calibration
    model.effect_sigma.fill_(0.0)
    model.sd_scale = 1.0
    if variant == "fluid_leak":
        assert env == "sepsis"
        from grey_wm.models.greybox import KE
        model.theta_pop[1, KE] = FLUID_LEAK_KE
    calib, holdout = calibration.randomised_sets(root, env)
    calib = calib.subset(slice(0, min(cfg["data"].get("calib_n", len(calib)), len(calib))))
    if variant in calibration.SHIFTS:
        calib = calibration.shifted_subset(calib, variant)
    test = splits["test"].subset(slice(0, min(e["test_trajectories"], len(splits["test"]))))

    run = f"{env}/{lam_dir}/{tag}/{seed_dir}"
    out: dict = {"run": run, "variant": variant, "env": env, "lam": float(lam_dir[3:]), "model": tag,
                 "seed": int(seed_dir[1:])}
    out.update(calibration.calibrate(model, calib, e["n_context"], e["cf_horizon"], **VARIANTS[variant]))
    p = "cal_"
    out.update(metrics.factual(model, test, e["n_context"], tuple(e["horizons"]), prefix=p))
    out.update({p + k: v for k, v in metrics.counterfactual(
        model, test, e["n_context"], horizon=e["cf_horizon"], effect_window=e["effect_window"]).items()})
    out.update({p + k: v for k, v in audit.run(
        model, test, holdout, e["n_context"], e["cf_horizon"], e["effect_window"]).items()})
    if env == "glucose":
        out.update({p + k: v for k, v in bolus_effects(model, test, e["n_context"], e["effect_window"]).items()})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs_v2")
    p.add_argument("--root", default="data")
    p.add_argument("--results", default="results_recal.jsonl")
    p.add_argument("--variants", nargs="+", default=["base"], choices=list(VARIANTS))
    p.add_argument("--only", nargs="*", default=[], help="substrings a run path must contain (any of them)")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    out = Path(args.results)
    done = set()
    if out.exists():
        done = {(r["run"], r["variant"]) for r in map(json.loads, filter(str.strip, out.read_text().splitlines()))}
    dirs = sorted(str(m.parent).replace("\\", "/") for m in Path(args.runs).glob("*/lam*/greybox*/s*/model/model.pt"))
    dirs = [x for x in dirs if not args.only or any(s in x for s in args.only)]
    todo = [(x, v) for x in dirs for v in args.variants if ("/".join(Path(x).parts[-5:-1]), v) not in done]
    print(f"{len(todo)} (model, variant) pairs to do, {len(done)} already done", flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(one, x, v, args.root, args.threads): (x, v) for x, v in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
                with out.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(f"[done] {r['run']} {r['variant']} skill={r.get('cal_skill_dose_up', float('nan')):.3f}", flush=True)
            except Exception:
                print(f"[failed] {futs[f]}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
