"""Baseline: fine-tune a trained black-box model on the small randomised set.

The grey-box model fits one effect multiplier per drug on the 100 randomised
calibration trajectories. The fair question for a black box is what it gains
from the same 100 trajectories, so here every weight is fine-tuned on them:

* 80 trajectories train, 20 pick the stopping point (rollout error, the same
  criterion used to select weights in training). The unchanged model is a
  candidate too, so fine-tuning cannot be selected if it makes things worse.
* Same loss as training, the training learning rate, batch 32, up to 1000
  steps, checked every 50, stop after 6 checks without improvement.
* Then the conformal sd multiplier is fitted on the 20 stopping trajectories,
  and the full evaluation and audit run as for every other model, with results
  under ``cal_`` keys so the same report code reads them.

The 72 audit trajectories and the test patients are never seen.

    python scripts/finetune.py --jobs 2                       # lam 0.5 and 1, both envs, 4 baselines, 3 seeds
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

N_TRAIN = 80
BATCH = 32
MAX_STEPS = 1000
EVERY = 50
PATIENCE = 6
# Post hoc variants (2026-11-16): "head" fine-tunes only the output layer; "l2sp" fine-tunes every
# weight with a penalty alpha/2 * ||w - w_pretrained||^2, alpha picked on the 20 stopping trajectories.
SUFFIX = {"full": "+ft", "head": "+fthead", "l2sp": "+ftl2sp"}
HEADS = {"gru": "decoder.", "gru-ode": "decoder.", "crn": "decoder.", "transformer": "head."}
L2SP_ALPHAS = (0.001, 0.01, 0.1, 1.0)


def finetune_one(env: str, lam: float, kind: str, seed: int, runs: str, root: str, threads: int,
                 mode: str = "full", subset: str = "") -> dict:
    import numpy as np
    import torch

    from grey_wm import audit, calibration, metrics
    from grey_wm.data import load_split
    from grey_wm.models import build_model, filter_rollout, masked_nll
    from grey_wm.tasks import PK_WARMUP

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    d = Path(runs) / env / f"lam{lam:.2f}" / kind / f"s{seed}" / "model"
    cfg = json.loads((d / "model.json").read_text())
    mcfg = dict(cfg["model"])
    model_kind = mcfg.pop("kind")
    o, e = cfg["optim"], cfg["eval"]
    splits = load_split(root, env, lam)
    model = build_model(model_kind, splits["train"].spec, **mcfg)
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))

    calib, holdout = calibration.randomised_sets(root, env)
    n = min(cfg.get("data", {}).get("calib_n", len(calib)), len(calib))  # size study: only the first n exist
    calib = calib.subset(slice(0, n))
    if subset:  # post hoc: half the randomised set, chosen by hidden severity
        calib = calibration.shifted_subset(calib, subset)
        n = len(calib)
    n_train = round(n * N_TRAIN / 100)
    train, val = calib.subset(slice(0, n_train)), calib.subset(slice(n_train, n))
    full = train.batch("cpu")
    steps = int(full["actions"].shape[1])  # as in training: actions are one step shorter
    H = e["select_horizon"]

    def val_rmse() -> float:
        model.eval()
        return metrics.factual(model, val, e["n_context"], (H,))[f"rmse_h{H}"]

    init = {k: v.clone() for k, v in model.state_dict().items()}
    head = HEADS[kind] if mode == "head" else None
    params = [p for name, p in model.named_parameters() if head is None or name.startswith(head)]
    anchor = [p.detach().clone() for p in params]
    if head is not None:
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith(head))

    def train(alpha: float) -> dict:
        model.load_state_dict(init)
        opt = torch.optim.Adam(params, lr=o["lr"])
        best = {"score": val_rmse(), "step": 0, "weights": {k: v.clone() for k, v in model.state_dict().items()}}
        before = best["score"]
        bad = 0
        for step in range(1, MAX_STEPS + 1):
            model.train()
            horizon = random.randint(1, o["horizon_max"])
            context = random.randint(min(o["context_min"], steps - horizon + 1), min(o["context_max"], steps - horizon + 1))
            start = random.randint(0, steps - (context + horizon - 1))
            stop = start + context + horizon
            idx = torch.randint(0, n_train, (BATCH,))
            history = full["actions"][idx, max(0, start - PK_WARMUP) : start]
            batch = {k: v[idx, start : stop - 1 if k == "actions" else stop] for k, v in full.items()}
            out = filter_rollout(model, batch, context, horizon, history=history)
            x, mask = batch["x"], batch["mask"]
            one = masked_nll(out.one_step_mean, out.one_step_logvar, x[:, 1:context], mask[:, 1:context])
            win = slice(context, context + horizon)
            roll = masked_nll(out.mean, out.logvar, x[:, win], mask[:, win])
            loss = one + o["rollout_weight"] * roll + sum(model.penalties(out.states).values())
            if alpha > 0:  # L2-SP: stay near the pretrained weights
                loss = loss + alpha / 2 * sum(((p - p0) ** 2).sum() for p, p0 in zip(params, anchor))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, o["max_grad_norm"])
            opt.step()
            if step % EVERY == 0:
                score = val_rmse()
                if score < best["score"]:
                    best = {"score": score, "step": step, "weights": {k: v.clone() for k, v in model.state_dict().items()}}
                    bad = 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
        return {**best, "before": before, "alpha": alpha}

    if mode == "l2sp":
        tried = [train(a) for a in L2SP_ALPHAS]
        best = min(tried, key=lambda t: t["score"])  # picked on the 20 stopping trajectories
    else:
        best = train(0.0)
    before = best["before"]
    model.load_state_dict(best["weights"])
    model.eval()
    tag = f"{kind}{SUFFIX[mode]}" + (f"_{subset}" if subset else "")
    save = Path(runs) / env / f"lam{lam:.2f}" / tag / f"s{seed}" / "model"
    save.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save / "model.pt")
    (save / "model.json").write_text(json.dumps(cfg, indent=2))

    sd = calibration.calibrate(model, val, e["n_context"], e["cf_horizon"])
    data = splits["test"].subset(slice(0, min(e["test_trajectories"], len(splits["test"]))))
    res = {
        **metrics.factual(model, data, e["n_context"], e["horizons"]),
        **metrics.counterfactual(model, data, e["n_context"], horizon=e["cf_horizon"], effect_window=e["effect_window"]),
        **audit.run(model, data, holdout, e["n_context"], e["cf_horizon"], e["effect_window"]),
    }
    return {
        "run": f"{env}/lam{lam:.2f}/{tag}/s{seed}", "env": env, "lam": lam,
        "model": tag, "seed": seed, "ft_step": best["step"],
        **({"ft_alpha": best["alpha"]} if mode == "l2sp" else {}), "ft_val_before": before, "ft_val_after": best["score"], "ft_secs": time.time() - t0,
        **sd, **{f"cal_{k}": v for k, v in res.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="+", default=["sepsis", "glucose"])
    p.add_argument("--lams", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--models", nargs="+", default=["gru", "gru-ode", "crn", "transformer"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--runs", default="runs_v2")
    p.add_argument("--root", default="data")
    p.add_argument("--results", default="results_ft.jsonl")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--mode", choices=sorted(SUFFIX), default="full")
    p.add_argument("--subset", default="", choices=["", "first50", "mild50", "sick50"])
    args = p.parse_args()

    out = Path(args.results)
    done = {json.loads(l)["run"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [(e, l, m, s) for e in args.envs for l in args.lams for m in args.models for s in args.seeds
            if f"{e}/lam{l:.2f}/{m}{SUFFIX[args.mode]}{'_' + args.subset if args.subset else ''}/s{s}" not in done]
    print(f"{len(todo)} runs to do, {len(done)} already done", flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(finetune_one, *t, args.runs, args.root, args.threads, args.mode, args.subset): t for t in todo}
        for f in as_completed(futs):
            t = futs[f]
            try:
                r = f.result()
                with out.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(f"[done] {r['run']} step={r['ft_step']} val {r['ft_val_before']:.3f}->{r['ft_val_after']:.3f} "
                      f"skill={r['cal_skill_dose_up']:.2f} {r['ft_secs']:.0f}s", flush=True)
            except Exception:
                print(f"[failed] {t}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
