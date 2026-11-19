"""Train every (env, lambda, model, seed) on Simulusion's Trainer; append final metrics.

    python scripts/sweep.py                                  # the full lambda grid
    python scripts/sweep.py --models gru greybox --lams 0.75 --seeds 0 --max-steps 500
    python scripts/sweep.py --models greybox --misspec       # k_e / EC50 prior sweep

Rerunning skips runs already in the results file; resume=auto picks up
interrupted ones from their last checkpoint.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

import grey_wm.tasks  # noqa: F401  (registers the tasks)
from grey_wm.tasks import MODEL_TASKS, nested
from simulusion.training.run import run

MISSPEC = (-0.5, -0.25, 0.25, 0.5, 1.0)  # relative error in the assumed prior
# Harsher errors: the effect size off by up to 4x, and the wrong dose-response shape.
MISSPEC_HARD = ({"emax_scale": s} for s in (0.25, 0.5, 2.0, 4.0))
MISSPEC_SHAPE = ({"hill_scale": s} for s in (0.5, 2.0))

# Per-env settings (grid steps): glucose steps are 5 min and insulin acts over
# hours, so it gets a 4 h context and a 2 h treatment-effect window.
# Insulin's effect is still building at 6 h, so glucose trains and scores over 6 h.
ENV_OVERRIDES = {
    "sepsis": {},
    "glucose": {
        "optim.context_min": 36, "optim.context_max": 72, "optim.horizon_max": 72,
        "eval.n_context": 48, "eval.effect_window": 72, "eval.cf_horizon": 72, "eval.select_horizon": 72,
        "eval.horizons": [1, 5, 20, 50, 72],
    },
}


def finished(results: Path) -> set[str]:
    if not results.exists():
        return set()
    return {json.loads(line)["run"] for line in results.read_text().splitlines() if line.strip()}


def grid(args: argparse.Namespace) -> list[tuple[str, float, str, int, dict[str, float]]]:
    if args.misspec_hard:
        perturb = [*MISSPEC_HARD, *MISSPEC_SHAPE]
        return [(e, 0.75, m, s, x) for e, m, s, x in itertools.product(args.envs, args.models, args.seeds, perturb)]
    if not args.misspec:
        return [(e, lam, m, s, {}) for e, lam, m, s in itertools.product(args.envs, args.lams, args.models, args.seeds)]
    perturb = [{}] + [{f"{p}_scale": 1 + d} for p in ("ke", "ec50") for d in MISSPEC]
    return [(e, 0.75, m, s, x) for e, m, s, x in itertools.product(args.envs, args.models, args.seeds, perturb)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="+", default=["sepsis"])
    p.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--models", nargs="+", default=list(MODEL_TASKS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--misspec", action="store_true", help="sweep k_e and EC50 prior errors at lambda=0.75")
    p.add_argument("--misspec-hard", action="store_true", help="sweep Emax and Hill errors at lambda=0.75")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--results", default="results.jsonl")
    p.add_argument("--device", default=None)
    p.add_argument("--jobs", type=int, default=1, help="parallel CPU workers, one thread each")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra overrides, e.g. model.prior_weight=0.1")
    p.add_argument("--tag", default="", help="suffix for run names, to keep variants apart")
    args = p.parse_args()

    results = Path(args.results)
    done = finished(results)
    pending = []
    for env, lam, model, seed, extra in grid(args):
        tag = "".join(f"_{k}{v:g}" for k, v in extra.items())
        name = f"{env}/lam{lam:.2f}/{model}{tag}{args.tag}/s{seed}"
        if name in done:
            continue
        overrides = {
            "trainer.seed": seed,
            "trainer.run_name": name,
            "trainer.out_dir": args.out_dir,
            "trainer.max_steps": args.max_steps,
            "trainer.resume": "auto",
            "trainer.eval_every": max(args.max_steps // 10, 1),
            "trainer.checkpoint_every": max(args.max_steps // 5, 1),
            "trainer.early_stopping_metric": "calib_rmse",
            "trainer.patience": 5,
            "trainer.export_world_model": False,
            "data.env": env,
            "data.confounding": lam,
            "data.root": args.data_root,
            **ENV_OVERRIDES[env],
            **{f"model.{k}": v for k, v in extra.items()},
            **{k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)},
        }
        if args.jobs > 1:
            overrides["trainer.device"] = "cpu"
            overrides["trainer.console"] = False
        elif args.device:
            overrides["trainer.device"] = args.device
        meta = {"run": name, "env": env, "lam": lam, "model": model + args.tag, "seed": seed, **extra}
        pending.append((model, overrides, meta))

    print(f"{len(pending)} runs to do, {args.jobs} at a time", flush=True)
    if args.jobs <= 1:
        for job in pending:
            try:
                record(results, train_one(*job))
            except Exception as exc:  # one bad run should not stop the grid
                print(f"[failed] {job[2]['run']}: {exc!r}", flush=True)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # workers inherit it: never touch the GPU
    with ProcessPoolExecutor(args.jobs, initializer=_single_thread) as pool:
        futures = {pool.submit(train_one, *job): job[2]["run"] for job in pending}
        for fut in as_completed(futures):
            try:
                record(results, fut.result())
            except Exception as exc:  # one bad run should not stop the grid
                print(f"[failed] {futures[fut]}: {exc!r}", flush=True)


def _single_thread() -> None:
    import torch

    torch.set_num_threads(1)


def train_one(model: str, overrides: dict, meta: dict) -> dict:
    res = run(task=model, overrides=nested(overrides))
    return {**meta, "steps": res.progress, "stopped_early": res.stopped_early, **res.metrics}


def record(results: Path, row: dict) -> None:
    with results.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[done] {row['run']}  cf_rmse={row.get('cf_rmse', float('nan')):.3f}", flush=True)


if __name__ == "__main__":
    main()
