"""Timing-aware effect test on saved models (post hoc; plan: notes/timing_effect_test_plan.md).

Scores every model in results_audit_v3.jsonl and results_audit_struct.jsonl; writes
results_audit_timing.jsonl (resumable) and the per-segment data to audit_timing/<run>.npz.

    python scripts/timing_audit.py --jobs 2
    python scripts/timing_audit.py --summary     # the analyses of the plan
"""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = Path("audit_timing")
SOURCES = ("results_audit_v3.jsonl", "results_audit_struct.jsonl")


def score_one(model_dir: str, root: str, threads: int) -> dict:
    import numpy as np
    import torch

    from grey_wm import audit, calibration
    from grey_wm.data import load_split
    from grey_wm.models import build_model

    torch.set_num_threads(threads)
    d = Path(model_dir)
    env, lam_dir, tag, seed_dir = d.parts[-5:-1]
    cfg = json.loads((d / "model.json").read_text())
    mcfg = dict(cfg["model"])
    kind = mcfg.pop("kind")
    e = cfg["eval"]
    spec = load_split(root, env, float(lam_dir[3:]))["train"].spec
    model = build_model(kind, spec, **mcfg)
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    model.eval()
    _, holdout = calibration.randomised_sets(root, env)
    data = audit.effect_test_timed_data(model, holdout, e["n_context"], e["cf_horizon"])
    run = f"{env}/{lam_dir}/{tag}/{seed_dir}"
    path = OUT / f"{run}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    t = audit.effect_test_timed(data)
    return {"run": run, "env": env, "model": tag, "timed_flag": t["flag"], "timed_zmax": t["zmax"],
            "timed_z": t["z"].tolist(), "timed_coef": t["coef"].tolist()}


def runs() -> list[str]:
    out = []
    for src in SOURCES:
        if Path(src).exists():
            out += [json.loads(l)["run"] for l in Path(src).read_text().splitlines() if l.strip()]
    return sorted(set(out))


def summary(results: str) -> None:
    from audit_v3_analysis import rows

    timed = {json.loads(l)["run"]: json.loads(l) for l in Path(results).read_text().splitlines() if l.strip()}
    main = [r for r in rows() if r["run"] in timed]
    bad = [r for r in main if r["broken_v2"]]
    good = [r for r in main if not r["broken_v2"]]
    f = lambda rs: sum(timed[r["run"]]["timed_flag"] for r in rs)  # noqa: E731
    either = lambda rs: sum(timed[r["run"]]["timed_flag"] or r["eff"] for r in rs)  # noqa: E731
    print(f"1. {len(main)} models of the main audit scored ({len(bad)} broken)")
    print(f"   timing test: caught {f(bad)}/{len(bad)} broken, flagged {f(good)}/{len(good)} of the rest")
    print(f"   effect test: caught {sum(r['eff'] for r in bad)}/{len(bad)}, flagged {sum(r['eff'] for r in good)}/{len(good)}")
    print(f"   either:      caught {either(bad)}/{len(bad)}, flagged {either(good)}/{len(good)}")
    struct = {json.loads(l)["run"]: json.loads(l) for l in Path("results_audit_struct.jsonl").read_text().splitlines() if l.strip()}
    groups = {"2. wrong form (no depot, no delay, added delay)": ("greybox_nodepot", "greybox_nodelay", "greybox_adddelay"),
              "2. linear dose-response": ("greybox_linear",),
              "3. PK/PD only": ("greybox_pkpd",), "3. fluid-only elimination": ("greybox_elimfluid",)}
    for name, tags in groups.items():
        rs = [r for k, r in timed.items() if k.split("/")[2] in tags]
        eff = sum(bool(struct[k]["eff_flag"]) for k in timed if k.split("/")[2] in tags and k in struct)
        print(f"{name}: timing test flags {sum(r['timed_flag'] for r in rs)}/{len(rs)}; effect test flags {eff}/{len(rs)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs_v2")
    p.add_argument("--root", default="data")
    p.add_argument("--results", default="results_audit_timing.jsonl")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--only", nargs="*", default=[], help="substrings a run must contain")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    if args.summary:
        summary(args.results)
        return
    out = Path(args.results)
    done = {json.loads(l)["run"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [str(Path(args.runs) / r / "model") for r in runs()
            if r not in done and all(s in r for s in args.only) and (Path(args.runs) / r / "model" / "model.pt").exists()]
    print(f"{len(todo)} models to score, {len(done)} already done", flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(score_one, d, args.root, args.threads): d for d in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
                with out.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(f"[done] {r['run']} timed_flag={r['timed_flag']} zmax={r['timed_zmax']:.1f}", flush=True)
            except Exception:
                print(f"[failed] {futs[f]}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
