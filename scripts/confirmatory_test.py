"""Confirmatory test on fresh seeds 5-14, exactly as in notes/confirmatory_seeds_preregistration.md.

40 one-sided tests (grey-box better): 2 simulators x 4 fine-tuned baselines x {effect skill, regret,
CF RMSE, open-loop planning regret, closed-loop planning regret}, lambda = 1. Exact permutation test
of the difference in means, 10 against 10 (all 184756 relabellings), Holm over all 40, alpha 0.05.
Also reported: the same tests on all 15 seeds (0-14), Monte Carlo with 200000 relabellings.

    python scripts/confirmatory_test.py     # prints every test; writes paper/tables/confirm.tex

The second test (notes/confirmatory_lam050_preregistration.md) uses the same code:

    python scripts/confirmatory_test.py --lam 0.5 --seeds 15-24   # writes paper/tables/confirm050.tex
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from paper_tables import NAMES, val
from stats_tests import holm

FRESH = tuple(range(5, 15))
ALL = tuple(range(15))
FT = ("gru+ft", "gru-ode+ft", "crn+ft", "transformer+ft")
TRAIN = ("results_v2.jsonl", "results_v2_transformer.jsonl", "results_ft.jsonl", "results_seeds.jsonl",
         "results_seeds_ft.jsonl", "results_seeds_transformer.jsonl", "results_confirm.jsonl",
         "results_confirm_transformer.jsonl", "results_confirm_ft.jsonl", "results_confirm050.jsonl",
         "results_confirm050_transformer.jsonl", "results_confirm050_ft.jsonl")
LAM = 1.0
PLANNING = {1.0: ("results_planning.jsonl", "results_planning_cl.jsonl"),
            0.5: ("results_planning_lam050.jsonl", "results_planning_cl_lam050.jsonl")}
METRICS = (("skill", "Effect skill"), ("regret", "Regret"), ("cf", "CF RMSE"),
           ("plan", "Planning regret, open loop"), ("cl", "Planning regret, closed loop"))
HIGHER = {"skill"}


def lines(path: str):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def collect() -> dict:
    """values[(env, model, metric)] = {seed: value} at lambda = 1."""
    out: dict = {}
    for path in TRAIN:
        for r in lines(path):
            if r["lam"] != LAM:
                continue
            model, seed = r["run"].split("/")[2], int(r["run"].split("/")[3][1:])
            for key, name in (("skill_dose_up", "skill"), ("raw:cal_regret", "regret"), ("raw:cal_cf_rmse", "cf")):
                out.setdefault((r["env"], model, name), {})[seed] = val([r], key)[0]
    for path, key, name in ((PLANNING[LAM][0], "plan_regret", "plan"), (PLANNING[LAM][1], "cl_regret", "cl")):
        for r in lines(path):
            if r["lam"] == LAM:
                out.setdefault((r["env"], r["model"], name), {})[r["seed"]] = r[key]
    return out


def exact(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided p of mean(a) - mean(b) over all relabellings."""
    pool, n = np.concatenate([a, b]), len(a)
    idx = np.array(list(itertools.combinations(range(len(pool)), n)))
    sa = pool[idx].sum(1)
    d = sa / n - (pool.sum() - sa) / (len(pool) - n)
    return float(np.mean(d >= a.mean() - b.mean() - 1e-12))


def monte_carlo(a: np.ndarray, b: np.ndarray, draws: int = 200_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    pool, n = np.concatenate([a, b]), len(a)
    obs, hits = a.mean() - b.mean(), 1
    for _ in range(draws // 10_000):
        perm = rng.permuted(np.tile(pool, (10_000, 1)), axis=1)
        hits += int(np.sum(perm[:, :n].mean(1) - perm[:, n:].mean(1) >= obs - 1e-12))
    return hits / (draws + 1)


def run(values: dict, seeds: tuple, method) -> list[dict]:
    rows = []
    for env in ("sepsis", "glucose"):
        for ft in FT:
            for m, label in METRICS:
                g, b = values.get((env, "greybox", m), {}), values.get((env, ft, m), {})
                sa, sb = [s for s in seeds if s in g], [s for s in seeds if s in b]
                sign = 1 if m in HIGHER else -1
                a, c = sign * np.array([g[s] for s in sa]), sign * np.array([b[s] for s in sb])
                complete = len(sa) == len(sb) == len(seeds)
                p = method(a, c) if len(a) and len(c) else float("nan")
                rows.append(dict(env=env, ft=ft, metric=label, n=(len(sa), len(sb)), complete=complete,
                                 grey=float(np.mean([g[s] for s in sa])) if sa else float("nan"),
                                 base=float(np.mean([b[s] for s in sb])) if sb else float("nan"), p=p))
    ps = [r["p"] if r["p"] == r["p"] else 1.0 for r in rows]
    for r, padj in zip(rows, holm(ps)):
        r["holm"] = padj
    return rows


def report(name: str, rows: list[dict]) -> None:
    print(f"\n== {name}")
    for r in rows:
        flag = "" if r["complete"] else "  INCOMPLETE"
        print(f"{r['env']:7s} {r['ft']:15s} {r['metric']:28s} n={r['n']} grey={r['grey']:7.3f} base={r['base']:7.3f} "
              f"p={r['p']:.5f} holm={r['holm']:.4f}{flag}")
    print(f"{name}: {sum(r['holm'] < 0.05 for r in rows)} of {len(rows)} significant after Holm; "
          f"{sum(not r['complete'] for r in rows)} incomplete")


def table(fresh: list[dict], every: "Sequence[dict | None]", out_path: str = "paper/tables/confirm.tex") -> None:
    f = lambda x: f"{x:.2f}".replace("-", "$-$")  # noqa: E731
    out = [r"\begin{tabular}{llccccc}", r"\toprule",
           r" & Fine-tuned & Skill & Regret & CF RMSE & Plan & Plan (CL) \\", r"\midrule"]
    k = len(METRICS)
    for e, env in enumerate(("sepsis", "glucose")):
        for i, ft in enumerate(FT):
            rs = fresh[(e * len(FT) + i) * k:(e * len(FT) + i + 1) * k]
            al = every[(e * len(FT) + i) * k:(e * len(FT) + i + 1) * k]
            cells = []
            for r, q in zip(rs, al):
                star = "$^{*}$" if r["holm"] < 0.05 else ""
                dag = "$^{\\dagger}$" if q is not None and q["holm"] < 0.05 else ""
                cells.append(f(r["grey"]) + " / " + f(r["base"]) + star + dag)
            out.append(" & ".join([env.capitalize() if i == 0 else "", NAMES[ft.split("+")[0]]] + cells) + r" \\")
        if env == "sepsis":
            out.append(r"\midrule")
    Path(out_path).write_text("\n".join(out + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def main() -> None:
    global LAM
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", default="5-14", help="fresh seeds, e.g. 15-24")
    a = ap.parse_args()
    LAM = a.lam
    lo, hi = map(int, a.seeds.split("-"))
    fresh_seeds = tuple(range(lo, hi + 1))
    values = collect()
    fresh = run(values, fresh_seeds, exact)
    report(f"fresh seeds {a.seeds} at lambda {LAM} (confirmatory, exact)", fresh)
    if LAM == 1.0 and fresh_seeds == FRESH:
        every = run(values, ALL, monte_carlo)
        report("all seeds 0-14 (Monte Carlo)", every)
        Path("results_confirmatory.json").write_text(json.dumps({"fresh": fresh, "all": every}, indent=1))
        table(fresh, every)
    else:
        Path(f"results_confirmatory_lam{LAM:.2f}.json").write_text(json.dumps({"fresh": fresh}, indent=1))
        table(fresh, [None] * len(fresh), f"paper/tables/confirm{int(LAM * 100):03d}.tex")


if __name__ == "__main__":
    main()
