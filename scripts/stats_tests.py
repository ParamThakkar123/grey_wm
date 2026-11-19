"""Seed-level tests: grey-box model against each fine-tuned baseline.

Each (simulator, lambda, model) has 3 independently trained seeds. For each
simulator, fine-tuned baseline and metric we test whether the grey-box model is
better, with an exact permutation test that shuffles the 6 seeds of a lambda
between the two models (stratified by lambda). Statistic: the difference in
means, summed over the lambdas tested. Two scopes:

* ``lam1``: lambda = 1 only (20 permutations, so p >= 0.05);
* ``conf``: the four confounded settings, lambda in {0.25, 0.5, 0.75, 1}
  (20^4 = 160000 permutations).
* ``lam1x5``: lambda = 1 with the two extra seeds (3, 4) added, 5 against 5
  (252 permutations, so p >= 0.004).

p values are one-sided and Holm-corrected over the 24 tests of each scope.
The interval is a 95% bootstrap interval of the mean difference (seeds resampled
within model and lambda, 20000 draws).

    python scripts/stats_tests.py     # prints all tests; writes paper/tables/stats.tex
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import json

from paper_tables import NAMES, load, val

EXTRA_SEEDS = ("results_seeds.jsonl", "results_seeds_ft.jsonl", "results_seeds_transformer.jsonl")

FT = ("gru+ft", "gru-ode+ft", "crn+ft", "transformer+ft")
# metric key, label, +1 if higher is better
METRICS = (("skill_dose_up", "Effect skill", 1), ("raw:cal_regret", "Regret", -1), ("raw:cal_cf_rmse", "CF RMSE", -1))
SCOPES = {"lam1": (1.0,), "conf": (0.25, 0.5, 0.75, 1.0), "lam1x5": (1.0,)}
RNG = np.random.default_rng(0)


def stratum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """All values of mean(a') - mean(b') over relabellings of the pooled seeds."""
    pool, n = np.concatenate([a, b]), len(a)
    out = []
    for idx in itertools.combinations(range(len(pool)), n):
        mask = np.zeros(len(pool), bool)
        mask[list(idx)] = True
        out.append(pool[mask].mean() - pool[~mask].mean())
    return np.array(out)


def test(pairs: list[tuple[np.ndarray, np.ndarray]], sign: int) -> tuple[float, float, float, float]:
    obs = sign * sum(a.mean() - b.mean() for a, b in pairs)
    dist = np.zeros(1)
    for a, b in pairs:
        dist = (dist[:, None] + sign * stratum(a, b)[None, :]).ravel()
    p = float(np.mean(dist >= obs - 1e-12))
    boots = np.zeros(20000)
    for a, b in pairs:
        boots += RNG.choice(a, (20000, len(a))).mean(1) - RNG.choice(b, (20000, len(b))).mean(1)
    k = len(pairs)
    lo, hi = np.percentile(boots / k, [2.5, 97.5])
    return sign * obs / k, float(lo), float(hi), p


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def with_extra_seeds(g: dict) -> dict:
    out = {k: list(v) for k, v in g.items()}
    for path in EXTRA_SEEDS:
        if Path(path).exists():
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    out.setdefault((r["env"], r["lam"], r["run"].split("/")[2]), []).append(r)
    return out


def main() -> None:
    base = load()
    rows = {}
    for scope, lams in SCOPES.items():
        g = with_extra_seeds(base) if scope == "lam1x5" else base
        keys, ps, res = [], [], []
        for env in ("sepsis", "glucose"):
            for ft in FT:
                for key, label, sign in METRICS:
                    pairs = [(np.array(val(g[(env, lam, "greybox")], key)), np.array(val(g[(env, lam, ft)], key)))
                             for lam in lams]
                    pairs = [(a[a == a], b[b == b]) for a, b in pairs]
                    d, lo, hi, p = test(pairs, sign)
                    keys.append((env, ft, label))
                    ps.append(p)
                    res.append((d, lo, hi, p))
        for k, (d, lo, hi, p), padj in zip(keys, res, holm(ps)):
            rows[(scope,) + k] = (d, lo, hi, p, padj)
            print(f"{scope:4s} {k[0]:7s} {k[1]:15s} {k[2]:12s} diff={d:8.3f} [{lo:8.3f}, {hi:8.3f}] "
                  f"p={p:.4f} holm={padj:.4f}")
        n = sum(rows[(scope,) + k][4] < 0.05 for k in keys)
        print(f"{scope}: {n} of {len(keys)} significant after Holm")
    table(rows)


def table(rows: dict) -> None:
    """Conf scope: mean difference (grey-box better is positive) with interval, and Holm p."""
    f = lambda x: f"{x:.2f}".replace("-", "$-$")  # noqa: E731
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r" & Fine-tuned & Effect skill & Regret & CF RMSE \\", r"\midrule"]
    for env in ("sepsis", "glucose"):
        for i, ft in enumerate(FT):
            name = NAMES[ft.split("+")[0]]
            rs = [rows[("conf", env, ft, lab)] for _, lab, _ in METRICS]
            top = [f(r[0]) + ("$^{*}$" if r[4] < 0.05 else "") for r in rs]
            low = [r"{\scriptsize[" + f(r[1]) + ", " + f(r[2]) + "]}" for r in rs]
            lines.append(" & ".join([env.capitalize() if i == 0 else "", name] + top) + r" \\")
            lines.append(" & ".join(["", ""] + low) + r" \\[2pt]")
        if env == "sepsis":
            lines.append(r"\midrule")
    Path("paper/tables/stats.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


if __name__ == "__main__":
    main()
