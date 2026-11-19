"""Summaries of the post hoc calibration variants (scripts/recalibrate.py).

    python scripts/recal_tables.py   # prints per (env, lambda, prior, variant); writes paper/tables/recal.tex
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

from grey_wm.audit import derived

ACTIONS = {"sepsis": ("norepinephrine", "fluid"), "glucose": ("basal", "bolus")}
KEYS = ("skill_dose_up", "regret_margin", "cfcov_worst", "rand_cov", "rand_gap", "course_worst", "hazard_recall")


def load(path: str = "results_recal.jsonl") -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = collections.defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            g[(r["env"], r["lam"], r["model"], r["variant"])].append(r)
    return g


def m(runs, key) -> float:
    xs = []
    for r in runs:
        d = derived(r, ACTIONS[r["env"]], "cal_")
        x = d[key] if key in d else r.get(key, r.get("cal_" + key, float("nan")))
        xs.append(x)
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def summary(g) -> dict:
    out = {}
    for (env, lam, model, variant), runs in sorted(g.items()):
        a, b = ACTIONS[env]
        keys = KEYS + (f"{a}_sensitivity", f"{b}_sensitivity", f"{a}_effect_scale", f"{b}_effect_scale",
                       f"{a}_ec50", f"{b}_ec50")
        if env == "glucose":
            keys += ("bolus_once_sensitivity", "bolus_x1.3_sensitivity")
        row = {k: m(runs, k) for k in keys}
        row["n"] = len(runs)
        out[(env, lam, model, variant)] = row
        print(f"{env:7s} {lam:.2f} {model:24s} {variant:8s} n={len(runs)} "
              + " ".join(f"{k}={v:.2f}" for k, v in row.items() if k != "n" and not math.isnan(v)))
    return out


def fmt(x: float, d: int = 2) -> str:
    return "--" if x != x else f"{x:.{d}f}".replace("-", "$-$")


def table(s: dict) -> None:
    """Main grey-box model at lambda = 1, the fourfold Emax over-estimate on sepsis, and glucose boluses."""
    names = {"base": "Paper", "perstep": "Per-step $s$", "ec50": r"$+\mathrm{EC}_{50}$", "loose": "Loose prior"}
    lines = [r"\begin{tabular}{lccccccccc}", r"\toprule",
             r" & \multicolumn{4}{c}{Sepsis, $\lambda=1$} & \multicolumn{3}{c}{Sepsis, $E_{\max}\times4$, $\lambda=0.75$}"
             r" & \multicolumn{2}{c}{Glucose, $\lambda=1$} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
             r"Calibration & Skill & Fluid & CF cov. & Course & Skill & NE & Fluid & Skill & CF cov. \\", r"\midrule"]
    for v, name in names.items():
        a = s.get(("sepsis", 1.0, "greybox", v), {})
        b = s.get(("sepsis", 0.75, "greybox_emax_scale4", v), {})
        c = s.get(("glucose", 1.0, "greybox", v), {})
        g = lambda d, k: fmt(d.get(k, float("nan")))  # noqa: E731
        cells = [g(a, "skill_dose_up"), g(a, "fluid_sensitivity"), g(a, "cfcov_worst"), g(a, "course_worst"),
                 g(b, "skill_dose_up"), g(b, "norepinephrine_sensitivity"), g(b, "fluid_sensitivity"),
                 g(c, "skill_dose_up"), g(c, "cfcov_worst")]
        if v == "perstep":
            cells[4:7] = ["--"] * 3  # not run on the wrong priors
        lines.append(" & ".join([name] + cells) + r" \\")
    Path("paper/tables/recal.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


if __name__ == "__main__":
    table(summary(load()))
