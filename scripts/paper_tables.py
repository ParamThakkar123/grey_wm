"""Tables and figures for the paper, from the v2 results files.

    python scripts/paper_tables.py     # writes paper/tables/*.tex and figures/v2_*.pdf, prints prose numbers

All numbers are after calibration on the randomised set (``cal_`` keys) unless
a name says ``raw``. Misspecified-prior runs are told apart by their run tag.
"""

from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from grey_wm.audit import derived  # noqa: E402

OUT = Path("paper/tables")
FIG = Path("figures")
RESULTS = ("results_v2.jsonl", "results_v2_transformer.jsonl", "results_ft.jsonl")
# Baselines added in the final week (3 seeds, same protocol); shown as extra rows only.
EXTRA = ("results_pkpd.jsonl", "results_ipw.jsonl")
ACTIONS = {"sepsis": ("norepinephrine", "fluid"), "glucose": ("basal", "bolus")}
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
BASE = ("greybox", "gru", "gru-ode", "crn", "transformer")
NAMES = {"greybox": "Grey-box (ours)","gru": "GRU", "gru-ode": "GRU-ODE", "crn": "CRN", "transformer": "Transformer",
         "greybox-res": "+ residual $R$", "greybox-actres": "Raw-action residual", "greybox-leak": "Action leak",
         "greybox-latent": "Effect into latent", "greybox_pkpd": "PK/PD only (ablation)", "gru-ipw": "GRU, IPW-weighted",
         **{f"{m}+ft": "\\quad + fine-tuned" for m in ("gru", "gru-ode", "crn", "transformer")}}
FT = ("gru+ft", "gru-ode+ft", "crn+ft", "transformer+ft")
COLORS = {"greybox": "#c0392b", "gru": "#2c3e50", "gru-ode": "#2980b9", "crn": "#27ae60", "transformer": "#8e44ad"}


def load() -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for path in RESULTS + tuple(x for x in EXTRA if Path(x).exists()):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                groups[(r["env"], r["lam"], r["run"].split("/")[2])].append(r)
    return groups


def val(runs: list[dict], key: str) -> list[float]:
    env = runs[0]["env"]
    out = []
    for r in runs:
        if key.startswith("raw:"):  # the derived names read calibrated keys
            out.append(r.get(key[4:], float("nan")))
            continue
        d = derived(r, ACTIONS[env], "cal_")
        out.append(d[key] if key in d else r.get(key, float("nan")))
    return out


def ms(runs, key, d=2, sd=True) -> str:
    if not runs:
        return "--"
    v = val(runs, key)
    m = f"{st.mean(v):.{d}f}".replace("-", "$-$") if float(f"{st.mean(v):.{d}f}") != 0 else f"{0:.{d}f}"
    if not sd or len(v) < 2:
        return m
    return f"{m}\\,$\\pm$\\,{st.stdev(v):.{d}f}"


def mean(runs, key) -> float:
    return st.mean(val(runs, key))


def row(cells: list[str], ours: bool = False) -> str:
    """One table row; ours in bold, numbers included."""
    if ours:
        cells = ["{\\bfseries\\boldmath " + c + "}" for c in cells]
    return " & ".join(cells) + " \\\\"


def write(name: str, lines: list[str]) -> None:
    (OUT / name).write_text("\n".join(lines) + "\n")


def main_table(g) -> None:
    """Effect skill under randomised and confounded data; the rest at lambda = 1."""
    out = ["\\begin{tabular}{lcccccc}", "\\toprule",
           " & \\multicolumn{2}{c}{Effect skill $\\uparrow$} & \\multicolumn{4}{c}{$\\lambda=1$} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}",
           "Model & $\\lambda=0$ & $\\lambda=1$ & CF RMSE $\\downarrow$ & Fact.\\ RMSE $\\downarrow$ & Regret $\\downarrow$ & CF cov. \\\\",
           "\\midrule"]
    for env in ACTIONS:
        fixed = mean(g[(env, 1.0, "greybox")], "cal_regret_fixed")
        out.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{env.capitalize()}}} (fixed best dose: regret {fixed:.1f})}} \\\\")
        order = [x for b in BASE for x in ((b, "greybox_pkpd") if b == "greybox" else
                                            (b, b + "+ft", "gru-ipw") if b == "gru" else (b, b + "+ft"))]
        for m in order:
            a, b = g.get((env, 0.0, m), []), g.get((env, 1.0, m), [])
            if not b:
                continue
            out.append(row([NAMES[m], ms(a, "skill_dose_up"), ms(b, "skill_dose_up"), ms(b, "cal_cf_rmse"),
                            ms(b, "cal_fact_rmse"), ms(b, "cal_regret", 1, False), ms(b, "cfcov_worst", 2, False)],
                           m == "greybox"))
    write("main.tex", out + ["\\bottomrule", "\\end{tabular}"])


def ablation_table(g) -> None:
    out = ["\\begin{tabular}{lcccccc}", "\\toprule",
           " & \\multicolumn{3}{c}{Sepsis} & \\multicolumn{3}{c}{Glucose} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           "Variant & Skill & Dir. & Gap & Skill & Dir. & Gap \\\\", "\\midrule"]
    for m in ("greybox", "greybox-res", "greybox-actres", "greybox-leak", "greybox-latent"):
        cells = []
        for env in ACTIONS:
            r = g[(env, 0.75, m)]
            cells += [ms(r, "skill_dose_up", 2, False), ms(r, "direction_worst", 2, False), ms(r, "rand_gap", 2, False)]
        out.append(row([NAMES[m]] + cells, m == "greybox"))
    write("ablation.tex", out + ["\\bottomrule", "\\end{tabular}"])


def misspec_table(g) -> None:
    out = ["\\begin{tabular}{lccccc}", "\\toprule",
           " & \\multicolumn{2}{c}{NE sensitivity} & & & \\\\", "\\cmidrule(lr){2-3}",
           "Assumed prior & raw & calibrated & $\\kappa_{\\mathrm{NE}}$ & Skill & Gap \\\\", "\\midrule"]
    rows = [("correct", "greybox")] + [(f"$E_{{\\max}}\\times{s:g}$", f"greybox_emax_scale{s:g}") for s in (0.25, 0.5, 2, 4)] \
        + [(f"$h\\times{s:g}$", f"greybox_hill_scale{s:g}") for s in (0.5, 2)]
    for label, tag in rows:
        r = g[("sepsis", 0.75, tag)]
        out.append(row([label, ms(r, "norepinephrine_sensitivity", 2, False), ms(r, "cal_norepinephrine_sensitivity", 2, False),
                        ms(r, "norepinephrine_effect_scale", 2, False), ms(r, "skill_dose_up", 2, False),
                        ms(r, "rand_gap", 2, False)], tag == "greybox"))
    write("misspec.tex", out + ["\\bottomrule", "\\end{tabular}"])


def fig_skill(g) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.2), sharey=True)
    for ax, env in zip(axes, ACTIONS):
        for m in BASE + FT:
            lams = [lam for lam in LAMS if (env, lam, m) in g]
            if not lams:
                continue
            ys = [val(g[(env, lam, m)], "skill_dose_up") for lam in lams]
            mu = [st.mean(y) for y in ys]
            base = m.removesuffix("+ft")
            ft = m.endswith("+ft")
            ax.plot(lams, mu, marker="o", ms=2, lw=2.2 if m == "greybox" else 1.0, ls="--" if ft else "-",
                    color=COLORS[base], label=None if ft else NAMES[m].replace(" (ours)", ""))
            if not ft:
                ax.fill_between(lams, [min(y) for y in ys], [max(y) for y in ys], color=COLORS[base], alpha=0.15, lw=0)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_title(env.capitalize(), fontsize=9)
        ax.set_xlabel("confounding $\\lambda$", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("effect skill", fontsize=8)
    handles, _ = axes[1].get_legend_handles_labels()
    handles.append(Line2D([], [], color="k", lw=1.0, ls="--", label="fine-tuned"))
    fig.set_size_inches(3.4, 2.6)
    fig.legend(handles=handles, fontsize=6, frameon=False, loc="lower center", ncol=3, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(FIG / "v2_skill_vs_lambda.pdf")


def fig_audit(g) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.0))
    for env, mk in (("sepsis", "o"), ("glucose", "s")):
        for (e, lam, tag), runs in g.items():
            if e != env:
                continue
            m = tag if tag in COLORS else ("greybox" if tag.startswith("greybox_") else None)
            if tag.startswith("greybox-"):
                m = None
            ft = tag.endswith("+ft")
            if ft:
                m = tag[:-3]
            x, y = mean(runs, "skill_dose_up"), mean(runs, "rand_gap")
            col = COLORS[m] if m else "#999999"
            style = dict(facecolors="none", edgecolors=col, lw=0.8) if ft else dict(color=col, lw=0)
            axes[0].scatter(x, y, marker=mk, s=14, alpha=0.85, **style)
    # bottom: effect ratio from the randomised effect test (no simulator) vs true sensitivity, one point per model and drug
    v2 = {r["run"]: r for runs in g.values() for r in runs}
    lo, hi = -1.0, 3.0
    for line in Path("results_audit_v3.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        r = v2.get(a["run"])
        if r is None:
            continue
        tag = a["model"]
        ft = tag.endswith("+ft")
        m = tag[:-3] if ft else (tag if tag in COLORS else ("greybox" if tag.startswith("greybox_") else None))
        col = COLORS[m] if m else "#999999"
        style = dict(facecolors="none", edgecolors=col, lw=0.6) if ft else dict(color=col, lw=0)
        for drug in ACTIONS[a["env"]]:
            t, e = r.get(f"cal_{drug}_sensitivity"), a[f"{drug}_eff_ratio"]
            if t is None:
                continue
            axes[1].scatter(min(max(t, lo), hi), min(max(e, lo), hi), marker="o" if a["env"] == "sepsis" else "s",
                            s=6, alpha=0.6, **style)
    axes[1].plot([lo, hi], [lo, hi], color="k", lw=0.6, ls="--")
    axes[1].set_xlim(lo - 0.1, hi + 0.1)
    axes[1].set_ylim(lo - 0.1, hi + 0.1)
    axes[1].set_xlabel("true sensitivity (needs simulator)", fontsize=8)
    axes[1].set_ylabel("effect ratio from randomised data", fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[0].axhline(1.25, color="k", lw=0.6, ls="--")
    axes[0].set_ylabel("randomised / own-policy error", fontsize=8)
    axes[0].set_xlabel("effect skill (needs simulator)", fontsize=8)
    axes[0].axvline(0, color="k", lw=0.6, ls=":")
    axes[0].set_yscale("log")
    axes[0].set_yticks([1, 1.5, 2, 3, 5, 8])
    axes[0].set_yticklabels(["1", "1.5", "2", "3", "5", "8"])
    axes[0].minorticks_off()
    axes[0].tick_params(labelsize=7)
    handles = [Line2D([], [], color=COLORS[m], marker="o", ls="", ms=4, label=NAMES[m]) for m in BASE]
    handles += [Line2D([], [], color="#999999", marker="o", ls="", ms=4, label="Ablations"),
                Line2D([], [], color="k", marker="o", ls="", ms=4, label="Sepsis"),
                Line2D([], [], color="k", marker="s", ls="", ms=4, label="Glucose"),
                Line2D([], [], color="k", marker="o", ls="", ms=4, mfc="none", label="Fine-tuned")]
    axes[0].legend(handles=handles, fontsize=5.5, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "v2_audit.pdf")


def prose(g) -> None:
    serious, flagged, false_alarm, n = 0, 0, [], 0
    for (env, lam, tag), runs in sorted(g.items()):
        n += 1
        bad = any(mean(runs, k) < 0 for k in ("skill_no_drug", "skill_dose_up", "skill_strong", "regret_margin"))
        flag = mean(runs, "direction_worst") < 0.99 or mean(runs, "rand_gap") > 1.25 or mean(runs, "rand_cov") < 0.8
        serious += bad
        flagged += bad and flag
        if flag and not bad:
            false_alarm.append((env, lam, tag, round(mean(runs, "rand_gap"), 2), round(mean(runs, "direction_worst"), 2)))
        if bad and not flag:
            print("MISSED", env, lam, tag)
    print(f"{n} configurations; {serious} seriously broken; deployable checks flag {flagged}")
    print("flagged but not seriously broken:", false_alarm)
    gap_only = sum(1 for (e, l, t), r in g.items()
                   if any(mean(r, k) < 0 for k in ("skill_no_drug", "skill_dose_up", "skill_strong", "regret_margin"))
                   and mean(r, "rand_gap") > 1.25)
    print("of the broken, caught by rand_gap alone:", gap_only)
    keys = ("skill_dose_up", "skill_no_drug", "cal_regret", "cal_regret_fixed", "hazard_recall", "cfcov_worst",
            "rand_gap", "direction_worst", "cal_fact_rmse", "cal_cf_rmse", "ft_step")
    for env in ACTIONS:
        for lam in LAMS:
            for m in FT:
                if (env, lam, m) in g:
                    a, b = g[(env, lam, m[:-3])], g[(env, lam, m)]
                    print(env, lam, m.ljust(15), " ".join(f"{k[:12]}={mean(a, k):.2f}->{mean(b, k):.2f}" for k in keys))
    for env in ACTIONS:
        for lam in LAMS:
            r = g[(env, lam, "greybox")]
            print(env, lam, "greybox raw skill", round(mean(r, "raw:skill_dose_up"), 2), "cal", round(mean(r, "cal_skill_dose_up"), 2),
                  "kappa", [round(mean(r, f"{a}_effect_scale"), 2) for a in ACTIONS[env]],
                  "sens", [round(mean(r, f"cal_{a}_sensitivity"), 2) for a in ACTIONS[env]],
                  "regret", round(mean(r, "cal_regret"), 1), "fixed", round(mean(r, "cal_regret_fixed"), 1),
                  "hazard", round(mean(r, "hazard_recall"), 2), "alarm", round(mean(r, "cal_alarm_recall"), 2), round(mean(r, "cal_alarm_rate"), 2))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    groups = load()
    main_table(groups)
    ablation_table(groups)
    misspec_table(groups)
    fig_skill(groups)
    fig_audit(groups)
    prose(groups)
