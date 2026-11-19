"""Tables for the two follow-up studies (randomised-set size; glucose prior misspecification).

    python scripts/weakness_tables.py   # writes paper/tables/size.tex and misspec_glucose.tex, prints prose numbers

Size study: models trained with only the first n randomised trajectories (n = 25, 50; n = 100 is the
main sweep) at lambda = 1. Misspecification: glucose grey-box at lambda = 0.75 with a wrong prior.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from paper_tables import NAMES, mean, ms, row, write

SIZE = ("results_size.jsonl", "results_size_ft.jsonl", "results_v2.jsonl", "results_ft.jsonl")
MISSPEC = ("results_misspec_glucose.jsonl", "results_v2.jsonl")
NS = (25, 50, 100)
SIZE_MODELS = ("greybox", "gru", "gru+ft", "gru-ode+ft", "crn+ft")


def load(paths) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = collections.defaultdict(list)
    for path in paths:
        if not Path(path).exists():
            continue
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                g[(r["env"], r["lam"], r["run"].split("/")[2])].append(r)
    return g


def tag(model: str, n: int) -> str:
    if n == 100:
        return model
    base, ft = (model[:-3], "+ft") if model.endswith("+ft") else (model, "")
    return f"{base}_n{n}{ft}"


def broken(runs) -> bool:
    return any(mean(runs, k) < 0 for k in ("skill_no_drug", "skill_dose_up", "skill_strong", "regret_margin"))


def size_table(g) -> None:
    out = ["\\begin{tabular}{llcccccc}", "\\toprule",
           " & & \\multicolumn{3}{c}{Effect skill $\\uparrow$} & \\multicolumn{3}{c}{Regret margin $\\uparrow$} \\\\",
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
           " & Model & $n{=}25$ & $50$ & $100$ & $25$ & $50$ & $100$ \\\\", "\\midrule"]
    for env in ("sepsis", "glucose"):
        for i, m in enumerate(SIZE_MODELS):
            runs = [g.get((env, 1.0, tag(m, n)), []) for n in NS]
            cells = [ms(r, "skill_dose_up", 2, False) if r else "--" for r in runs]
            cells += [ms(r, "regret_margin", 1, False) if r else "--" for r in runs]
            label = {"gru": "GRU (no fine-tune)", "gru+ft": "GRU fine-tuned", "gru-ode+ft": "GRU-ODE fine-tuned",
                     "crn+ft": "CRN fine-tuned"}.get(m, NAMES.get(m, m))
            out.append(row([env.capitalize() if i == 0 else "", label] + cells, m == "greybox"))
            for n, r in zip(NS, runs):
                print(f"size {env} {m:11s} n={n:3d} seeds={len(r)} "
                      + (f"skill={mean(r, 'skill_dose_up'):.2f} regret_margin={mean(r, 'regret_margin'):.1f} "
                         f"cf_rmse={mean(r, 'cal_cf_rmse'):.3f} broken={broken(r)}" if r else "missing"))
        if env == "sepsis":
            out.append("\\midrule")
    write("size.tex", out + ["\\bottomrule", "\\end{tabular}"])


def misspec_table(g) -> None:
    out = ["\\begin{tabular}{lcccccc}", "\\toprule",
           " & \\multicolumn{2}{c}{Bolus sensitivity} & & & & \\\\", "\\cmidrule(lr){2-3}",
           "Assumed prior & raw & calibrated & $\\kappa_{\\mathrm{bolus}}$ & $\\kappa_{\\mathrm{basal}}$ & Skill & Gap \\\\",
           "\\midrule"]
    rows = [("correct", "greybox")] \
        + [(f"$E_{{\\max}}\\times{s:g}$", f"greybox_emax_scale{s:g}") for s in (0.25, 0.5, 2, 4)] \
        + [(f"$h\\times{s:g}$", f"greybox_hill_scale{s:g}") for s in (0.5, 2)] \
        + [(f"$k_e\\times{s:g}$", f"greybox_ke_scale{s:g}") for s in (0.5, 2)]
    for label, t in rows:
        r = g.get(("glucose", 0.75, t), [])
        if not r:
            print("misspec missing", t)
            out.append(row([label] + ["--"] * 6))
            continue
        out.append(row([label, ms(r, "bolus_sensitivity", 2, False), ms(r, "cal_bolus_sensitivity", 2, False),
                        ms(r, "bolus_effect_scale", 2, False), ms(r, "basal_effect_scale", 2, False),
                        ms(r, "skill_dose_up", 2, False), ms(r, "rand_gap", 2, False)], t == "greybox"))
        print(f"misspec {t:24s} seeds={len(r)} bolus raw={mean(r, 'bolus_sensitivity'):.2f} "
              f"cal={mean(r, 'cal_bolus_sensitivity'):.2f} basal cal={mean(r, 'cal_basal_sensitivity'):.2f} "
              f"k_bolus={mean(r, 'bolus_effect_scale'):.2f} k_basal={mean(r, 'basal_effect_scale'):.2f} "
              f"skill={mean(r, 'skill_dose_up'):.2f} gap={mean(r, 'rand_gap'):.2f} "
              f"regret_margin={mean(r, 'regret_margin'):.1f} broken={broken(r)} "
              f"dir={mean(r, 'direction_worst'):.2f}")
    write("misspec_glucose.tex", out + ["\\bottomrule", "\\end{tabular}"])


def audit_summary() -> None:
    """The pre-registered effect test (and the old gates) on the new models; same definitions as audit v3."""
    from audit_v3_analysis import rows
    if not Path("results_audit_weak.jsonl").exists():
        print("no audit file yet")
        return
    rs = rows(("results_size.jsonl", "results_size_ft.jsonl", "results_misspec_glucose.jsonl"), "results_audit_weak.jsonl")
    groups = {"glucose misspec": [r for r in rs if "_scale" in r["tag"]],
              "size, fine-tuned": [r for r in rs if "_n" in r["tag"] and r["ft"]],
              "size, unadjusted": [r for r in rs if "_n" in r["tag"] and not r["ft"] and not r["tag"].startswith("greybox")],
              "size, grey-box": [r for r in rs if r["tag"].startswith("greybox_n")]}
    for name, g in groups.items():
        bad = [r for r in g if r["broken_v2"]]
        good = [r for r in g if not r["broken_v2"]]
        print(f"audit {name}: n={len(g)} broken={len(bad)} | effect test caught {sum(r['eff'] for r in bad)}/{len(bad)}"
              f" false alarms {sum(r['eff'] for r in good)}/{len(good)} | old gates caught {sum(r['old'] for r in bad)}/{len(bad)}"
              f" false {sum(r['old'] for r in good)}/{len(good)} | gap caught {sum(r['gap'] for r in bad)}/{len(bad)}")
    for r in groups["glucose misspec"]:
        print(f"  {r['run']:44s} broken={r['broken_v2']} eff={r['eff']} ratio={[round(x, 2) for x in r['ratio']]} "
              f"true={[round(x, 2) for x in r['true_sens']]}")


if __name__ == "__main__":
    size_table(load(SIZE))
    misspec_table(load(MISSPEC))
    audit_summary()
