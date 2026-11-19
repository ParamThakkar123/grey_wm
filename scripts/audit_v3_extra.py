"""Post hoc extras for audit v3 (not in the pre-registration; reported as post hoc).

    python scripts/audit_v3_extra.py
"""
import collections, json
import numpy as np
import sys; sys.path.insert(0, "scripts")
from audit_v3_analysis import rows, auc, rates

rs = rows()
ft = [r for r in rs if r["ft"]]
print("broken fine-tuned models and which checks catch them:")
for r in ft:
    if r["broken_v2"]:
        print(f"  {r['run']:35s} dir={r['dir']:d} gap={r['gap']:d} cov={r['cov']:d} eff={r['eff']:d} gapval={r['s_gap']:.2f} z={r['s_eff']:.1f}")
print("\nAUC within fine-tuned:", {s: round(auc(np.array([r[s] for r in ft]), np.array([r['broken_v2'] for r in ft])), 2) for s in ("s_gap","s_dir","s_eff")})

# 'flawed' = broken_v2 or any drug's true sensitivity outside [0.5, 2] (post hoc)
for r in rs:
    s = [x for x in r["true_sens"] if np.isfinite(x)]
    r["mis_sized"] = any(x < 0.5 or x > 2 for x in s)
    r["flawed"] = r["broken_v2"] or r["mis_sized"]
for g, name in ((rs, "all"), (ft, "fine-tuned")):
    for flag in ("old", "eff", "combined"):
        k = rates(g, flag, "flawed")
        print(f"flawed  {name:10s} {flag:9s} caught {k['caught']}/{k['broken']}  false alarms {k['false_alarms']}/{k['ok']} ({k['fa_rate']:.2f})")
fa = [r for r in rs if r["eff"] and not r["broken_v2"]]
print(f"\neffect-test false alarms (v2 truth): {len(fa)}; of which a drug's true sensitivity is outside [0.5,2]: {sum(r['mis_sized'] for r in fa)}")
ok = [r for r in rs if not r["eff"] and not r["broken_v2"]]
print(f"passes (not broken): {len(ok)}; mis-sized: {sum(r['mis_sized'] for r in ok)}")
def dev(r): return max(abs(np.log(max(x,1e-3))) for x in r["true_sens"] if np.isfinite(x))
print("median worst |log sensitivity|: false alarms", round(float(np.median([dev(r) for r in fa])),2), " passes", round(float(np.median([dev(r) for r in ok])),2))

# seed-mean (configuration) comparison with v2 paper numbers
cfg = collections.defaultdict(list)
for r in rs: cfg[(r["env"], r["lam"], r["tag"])].append(r)
n = len(cfg); out = collections.Counter()
for k, v in cfg.items():
    b = np.mean([x["broken_v2"] for x in v]) > 0.5
    for f in ("old","eff","combined"):
        fl = np.mean([x[f] for x in v]) > 0.5
        out[(f, b, fl)] += 1
print(f"\nconfigs {n} (majority of seeds):")
for f in ("old","eff","combined"):
    print(f"  {f:9s} broken caught {out[(f,True,True)]}/{out[(f,True,True)]+out[(f,True,False)]}  false alarms {out[(f,False,True)]}/{out[(f,False,True)]+out[(f,False,False)]}")
gb = [r for r in rs if r["tag"] == "greybox"]
for r in gb:
    if r["eff"]: print("greybox flagged", r["run"], "ratio", [round(x,2) for x in r["ratio"]], "true", [round(x,2) for x in r["true_sens"]])

# table for the paper: checks per model (312), broken_v2 truth
S = json.loads(open("audit_v3_summary.json").read())
names = [("dir", "s_dir", "Direction"), ("gap", "s_gap", "Randomised gap"), ("cov", "s_cov", "Randomised coverage"),
         ("eff", "s_eff", r"\textbf{Effect test (new)}"), ("old", None, "First three"), ("combined", None, "All four")]
lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{All 312 models} & \multicolumn{2}{c}{Fine-tuned (120)} & \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Check & caught & false al. & caught & false al. & AUC \\", r"\midrule"]
for flag, score, label in names:
    a, f = S[f"broken_v2|all|{flag}"], S[f"broken_v2|fine-tuned|{flag}"]
    au = f"{S[f'auc|{score}|pooled']:.2f}" if score else "--"
    lines.append(f"{label} & {a['caught']}/{a['broken']} & {a['false_alarms']}/{a['ok']} & {f['caught']}/{f['broken']} & {f['false_alarms']}/{f['ok']} & {au} " + "\\\\")
    if flag == "eff":
        lines.append(r"\midrule")
open("paper/tables/audit.tex", "w").write("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
print(open("paper/tables/audit.tex").read())
