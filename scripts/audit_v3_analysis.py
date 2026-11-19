"""Audit v3 analysis, as fixed in notes/audit_v3_preregistration.md.

    python scripts/audit_v3_analysis.py      # prints everything, writes audit_v3_summary.json

Unit = one trained model (per seed). Reads the v2 results (benchmark metrics and
old deployable gates, calibrated keys) and results_audit_v3.jsonl (both drugs'
effect skill, randomised effect test), plus audit_v3/<run>.npz for the holdout-size
study.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from grey_wm.audit import derived

RESULTS = ("results_v2.jsonl", "results_v2_transformer.jsonl", "results_ft.jsonl")
ACTIONS = {"sepsis": ("norepinephrine", "fluid"), "glucose": ("basal", "bolus")}
PULSE = {"bolus"}
SIZES = (18, 36, 54)
DRAWS = 10
BOOT = 2000


def rows(results=RESULTS, audit_file="results_audit_v3.jsonl") -> list[dict]:
    v2 = {}
    for f in results:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                v2[r["run"]] = r
    out = []
    for line in Path(audit_file).read_text().splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        r = v2.get(a["run"])
        if r is None:
            print("no v2 row for", a["run"])
            continue
        env = a["env"]
        first, second = ACTIONS[env]
        d = derived(r, ACTIONS[env], "cal_")
        pols = ("no_drug", "dose_up") + (() if second in PULSE else ("strong",))
        b1 = any(d[k] < 0 for k in ("skill_no_drug", "skill_dose_up", "skill_strong", "regret_margin"))
        b2 = b1 or any(a[f"{second}_skill_{p}"] < 0 for p in pols)
        check = abs(a[f"{first}_skill_dose_up"] - r["cal_skill_dose_up"])
        out.append({
            "run": a["run"], "env": env, "lam": a["lam"], "tag": a["model"], "seed": a["seed"],
            "ft": a["model"].endswith("+ft"), "broken_v1": b1, "broken_v2": b2, "repro_err": check,
            "dir": d["direction_worst"] < 0.99, "gap": d["rand_gap"] > 1.25, "cov": d["rand_cov"] < 0.8,
            "eff": bool(a["eff_flag"]),
            "s_gap": d["rand_gap"], "s_dir": 1 - d["direction_worst"], "s_cov": 1 - d["rand_cov"], "s_eff": a["eff_zmax"],
            "ratio": [a[f"{x}_eff_ratio"] for x in ACTIONS[env]],
            "true_sens": [r.get(f"cal_{x}_sensitivity", float("nan")) for x in ACTIONS[env]],
        })
    for r in out:
        r["old"] = r["dir"] or r["gap"] or r["cov"]
        r["combined"] = r["old"] or r["eff"]
    return out


def rates(rs: list[dict], flag: str, truth: str) -> dict:
    bad = [r for r in rs if r[truth]]
    good = [r for r in rs if not r[truth]]
    tp = sum(r[flag] for r in bad)
    fp = sum(r[flag] for r in good)
    return {"broken": len(bad), "caught": tp, "sens": tp / max(len(bad), 1),
            "ok": len(good), "false_alarms": fp, "fa_rate": fp / max(len(good), 1),
            "precision": tp / max(tp + fp, 1)}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def youden(score: np.ndarray, label: np.ndarray) -> float:
    best, thr = -1.0, float("nan")
    for t in np.unique(score):
        flag = score >= t
        j = flag[label].mean() + (~flag[~label]).mean() - 1
        if j > best:
            best, thr = j, float(t)
    return thr


def fast_test(d: dict, patients: np.ndarray, rng: np.random.Generator, boot: int = BOOT) -> bool:
    """effect_test via per-patient sufficient statistics; same draws as audit.effect_test."""
    X = np.column_stack([np.ones(len(d["resid"])), d["dose"]])
    k = X.shape[1]
    xtx = np.stack([X[d["patient"] == p].T @ X[d["patient"] == p] for p in patients])
    xty = np.stack([X[d["patient"] == p].T @ d["resid"][d["patient"] == p] for p in patients])
    idx = {p: i for i, p in enumerate(patients)}
    counts = np.zeros((boot, len(patients)))
    for b in range(boot):
        for p in rng.choice(patients, len(patients)):
            counts[b, idx[p]] += 1
    A = (counts @ xtx.reshape(len(patients), -1)).reshape(boot, k, k)
    y = counts @ xty
    coef = np.linalg.solve(A, y[..., None])[..., 0][:, 1:]
    lo, hi = np.percentile(coef, [2.5, 97.5], axis=0)
    return bool(((lo > 0) | (hi < 0)).any())


def main() -> None:
    rs = rows()
    summary: dict = {"n_models": len(rs)}
    print(f"{len(rs)} models; max |drug-0 skill - v2 value| = {max(r['repro_err'] for r in rs):.2e}")
    groups = {"all": rs, "not fine-tuned": [r for r in rs if not r["ft"]], "fine-tuned": [r for r in rs if r["ft"]]}
    for truth in ("broken_v1", "broken_v2"):
        for gname, g in groups.items():
            for flag in ("dir", "gap", "cov", "old", "eff", "combined"):
                k = rates(g, flag, truth)
                summary[f"{truth}|{gname}|{flag}"] = k
                print(f"{truth:9s} {gname:15s} {flag:9s} caught {k['caught']:3d}/{k['broken']:3d} ({k['sens']:.2f})"
                      f"  false alarms {k['false_alarms']:3d}/{k['ok']:3d} ({k['fa_rate']:.2f})  precision {k['precision']:.2f}")
    missed = [(r["run"]) for r in rs if r["broken_v2"] and not r["combined"]]
    print("missed by combined gate (v2 truth):", missed)
    summary["missed_combined_v2"] = missed

    print("\nAUC for broken_v2 (higher score = more suspicious)")
    for s in ("s_gap", "s_dir", "s_cov", "s_eff"):
        line = []
        for env in ("sepsis", "glucose", None):
            g = [r for r in rs if env is None or r["env"] == env]
            a = auc(np.array([r[s] for r in g]), np.array([r["broken_v2"] for r in g]))
            summary[f"auc|{s}|{env or 'pooled'}"] = a
            line.append(f"{env or 'pooled'} {a:.2f}")
        print(f"  {s}: " + ", ".join(line))

    print("\nTransfer: threshold by Youden on one simulator, tested on the other (broken_v2)")
    for s in ("s_gap", "s_dir", "s_cov", "s_eff"):
        for src, dst in (("sepsis", "glucose"), ("glucose", "sepsis")):
            a = [r for r in rs if r["env"] == src]
            b = [r for r in rs if r["env"] == dst]
            thr = youden(np.array([r[s] for r in a]), np.array([r["broken_v2"] for r in a]))
            sb, lb = np.array([r[s] for r in b]), np.array([r["broken_v2"] for r in b])
            sens, spec = float((sb >= thr)[lb].mean()), float((sb < thr)[~lb].mean())
            summary[f"transfer|{s}|{src}->{dst}"] = {"threshold": thr, "sens": sens, "spec": spec}
            print(f"  {s} {src}->{dst}: threshold {thr:.3f}, sens {sens:.2f}, spec {spec:.2f}")

    print("\nHoldout size (effect test alone, broken_v2)")
    rng = np.random.default_rng(0)
    data = {r["run"]: dict(np.load(Path("audit_v3") / f"{r['run']}.npz")) for r in rs}
    agree = sum(fast_test(data[r["run"]], np.unique(data[r["run"]]["patient"]), np.random.default_rng(0)) == r["eff"] for r in rs)
    print(f"  fast test reproduces the recorded flag for {agree}/{len(rs)} models")
    for n in SIZES:
        flags = {r["run"]: [] for r in rs}
        for _ in range(DRAWS):
            for r in rs:
                ids = np.unique(data[r["run"]]["patient"])
                pick = np.sort(rng.choice(ids, min(n, len(ids)), replace=False))
                flags[r["run"]].append(fast_test(data[r["run"]], pick, rng))
        bad = [np.mean(flags[r["run"]]) for r in rs if r["broken_v2"]]
        good = [np.mean(flags[r["run"]]) for r in rs if not r["broken_v2"]]
        summary[f"size|{n}"] = {"sens": float(np.mean(bad)), "fa_rate": float(np.mean(good))}
        print(f"  {n} patients: sens {np.mean(bad):.2f}, false-alarm rate {np.mean(good):.2f}")

    ratio = np.array([x for r in rs for x in r["ratio"]])
    true = np.array([x for r in rs for x in r["true_sens"]])
    ok = np.isfinite(ratio) & np.isfinite(true) & (np.abs(true) < 10)
    summary["ratio_vs_true_corr"] = float(np.corrcoef(ratio[ok], true[ok])[0, 1])
    print(f"\nEffect ratio from randomised data vs simulator sensitivity: r = {summary['ratio_vs_true_corr']:.2f} (n={ok.sum()})")

    by_cfg = collections.defaultdict(list)
    for r in rs:
        by_cfg[(r["env"], r["lam"], r["tag"])].append(r)
    gb = {k: v for k, v in by_cfg.items() if k[2] == "greybox"}
    print("\nGrey-box flags by effect test (seeds flagged / 3):",
          {f"{k[0]} {k[1]}": sum(r["eff"] for r in v) for k, v in sorted(gb.items())})
    Path("audit_v3_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
