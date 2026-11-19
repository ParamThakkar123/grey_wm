"""Post hoc: the effect test at stricter thresholds, and what its other flags point at.

    python scripts/effect_test_thresholds.py
"""

from __future__ import annotations


from audit_v3_analysis import rows

THRESHOLDS = (("95% (paper)", 1.96), ("95%, Bonferroni over 2 drugs", 2.24), ("99%", 2.576), ("99.9%", 3.29))


NO_BOLUS = False  # set by main: leave out the glucose bolus, whose sensitivity uses an unrealistic dose change


def off(r: dict, factor: float) -> bool:
    """Some drug's true sensitivity is off by more than ``factor`` (or reversed)."""
    sens = r["true_sens"][:1] if NO_BOLUS and r["env"] == "glucose" else r["true_sens"]
    return any(s == s and (s <= 0 or max(s, 1 / s) > factor) for s in sens)


def main() -> None:
    global NO_BOLUS
    for NO_BOLUS in (False, True):
        print("\n== glucose bolus " + ("left out" if NO_BOLUS else "included") + " when judging how far off a model is")
        report()


def report() -> None:
    rs = rows()
    bad = [r for r in rs if r["broken_v2"]]
    good = [r for r in rs if not r["broken_v2"]]
    print(f"{len(rs)} models, {len(bad)} broken")
    for name, z in THRESHOLDS:
        tp = sum(r["s_eff"] > z for r in bad)
        fp = [r for r in good if r["s_eff"] > z]
        print(f"{name:30s} caught {tp}/{len(bad)}  flagged {len(fp)}/{len(good)} of the rest; "
              + ", ".join(f"{sum(off(r, f) for r in fp)} of those off >{f}x" for f in (1.25, 1.5, 2)))
    for f in (1.25, 1.5, 2):
        sub = [r for r in good if off(r, f)]
        rest = [r for r in good if not off(r, f)]
        print(f"not broken, some drug off >{f}x: {len(sub)}, flagged {sum(r['eff'] for r in sub)}; "
              f"within {f}x: {len(rest)}, flagged {sum(r['eff'] for r in rest)}")


if __name__ == "__main__":
    main()
