"""Table of planning regret (open and closed loop) at lambda = 1: paper/tables/planning.tex.

    python scripts/planning_table.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

ROWS = (("greybox", r"\textbf{Grey-box (ours)}"), ("greybox_pkpd", "PK/PD only"),
        ("gru+ft", "GRU fine-tuned"), ("gru-ode+ft", "GRU-ODE fine-tuned"), ("crn+ft", "CRN fine-tuned"),
        ("transformer+ft", "Transformer fine-tuned"), ("gru-ipw", "GRU, IPW-weighted"),
        ("gru", "GRU"), ("gru-ode", "GRU-ODE"), ("crn", "CRN"), ("transformer", "Transformer"))


def load(path: str, key: str) -> dict:
    g = collections.defaultdict(list)
    for r in map(json.loads, Path(path).read_text().splitlines()):
        if r["seed"] < 5:  # seeds 0-4; the fresh seeds 5-14 are in the confirmatory table
            g[(r["env"], r["model"])].append(r)
    return {k: (sum(r[key] for r in v) / len(v), len(v), v[0]["plan_regret_fixed"]) for k, v in g.items()}


def main() -> None:
    op, cl = load("results_planning.jsonl", "plan_regret"), load("results_planning_cl.jsonl", "cl_regret")
    f = lambda x: f"{x:.2f}"  # noqa: E731
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Sepsis} & \multicolumn{2}{c}{Glucose} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Model & Seeds & Open & Closed & Open & Closed \\", r"\midrule",
             " & ".join(["Fixed schedule", "", f(op[("sepsis", "greybox")][2]), f(cl[("sepsis", "greybox")][2]),
                         f(op[("glucose", "greybox")][2]), f(cl[("glucose", "greybox")][2])]) + r" \\", r"\midrule"]
    for tag, name in ROWS:
        cells = [f(d[(env, tag)][0]) for env in ("sepsis", "glucose") for d in (op, cl)]
        if tag == "greybox":
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join([name, str(op[("sepsis", tag)][1])] + cells) + r" \\")
    Path("paper/tables/planning.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


if __name__ == "__main__":
    main()
