"""Paper figures and the main table, from results.jsonl and the exported models.

    python scripts/figures.py [--results results.jsonl] [--runs runs] [--out figures]

Writes, per env where data exists:

* ``sign_vs_lambda_<env>.pdf``   fig 2: treatment-effect sign agreement vs lambda
* ``factual_vs_cf_<env>.pdf``    fig 3: factual vs counterfactual RMSE
* ``misspec_<env>.pdf``          fig 4: prior misspecification vs CF RMSE and residual norm
* ``rollout_<env>.pdf``          fig 5: one patient under a counterfactual dose change
* ``table_<env>.tex``            main table at lambda = 0.75, mean +- std over seeds
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from grey_wm.data import load_split  # noqa: E402
from grey_wm.metrics import _truth  # noqa: E402
from grey_wm.models import build_model, filter_rollout  # noqa: E402

LABELS = {
    "greybox": "Grey-box (ours)",
    "greybox-latent": "PK/PD, no sign constraint",
    "greybox-actres": "Grey-box, raw-action residual",
    "greybox-nores": "Grey-box, no residual",
    "greybox-leak": "Grey-box + raw action",
    "gru": "GRU",
    "gru-ode": "GRU-ODE",
    "crn": "CRN",
    "transformer": "Transformer",
}
ORDER = list(LABELS)
COLORS = dict(zip(ORDER, ["#c0392b", "#e67e22", "#a04000", "#f1948a", "#d35400", "#2c3e50", "#2980b9", "#27ae60", "#8e44ad"]))
TABLE_KEYS = [
    ("rmse_h1", "RMSE@1"), ("rmse_h20", "RMSE@20"), ("rmse_h50", "RMSE@50"), ("cf_rmse", "CF RMSE"),
    ("{a}_sign_agree", "Sign agr."), ("{a}_ite_rmse", "ITE err."), ("{a}_sensitivity", "Sens. ratio"),
    ("{a}_monotone_violation", "Mono. viol."), ("coverage_h20", "Cov.@20"),
]
MAIN_ACTION = {"sepsis": "norepinephrine", "glucose": "basal"}


def load(results: Path) -> list[dict]:
    return [json.loads(line) for line in results.read_text().splitlines() if line.strip()]


def base_runs(rows: list[dict]) -> list[dict]:
    return [r for r in rows if "ke_scale" not in r and "ec50_scale" not in r]


def by(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def mean_std(rows, key):
    v = [r[key] for r in rows if key in r]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)


def fig_sign_vs_lambda(rows, env, actions, out):
    fig, axes = plt.subplots(1, len(actions), figsize=(4.2 * len(actions), 3.5), squeeze=False)
    groups = by([r for r in rows if r["env"] == env], "model", "lam")
    curves = [m for m in ORDER if sum(g[0] == m for g in groups) > 1]  # ablations run at one lambda only
    for ax, action in zip(axes[0], actions):
        for model in curves:
            lams = sorted({lam for m, lam in groups if m == model})
            stats = [mean_std(groups[(model, lam)], f"{action}_sign_agree") for lam in lams]
            m, s = np.array(stats).T
            ax.errorbar(lams, m, yerr=s, label=LABELS[model], color=COLORS[model], marker="o", ms=3, capsize=2,
                        lw=2.2 if model == "greybox" else 1.2)
        ax.axhline(0.5, color="grey", ls=":", lw=0.8)
        ax.set(xlabel=r"confounding $\lambda$", ylabel="treatment-effect sign agreement", title=action, ylim=(-0.03, 1.03))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=len(curves))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / f"sign_vs_lambda_{env}.pdf")
    plt.close(fig)


def fig_factual_vs_cf(rows, env, out):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for r in rows:
        if r["env"] != env or "cf_rmse" not in r:
            continue
        ax.scatter(r["rmse_h20"], r["cf_rmse"], color=COLORS[r["model"]], s=10 + 40 * r["lam"], alpha=0.75,
                   edgecolor="none")
    for model in ORDER:
        ax.scatter([], [], color=COLORS[model], label=LABELS[model], s=15)
    ax.set(xlabel="factual rollout RMSE (h=20)", ylabel="counterfactual RMSE")
    ax.text(0.98, 0.02, r"marker size $\propto\lambda$", transform=ax.transAxes, ha="right", fontsize=7)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"factual_vs_cf_{env}.pdf")
    plt.close(fig)


def fig_misspec(rows, env, out):
    mis = [r for r in rows if r["env"] == env and r["lam"] == 0.75 and r["model"] in ("greybox", "greybox-nores")]
    if not any("ke_scale" in r or "ec50_scale" in r for r in mis):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    for ax, param in zip(axes, ("ke_scale", "ec50_scale")):
        for model in ("greybox", "greybox-nores"):
            pts = defaultdict(list)
            for r in mis:
                if r["model"] != model:
                    continue
                other = "ec50_scale" if param == "ke_scale" else "ke_scale"
                if r.get(other, 1.0) != 1.0:
                    continue
                pts[r.get(param, 1.0)].append(r)
            xs = sorted(pts)
            if not xs:
                continue
            m, s = np.array([mean_std(pts[x], "cf_rmse") for x in xs]).T
            ax.errorbar(xs, m, yerr=s, marker="o", ms=3, capsize=2, color=COLORS[model], label=f"{LABELS[model]}: CF RMSE")
            if model == "greybox":
                twin = ax.twinx()
                m2, s2 = np.array([mean_std(pts[x], "residual_norm") for x in xs]).T
                twin.errorbar(xs, m2, yerr=s2, marker="s", ms=3, capsize=2, ls="--", color="grey", label="residual norm")
                twin.set_ylabel("residual pathway norm", color="grey")
        ax.axvline(1.0, color="k", lw=0.6, ls=":")
        ax.set(xlabel=f"assumed {param.split('_')[0]} / true (log scale)", ylabel="counterfactual RMSE", xscale="log")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"misspec_{env}.pdf")
    plt.close(fig)


def load_model(run_dir: Path, spec):
    cfg = json.loads((run_dir / "model" / "model.json").read_text())["model"]
    kind = cfg.pop("kind")
    model = build_model(kind, spec, **cfg)
    model.load_state_dict(torch.load(run_dir / "model" / "model.pt", map_location="cpu", weights_only=True))
    return model.eval()


@torch.no_grad()
def fig_rollout(runs: Path, data_root: str, env: str, out: Path, lam: float = 0.75, seed: int = 0, patient: int = 0):
    try:
        test = load_split(data_root, env, lam)["test"]
    except FileNotFoundError:
        return
    one = test.subset(np.array([patient]))
    spec, o = one.spec, one.spec.outcome_index
    n_context = 24 if env == "sepsis" else 48
    horizon = 40 if env == "sepsis" else 60
    k = n_context - 1
    factual = one["actions"][:, k : k + horizon]
    raised = factual.copy()
    raised[..., 0] = np.minimum(raised[..., 0] + 0.4 * spec.action_max[0], spec.action_max[0])
    hours = spec.dt * np.arange(n_context + horizon)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, (name, acts) in zip(axes, (("factual dose", factual), (f"{spec.action_names[0]} raised", raised))):
        truth = _truth(one, n_context, acts)[0, :, o]
        ax.plot(hours[:n_context], one["clean"][0, :n_context, o], color="k", lw=1)
        ax.plot(hours[n_context:], truth, color="k", lw=2, label="simulator (truth)")
        batch = one.batch("cpu")
        for model in ORDER:
            run_dir = runs / env / f"lam{lam:.2f}" / model / f"s{seed}"
            if not (run_dir / "model" / "model.pt").exists():
                continue
            m = load_model(run_dir, spec)
            pred = filter_rollout(m, batch, n_context, horizon, torch.as_tensor(acts, dtype=torch.float32)).mean
            pred = pred[0, :, o].numpy() * spec.obs_std[o] + spec.obs_mean[o]
            ax.plot(hours[n_context:], pred, color=COLORS[model], lw=1.8 if model == "greybox" else 1, label=LABELS[model])
        ax.axvline(hours[n_context - 1], color="grey", ls=":", lw=0.8)
        ax.set(title=name, xlabel="hours", ylabel=spec.outcome)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / f"rollout_{env}.pdf")
    plt.close(fig)


def table(rows, env, out, lam=0.75):
    action = MAIN_ACTION[env]
    groups = by([r for r in rows if r["env"] == env and r["lam"] == lam], "model")
    if not groups:
        return
    cols = [(k.format(a=action), name) for k, name in TABLE_KEYS]
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Model & " + " & ".join(name for _, name in cols) + r" \\", r"\midrule"]
    for model in ORDER:
        if (model,) not in groups:
            continue
        cells = []
        for key, _ in cols:
            m, s = mean_std(groups[(model,)], key)
            cells.append("--" if np.isnan(m) else f"{m:.3f}$\\pm${s:.3f}")
        lines.append(f"{LABELS[model]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / f"table_{env}_lam{lam:.2f}.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["results.jsonl", "results_transformer.jsonl"])
    p.add_argument("--runs", default="runs")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rows = [r for path in args.results if Path(path).exists() for r in load(Path(path))]
    base = base_runs(rows)
    actions = {"sepsis": ["norepinephrine", "fluid"], "glucose": ["basal", "bolus"]}
    for env in sorted({r["env"] for r in rows}):
        fig_sign_vs_lambda(base, env, actions[env], out)
        fig_factual_vs_cf(base, env, out)
        fig_misspec(rows, env, out)
        fig_rollout(Path(args.runs), args.data_root, env, out)
        table(base, env, out, lam=0.75)
        table(base, env, out, lam=1.0)
        print(f"{env}: figures and table written to {out}/")


if __name__ == "__main__":
    main()
