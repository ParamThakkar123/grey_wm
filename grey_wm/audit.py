"""Can we tell that a world model is wrong? The checks factual error cannot make.

Two tiers, because a deployed model has no simulator behind it:

Benchmark tier (needs counterfactual truth):

* ``skill_{policy}``: error in the predicted *change* in outcome, against assuming
  the drug does nothing. 1 is perfect, 0 no better than no effect, negative worse.
* ``cfcov_{policy}``: 90% interval coverage under a counterfactual policy.
* ``regret`` / ``regret_fixed``: extra distance from the target when dosing by the
  model, and when giving everyone the single best dose. A model that does not
  beat ``regret_fixed`` has no use as a personal dosing tool.
* ``hazard_recall``: share of dangerous lows the model sees coming.
* ``{action}_sens_sick`` / ``_sens_rest``: effect size in the sickest quarter and
  the rest; an average can hide the patients who need the drug most.
* ``{action}_course_{h}h``: predicted over true effect at ``h`` hours, left out
  while the true effect is under a quarter of its largest size in the window.

Deployable tier (needs only a small randomised set):

* ``rand_rmse``, ``rand_cov``: rollout error and 90% coverage on held-out
  randomised dosing. ``fact_rmse`` is the same error on the model's own policy;
  a model that only forecasts because doses give severity away shows a large gap.
* ``{action}_direction``: share of patients whose predicted effect has the
  direction pharmacology says it must. Needs no outcome data at all.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from grey_wm.data import TrajectorySet
from grey_wm.metrics import _nrmse, _Predictor, _truth
from grey_wm.models.base import SequenceModel

Z90 = 1.6449
DOSE_LEVELS = 5
COURSE_HOURS = (1, 2, 4, 6)
# (outcome target, danger threshold, dose fraction that exposes the danger)
TARGETS = {"sepsis": (70.0, 65.0, 0.0), "glucose": (130.0, 70.0, 1.0)}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def _at(actions: np.ndarray, dose: float) -> np.ndarray:
    fixed = actions.copy()
    fixed[..., 0] = dose
    return fixed


def _policies(base: np.ndarray, high: np.ndarray) -> dict[str, np.ndarray]:
    """Changes to the first drug only: raising a bolus at every step is not a treatment."""
    none, up, strong = base.copy(), base.copy(), base.copy()
    none[..., 0] = 0.0
    up[..., 0] = np.minimum(1.3 * base[..., 0], high[0])
    strong[..., 0] = 0.8 * high[0]
    return {"no_drug": none, "dose_up": up, "strong": strong}


@torch.no_grad()
def run(
    model: SequenceModel,
    data: TrajectorySet,
    holdout: TrajectorySet,
    n_context: int,
    horizon: int,
    effect_window: int,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    spec = data.spec
    predict = _Predictor(model, data, n_context, device)
    k, o = n_context - 1, spec.outcome_index
    high = np.asarray(spec.action_max)
    rng = np.random.default_rng(0)
    steps = max(horizon, effect_window, *(int(round(h / spec.dt)) for h in COURSE_HOURS))
    steps = min(steps, data["actions"].shape[1] - n_context + 1)
    factual = data["actions"][:, k : k + steps]
    out: dict[str, float] = {}

    base = factual[:, :horizon]
    base_model, _, _ = predict(horizon, base)
    base_true = _truth(data, n_context, base)
    for name, future in _policies(base, high).items():
        mean, sd, _ = predict(horizon, future)
        true = _truth(data, n_context, future)
        change_model, change_true = (mean - base_model)[..., o], (true - base_true)[..., o]
        error = np.sqrt(np.mean((change_model - change_true) ** 2))
        out[f"skill_{name}"] = float(1 - error / max(np.sqrt(np.mean(change_true**2)), 1e-9))
        noisy = true + rng.standard_normal(true.shape) * np.asarray(spec.obs_noise)
        out[f"cfcov_{name}"] = float(np.mean(np.abs(noisy - mean) <= Z90 * sd))
        out[f"cfwidth_{name}"] = float(np.mean(2 * Z90 * sd[..., o]))

    window = factual[:, :effect_window]
    model_base = predict(effect_window, window)[0][:, -1, o]
    true_base = _truth(data, n_context, window)[:, -1, o]
    sickness = data.sim.sickness(data["states"][:, k].astype(np.float64))
    sick = sickness >= np.quantile(sickness, 0.75)
    for c, action in enumerate(spec.action_names):
        raised = window.copy()
        raised[..., c] = np.minimum(raised[..., c] + 0.2 * high[c], high[c])
        ite_model = predict(effect_window, raised)[0][:, -1, o] - model_base
        ite_true = _truth(data, n_context, raised)[:, -1, o] - true_base
        out[f"{action}_sens_sick"] = float(ite_model[sick].mean() / ite_true[sick].mean())
        out[f"{action}_sens_rest"] = float(ite_model[~sick].mean() / ite_true[~sick].mean())
        out[f"{action}_ite_corr"] = _corr(ite_model, ite_true)
        expected = np.sign(spec.pk_prior[c].signs[o]) if spec.pk_prior[c].signs else 0.0
        out[f"{action}_direction"] = float(np.mean(np.sign(ite_model) == expected))

        long = factual.copy()
        long[..., c] = np.minimum(long[..., c] + 0.2 * high[c], high[c])
        dm = (predict(steps, long)[0] - predict(steps, factual)[0])[..., o].mean(0)
        dt = (_truth(data, n_context, long) - _truth(data, n_context, factual))[..., o].mean(0)
        floor = 0.25 * np.abs(dt).max()
        for h in COURSE_HOURS:
            i = int(round(h / spec.dt)) - 1
            if i < steps:
                out[f"{action}_course_{h}h"] = float(dm[i] / dt[i]) if abs(dt[i]) >= max(floor, 1e-6) else float("nan")

    target, danger, exposed = TARGETS[spec.name]
    levels = np.linspace(0.0, 1.0, DOSE_LEVELS)
    pred, true = [], []
    for level in levels:
        dosed = window.copy()
        dosed[..., 0] = level * high[0]
        pred.append(predict(effect_window, dosed)[0][:, -1, o])
        true.append(_truth(data, n_context, dosed)[:, -1, o])
    pred, true = np.stack(pred, 1), np.stack(true, 1)
    rows = np.arange(len(pred))
    chosen, oracle = np.abs(pred - target).argmin(1), np.abs(true - target).argmin(1)
    best = np.abs(true[rows, oracle] - target)
    out["regret"] = float(np.mean(np.abs(true[rows, chosen] - target) - best))
    out["regret_fixed"] = float(min(np.mean(np.abs(true[:, j] - target) - best) for j in range(DOSE_LEVELS)))
    out["dose_chosen"] = float(levels[chosen].mean())
    out["dose_oracle"] = float(levels[oracle].mean())
    j = int(np.argmin(np.abs(levels - exposed)))
    event = true[:, j] < danger
    out["hazard_n"] = float(event.sum())
    out["hazard_recall"] = float((pred[:, j] < danger)[event].mean()) if event.any() else float("nan")
    sd_j = predict(effect_window, _at(window, levels[j] * high[0]))[1][:, -1, o]
    alarm = pred[:, j] - 1.2816 * sd_j < danger  # the 10% lower bound crosses the line
    out["alarm_recall"] = float(alarm[event].mean()) if event.any() else float("nan")
    out["alarm_rate"] = float(alarm.mean())

    out["fact_rmse"] = _nrmse(base_model, data["clean"][:, n_context : n_context + horizon], predict.std)
    held = _Predictor(model, holdout, n_context, device)
    mean, sd, _ = held(horizon)
    win = slice(n_context, n_context + horizon)
    mask = holdout["mask"][:, win].astype(bool)
    out["rand_rmse"] = _nrmse(mean, holdout["clean"][:, win], held.std)
    out["rand_cov"] = float((np.abs(holdout["y"][:, win] - mean) <= Z90 * sd)[mask].mean())
    return out


@torch.no_grad()
def drug_skill(model: SequenceModel, data: TrajectorySet, n_context: int, horizon: int, channel: int) -> dict[str, float]:
    """Effect skill for one drug: stop it, raise it 30%, and (unless it is given as
    occasional pulses) set it to 80% of maximum. ``channel = 0`` repeats ``run``."""
    model.eval()
    spec = data.spec
    predict = _Predictor(model, data, n_context, next(model.parameters()).device)
    k, o = n_context - 1, spec.outcome_index
    high = np.asarray(spec.action_max)
    base = data["actions"][:, k : k + horizon]
    base_model, base_true = predict(horizon, base)[0], _truth(data, n_context, base)
    pulse = data.sim.action_pulse
    policies = {"no_drug": 0.0, "dose_up": None}
    if not (pulse and pulse[channel] < 1.0):
        policies["strong"] = 0.8 * high[channel]
    out: dict[str, float] = {}
    for name, level in policies.items():
        future = base.copy()
        future[..., channel] = np.minimum(1.3 * base[..., channel], high[channel]) if level is None else level
        change_model = (predict(horizon, future)[0] - base_model)[..., o]
        change_true = (_truth(data, n_context, future) - base_true)[..., o]
        error = np.sqrt(np.mean((change_model - change_true) ** 2))
        out[f"skill_{name}"] = float(1 - error / max(np.sqrt(np.mean(change_true**2)), 1e-9))
    return out


EFFECT_STARTS = 4


@torch.no_grad()
def effect_test_data(model: SequenceModel, holdout: TrajectorySet, n_context: int, horizon: int) -> dict[str, np.ndarray]:
    """Per-window inputs of the randomised effect test (notes/audit_v3_preregistration.md).

    For each held-out randomised patient and each of ``EFFECT_STARTS`` start times:
    the mean normalised outcome residual (observed minus predicted, over observed
    steps), the mean predicted and observed outcome, and each drug's mean dose over
    the window as a fraction of its maximum. The test itself is ``effect_test``.
    """
    model.eval()
    spec = holdout.spec
    device = next(model.parameters()).device
    o, sd = spec.outcome_index, float(spec.obs_std[spec.outcome_index])
    high = np.asarray(spec.action_max)
    steps = holdout["actions"].shape[1]
    last = steps - horizon + 1
    starts = [n_context + i * (last - n_context) // (EFFECT_STARTS - 1) for i in range(EFFECT_STARTS)]
    rows: dict[str, list] = {"patient": [], "start": [], "resid": [], "pred": [], "obs": [], "dose": []}
    for k in starts:
        mean = _Predictor(model, holdout, k, device)(horizon)[0][..., o]
        win = slice(k, k + horizon)
        y, seen = holdout["y"][:, win, o], holdout["mask"][:, win, o].astype(bool)
        n_seen = seen.sum(1)
        keep = n_seen > 0
        avg = lambda a: (np.where(seen, a, 0.0).sum(1) / np.maximum(n_seen, 1))  # noqa: E731
        dose = holdout["actions"][:, k - 1 : k + horizon - 1].mean(1) / high
        rows["patient"].append(np.flatnonzero(keep))
        rows["start"].append(np.full(keep.sum(), k))
        rows["resid"].append((avg(y) - avg(mean))[keep] / sd)
        rows["pred"].append(avg(mean)[keep] / sd)
        rows["obs"].append(avg(y)[keep] / sd)
        rows["dose"].append(dose[keep])
    return {key: np.concatenate(v) for key, v in rows.items()}


TIMING_SEGMENTS = 3
TIMING_Z = 2.394  # two-sided 95% with Bonferroni over the 3 segments


@torch.no_grad()
def effect_test_timed_data(model: SequenceModel, holdout: TrajectorySet, n_context: int, horizon: int,
                           segments: int = TIMING_SEGMENTS) -> dict[str, np.ndarray]:
    """Per-window, per-segment inputs of the timing-aware effect test (notes/timing_effect_test_plan.md).

    Same patients, start times and horizon as ``effect_test_data``; each window is cut into
    ``segments`` equal parts. For segment j: the mean normalised residual over observed steps in
    the segment, and each drug's mean dose (fraction of maximum) from the step before the window
    to the end of the segment.
    """
    model.eval()
    spec = holdout.spec
    device = next(model.parameters()).device
    o, sd = spec.outcome_index, float(spec.obs_std[spec.outcome_index])
    high = np.asarray(spec.action_max)
    steps = holdout["actions"].shape[1]
    last = steps - horizon + 1
    starts = [n_context + i * (last - n_context) // (EFFECT_STARTS - 1) for i in range(EFFECT_STARTS)]
    edges = np.linspace(0, horizon, segments + 1).round().astype(int)
    rows: dict[str, list] = {"patient": [], "segment": [], "resid": [], "dose": []}
    for k in starts:
        mean = _Predictor(model, holdout, k, device)(horizon)[0][..., o]
        y, seen = holdout["y"][:, k:k + horizon, o], holdout["mask"][:, k:k + horizon, o].astype(bool)
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            s = seen[:, a:b]
            n = s.sum(1)
            keep = n > 0
            resid = (np.where(s, y[:, a:b] - mean[:, a:b], 0.0).sum(1) / np.maximum(n, 1)) / sd
            dose = holdout["actions"][:, k - 1:k + b - 1].mean(1) / high
            rows["patient"].append(np.flatnonzero(keep))
            rows["segment"].append(np.full(keep.sum(), j))
            rows["resid"].append(resid[keep])
            rows["dose"].append(dose[keep])
    return {key: np.concatenate(v) for key, v in rows.items()}


def effect_test_timed(d: dict[str, np.ndarray], boot: int = 2000, seed: int = 0, z_flag: float = TIMING_Z) -> dict:
    """Per segment, regress the residual on the doses; patient-level bootstrap gives the
    standard errors; flag if any (segment, drug) |z| exceeds ``z_flag``."""
    ids = np.unique(d["patient"])
    segs = np.unique(d["segment"])
    idx = {(p, j): np.flatnonzero((d["patient"] == p) & (d["segment"] == j)) for p in ids for j in segs}
    coef = np.stack([_ols(d["resid"][d["segment"] == j], d["dose"][d["segment"] == j]) for j in segs])
    rng = np.random.default_rng(seed)
    draws = np.empty((boot,) + coef.shape)
    for b in range(boot):
        pick = rng.choice(ids, len(ids))
        for j in segs:
            r = np.concatenate([idx[(p, j)] for p in pick])
            draws[b, j] = _ols(d["resid"][r], d["dose"][r])
    z = np.abs(coef) / np.maximum(draws.std(0), 1e-12)  # (segments, drugs)
    return {"coef": coef, "z": z, "zmax": float(z.max()), "flag": bool((z > z_flag).any())}


def _ols(y: np.ndarray, dose: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), dose])
    return np.linalg.lstsq(design, y, rcond=None)[0][1:]


def effect_test(d: dict[str, np.ndarray], patients: np.ndarray | None = None, boot: int = 2000, seed: int = 0) -> dict:
    """Regress the residual on the doses; patient-level bootstrap; flag if any drug's
    95% interval excludes zero. ``patients`` restricts the test to a subset."""
    ids = np.unique(d["patient"]) if patients is None else np.asarray(patients)
    rows = {p: np.flatnonzero(d["patient"] == p) for p in ids}
    use = np.concatenate([rows[p] for p in ids])
    coef = _ols(d["resid"][use], d["dose"][use])
    rng = np.random.default_rng(seed)
    draws = np.empty((boot, len(coef)))
    for b in range(boot):
        pick = np.concatenate([rows[p] for p in rng.choice(ids, len(ids))])
        draws[b] = _ols(d["resid"][pick], d["dose"][pick])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    se = draws.std(0)
    slope_pred, slope_obs = _ols(d["pred"][use], d["dose"][use]), _ols(d["obs"][use], d["dose"][use])
    return {
        "coef": coef, "lo": lo, "hi": hi, "z": np.abs(coef) / np.maximum(se, 1e-12),
        "ratio": slope_pred / np.where(np.abs(slope_obs) > 1e-12, slope_obs, np.nan),
        "flag": bool(((lo > 0) | (hi < 0)).any()),
    }


GATES: Sequence[tuple[str, str, str, float]] = (
    ("benchmark", "skill_no_drug", "ge", 0.0),
    ("benchmark", "skill_dose_up", "ge", 0.0),
    ("benchmark", "skill_strong", "ge", 0.0),
    ("benchmark", "cfcov_worst", "ge", 0.8),
    ("benchmark", "regret_margin", "ge", 0.0),
    ("benchmark", "hazard_recall", "ge", 0.7),
    ("benchmark", "course_worst", "ge", 0.5),
    ("deployable", "direction_worst", "ge", 0.99),
    ("deployable", "rand_gap", "le", 1.25),
    ("deployable", "rand_cov", "ge", 0.8),
)


def derived(r: dict[str, float], actions: Sequence[str], prefix: str = "") -> dict[str, float]:
    """The single numbers the gates read, from one run's results."""
    get = lambda k: r.get(prefix + k, float("nan"))  # noqa: E731
    course = [get(f"{a}_course_{h}h") for a in actions for h in COURSE_HOURS]
    course = [min(c, 1 / c) if c > 0 else c for c in course if c == c]
    return {
        "skill_no_drug": get("skill_no_drug"),
        "skill_dose_up": get("skill_dose_up"),
        "skill_strong": get("skill_strong"),
        "cfcov_worst": min(get(f"cfcov_{p}") for p in ("no_drug", "dose_up", "strong")),
        "regret_margin": get("regret_fixed") - get("regret"),
        "hazard_recall": get("hazard_recall"),
        "course_worst": min(course) if course else float("nan"),
        "direction_worst": min(get(f"{a}_direction") for a in actions),
        "rand_gap": get("rand_rmse") / get("fact_rmse"),
        "rand_cov": get("rand_cov"),
    }
