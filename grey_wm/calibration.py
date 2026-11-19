"""Calibration on a small randomised set, the stand-in for a pilot trial.

Confounded data cannot separate a drug's effect from the severity that drove its
dose, so something outside them has to set the size of an effect. Here that is
100 trajectories (5% of a dataset) dosed at random, taken from the validation
patients of the ``lambda = 0`` file; 72 more are kept back for the audit. Neither
set is trained on, and both use only what a real trial would give: observed
outcomes under randomised doses, never simulator ground truth.

Three things are fitted, after training, in this order:

* ``effect_scale``: one positive multiplier per drug. It can raise or lower an
  effect and cannot reverse it. Grey-box models only.
* ``effect_sigma``: log-sd of that multiplier across patients, which widens
  intervals in proportion to the size of the predicted effect. Grey-box only.
* ``sd_scale``: a conformal multiplier on every predictive sd. Every model gets
  this one, so all models' intervals are compared on equal terms.
"""

from __future__ import annotations

import numpy as np
import torch

from grey_wm.data import TrajectorySet, load_split
from grey_wm.metrics import _Predictor
from grey_wm.models import filter_rollout, masked_nll
from grey_wm.models.base import SequenceModel
from grey_wm.models.greybox import EC50, GreyBoxModel

CALIB_FROM, CALIB_TO = 128 / 300, 228 / 300  # of the 300 validation patients: after the 128 used for val_*
SCALE_STEPS = 40
SCALE_LR = 0.05
SCALE_PRIOR_SD = 0.5  # log-normal prior on the multiplier: a factor of about 1.6
STARTS = 4
SIGMA_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
COVERAGE = 0.90


def randomised_sets(root: str, env: str) -> tuple[TrajectorySet, TrajectorySet]:
    val = load_split(root, env, 0.0)["val"]
    a, b = round(CALIB_FROM * len(val)), round(CALIB_TO * len(val))
    return val.subset(slice(a, b)), val.subset(slice(b, len(val)))


SHIFTS = ("first50", "mild50", "sick50")


def shifted_subset(calib: TrajectorySet, how: str) -> TrajectorySet:
    """Post hoc (2026-11-16): half of the randomised calibration set, chosen by hidden severity.

    ``first50``: the first 50 (control for size); ``mild50`` / ``sick50``: the 50 patients with the
    lowest / highest mean hidden sickness over their trajectory (a trial population unlike the clinic's).
    """
    import numpy as np

    n = len(calib) // 2
    if how == "first50":
        return calib.subset(slice(0, n))
    states = calib["states"].astype(np.float64)
    sick = np.stack([calib.sim.sickness(states[:, t]) for t in range(states.shape[1])], 1).mean(1)
    order = np.argsort(sick, kind="stable")
    if how == "mild50":
        return calib.subset(np.sort(order[:n]))
    if how == "sick50":
        return calib.subset(np.sort(order[-n:]))
    raise ValueError(how)


def _rollout_nll(model: SequenceModel, batch: dict, n_context: int, horizon: int) -> torch.Tensor:
    """Mean rollout NLL from several start times: one start says little about a small effect."""
    steps = batch["actions"].shape[1]
    last = steps - horizon + 1
    starts = [n_context + i * (last - n_context) // (STARTS - 1) for i in range(STARTS)]
    losses = []
    for k in starts:
        out = filter_rollout(model, batch, k, horizon)
        win = slice(k, k + horizon)
        losses.append(masked_nll(out.mean, out.logvar, batch["x"][:, win], batch["mask"][:, win]))
    return torch.stack(losses).mean()


def _fit_effect_scale(model: GreyBoxModel, batch: dict, n_context: int, horizon: int,
                      prior_sd: float = SCALE_PRIOR_SD, fit_ec50: bool = False) -> None:
    """``fit_ec50`` (post hoc variant) also fits one EC50 multiplier per drug, with the same prior."""
    log_k = torch.zeros_like(model.effect_scale, requires_grad=True)
    log_e = torch.zeros_like(model.effect_scale, requires_grad=True)
    pop = model.theta_pop.clone()
    opt = torch.optim.Adam([log_k, log_e] if fit_ec50 else [log_k], lr=SCALE_LR)
    for p in model.parameters():
        p.requires_grad_(False)
    for _ in range(SCALE_STEPS):
        opt.zero_grad()
        model.effect_scale = torch.exp(log_k)
        if fit_ec50:
            model.theta_pop = torch.cat([pop[:, :EC50], pop[:, EC50:] * torch.exp(log_e)[:, None]], 1)
        observed = float(batch["mask"].sum()) * horizon / batch["mask"].shape[1] * STARTS
        penalty = (log_k**2).sum() + (log_e**2).sum()
        loss = _rollout_nll(model, batch, n_context, horizon) + penalty / (2 * prior_sd**2 * observed)
        loss.backward()
        opt.step()
    model.effect_scale = torch.exp(log_k).detach()
    if fit_ec50:
        model.theta_pop = torch.cat([pop[:, :EC50], pop[:, EC50:] * torch.exp(log_e.detach())[:, None]], 1)


@torch.no_grad()
def _fit_effect_sigma(model: GreyBoxModel, batch: dict, n_context: int, horizon: int) -> None:
    for c in range(model.effect_sigma.shape[0]):
        scores = {}
        for s in SIGMA_GRID:
            model.effect_sigma[c] = s
            scores[s] = float(_rollout_nll(model, batch, n_context, horizon))
        model.effect_sigma[c] = min(scores, key=scores.get)


def _quantile(scores: np.ndarray) -> float:
    level = min(1.0, COVERAGE * (len(scores) + 1) / len(scores))
    return float(np.quantile(scores, level) / 1.6449)


@torch.no_grad()
def _conformal_scale(model: SequenceModel, data: TrajectorySet, n_context: int, horizon: int,
                     per_step: bool = False):
    """One multiplier, or (post hoc variant) one per rollout step: errors grow along a rollout."""
    model.sd_scale = 1.0
    predict = _Predictor(model, data, n_context, next(model.parameters()).device)
    mean, sd, _ = predict(horizon)
    win = slice(n_context, n_context + horizon)
    mask = data["mask"][:, win].astype(bool)
    z = np.abs(data["y"][:, win] - mean) / sd
    if not per_step:
        return _quantile(z[mask])
    return np.array([_quantile(z[:, h][mask[:, h]]) for h in range(z.shape[1])], np.float32)


def calibrate(model: SequenceModel, data: TrajectorySet, n_context: int, horizon: int,
              prior_sd: float = SCALE_PRIOR_SD, fit_ec50: bool = False, per_step: bool = False) -> dict[str, float]:
    """Fit in place; returns what was fitted, for the results file. The keyword options are the
    post hoc variants of ``scripts/recalibrate.py``; their defaults are the paper's calibration."""
    model.eval()
    device = next(model.parameters()).device
    batch = data.batch(device)
    out: dict[str, float] = {}
    if isinstance(model, GreyBoxModel) and model.additive:
        with torch.enable_grad():
            _fit_effect_scale(model, batch, n_context, horizon, prior_sd, fit_ec50)
        _fit_effect_sigma(model, batch, n_context, horizon)
        for c, action in enumerate(model.spec.action_names):
            out[f"{action}_effect_scale"] = float(model.effect_scale[c])
            out[f"{action}_effect_sigma"] = float(model.effect_sigma[c])
            if fit_ec50:
                out[f"{action}_ec50"] = float(model.theta_pop[c, EC50])
    model.sd_scale = _conformal_scale(model, data, n_context, horizon, per_step)
    out["sd_scale"] = float(np.mean(model.sd_scale))
    return out
