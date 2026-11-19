"""Every number in the paper, for any :class:`~grey_wm.models.base.SequenceModel`.

All evaluations share one protocol: observe ``n_context`` steps of a test
trajectory, then roll out open loop. Ground truth for a counterfactual comes
from replaying the simulator from the true state at step ``K - 1`` with the
trajectory's own process noise and the counterfactual actions.

* ``rmse_h{h}`` / ``map_rmse_h{h}``: factual rollout error ``h`` steps after the
  context (normalised, averaged over variables / outcome in raw units).
* ``coverage_h{h}``: share of noisy observations inside the 90% interval.
* ``cf_rmse_{policy}``: counterfactual rollout error per evaluation policy,
  over the whole horizon; ``cf_rmse`` averages them.
* ``{action}_sign_agree``, ``{action}_ite_rmse``, ``{action}_sensitivity``:
  the effect on the outcome of raising one action for a window, model vs. truth
  (sensitivity < 0 means the model learned the sign backwards).
* ``{action}_monotone_violation``: share of adjacent dose levels where the
  predicted outcome goes down as the dose goes up.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from grey_wm.data import TrajectorySet
from grey_wm.models.base import SequenceModel, filter_rollout
from grey_wm.models.greybox import GreyBoxModel
from grey_wm.policies import EVAL_POLICIES, eval_actions

Z90 = 1.6449


class _Predictor:
    """Rolls a model out on a fixed batch and returns raw-unit predictions."""

    def __init__(self, model: SequenceModel, data: TrajectorySet, n_context: int, device: torch.device) -> None:
        self.model = model
        self.data = data
        self.n_context = n_context
        self.device = device
        self.batch = data.batch(device)
        self.mean = np.asarray(data.spec.obs_mean, np.float32)
        self.std = np.asarray(data.spec.obs_std, np.float32)

    def __call__(self, horizon: int, future: Optional[np.ndarray] = None):
        fut = None if future is None else torch.as_tensor(future, dtype=torch.float32, device=self.device)
        out = filter_rollout(self.model, self.batch, self.n_context, horizon, fut)
        mean = out.mean.float().cpu().numpy() * self.std + self.mean
        scale = np.asarray(self.model.sd_scale, np.float32)
        if scale.ndim:  # one multiplier per rollout step; later steps reuse the last one
            scale = scale[np.minimum(np.arange(out.mean.shape[1]), len(scale) - 1)][None, :, None]
        sd = np.exp(0.5 * out.logvar.float().cpu().numpy()) * self.std * scale
        return mean, sd, out.states


def _truth(data: TrajectorySet, n_context: int, future: np.ndarray) -> np.ndarray:
    """Clean observations ``(n, H, D)`` the simulator produces under ``future``."""
    k = n_context - 1
    horizon = future.shape[1]
    states, _ = data.sim.rollout(
        data["params"].astype(np.float64),
        data["states"][:, k].astype(np.float64),
        future.astype(np.float64),
        data["noise"][:, k : k + horizon].astype(np.float64),
        data["exog"][:, k : k + horizon + 1].astype(np.float64),
    )
    return data.sim.observe_clean(states[:, 1:])


def _nrmse(pred: np.ndarray, truth: np.ndarray, std: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((pred - truth) / std) ** 2)))


@torch.no_grad()
def factual(
    model: SequenceModel, data: TrajectorySet, n_context: int, horizons: Sequence[int], prefix: str = ""
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    predict = _Predictor(model, data, n_context, device)
    horizon = min(max(horizons), data["actions"].shape[1] - n_context + 1)
    mean, sd, states = predict(horizon)
    window = slice(n_context, n_context + horizon)
    clean, y = data["clean"][:, window], data["y"][:, window]
    out_i = data.spec.outcome_index
    out: dict[str, float] = {}
    for h in horizons:
        if h > horizon:
            continue
        i = h - 1
        out[f"{prefix}rmse_h{h}"] = _nrmse(mean[:, i], clean[:, i], predict.std)
        out[f"{prefix}map_rmse_h{h}"] = float(np.sqrt(np.mean((mean[:, i, out_i] - clean[:, i, out_i]) ** 2)))
        inside = np.abs(y[:, i] - mean[:, i]) <= Z90 * sd[:, i]
        out[f"{prefix}coverage_h{h}"] = float(inside.mean())
    if isinstance(model, GreyBoxModel) and model.residual:
        out[f"{prefix}residual_norm"] = float(GreyBoxModel.residual_norm(states))
    return out


@torch.no_grad()
def counterfactual(
    model: SequenceModel,
    data: TrajectorySet,
    n_context: int,
    horizon: int = 20,
    effect_window: int = 8,
    effect_delta: float = 0.2,
    dose_levels: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    spec = data.spec
    predict = _Predictor(model, data, n_context, device)
    rng = np.random.default_rng(seed)
    k = n_context - 1
    factual_actions = data["actions"][:, k : k + max(horizon, effect_window)]
    out: dict[str, float] = {}

    errors = []
    for name in EVAL_POLICIES:
        future = eval_actions(name, factual_actions[:, :horizon], spec.action_max, rng, data.sim.action_pulse)
        mean, _, _ = predict(horizon, future)
        out[f"cf_rmse_{name}"] = _nrmse(mean, _truth(data, n_context, future), predict.std)
        errors.append(out[f"cf_rmse_{name}"])
    out["cf_rmse"] = float(np.mean(errors))

    o = spec.outcome_index
    high = np.asarray(spec.action_max)
    base = factual_actions[:, :effect_window]
    base_model = predict(effect_window, base)[0][:, -1, o]
    base_true = _truth(data, n_context, base)[:, -1, o]
    for c, action in enumerate(spec.action_names):
        raised = base.copy()
        raised[..., c] = np.minimum(raised[..., c] + effect_delta * high[c], high[c])
        ite_model = predict(effect_window, raised)[0][:, -1, o] - base_model
        ite_true = _truth(data, n_context, raised)[:, -1, o] - base_true
        out[f"{action}_sign_agree"] = float(np.mean(np.sign(ite_model) == np.sign(ite_true)))
        out[f"{action}_ite_rmse"] = float(np.sqrt(np.mean((ite_model - ite_true) ** 2)))
        true_mean = float(np.mean(ite_true))
        safe = true_mean if abs(true_mean) > 1e-6 else 1e-6
        out[f"{action}_sensitivity"] = float(np.mean(ite_model) / safe)  # 1 = right size, < 0 = backwards

        finals, true_finals = [], []
        for level in np.linspace(0.0, 1.0, dose_levels):
            dosed = base.copy()
            dosed[..., c] = level * high[c]
            finals.append(predict(effect_window, dosed)[0][:, -1, o])
            true_finals.append(_truth(data, n_context, dosed)[:, -1, o])
        steps = np.diff(np.stack(finals, axis=1), axis=1)
        true_steps = np.diff(np.stack(true_finals, axis=1), axis=1)
        # A violation is a dose step where the model moves the outcome against the
        # simulator's direction (up for norepinephrine on MAP, down for insulin).
        clear = np.abs(true_steps) > 1e-2
        wrong = np.sign(steps) != np.sign(true_steps)
        out[f"{action}_monotone_violation"] = float(wrong[clear].mean()) if clear.any() else 0.0
    return out
