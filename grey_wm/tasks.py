"""Every model as a Simulusion ``TrainingTask``.

Importing this module registers the tasks; the Simulusion ``Trainer`` then owns
seeding, the device, checkpoints and resume, eval cadence, early stopping and
the run directory. ``scripts/sweep.py`` drives the grid.

Training samples, per batch, a context length ``K`` and a rollout horizon ``H``
(its cap annealed from 1 to ``optim.horizon_max``), and minimises the masked
one-step NLL over the context, the masked NLL of the open-loop rollout, and the
model's own penalties (residual L1, theta prior).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from simulusion.training.task import TrainingTask, register_task

from grey_wm import audit, calibration, metrics
from grey_wm.data import TorchTrajectories, load_split, model_inputs
from grey_wm.models import build_model, filter_rollout, masked_nll


PK_WARMUP = 96  # steps of pre-window actions replayed through the PK compartments


@dataclass
class ModelConfig:
    kind: str = "gru"
    latent_dim: int = 32
    hidden_dim: int = 64
    effect_mode: str = "additive"  # greybox: "additive" (sign-constrained) or "latent" (ablation)
    residual: bool = True
    residual_mode: str = "effect"  # greybox: "effect" (monotone) or "action" (raw-action g, ablation)
    learn_effect_scale: bool = False
    action_leak: bool = False
    dynamics: str = "euler"  # greybox: latent update, "euler" (f_phi step), "gru" (recurrent cell) or "frozen"
    delay: str = "prior"  # greybox, structural misspecification: "prior", "none" or "add"
    depot: bool = True  # greybox, structural misspecification: False drops the subcutaneous depot
    elim: str = "fixed"  # greybox: "state" lets elimination rates depend on the latent state
    elim_only: int = -1  # greybox: with elim="state", only this drug (-1: all)
    pd: str = "emax"  # greybox, structural misspecification: "linear" dose-response
    l1: float = 0.05
    prior_weight: float = 0.1
    prior_sigma: float = 0.3  # log-normal sd: typical between-patient PK/PD variability
    ke_scale: float = 1.0
    ec50_scale: float = 1.0
    emax_scale: float = 1.0
    hill_scale: float = 1.0
    # transformer
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    # crn
    adv_weight: float = 1.0


@dataclass
class DataConfig:
    env: str = "sepsis"
    confounding: float = 0.75
    root: str = "data"
    batch_size: int = 64
    calib_n: int = 100  # randomised trajectories available (first n of the 100); size study only


@dataclass
class OptimConfig:
    lr: float = 1e-3
    max_grad_norm: float = 10.0
    # Training windows start anywhere in a trajectory, with PK state and latent
    # reset to zero; context_min steps of burn-in drain the reset (fluid PK
    # memory is ~0.86 per step) and keep each step's loop short.
    context_min: int = 24
    context_max: int = 48
    horizon_max: int = 20
    anneal_steps: int = 2000
    rollout_weight: float = 1.0
    ipw: bool = False  # inverse-probability-weighted loss (a simplified RMSN; see stabilised_log_weights)
    ipw_oracle: bool = False  # control: the propensity model also sees the true hidden severity
    ipw_clip: float = 20.0  # cap on a window's stabilised weight (and 1 / cap as floor)
    # "full": the weight multiplies over every dose in the window (context and rollout);
    # "horizon": only over the doses in the rollout, as in RMSN's decoder, with each step's
    # log weight truncated at its 1st/99th percentile on the training set.
    ipw_window: str = "full"


@dataclass
class EvalConfig:
    n_context: int = 24  # 6 h on the sepsis grid
    horizons: tuple[int, ...] = (1, 5, 20, 50)
    val_trajectories: int = 128
    test_trajectories: int = 300
    cf_horizon: int = 20
    effect_window: int = 8  # steps the dose is raised for the treatment-effect metrics
    select_horizon: int = 20  # rollout steps of the randomised error used to pick weights


@dataclass
class SequenceTaskConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _gaussian_logpdf(a: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """Log density of ``a`` under a least-squares linear-Gaussian model on ``feats``."""
    coef, *_ = np.linalg.lstsq(feats, a, rcond=None)
    resid = a - feats @ coef
    var = max(float(resid.var()), 1e-8)
    return -0.5 * (resid**2 / var + np.log(2 * np.pi * var))


def stabilised_log_weights(data, oracle: bool = False, trunc_pct: float = 0.0) -> np.ndarray:
    """Per-step log stabilised weight ``log p(a_t | a_{t-1}) - log p(a_t | H_t)``, shape ``(N, T)``.

    As in RMSN, the denominator models each dose given the recorded history (last
    measured value of every observed variable, previous doses, exogenous inputs),
    the numerator given previous doses only. Both are linear-Gaussian, fitted on
    the training set; drugs are treated as independent. The variable that drives
    dosing is hidden, so the weights can only remove confounding by recorded variables.
    """
    inp = model_inputs(data)
    x, mask, acts, exog = inp["x"], inp["mask"], inp["actions"].astype(np.float64), inp["exog"]
    n, t, _ = acts.shape
    last = np.zeros_like(x)
    for i in range(x.shape[1]):  # forward-fill the last measured value
        last[:, i] = np.where(mask[:, i] > 0, x[:, i], last[:, i - 1] if i else 0.0)
    prev = np.concatenate([np.zeros_like(acts[:, :1]), acts[:, :-1]], 1)
    one = np.ones((n, t, 1))
    parts = [one, last[:, :t], prev, exog[:, :t]]
    if oracle:  # control only: the confounder no model sees
        sev = np.stack([data.sim.sickness(data["states"][:, i].astype(np.float64)) for i in range(t)], 1)
        parts.append(sev[..., None])
    hist = np.concatenate(parts, -1).reshape(n * t, -1)
    marg = np.concatenate([one, prev], -1).reshape(n * t, -1)
    logw = np.zeros(n * t)
    for c in range(acts.shape[-1]):
        a = acts[..., c].reshape(-1)
        logw += _gaussian_logpdf(a, marg) - _gaussian_logpdf(a, hist)
    if trunc_pct > 0:  # per-step truncation (percentiles of the training set)
        lo, hi = np.percentile(logw, [trunc_pct, 100 - trunc_pct])
        logw = np.clip(logw, lo, hi)
    return logw.reshape(n, t).astype(np.float32)


class SequenceModelTask(TrainingTask):
    config_cls = SequenceTaskConfig
    default_max_steps = 5000

    def setup(self, trainer) -> None:
        c = self.config
        self.splits = load_split(c.data.root, c.data.env, c.data.confounding)
        spec = self.splits["train"].spec
        kwargs = {k: v for k, v in asdict(c.model).items() if k != "kind"}
        self.model = trainer.prepare_module(build_model(c.model.kind, spec, **kwargs))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=c.optim.lr)
        self.update = trainer.gradient_update(
            self.optimizer, self.model.parameters(), max_grad_norm=c.optim.max_grad_norm
        )
        dataset = TorchTrajectories(self.splits["train"])
        if c.optim.ipw:
            trunc = 1.0 if c.optim.ipw_window == "horizon" else 0.0
            dataset.inputs["logw"] = torch.from_numpy(
                stabilised_log_weights(self.splits["train"], c.optim.ipw_oracle, trunc_pct=trunc))
        loader = DataLoader(dataset, batch_size=c.data.batch_size, shuffle=True, drop_last=True)
        self.batches = trainer.infinite(trainer.prepare_dataloader(loader))
        self.steps = int(self.splits["train"]["actions"].shape[1])
        # The small randomised set: never trained on, picks the weights, calibrates them.
        self.calib, self.holdout = calibration.randomised_sets(c.data.root, c.data.env)
        self.calib = self.calib.subset(slice(0, min(c.data.calib_n, len(self.calib))))
        self.best: dict[str, Any] | None = None

    def _sample_window(self) -> tuple[int, int, int]:
        """Start offset, context length and rollout horizon for one batch."""
        o = self.config.optim
        frac = min(1.0, self.trainer.iteration / max(o.anneal_steps, 1))
        h_cap = max(1, round(1 + frac * (o.horizon_max - 1)))
        horizon = random.randint(1, h_cap)
        context = random.randint(min(o.context_min, self.steps - horizon + 1), min(o.context_max, self.steps - horizon + 1))
        start = random.randint(0, self.steps - (context + horizon - 1))
        return start, context, horizon

    def train_step(self) -> dict[str, float]:
        self.model.train()
        start, context, horizon = self._sample_window()
        stop = start + context + horizon
        full = next(self.batches)
        history = full["actions"][:, max(0, start - PK_WARMUP) : start]
        logw = full.pop("logw", None)
        batch = {k: v[:, start : stop - 1 if k == "actions" else stop] for k, v in full.items()}
        with self.trainer.precision.autocast():
            out = filter_rollout(self.model, batch, context, horizon, history=history)
            x, mask = batch["x"], batch["mask"]
            if logw is not None:
                # One stabilised weight per window: the product over the doses it contains,
                # capped, normalised to mean 1 over the batch, applied to every target.
                clip = self.config.optim.ipw_clip
                first = start + context - 1 if self.config.optim.ipw_window == "horizon" else start
                w = torch.exp(logw[:, first : stop - 1].sum(1)).clamp(1 / clip, clip)
                w = (w / w.mean())[:, None, None].to(x.dtype)
                mask = mask * w
            one = masked_nll(out.one_step_mean, out.one_step_logvar, x[:, 1:context], mask[:, 1:context])
            win = slice(context, context + horizon)
            roll = masked_nll(out.mean, out.logvar, x[:, win], mask[:, win])
            pens = self.model.penalties(out.states)
            loss = one + self.config.optim.rollout_weight * roll + sum(pens.values())
        self.update(loss)
        return {"loss": loss.item(), "nll_one_step": one.item(), "nll_rollout": roll.item(),
                **{k: v.item() for k, v in pens.items()}, "horizon": float(horizon)}

    def _subset(self, split: str, n: int):
        data = self.splits[split]
        return data.subset(slice(0, min(n, len(data))))

    def evaluate(self) -> dict[str, float]:
        e = self.config.eval
        data = self._subset("val", e.val_trajectories)
        horizons = tuple(h for h in e.horizons if h <= 20)
        out = metrics.factual(self.model, data, e.n_context, horizons, prefix="val_")
        # Early stopping and the final weights follow error on randomised dosing, not
        # factual error on the confounded policy, which rewards the confounded fit.
        out["calib_rmse"] = metrics.factual(self.model, self.calib, e.n_context, (e.select_horizon,))[
            f"rmse_h{e.select_horizon}"
        ]
        if self.best is None or out["calib_rmse"] < self.best["score"]:
            weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.best = {"score": out["calib_rmse"], "weights": weights, "step": self.trainer.iteration}
        return out

    def final_evaluation(self) -> dict[str, float]:
        e = self.config.eval
        data = self._subset("test", e.test_trajectories)
        out: dict[str, Any] = {}
        if self.best is not None:
            self.model.load_state_dict(self.best["weights"])
            out["selected_step"] = self.best["step"]

        def score(prefix: str) -> dict[str, float]:
            return {
                **metrics.factual(self.model, data, e.n_context, e.horizons, prefix=prefix),
                **{f"{prefix}{k}": v for k, v in metrics.counterfactual(
                    self.model, data, e.n_context, horizon=e.cf_horizon, effect_window=e.effect_window
                ).items()},
                **{f"{prefix}{k}": v for k, v in audit.run(
                    self.model, data, self.holdout, e.n_context, e.cf_horizon, e.effect_window
                ).items()},
            }

        out.update(score(""))
        out.update(calibration.calibrate(self.model, self.calib, e.n_context, e.cf_horizon))
        out.update(score("cal_"))
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "best": self.best}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.best = state.get("best")

    def export(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        (directory / "model.json").write_text(json.dumps(asdict(self.config), indent=2))
        return directory


def _variant(optim: dict[str, Any] | None = None, **model: Any):
    @dataclass
    class Config(SequenceTaskConfig):
        model: ModelConfig = field(default_factory=lambda: ModelConfig(**model))
        optim: OptimConfig = field(default_factory=lambda: OptimConfig(**(optim or {})))

    return Config


@register_task("gru")
class GRUTask(SequenceModelTask):
    config_cls = _variant(kind="gru")


@register_task("gru-ipw")
class GRUIPWTask(SequenceModelTask):
    """GRU with an inverse-probability-weighted loss (simplified RMSN)."""

    config_cls = _variant(kind="gru", optim={"ipw": True})


@register_task("gru-ipw-oracle")
class GRUIPWOracleTask(SequenceModelTask):
    """Control: IPW-weighted GRU whose propensity model sees the hidden severity."""

    config_cls = _variant(kind="gru", optim={"ipw": True, "ipw_oracle": True})


@register_task("gru-ipw-h")
class GRUIPWHorizonTask(SequenceModelTask):
    """IPW over the rollout doses only, per-step weights truncated at 1st/99th percentiles."""

    config_cls = _variant(kind="gru", optim={"ipw": True, "ipw_window": "horizon"})


@register_task("gru-ipw-h-oracle")
class GRUIPWHorizonOracleTask(SequenceModelTask):
    """Control: gru-ipw-h whose propensity model sees the hidden severity."""

    config_cls = _variant(kind="gru", optim={"ipw": True, "ipw_window": "horizon", "ipw_oracle": True})


@register_task("transformer")
class TransformerTask(SequenceModelTask):
    config_cls = _variant(kind="transformer", optim={"lr": 3e-4})


@register_task("gru-ode")
class GRUODETask(SequenceModelTask):
    config_cls = _variant(kind="gru-ode")


@register_task("crn")
class CRNTask(SequenceModelTask):
    config_cls = _variant(kind="crn")


@register_task("greybox")
class GreyBoxTask(SequenceModelTask):
    config_cls = _variant(kind="greybox", residual=False)


@register_task("greybox-res")
class GreyBoxResidualTask(SequenceModelTask):
    """The monotone, add-only effect residual on (ablation)."""

    config_cls = _variant(kind="greybox")


@register_task("greybox-latent")
class GreyBoxLatentTask(SequenceModelTask):
    """PK/PD effect fed into the free latent ODE: no sign constraint (ablation)."""

    config_cls = _variant(kind="greybox", effect_mode="latent", residual=False)


@register_task("greybox-actres")
class GreyBoxActionResidualTask(SequenceModelTask):
    """Unconstrained raw-action residual g(z, a) in the latent dynamics (ablation)."""

    config_cls = _variant(kind="greybox", residual_mode="action")


@register_task("greybox-leak")
class GreyBoxLeakTask(SequenceModelTask):
    config_cls = _variant(kind="greybox", action_leak=True, residual=False)


MODEL_TASKS = (
    "greybox", "greybox-res", "greybox-latent", "greybox-actres", "greybox-leak", "gru", "gru-ipw", "gru-ipw-oracle", "gru-ipw-h", "gru-ipw-h-oracle", "gru-ode", "crn", "transformer",
)


def nested(flat: dict[str, Any]) -> dict[str, Any]:
    """``{"trainer.seed": 0}`` -> ``{"trainer": {"seed": 0}}``, the form ``run()`` merges."""
    out: dict[str, Any] = {}
    for key, value in flat.items():
        *parents, leaf = key.split(".")
        node = out
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return out
