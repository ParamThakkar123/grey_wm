"""The interface every model implements, and the one filter-then-rollout loop.

A model is a filter over the time grid with three moves:

* ``observe(state, x_t, mask_t, dt_since_t, u_t)``: fold in what was measured at ``t``;
* ``transition(state, a_t, u_t)``: advance to ``t + 1`` under action ``a_t``;
* ``emit(state)``: Gaussian over the observations at the state's time.

:func:`filter_rollout` observes the first ``K`` steps (context), then keeps
transitioning without observing for ``H`` steps. Predictions during the context
are one-step-ahead; after it they are open-loop rollouts, under the factual
actions or under ``future_actions`` for a counterfactual.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from grey_wm.sims import EnvSpec

State = dict[str, torch.Tensor]
LOGVAR_MIN, LOGVAR_MAX = -8.0, 4.0


def mlp(inp: int, hidden: int, out: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    size = inp
    for _ in range(layers - 1):
        mods += [nn.Linear(size, hidden), nn.Tanh()]
        size = hidden
    mods.append(nn.Linear(size, out))
    return nn.Sequential(*mods)


class SequenceModel(nn.Module, ABC):
    def __init__(self, spec: EnvSpec) -> None:
        super().__init__()
        self.spec = spec
        self.register_buffer("action_max", torch.tensor(spec.action_max, dtype=torch.float32))
        self.sd_scale = 1.0  # conformal widening of the predictive sd, set by calibration

    @property
    def obs_input_dim(self) -> int:
        """Size of ``[x * mask, mask, dt_since, u]``."""
        return 3 * self.spec.obs_dim + self.spec.exog_dim

    def obs_input(self, x, mask, dt_since, exog) -> torch.Tensor:
        return torch.cat([x * mask, mask, dt_since / 6.0, exog], dim=-1)

    @abstractmethod
    def init_state(self, batch: int, device: torch.device) -> State: ...

    @abstractmethod
    def observe(self, state: State, x, mask, dt_since, exog) -> State: ...

    @abstractmethod
    def transition(self, state: State, action: torch.Tensor, exog: torch.Tensor) -> State: ...

    @abstractmethod
    def emit(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log-variance of the normalised observations."""

    def penalties(self, states: list[State]) -> dict[str, torch.Tensor]:
        """Regularisers over the states :func:`filter_rollout` visited."""
        return {}

    def warm_start(self, state: State, history: torch.Tensor) -> State:
        """Advance ``state`` over actions before the window; stateless by default."""
        return state


def gaussian_head(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, raw = out.chunk(2, dim=-1)
    span = LOGVAR_MAX - LOGVAR_MIN
    return mean, LOGVAR_MIN + span * torch.sigmoid(raw)


@dataclass
class RolloutOut:
    one_step_mean: torch.Tensor  # (B, K-1, D): predictions of x_1 .. x_{K-1}
    one_step_logvar: torch.Tensor
    mean: torch.Tensor  # (B, H, D): open-loop predictions of x_K .. x_{K+H-1}
    logvar: torch.Tensor
    states: list[State]  # every state after a transition


def filter_rollout(
    model: SequenceModel,
    batch: dict[str, torch.Tensor],
    n_context: int,
    horizon: int,
    future_actions: Optional[torch.Tensor] = None,
    history: Optional[torch.Tensor] = None,
) -> RolloutOut:
    """Observe steps ``0..K-1``, then roll out ``H`` steps open loop.

    ``future_actions`` ``(B, H, A)``, when given, replaces the actions
    ``a_{K-1} .. a_{K+H-2}`` that drive the rollout. ``history`` holds the actions
    before a training window, for models with drug still on board at its start.
    """
    x, mask, dts, acts, exog = (batch[k] for k in ("x", "mask", "dt_since", "actions", "exog"))
    if n_context < 1 or n_context + horizon - 1 > acts.shape[1]:
        raise ValueError(f"context {n_context} + horizon {horizon} exceeds {acts.shape[1]} steps")
    parallel = getattr(model, "parallel_rollout", None)
    if parallel is not None:  # models that see the whole sequence at once (transformer)
        return parallel(batch, n_context, horizon, future_actions)

    def action(t: int) -> torch.Tensor:
        if future_actions is not None and t >= n_context - 1:
            return future_actions[:, t - n_context + 1]
        return acts[:, t]

    state = model.init_state(x.shape[0], x.device)
    if history is not None and history.shape[1] > 0:
        state = model.warm_start(state, history)
    one_m, one_lv, roll_m, roll_lv, states = [], [], [], [], []
    for t in range(n_context):
        state = model.observe(state, x[:, t], mask[:, t], dts[:, t], exog[:, t])
        state = model.transition(state, action(t), exog[:, t])
        states.append(state)
        m, lv = model.emit(state)
        (one_m if t < n_context - 1 else roll_m).append(m)
        (one_lv if t < n_context - 1 else roll_lv).append(lv)
    for t in range(n_context, n_context + horizon - 1):
        state = model.transition(state, action(t), exog[:, t])
        states.append(state)
        m, lv = model.emit(state)
        roll_m.append(m)
        roll_lv.append(lv)

    def stack(items: list[torch.Tensor]) -> torch.Tensor:
        if not items:
            return x.new_zeros(x.shape[0], 0, x.shape[-1])
        return torch.stack(items, dim=1)

    return RolloutOut(stack(one_m), stack(one_lv), stack(roll_m), stack(roll_lv), states)


def masked_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    nll = 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)
