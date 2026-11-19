"""Baseline (d): CRN-style balanced representation (Bica et al., ICLR 2020).

The GRU baseline's representation is trained adversarially, through a gradient
reversal layer, to carry no information about the action about to be taken:
an adversary regresses the (scaled, continuous) dose from the representation,
and the reversed gradient pushes the representation to defeat it. This removes
the bias from time-varying confounders that the history reveals, which is the
causal baseline's job; it does not add pharmacological structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from grey_wm.models.base import SequenceModel, State, gaussian_head, mlp
from grey_wm.sims import EnvSpec


class _ReverseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


class CRNModel(SequenceModel):
    def __init__(
        self, spec: EnvSpec, latent_dim: int = 32, hidden_dim: int = 64, adv_weight: float = 1.0, grl_scale: float = 1.0
    ) -> None:
        super().__init__(spec)
        self.latent_dim = latent_dim
        self.adv_weight = adv_weight
        self.grl_scale = grl_scale
        self.obs_cell = nn.GRUCell(self.obs_input_dim, latent_dim)
        self.dyn_cell = nn.GRUCell(spec.action_dim + spec.exog_dim, latent_dim)
        self.decoder = mlp(latent_dim, hidden_dim, 2 * spec.obs_dim)
        self.adversary = mlp(latent_dim, hidden_dim, spec.action_dim)

    def init_state(self, batch: int, device: torch.device) -> State:
        z = torch.zeros(batch, self.latent_dim, device=device)
        return {"z": z, "adv": torch.zeros(batch, device=device)}

    def observe(self, state, x, mask, dt_since, exog):
        return {**state, "z": self.obs_cell(self.obs_input(x, mask, dt_since, exog), state["z"])}

    def transition(self, state, action, exog):
        z, scaled = state["z"], action / self.action_max
        guess = self.adversary(_ReverseGrad.apply(z, self.grl_scale))
        adv = ((guess - scaled) ** 2).mean(-1)
        return {"z": self.dyn_cell(torch.cat([scaled, exog], -1), z), "adv": adv}

    def emit(self, state):
        return gaussian_head(self.decoder(state["z"]))

    def penalties(self, states):
        return {"adversary": self.adv_weight * torch.stack([s["adv"] for s in states]).mean()}
