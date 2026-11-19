"""Baseline (a): action-conditioned GRU. A pure black box."""

from __future__ import annotations

import torch
import torch.nn as nn

from grey_wm.models.base import SequenceModel, State, gaussian_head, mlp
from grey_wm.sims import EnvSpec


class GRUModel(SequenceModel):
    def __init__(self, spec: EnvSpec, latent_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__(spec)
        self.latent_dim = latent_dim
        self.obs_cell = nn.GRUCell(self.obs_input_dim, latent_dim)
        self.dyn_cell = nn.GRUCell(spec.action_dim + spec.exog_dim, latent_dim)
        self.decoder = mlp(latent_dim, hidden_dim, 2 * spec.obs_dim)

    def init_state(self, batch: int, device: torch.device) -> State:
        return {"z": torch.zeros(batch, self.latent_dim, device=device)}

    def observe(self, state, x, mask, dt_since, exog):
        return {"z": self.obs_cell(self.obs_input(x, mask, dt_since, exog), state["z"])}

    def transition(self, state, action, exog):
        return {"z": self.dyn_cell(torch.cat([action / self.action_max, exog], -1), state["z"])}

    def emit(self, state):
        return gaussian_head(self.decoder(state["z"]))
