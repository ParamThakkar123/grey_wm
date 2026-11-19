"""Baseline (c): GRU-ODE with the action concatenated into the vector field.

Between observations the latent flows as ``dz/dt = f(z, a, u)`` (Euler
substeps); at an observation a GRU cell jumps it, as in GRU-ODE-Bayes. The
continuous-time structure is the same as the grey-box model's; the difference
is that the raw action enters the free vector field.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from grey_wm.models.base import SequenceModel, State, gaussian_head, mlp
from grey_wm.sims import EnvSpec


class GRUODEModel(SequenceModel):
    def __init__(self, spec: EnvSpec, latent_dim: int = 32, hidden_dim: int = 64, substeps: int = 2) -> None:
        super().__init__(spec)
        self.latent_dim = latent_dim
        self.substeps = substeps
        self.obs_cell = nn.GRUCell(self.obs_input_dim, latent_dim)
        self.f = mlp(latent_dim + spec.action_dim + spec.exog_dim, hidden_dim, latent_dim)
        self.decoder = mlp(latent_dim, hidden_dim, 2 * spec.obs_dim)

    def init_state(self, batch: int, device: torch.device) -> State:
        return {"z": torch.zeros(batch, self.latent_dim, device=device)}

    def observe(self, state, x, mask, dt_since, exog):
        return {"z": self.obs_cell(self.obs_input(x, mask, dt_since, exog), state["z"])}

    def transition(self, state, action, exog):
        z = state["z"]
        h = self.spec.dt / self.substeps
        inputs = torch.cat([action / self.action_max, exog], -1)
        for _ in range(self.substeps):
            z = z + h * self.f(torch.cat([z, inputs], -1))
        return {"z": z}

    def emit(self, state):
        return gaussian_head(self.decoder(state["z"]))
