"""Baseline (b): a ~3M-parameter causal decoder over (dt, obs, action) tokens.

Token ``t`` carries what was measured at ``t`` (zeros and mask 0 when nothing
was), the time since each variable was measured, the exogenous input and the
action ``a_t``; its output predicts ``x_{t+1}``. An open-loop rollout is the
same sequence with the observation slots masked after the context, so filtering
and rollout together are one causal pass (:meth:`parallel_rollout`), with no
step-by-step loop.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from grey_wm.models.base import RolloutOut, SequenceModel, State, gaussian_head
from grey_wm.sims import EnvSpec

MAX_TOKENS = 512


class TransformerModel(SequenceModel):
    def __init__(
        self, spec: EnvSpec, d_model: int = 256, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__(spec)
        token_dim = self.obs_input_dim + spec.action_dim
        self.embed = nn.Linear(token_dim, d_model)
        self.pos = nn.Embedding(MAX_TOKENS, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2 * spec.obs_dim)

    def parallel_rollout(
        self,
        batch: dict[str, torch.Tensor],
        n_context: int,
        horizon: int,
        future_actions: Optional[torch.Tensor] = None,
    ) -> RolloutOut:
        x, mask, dts, acts, exog = (batch[k] for k in ("x", "mask", "dt_since", "actions", "exog"))
        length = n_context + horizon - 1  # tokens 0 .. K+H-2 predict x_1 .. x_{K+H-1}
        actions = acts[:, :length]
        if future_actions is not None:
            actions = torch.cat([acts[:, : n_context - 1], future_actions[:, :horizon]], 1)
        observed = torch.arange(length, device=x.device) < n_context
        keep = observed[None, :, None].to(x.dtype)
        m = mask[:, :length] * keep
        # After the context nothing is measured: time-since keeps growing.
        steps_after = (torch.arange(length, device=x.device) - n_context + 1).clamp_min(0).to(x.dtype)
        last = dts[:, n_context - 1 : n_context]
        dt_roll = torch.where(observed[None, :, None], dts[:, :length], last + steps_after[None, :, None] * self.spec.dt)
        tokens = torch.cat([self.obs_input(x[:, :length], m, dt_roll, exog[:, :length]), actions / self.action_max], -1)
        pos = torch.arange(length, device=x.device)
        causal = nn.Transformer.generate_square_subsequent_mask(length, device=x.device)
        h = self.blocks(self.embed(tokens) + self.pos(pos)[None], mask=causal, is_causal=True)
        mean, logvar = gaussian_head(self.head(self.norm(h)))
        k = n_context - 1
        return RolloutOut(mean[:, :k], logvar[:, :k], mean[:, k:], logvar[:, k:], [])

    # The step-by-step interface is not used: filter_rollout takes the parallel path.
    def init_state(self, batch: int, device: torch.device) -> State:
        raise NotImplementedError("TransformerModel rolls out in parallel; use filter_rollout")

    def observe(self, state, x, mask, dt_since, exog):
        raise NotImplementedError

    def transition(self, state, action, exog):
        raise NotImplementedError

    def emit(self, state):
        raise NotImplementedError
