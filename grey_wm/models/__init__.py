"""Models, all behind :class:`~grey_wm.models.base.SequenceModel`.

Baselines (a) ``gru``, (b) ``transformer``, (c) ``gru-ode``, (d) ``crn``; the
grey-box model and its ablations (e) ``residual=False`` and (f)
``action_leak=True``.
"""

from __future__ import annotations

import inspect
from typing import Any

from grey_wm.models.base import SequenceModel, filter_rollout, masked_nll
from grey_wm.models.crn import CRNModel
from grey_wm.models.greybox import GreyBoxModel
from grey_wm.models.gru import GRUModel
from grey_wm.models.gru_ode import GRUODEModel
from grey_wm.models.transformer import TransformerModel
from grey_wm.sims import EnvSpec

MODELS: dict[str, type[SequenceModel]] = {
    "gru": GRUModel,
    "transformer": TransformerModel,
    "gru-ode": GRUODEModel,
    "crn": CRNModel,
    "greybox": GreyBoxModel,
}


def build_model(kind: str, spec: EnvSpec, **kwargs: Any) -> SequenceModel:
    """Build ``kind``, passing only the settings its constructor takes."""
    if kind not in MODELS:
        raise KeyError(f"Unknown model kind '{kind}'. Known: {', '.join(MODELS)}")
    cls = MODELS[kind]
    accepted = inspect.signature(cls.__init__).parameters
    return cls(spec, **{k: v for k, v in kwargs.items() if k in accepted})


__all__ = [
    "CRNModel", "GRUModel", "GRUODEModel", "GreyBoxModel", "MODELS", "SequenceModel",
    "TransformerModel", "build_model", "filter_rollout", "masked_nll",
]
