"""Generate confounded datasets, save them, and serve them to models.

One file per ``(env, lambda)``: ``<root>/<env>/lam0.75.npz``. Patients, process
noise, measurement noise, observation masks and the train/val/test split come
from ``seed`` alone, so every lambda shares them and only the behaviour policy
differs between files: the confounding knob is the one thing that changes.

Arrays (``N`` trajectories, ``T`` steps):

==============  ====================  =========================================
``states``      ``(N, T+1, S)``       full simulator state, incl. latent ones
``clean``       ``(N, T+1, D)``       noise-free observations
``y``           ``(N, T+1, D)``       noisy observations, every step
``mask``        ``(N, T+1, D)``       which of ``y`` a model may see
``actions``     ``(N, T, A)``         behaviour actions, raw units
``exog``        ``(N, T+1, U)``       exogenous inputs
``noise``       ``(N, T, K)``         process noise, for counterfactual replays
``params``      ``(N, P)``            true patient parameters (theta)
``split``       ``(N,)``              0 train, 1 val, 2 test
==============  ====================  =========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from grey_wm.policies import BehaviorPolicy
from grey_wm.sims import EnvSpec, Simulator, get_sim

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLITS = ("train", "val", "test")
DT_SINCE_CAP = 24.0  # hours


def dataset_path(root: str | Path, env: str, lam: float) -> Path:
    return Path(root) / env / f"lam{lam:.2f}.npz"


def generate(env: str, lam: float, n: int, seed: int = 0, steps: Optional[int] = None) -> dict[str, np.ndarray]:
    sim = get_sim(env)
    spec = sim.spec
    steps = steps or spec.horizon
    rng = np.random.default_rng(seed)
    params = sim.sample_patients(n, rng)
    x0 = sim.initial_state(params, rng)
    noise = sim.sample_noise(n, steps, rng)
    exog = sim.exog(rng.uniform(0, 24, n), steps, rng)
    obs_noise = rng.standard_normal((n, steps + 1, spec.obs_dim)) * np.asarray(spec.obs_noise)
    mask = rng.random((n, steps + 1, spec.obs_dim)) < np.asarray(spec.obs_prob)
    mask[:, 0] = True  # admission: everything measured
    split = np.zeros(n, dtype=np.int8)
    order = rng.permutation(n)
    n_train, n_val = int(SPLIT_FRACTIONS[0] * n), int(SPLIT_FRACTIONS[1] * n)
    split[order[n_train : n_train + n_val]] = 1
    split[order[n_train + n_val :]] = 2

    policy_rng = np.random.default_rng([seed, int(round(lam * 1000))])
    policy = BehaviorPolicy(sim, lam, n, policy_rng)
    states, actions = sim.rollout(params, x0, policy, noise, exog, steps)
    clean = sim.observe_clean(states)
    return {
        "states": states.astype(np.float32),
        "clean": clean.astype(np.float32),
        "y": (clean + obs_noise).astype(np.float32),
        "mask": mask,
        "actions": actions.astype(np.float32),
        "exog": exog.astype(np.float32),
        "noise": noise.astype(np.float32),
        "params": params.astype(np.float32),
        "split": split,
        "lam": np.asarray(lam, dtype=np.float32),
    }


def save(arrays: dict[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def time_since_observed(mask: np.ndarray, dt: float) -> np.ndarray:
    """Hours since each variable was last measured (0 when measured now)."""
    out = np.zeros(mask.shape, dtype=np.float32)
    last = np.zeros(mask.shape[:1] + mask.shape[2:], dtype=np.float32)
    for t in range(mask.shape[1]):
        last = np.where(mask[:, t], 0.0, last + dt)
        out[:, t] = last
    return np.minimum(out, DT_SINCE_CAP)


@dataclass
class TrajectorySet:
    """One split of a dataset, plus what a model needs to read it."""

    spec: EnvSpec
    sim: Simulator
    arrays: dict[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.arrays["y"].shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def subset(self, idx: np.ndarray) -> "TrajectorySet":
        return TrajectorySet(self.spec, self.sim, {k: v[idx] for k, v in self.arrays.items()})

    def normalize(self, y: np.ndarray | torch.Tensor):
        return (y - np.asarray(self.spec.obs_mean, np.float32)) / np.asarray(self.spec.obs_std, np.float32)

    def batch(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """The whole set as model inputs (see :class:`TorchTrajectories`)."""
        return {k: torch.as_tensor(v, device=device) for k, v in model_inputs(self).items()}


def model_inputs(data: TrajectorySet) -> dict[str, np.ndarray]:
    """What a model sees: normalised observations zeroed where unmeasured."""
    mask = data["mask"].astype(np.float32)
    x = data.normalize(data["y"]).astype(np.float32) * mask
    return {
        "x": x,
        "mask": mask,
        "dt_since": time_since_observed(data["mask"], data.spec.dt),
        "actions": data["actions"],
        "exog": data["exog"],
    }


class TorchTrajectories(Dataset):
    def __init__(self, data: TrajectorySet) -> None:
        self.inputs = {k: torch.from_numpy(v) for k, v in model_inputs(data).items()}

    def __len__(self) -> int:
        return int(self.inputs["x"].shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.inputs.items()}


def load_split(root: str | Path, env: str, lam: float) -> dict[str, TrajectorySet]:
    path = dataset_path(root, env, lam)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/generate_data.py --env {env} --lams {lam}")
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files if k != "lam"}
    sim = get_sim(env)
    full = TrajectorySet(sim.spec, sim, arrays)
    return {name: full.subset(np.flatnonzero(arrays["split"] == i)) for i, name in enumerate(SPLITS)}
