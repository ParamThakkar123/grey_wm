"""What every benchmark simulator provides.

Time runs on a fixed grid of ``T`` steps of ``spec.dt`` hours. ``states[:, t]``
is the full simulator state at step ``t`` (``T + 1`` of them) and
``actions[:, t]`` the action held constant from step ``t`` to ``t + 1``.
Irregular observation is modelled on top of the grid by a mask, so the
simulator itself only ever steps on the grid.

A simulator is deterministic given its patient parameters, initial state,
pre-sampled process noise and exogenous inputs. That is what gives
counterfactual ground truth: re-running :meth:`Simulator.rollout` from a saved
state with the *same* noise and different actions answers "what would have
happened to this patient".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Union

import numpy as np

Policy = Callable[[int, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PKPrior:
    """Population PK/PD values for one action channel (published, for real drugs).

    The PD curve is ``sens * emax * C**hill / (ec50**hill + C**hill)``. With
    ``ka`` set, the dose first fills a depot that releases into the central
    compartment at rate ``ka`` (subcutaneous route); without it the dose goes
    straight into the central compartment (intravenous).
    """

    ke: float  # elimination rate, 1/h
    sens: float  # per-patient sensitivity multiplier (population value)
    emax: float  # maximal effect, in outcome units
    ec50: float  # concentration at half-maximal effect
    hill: float  # Hill coefficient, treated as known
    ka: Optional[float] = None  # absorption rate from a depot, 1/h
    input_scale: float = 1.0  # action units -> dose rate per hour
    # Known direction of the drug's direct effect on each observation:
    # +1 raises it, -1 lowers it, 0 no direct effect (only via the residual).
    signs: tuple[float, ...] = ()
    # Effect-compartment rate, 1/h: the PD curve reads C_e, a first-order lag of the
    # concentration, for a drug that acts later than it arrives. None: C_e = C.
    ke0: Optional[float] = None
    # Dose rate (per hour) the patient already receives at t = 0. The compartments
    # start at its steady state, and effects are relative to it.
    baseline_rate: float = 0.0


@dataclass(frozen=True)
class EnvSpec:
    name: str
    obs_names: tuple[str, ...]
    action_names: tuple[str, ...]
    action_max: tuple[float, ...]
    exog_dim: int
    dt: float  # hours per grid step
    horizon: int  # grid steps per trajectory
    obs_mean: tuple[float, ...]  # normalisation constants
    obs_std: tuple[float, ...]
    obs_noise: tuple[float, ...]  # measurement noise std, raw units
    obs_prob: tuple[float, ...]  # chance each variable is measured at a step
    outcome: str  # the observation treatment effects are measured on
    pk_prior: tuple[PKPrior, ...]  # one per action channel

    @property
    def obs_dim(self) -> int:
        return len(self.obs_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    @property
    def outcome_index(self) -> int:
        return self.obs_names.index(self.outcome)


class Simulator(ABC):
    spec: ClassVar[EnvSpec]
    param_names: ClassVar[tuple[str, ...]]
    state_names: ClassVar[tuple[str, ...]]
    noise_dim: ClassVar[int]
    #: Per action channel, the chance the behaviour policy acts at a step
    #: (1 = continuous, like an infusion; < 1 = occasional, like a bolus).
    action_pulse: ClassVar[tuple[float, ...]] = ()

    @abstractmethod
    def sample_patients(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, P)`` patient parameters, columns as :attr:`param_names`."""

    @abstractmethod
    def initial_state(self, params: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """``(n, S)`` states at admission, before any treatment."""

    @abstractmethod
    def step(
        self, x: np.ndarray, a: np.ndarray, params: np.ndarray, noise: np.ndarray, exog: np.ndarray
    ) -> np.ndarray:
        """Advance ``(n, S)`` states one grid step under actions ``(n, A)``."""

    @abstractmethod
    def observe_clean(self, states: np.ndarray) -> np.ndarray:
        """Noise-free observations ``(..., D)`` of states ``(..., S)``."""

    @abstractmethod
    def sickness(self, x: np.ndarray) -> np.ndarray:
        """``(n,)`` latent sickness in ``[0, 1]``: what a confounded clinician reacts to."""

    def sample_noise(self, n: int, steps: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, steps, self.noise_dim))

    def exog(self, start_hour: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, steps + 1, U)`` exogenous inputs; time of day by default."""
        del rng
        return self.time_of_day(start_hour, steps)

    def time_of_day(self, start_hour: np.ndarray, steps: int) -> np.ndarray:
        hours = start_hour[:, None] + self.spec.dt * np.arange(steps + 1)[None, :]
        phase = 2 * np.pi * hours / 24.0
        return np.stack([np.sin(phase), np.cos(phase)], axis=-1)

    def rollout(
        self,
        params: np.ndarray,
        x0: np.ndarray,
        actions: Union[np.ndarray, Policy],
        noise: np.ndarray,
        exog: np.ndarray,
        steps: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """States ``(n, steps + 1, S)`` and the actions ``(n, steps, A)`` taken.

        ``actions`` is either an open-loop array or a closed-loop policy
        ``policy(t, x_t) -> a_t``.
        """
        if steps is None:
            steps = int(actions.shape[1] if isinstance(actions, np.ndarray) else noise.shape[1])
        n = x0.shape[0]
        states = np.empty((n, steps + 1, x0.shape[1]))
        taken = np.empty((n, steps, self.spec.action_dim))
        states[:, 0] = x0
        high = np.asarray(self.spec.action_max)
        for t in range(steps):
            a = actions[:, t] if isinstance(actions, np.ndarray) else actions(t, states[:, t])
            a = np.clip(a, 0.0, high)
            taken[:, t] = a
            states[:, t + 1] = self.step(states[:, t], a, params, noise[:, t], exog[:, t])
        return states, taken
