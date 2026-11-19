"""The confounded behaviour policy and the open-loop evaluation policies.

Behaviour (generates the training data), per action channel::

    a_t = clip(lam * dose(sickness_t) + (1 - lam) * random_t + noise, 0, a_max)

``dose`` reads the simulator's *latent* sickness, which the model never sees:
``lam = 0`` is a randomised trial, ``lam = 1`` a clinician dosing purely by
indication. ``random_t`` is piecewise constant, so the data contains sustained
dose changes and not only white noise.

Evaluation policies are open loop: they replace the actions after the context
window, and the same sequence goes to the model and to the simulator.
"""

from __future__ import annotations

import numpy as np

from grey_wm.sims.base import Simulator

EVAL_POLICIES = ("random", "aggressive", "conservative", "dose_down", "dose_up", "flip")


class BehaviorPolicy:
    def __init__(
        self,
        sim: Simulator,
        lam: float,
        n: int,
        rng: np.random.Generator,
        switch_prob: float = 0.1,
        noise: float = 0.05,
    ) -> None:
        self.sim = sim
        self.lam = lam
        self.rng = rng
        self.high = np.asarray(sim.spec.action_max)
        self.switch_prob = switch_prob
        self.noise = noise
        self.level = rng.uniform(0, 1, (n, len(self.high))) * self.high

    def __call__(self, t: int, x: np.ndarray) -> np.ndarray:
        switch = self.rng.random(self.level.shape) < self.switch_prob
        fresh = self.rng.uniform(0, 1, self.level.shape) * self.high
        self.level = np.where(switch, fresh, self.level)
        indicated = np.clip((self.sim.sickness(x) - 0.2) / 0.6, 0.0, 1.0)[:, None] * self.high
        a = self.lam * indicated + (1 - self.lam) * self.level
        a = a + self.noise * self.high * self.rng.standard_normal(a.shape)
        if self.sim.action_pulse:  # occasional channels (boluses) act only on some steps
            a = a * (self.rng.random(a.shape) < np.asarray(self.sim.action_pulse))
        return np.clip(a, 0.0, self.high)


def eval_actions(
    name: str,
    factual: np.ndarray,
    action_max: tuple[float, ...],
    rng: np.random.Generator,
    pulse: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Open-loop actions ``(n, H, A)`` for evaluation policy ``name``.

    ``factual`` holds the behaviour policy's actions over the same window, for
    the policies defined relative to it.
    """
    high = np.asarray(action_max)
    n, horizon, dim = factual.shape
    if name == "random":
        blocks = max(1, horizon // 8)
        levels = rng.uniform(0, 1, (n, blocks, dim)) * high
        out = np.repeat(levels, int(np.ceil(horizon / blocks)), axis=1)[:, :horizon]
    elif name == "aggressive":
        out = np.broadcast_to(0.8 * high, factual.shape) + 0.05 * high * rng.standard_normal(factual.shape)
    elif name == "conservative":
        out = np.broadcast_to(0.1 * high, factual.shape) + 0.02 * high * rng.standard_normal(factual.shape)
    elif name == "dose_down":
        out = 0.7 * factual
    elif name == "dose_up":
        out = 1.3 * factual
    elif name == "flip":
        out = factual.copy()
        t = rng.integers(0, horizon, n)
        out[np.arange(n), t] = high - factual[np.arange(n), t]
    else:
        raise KeyError(f"Unknown eval policy '{name}'. Known: {', '.join(EVAL_POLICIES)}")
    if pulse and name in ("random", "aggressive", "conservative"):
        # An occasional channel keeps its schedule: a bolus every step is no treatment.
        occasional = np.asarray(pulse) < 1.0
        out = np.where(occasional, out * (factual > 0), out)
    return np.clip(out, 0.0, high)
