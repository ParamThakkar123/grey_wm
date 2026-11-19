"""Where the insulin prior in ``grey_wm/sims/glucose.py`` comes from.

Steps basal insulin and gives single boluses to the 10 adult patients with no
meals and no noise, then fits a chain of three first-order lags to the mean change
in glucose. This plays the part of a published dose-finding study: it uses the
simulator's response to clean test doses, never a benchmark trajectory or metric.

The fitted gain is per U/h at steady state. ``emax`` and ``ec50`` are then set so
the curve has that slope at the basal dose and stays nearly linear above it.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from grey_wm.sims.glucose import GlucoseSim

H, DT = 144, 5 / 60  # 12 h on the 5-minute grid
EC50 = 20.0


def chain(rates: np.ndarray, gain: float, rate: np.ndarray) -> np.ndarray:
    y = rate.astype(float).copy()
    for k in rates:
        out, state, decay = np.zeros(H), 0.0, np.exp(-k * DT)
        for t in range(H):
            state = state * decay + y[t] * (1 - decay)
            out[t] = state
        y = out
    return -gain * y


def main() -> None:
    sim = GlucoseSim()
    n = 10
    params = sim.sample_patients(n, np.random.default_rng(0))
    start = sim.initial_state(params, np.random.default_rng(1))
    col = sim._col
    basal = params[:, col["u2ss"]] * params[:, col["BW"]] * 60 / 6000
    exog, noise = np.zeros((n, H + 1, 3)), np.zeros((n, H, 1))

    def glucose(actions: np.ndarray) -> np.ndarray:
        states, _ = sim.rollout(params, start, actions, noise, exog)
        return sim.observe_clean(states[:, 1:])[..., 0]

    base = np.zeros((n, H, 2))
    base[..., 0] = basal[:, None]
    reference = glucose(base)
    cases = []
    for step in (0.1, 0.25, 0.5):
        dosed = base.copy()
        dosed[..., 0] += step
        cases.append((np.full(H, step), (glucose(dosed) - reference).mean(0)))
    for units in (0.5, 1.0, 2.0):
        dosed = base.copy()
        dosed[:, 6, 1] = units
        rate = np.zeros(H)
        rate[6] = units * 12
        cases.append((rate, (glucose(dosed) - reference).mean(0)))

    def residual(q: np.ndarray) -> np.ndarray:
        return np.concatenate([chain(np.exp(q[:3]), np.exp(q[3]), r) - d for r, d in cases])

    fit = least_squares(residual, np.log([0.5, 1.0, 2.0, 100.0]))
    rates, gain = np.sort(np.exp(fit.x[:3])), float(np.exp(fit.x[3]))
    print(f"rates per hour {np.round(rates, 3)}, gain {gain:.1f} mg/dL per U/h")
    print(f"fit RMSE {np.sqrt(np.mean(fit.fun**2)):.2f} mg/dL, mean basal {basal.mean():.2f} U/h")
    ke = float(rates[1])
    conc = basal.mean() / ke
    print(f"ke {ke:.3f}, emax for ec50 = {EC50:g}: {gain * ke * (EC50 + conc) ** 2 / EC50:.0f}")


if __name__ == "__main__":
    main()
