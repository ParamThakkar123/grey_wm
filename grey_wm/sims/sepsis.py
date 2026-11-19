"""Env B: sepsis hemodynamics under norepinephrine and fluids.

A small cardiovascular ODE, stepped on a 15-minute grid:

* **Severity** ``s`` in ``[0, 1]`` is latent (never observed). It drifts toward a
  per-patient target and is worsened by sustained hypotension, so treatment
  matters beyond the MAP it buys directly.
* **Norepinephrine** follows one-compartment PK (exact for a constant infusion
  over a step) and an Emax PD curve on MAP, scaled by a per-patient
  sensitivity.
* **Fluids** fill an excess-volume compartment that leaks faster in sicker
  patients and raises MAP through a saturating curve.
* **MAP, HR, lactate** relax toward severity- and treatment-dependent set
  points. Only these three are observed, irregularly (arterial line vs. labs).

Confounding comes from the behaviour policy reacting to ``s``: sicker patients
get more norepinephrine *and* have lower MAP, so a model that learns from the
correlation sees "more drug, lower pressure".
"""

from __future__ import annotations

import numpy as np

from grey_wm.sims.base import EnvSpec, PKPrior, Simulator

MAP, HR, LAC, VOL, SEV, CNE = range(6)
KE_NE, SENS_NE, EMAX_NE, EC50_NE, KE_F, EMAX_F, MAP0, HR0, SEV_TARGET, SEV_RATE = range(10)

_NE_PRIOR = PKPrior(ke=15.0, sens=1.0, emax=25.0, ec50=0.012, hill=1.5, signs=(1.0, 0.0, 0.0))
_FLUID_PRIOR = PKPrior(ke=0.6, sens=1.0, emax=10.0, ec50=0.5, hill=1.0, signs=(1.0, 0.0, 0.0))
_SUBSTEPS = 10
SEV_MAP_DROP = 0.7  # fraction of baseline MAP lost at full severity; must outweigh
# what indicated dosing buys back, or confounding never flips the observed sign.


def _relu(v: np.ndarray) -> np.ndarray:
    return np.maximum(v, 0.0)


class SepsisSim(Simulator):
    spec = EnvSpec(
        name="sepsis",
        obs_names=("map", "hr", "lactate"),
        action_names=("norepinephrine", "fluid"),
        action_max=(0.5, 0.5),  # mcg/kg/min, L/h
        exog_dim=2,
        dt=0.25,
        horizon=192,  # 48 h
        obs_mean=(70.0, 100.0, 2.5),
        obs_std=(12.0, 15.0, 1.5),
        obs_noise=(3.0, 4.0, 0.2),
        obs_prob=(0.7, 0.7, 0.08),
        outcome="map",
        pk_prior=(_NE_PRIOR, _FLUID_PRIOR),
    )
    param_names = (
        "ke_ne", "sens_ne", "emax_ne", "ec50_ne", "ke_fluid",
        "emax_fluid", "map0", "hr0", "sev_target", "sev_rate",
    )
    state_names = ("map", "hr", "lactate", "volume", "severity", "ne_conc")
    noise_dim = 3  # severity, MAP, HR

    def sample_patients(self, n: int, rng: np.random.Generator) -> np.ndarray:
        def lognormal(mean: float, sigma: float) -> np.ndarray:
            return mean * np.exp(sigma * rng.standard_normal(n))

        p = np.empty((n, len(self.param_names)))
        p[:, KE_NE] = lognormal(_NE_PRIOR.ke, 0.25)
        p[:, SENS_NE] = lognormal(1.0, 0.35)  # the per-patient sensitivity
        p[:, EMAX_NE] = lognormal(_NE_PRIOR.emax, 0.15)
        p[:, EC50_NE] = lognormal(_NE_PRIOR.ec50, 0.25)
        p[:, KE_F] = lognormal(_FLUID_PRIOR.ke, 0.25)
        p[:, EMAX_F] = lognormal(_FLUID_PRIOR.emax, 0.2)
        p[:, MAP0] = rng.normal(85.0, 5.0, n)
        p[:, HR0] = rng.normal(80.0, 8.0, n)
        # Some patients recover, some deteriorate.
        p[:, SEV_TARGET] = np.where(rng.random(n) < 0.5, rng.uniform(0.0, 0.3, n), rng.uniform(0.6, 0.95, n))
        p[:, SEV_RATE] = lognormal(0.05, 0.3)
        return p

    def initial_state(self, params: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = params.shape[0]
        s = rng.uniform(0.2, 0.8, n)
        x = np.zeros((n, len(self.state_names)))
        x[:, SEV] = s
        x[:, MAP] = params[:, MAP0] * (1 - SEV_MAP_DROP * s)
        x[:, HR] = params[:, HR0] + 40 * s + 0.8 * _relu(75 - x[:, MAP])
        x[:, LAC] = 1.0 + 0.8 * s * (1 + _relu(65 - x[:, MAP]) / 20) / 0.3
        return x

    def ne_effect(self, conc: np.ndarray, params: np.ndarray) -> np.ndarray:
        h = _NE_PRIOR.hill
        ch = np.maximum(conc, 0.0) ** h
        return params[:, SENS_NE] * params[:, EMAX_NE] * ch / (params[:, EC50_NE] ** h + ch)

    def step(self, x, a, params, noise, exog):
        dt = self.spec.dt
        x = x.copy()
        ke = params[:, KE_NE]
        decay = np.exp(-ke * dt)
        x[:, CNE] = x[:, CNE] * decay + a[:, 0] / ke * (1 - decay)
        e_ne = self.ne_effect(x[:, CNE], params)
        circadian = 3.0 * exog[:, 0]

        h = dt / _SUBSTEPS
        for _ in range(_SUBSTEPS):
            s, m, v = x[:, SEV], x[:, MAP], x[:, VOL]
            e_f = params[:, EMAX_F] * v / (_FLUID_PRIOR.ec50 + v)
            map_ss = params[:, MAP0] * (1 - SEV_MAP_DROP * s) + e_ne + e_f
            hr_ss = params[:, HR0] + 40 * s + 0.8 * _relu(75 - m) + 10 * e_ne / _NE_PRIOR.emax + circadian
            hypo = _relu(65 - m)
            x[:, MAP] += h * (map_ss - m) / 0.2
            x[:, HR] += h * (hr_ss - x[:, HR]) / 0.3
            x[:, LAC] += h * (0.8 * s * (1 + hypo / 20) - 0.3 * (x[:, LAC] - 1.0))
            x[:, VOL] += h * (a[:, 1] - (params[:, KE_F] + 0.5 * s) * v)
            x[:, SEV] += h * (params[:, SEV_RATE] * (params[:, SEV_TARGET] - s) + 0.3 * hypo / 65 * (1 - s))

        root = np.sqrt(dt)
        x[:, SEV] = np.clip(x[:, SEV] + 0.03 * root * noise[:, 0], 0.0, 1.0)
        x[:, MAP] += 2.0 * root * noise[:, 1]
        x[:, HR] += 3.0 * root * noise[:, 2]
        x[:, LAC] = np.maximum(x[:, LAC], 0.3)
        x[:, VOL] = np.maximum(x[:, VOL], 0.0)
        return x

    def observe_clean(self, states: np.ndarray) -> np.ndarray:
        return states[..., [MAP, HR, LAC]]

    def sickness(self, x: np.ndarray) -> np.ndarray:
        return x[:, SEV]
