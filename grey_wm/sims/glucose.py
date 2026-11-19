"""Env A: glucose-insulin, the UVA/Padova T1D model of simglucose, vectorised.

The equations and the 10 adult virtual patients' parameters are simglucose's
(``T1DPatient.model``, ``vpatient_params.csv``; MIT licence). simglucose steps
one patient at a time through scipy; here every trajectory steps together with
Euler substeps of 0.5 min (the adult rate constants are all below 0.6/min), so
a whole dataset generates in seconds and any saved state can be replayed under
other actions for counterfactual ground truth.

* Grid: 5-minute steps, 24-hour trajectories starting at a random clock time.
* Observed: CGM only (subcutaneous glucose), with measurement noise and dropout.
* Actions: basal rate (U/h) and bolus (U, delivered within the step). Boluses
  are occasional (``action_pulse``).
* Exogenous: carbohydrate eaten during each step (meals at 5 g/min) and time
  of day. Meals are breakfast, lunch, dinner and an optional snack, jittered.
* Latent sickness for the confounded policy: how far plasma glucose is above
  target, so insulin is given when glucose is high.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from grey_wm.sims.base import EnvSpec, PKPrior, Simulator

N_ODE = 13
QSTO, FOOD, PREV_CARB, VG = 13, 14, 15, 16  # meal bookkeeping and Vg, carried in the state
EAT_RATE = 5.0  # g/min
SUBSTEP_MIN = 0.5
STEP_MIN = 5.0

# Reduced insulin PK/PD for the grey-box model: three first-order lags (depot,
# central, effect) and a nearly linear curve. Fitted by scripts/derive_insulin_prior.py
# to the simulator's mean response to basal steps and single boluses in meal-free
# patients: never to trajectories, metrics or any data a model is scored on.
_INSULIN = dict(ke=0.576, sens=1.0, emax=1810.0, ec50=20.0, hill=1.0, ka=0.576, ke0=0.282, signs=(-1.0,))
BASAL_U_PER_H = 1.25  # mean steady-state basal of the 10 adults (u2ss)


#: simglucose's virtual-patient parameters (MIT licence), vendored so the
#: simulator needs neither simglucose nor its gym dependency at run time.
PARAMS_CSV = os.path.join(os.path.dirname(__file__), "data", "vpatient_params.csv")


@lru_cache(maxsize=1)
def _adults() -> tuple[tuple[str, ...], np.ndarray]:
    import csv

    with open(PARAMS_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["Name"].startswith("adult")]
    columns = [c for c in rows[0] if c not in ("Name", "patient_history")]
    names = tuple(c.replace(" ", "") for c in columns)
    return names, np.array([[float(r[c]) for c in columns] for r in rows], dtype=np.float64)


class GlucoseSim(Simulator):
    spec = EnvSpec(
        name="glucose",
        obs_names=("cgm",),
        action_names=("basal", "bolus"),
        action_max=(2.5, 2.0),  # U/h, U per step
        exog_dim=3,
        dt=STEP_MIN / 60.0,
        horizon=288,  # 24 h
        obs_mean=(140.0,),
        obs_std=(50.0,),
        obs_noise=(8.0,),
        obs_prob=(0.9,),
        outcome="cgm",
        pk_prior=(
            PKPrior(**_INSULIN, input_scale=1.0, baseline_rate=BASAL_U_PER_H),  # basal: U/h already
            PKPrior(**_INSULIN, input_scale=60.0 / STEP_MIN),  # bolus: U per step -> U/h
        ),
    )
    state_names = tuple(f"x{i + 1}" for i in range(N_ODE)) + ("last_qsto", "last_food", "prev_carbs", "vg")
    noise_dim = 1
    action_pulse = (1.0, 0.08)

    def __init__(self) -> None:
        names, table = _adults()
        self.param_names = names
        self._table = table
        self._col = {n: i for i, n in enumerate(names)}

    def sample_patients(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._table[np.arange(n) % len(self._table)].copy()  # 10 adults, evenly

    def initial_state(self, params: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = params.shape[0]
        x = np.zeros((n, len(self.state_names)))
        x[:, :N_ODE] = params[:, [self._col[f"x0_{i + 1}"] for i in range(N_ODE)]]
        jitter = np.exp(0.15 * rng.standard_normal(n))
        x[:, [3, 4, 12]] *= jitter[:, None]  # random initial glucose, as simglucose's random_init_bg
        x[:, QSTO] = x[:, 0] + x[:, 1]
        x[:, VG] = params[:, self._col["Vg"]]
        return x

    def exog(self, start_hour: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
        n = start_hour.shape[0]
        carbs = np.zeros((n, steps + 1))
        meals = [(7.0, 0.5, 45, 70, 1.0), (12.5, 0.5, 60, 85, 1.0), (19.0, 0.5, 60, 90, 1.0), (15.5, 1.0, 10, 25, 0.5)]
        for day in (0, 24):
            for clock, jitter, lo, hi, prob in meals:
                when = (clock + day + jitter * rng.standard_normal(n) - start_hour) / self.spec.dt
                grams = rng.uniform(lo, hi, n) * (rng.random(n) < prob)
                start = np.floor(when).astype(int)
                for i in np.flatnonzero((start >= 0) & (start <= steps) & (grams > 0)):
                    left, t = grams[i], start[i]
                    while left > 0 and t <= steps:
                        bite = min(EAT_RATE * STEP_MIN, left)
                        carbs[i, t] += bite
                        left -= bite
                        t += 1
        return np.concatenate([carbs[..., None] / (EAT_RATE * STEP_MIN), self.time_of_day(start_hour, steps)], -1)

    def step(self, x, a, params, noise, exog):
        c = self._col
        p = {k: params[:, c[k]] for k in (
            "BW", "kmax", "kmin", "kabs", "b", "d", "f", "kp1", "kp2", "kp3", "Fsnc", "ke1", "ke2", "k1", "k2",
            "Vm0", "Vmx", "Km0", "m1", "m2", "m4", "m30", "ka1", "ka2", "kd", "ksc", "p2u", "ki", "Ib", "Vi",
        )}
        x = x.copy()
        carbs = exog[:, 0] * EAT_RATE * STEP_MIN  # g eaten this step
        starting = (carbs > 0) & (x[:, PREV_CARB] <= 0)
        x[:, QSTO] = np.where(starting, x[:, 0] + x[:, 1], x[:, QSTO])
        x[:, FOOD] = np.where(starting, 0.0, x[:, FOOD]) + carbs
        x[:, PREV_CARB] = carbs
        d = carbs / STEP_MIN * 1000.0  # mg/min
        insulin = (a[:, 0] / 60.0 + a[:, 1] / STEP_MIN) * 6000.0 / p["BW"]  # pmol/kg/min
        dbar = x[:, QSTO] + x[:, FOOD] * 1000.0

        s = x[:, :N_ODE]
        h = SUBSTEP_MIN
        for _ in range(int(STEP_MIN / SUBSTEP_MIN)):
            qsto = s[:, 0] + s[:, 1]
            safe = np.maximum(dbar, 1e-9)
            aa = 5.0 / (2.0 * safe * (1 - p["b"]))
            cc = 5.0 / (2.0 * safe * p["d"])
            kgut = np.where(
                dbar > 0,
                p["kmin"] + (p["kmax"] - p["kmin"]) / 2
                * (np.tanh(aa * (qsto - p["b"] * dbar)) - np.tanh(cc * (qsto - p["d"] * dbar)) + 2),
                p["kmax"],
            )
            ds = np.empty_like(s)
            ds[:, 0] = -p["kmax"] * s[:, 0] + d
            ds[:, 1] = p["kmax"] * s[:, 0] - s[:, 1] * kgut
            ds[:, 2] = kgut * s[:, 1] - p["kabs"] * s[:, 2]
            rat = p["f"] * p["kabs"] * s[:, 2] / p["BW"]
            egp = p["kp1"] - p["kp2"] * s[:, 3] - p["kp3"] * s[:, 8]
            et = np.where(s[:, 3] > p["ke2"], p["ke1"] * (s[:, 3] - p["ke2"]), 0.0)
            ds[:, 3] = np.maximum(egp, 0) + rat - p["Fsnc"] - et - p["k1"] * s[:, 3] + p["k2"] * s[:, 4]
            uid = (p["Vm0"] + p["Vmx"] * s[:, 6]) * s[:, 4] / (p["Km0"] + s[:, 4])
            ds[:, 4] = -uid + p["k1"] * s[:, 3] - p["k2"] * s[:, 4]
            ds[:, 5] = -(p["m2"] + p["m4"]) * s[:, 5] + p["m1"] * s[:, 9] + p["ka1"] * s[:, 10] + p["ka2"] * s[:, 11]
            it = s[:, 5] / p["Vi"]
            ds[:, 6] = -p["p2u"] * s[:, 6] + p["p2u"] * (it - p["Ib"])
            ds[:, 7] = -p["ki"] * (s[:, 7] - it)
            ds[:, 8] = -p["ki"] * (s[:, 8] - s[:, 7])
            ds[:, 9] = -(p["m1"] + p["m30"]) * s[:, 9] + p["m2"] * s[:, 5]
            ds[:, 10] = insulin - (p["ka1"] + p["kd"]) * s[:, 10]
            ds[:, 11] = p["kd"] * s[:, 10] - p["ka2"] * s[:, 11]
            ds[:, 12] = -p["ksc"] * s[:, 12] + p["ksc"] * s[:, 3]
            for i in (3, 4, 5, 9, 10, 11, 12):  # simglucose's non-negativity guards
                ds[:, i] *= s[:, i] >= 0
            s = np.maximum(s + h * ds, np.where(np.arange(N_ODE) == 6, -np.inf, 0.0))
        s[:, 3] = np.maximum(s[:, 3] + 1.0 * np.sqrt(STEP_MIN) * noise[:, 0], 0.0)  # small EGP noise, mg/kg
        x[:, :N_ODE] = s
        return x

    def observe_clean(self, states: np.ndarray) -> np.ndarray:
        return (states[..., 12] / states[..., VG])[..., None]

    def sickness(self, x: np.ndarray) -> np.ndarray:
        bg = x[:, 3] / x[:, VG]
        return np.clip((bg - 90.0) / 160.0, 0.0, 1.0)
