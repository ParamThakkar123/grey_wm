"""The grey-box model: actions reach the outcome only through sign-constrained PK/PD.

Per action channel ``c``, with per-patient ``theta_c = (k_e, S, Emax, EC50)``::

    C_c(t + dt) = PK_c(C_c, a_c)                          # exact IV / depot + central (SC)
    Ce_c(t+dt)  = lag(Ce_c, C_c; ke0)                     # effect compartment, if the drug has one
    H(C)        = S Emax C^h / (EC50^h + C^h)             # >= 0, monotone in dose
    E_c         = k_c [ H(Ce_c) - H(Ce_c at baseline) ]   # relative to the baseline dose
    z(t + dt)   = z + dt * [ f_phi(z, u) + g_psi(z, a) ]  # untreated course + sparse residual
    x_t         ~ N( h(z_t) + sum_c M_c E_c ,  sigma(z_t)^2 )
    M_c         = sign_c * softplus(w_c) / obs_std        # sign from pharmacology, size learned

``effect_mode="additive"`` (the model): ``f_phi`` sees neither the action nor
its effect, so the latent learns the untreated disease course, and the drug
moves each observation only along its known direction (``PKPrior.signs``). The
encoder filters observations with the predicted drug effect subtracted. The
predicted outcome is monotone in dose by construction; confounding can bias the
size of an effect but cannot flip its sign.

``effect_mode="latent"`` (ablation): the PD effect is fed into ``f_phi(z, e, u)``
as in a plain hybrid ODE. ``e`` is still monotone in dose, but a free network
can couple it to the outcome with either sign, and under confounding it learns
the wrong one, as a black box does.

* ``theta`` comes from a softplus head on the filtered latent, initialised at
  the published population values and refined at every observation; it is held
  fixed during an open-loop rollout. A log-normal prior keeps it near them.
* ``k_c`` (``effect_scale``) and the predictive uncertainty of an effect
  (``effect_sigma``) stay at 1 and 0 in training. Both are fitted afterwards on a
  small randomised set (:mod:`grey_wm.calibration`), because confounded data can
  shrink an effect's size: size is calibrated, direction is fixed.
* The residual is off in the model (``greybox-res`` turns it on): it can only add
  to an effect, and in the misspecification sweeps it gave no measurable gain.
  ``residual_mode="effect"`` adds to each drug's effect a correction
  ``R_c(C) = sum_k a_k(z) [sigmoid(alpha_k C / EC50 + beta_k) - sigmoid(beta_k)]``
  with ``a, alpha >= 0``: monotone in concentration and zero at zero dose, so it
  can reshape a dose-response curve but never reverse it; it has no sparsity
  penalty. ``residual_mode="action"`` is the unconstrained ``g_psi(z, a)`` on the
  raw action, added to ``dz/dt`` with an L1 penalty; under confounding it learns
  the confounded association and overrides the mechanistic pathway (ablation).
* ``action_leak`` also gives ``f_phi`` the raw action (ablation). ``ke_scale`` /
  ``ec50_scale`` / ``emax_scale`` misspecify the priors on purpose; ``hill_scale``
  misspecifies the assumed shape of the dose-response curve.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from grey_wm.models.base import SequenceModel, State, gaussian_head, mlp
from grey_wm.sims import EnvSpec

KE, SENS, EMAX, EC50 = range(4)
PK_SUBSTEPS = 4
EFFECT_MODES = ("additive", "latent")
RESIDUAL_MODES = ("effect", "action")
RESIDUAL_BASIS = 8


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class GreyBoxModel(SequenceModel):
    def __init__(
        self,
        spec: EnvSpec,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        effect_mode: str = "additive",
        residual: bool = True,
        residual_mode: str = "effect",
        learn_effect_scale: bool = False,
        action_leak: bool = False,
        dynamics: str = "euler",
        l1: float = 0.05,
        prior_weight: float = 0.1,
        prior_sigma: float = 0.3,
        ke_scale: float = 1.0,
        ec50_scale: float = 1.0,
        emax_scale: float = 1.0,
        hill_scale: float = 1.0,
        delay: str = "prior",
        depot: bool = True,
        elim: str = "fixed",
        elim_only: int = -1,
        pd: str = "emax",
    ) -> None:
        super().__init__(spec)
        if effect_mode not in EFFECT_MODES:
            raise ValueError(f"effect_mode must be one of {EFFECT_MODES}, not '{effect_mode}'")
        if residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"residual_mode must be one of {RESIDUAL_MODES}, not '{residual_mode}'")
        self.latent_dim = latent_dim
        self.additive = effect_mode == "additive"
        self.residual = residual
        self.effect_residual = residual and residual_mode == "effect"
        self.action_leak = action_leak
        self.l1 = l1
        self.prior_weight = prior_weight
        self.prior_sigma = prior_sigma
        pop = torch.tensor(
            [[p.ke * ke_scale, p.sens, p.emax * emax_scale, p.ec50 * ec50_scale] for p in spec.pk_prior], dtype=torch.float32
        )
        self.register_buffer("theta_pop", pop)  # (A, 4)
        self.register_buffer("hill", torch.tensor([p.hill * hill_scale for p in spec.pk_prior], dtype=torch.float32))
        # Subcutaneous channels absorb from a depot at a fixed rate ka; 0 = intravenous.
        # Structural misspecification (wrong form of the prior, not wrong values):
        # depot=False drops the subcutaneous depot (insulin acts like an infusion);
        # delay="none" removes the effect-compartment delay, delay="add" puts a 1/h
        # delay on drugs that have none.
        self.register_buffer("ka", torch.tensor([(p.ka or 0.0) if depot else 0.0 for p in spec.pk_prior],
                                                dtype=torch.float32))
        self.register_buffer("input_scale", torch.tensor([p.input_scale for p in spec.pk_prior], dtype=torch.float32))
        ke0 = {"prior": lambda p: p.ke0 or 0.0, "none": lambda p: 0.0, "add": lambda p: p.ke0 or 1.0}[delay]
        self.register_buffer("ke0", torch.tensor([ke0(p) for p in spec.pk_prior], dtype=torch.float32))
        self.register_buffer("base_rate", torch.tensor([p.baseline_rate for p in spec.pk_prior], dtype=torch.float32))
        self.register_buffer("effect_scale", torch.ones(len(spec.pk_prior)))  # k_c, calibrated
        self.register_buffer("effect_sigma", torch.zeros(len(spec.pk_prior)))  # log-sd of k_c, calibrated
        n_act = spec.action_dim

        self.obs_cell = nn.GRUCell(self.obs_input_dim, latent_dim)
        self.theta_head = nn.Linear(latent_dim, 4 * n_act)
        nn.init.zeros_(self.theta_head.weight)
        nn.init.zeros_(self.theta_head.bias)
        extra = spec.exog_dim + (0 if self.additive else n_act) + (n_act if action_leak else 0)
        f_in = latent_dim + extra
        # "gru": a recurrent cell for the untreated course. It sees exactly what f sees,
        # so the action still reaches the outcome only through the PK/PD block.
        # "frozen": no learned course at all (mechanistic-only baseline); the encoder
        # and decoder still read the patient's state from the context.
        self.frozen = dynamics == "frozen"
        self.cell = nn.GRUCell(extra, latent_dim) if dynamics == "gru" else None
        self.f = None if dynamics == "gru" else mlp(f_in, hidden_dim, latent_dim)
        self.g = mlp(latent_dim + n_act, hidden_dim, latent_dim) if residual and not self.effect_residual else None
        if self.effect_residual:
            k = RESIDUAL_BASIS
            self.res_head = nn.Linear(latent_dim, n_act * k)  # per-patient basis weights a_k(z) >= 0
            nn.init.zeros_(self.res_head.weight)
            nn.init.constant_(self.res_head.bias, -4.0)  # softplus(-4) ~ 0.02: starts near zero
            self.res_alpha = nn.Parameter(torch.linspace(-1.0, 2.0, k).repeat(n_act, 1))  # log-ish slopes
            self.res_beta = nn.Parameter(torch.linspace(-3.0, 1.0, k).repeat(n_act, 1))
        # elim="state": each drug's elimination rate is scaled by exp(tanh(w . z_t)), a
        # factor in [1/e, e] set by the patient's current (untreated) state, e.g. a leak
        # that speeds up in sicker patients. The dose still acts only through the PK/PD
        # block, and more drug still means more concentration, so the direction holds.
        # elim_only >= 0 restricts the state-dependent factor to that drug.
        # pd="linear": a dose-response of the wrong family, S Emax C / EC50 (no ceiling).
        self.elim_only = elim_only
        self.linear_pd = pd == "linear"
        self.elim_head = None
        if elim == "state":
            self.elim_head = nn.Linear(latent_dim, n_act)
            nn.init.zeros_(self.elim_head.weight)
            nn.init.zeros_(self.elim_head.bias)
        self.decoder = mlp(latent_dim, hidden_dim, 2 * spec.obs_dim)
        if self.additive:
            signs = [p.signs or (0.0,) * spec.obs_dim for p in spec.pk_prior]
            self.register_buffer("signs", torch.tensor(signs, dtype=torch.float32))  # (A, D)
            self.register_buffer("obs_std", torch.tensor(spec.obs_std, dtype=torch.float32))
            # The size of each effect is set by the published Emax (and per-patient S);
            # a learned scale on top is redundant with them and lets confounding shrink
            # the effect through their product, so by default it is fixed at 1.
            init = torch.full_like(self.signs, float(inv_softplus(torch.tensor(1.0))))
            if learn_effect_scale:
                self.effect_weight = nn.Parameter(init)
            else:
                self.register_buffer("effect_weight", init)

    def theta(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.theta_head(z).view(-1, *self.theta_pop.shape)
        return F.softplus(raw + inv_softplus(self.theta_pop))

    def init_state(self, batch: int, device: torch.device) -> State:
        z = torch.zeros(batch, self.latent_dim, device=device)
        conc = (self.base_rate / self.theta_pop[:, KE]).expand(batch, -1)
        depot = (self.base_rate / self.ka.clamp_min(1e-6) * (self.ka > 0)).expand(batch, -1)
        return {
            "z": z,
            "conc": conc,
            "depot": depot,
            "ce": conc,
            "theta": self.theta(z),
            "shift": torch.zeros(batch, self.spec.obs_dim, device=device),
            "shift_var": torch.zeros(batch, self.spec.obs_dim, device=device),
            "g_abs": torch.zeros(batch, device=device),
        }

    def warm_start(self, state, history):
        """Run the compartments over the actions before a window, at population rates."""
        ke = self.theta_pop[:, KE].expand(state["conc"].shape)
        conc, depot, ce = state["conc"], state["depot"], state["ce"]
        with torch.no_grad():
            for t in range(history.shape[1]):
                conc, depot = self.pk(conc, depot, history[:, t] * self.input_scale, ke)
                ce = self.effect_lag(ce, conc)
        return {**state, "conc": conc, "depot": depot, "ce": ce}

    def effect_lag(self, ce: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        """One exact step of ``C_e' = ke0 (C - C_e)``; ``C_e = C`` without a lag."""
        decay = torch.exp(-self.ke0 * self.spec.dt)
        return torch.where(self.ke0 > 0, ce * decay + conc * (1 - decay), conc)

    def observe(self, state, x, mask, dt_since, exog):
        # The latent tracks the untreated course: filter the observations with
        # the current drug effect taken out.
        x = x - state["shift"] * mask
        z = self.obs_cell(self.obs_input(x, mask, dt_since, exog), state["z"])
        return {**state, "z": z, "theta": self.theta(z)}

    def effect(self, conc: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """PD effect per channel, in outcome units; >= 0 and monotone in concentration."""
        if self.linear_pd:
            return theta[..., SENS] * theta[..., EMAX] * conc.clamp_min(0.0) / theta[..., EC50]
        ch = conc.clamp_min(0.0) ** self.hill
        return theta[..., SENS] * theta[..., EMAX] * ch / (theta[..., EC50] ** self.hill + ch)

    def effect_correction(self, conc: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Residual ``R_c(C)`` in outcome units: >= 0, monotone in ``C``, zero at ``C = 0``."""
        a = F.softplus(self.res_head(z)).view(z.shape[0], *self.res_alpha.shape)  # (B, A, K)
        u = (conc.clamp_min(0.0) / self.theta_pop[:, EC50])[..., None]  # (B, A, 1)
        basis = torch.sigmoid(F.softplus(self.res_alpha) * u + self.res_beta) - torch.sigmoid(self.res_beta)
        return (a * basis).sum(-1) * self.theta_pop[:, EMAX]

    def effect_matrix(self) -> torch.Tensor:
        """``(A, D)``: normalised shift of each observation per unit of each effect."""
        return self.signs * F.softplus(self.effect_weight) / self.obs_std

    def pk(self, conc, depot, rate, ke):
        """Advance the compartments one step under a constant dose ``rate`` (per hour).

        Intravenous: ``C' = -ke C + rate``, solved exactly. Subcutaneous:
        ``D' = -ka D + rate`` (exact) and ``C' = ka D - ke C`` (Euler substeps).
        """
        dt = self.spec.dt
        decay = torch.exp(-ke * dt)
        iv = conc * decay + rate / ke * (1 - decay)
        if not bool((self.ka > 0).any()):
            return iv, depot
        ka = self.ka.clamp_min(1e-6)
        sc_conc, sc_depot = conc, depot
        h = dt / PK_SUBSTEPS
        for _ in range(PK_SUBSTEPS):
            sc_depot = sc_depot * torch.exp(-ka * h) + rate / ka * (1 - torch.exp(-ka * h))
            sc_conc = sc_conc + h * (ka * sc_depot - ke * sc_conc)
        sub = self.ka > 0
        return torch.where(sub, sc_conc, iv), torch.where(sub, sc_depot, depot)

    def transition(self, state, action, exog):
        dt = self.spec.dt
        theta, z = state["theta"], state["z"]
        ke = theta[..., KE]
        if self.elim_head is not None:
            factor = torch.exp(torch.tanh(self.elim_head(z)))
            if self.elim_only >= 0:
                keep = torch.zeros_like(factor[0], dtype=torch.bool)
                keep[self.elim_only] = True
                factor = torch.where(keep, factor, torch.ones_like(factor))
            ke = ke * factor
        conc, depot = self.pk(state["conc"], state["depot"], action * self.input_scale, ke)
        ce = self.effect_lag(state["ce"], conc)
        rest = (self.base_rate / self.theta_pop[:, KE]).expand(ce.shape)
        effect = self.effect(ce, theta) - self.effect(rest, theta)  # relative to the baseline dose
        g_abs = torch.zeros_like(state["g_abs"])
        if self.effect_residual:
            correction = self.effect_correction(ce, z)
            effect = effect + correction
            g_abs = (correction / self.theta_pop[:, EMAX]).abs().mean(-1)
        scaled = action / self.action_max
        f_in = [z, exog]
        shift_var = state["shift_var"]
        if self.additive:
            per_drug = effect * self.effect_scale
            shift = per_drug @ self.effect_matrix()
            shift_var = (per_drug * self.effect_sigma) ** 2 @ self.effect_matrix() ** 2
        else:
            f_in.append(effect / self.theta_pop[:, EMAX])  # ~[0, S]
            shift = state["shift"]
        if self.action_leak:
            f_in.append(scaled)
        inputs = torch.cat(f_in, -1)
        if self.frozen:
            z_next = z  # PK/PD only: the untreated course is held where the filter left it
        elif self.cell is not None:
            z_next = self.cell(inputs[..., self.latent_dim :], z)
        else:
            z_next = z + dt * self.f(inputs)
        if self.g is not None:
            g = self.g(torch.cat([z, scaled], -1))
            z_next = z_next + dt * g
            g_abs = g.abs().mean(-1)
        return {
            "z": z_next, "conc": conc, "depot": depot, "ce": ce, "theta": theta,
            "shift": shift, "shift_var": shift_var, "g_abs": g_abs,
        }

    def emit(self, state):
        mean, logvar = gaussian_head(self.decoder(state["z"]))
        if bool((self.effect_sigma > 0).any()):
            logvar = torch.log(torch.exp(logvar) + state["shift_var"] + 1e-12)
        return mean + state["shift"], logvar

    def penalties(self, states):
        out: dict[str, torch.Tensor] = {}
        if self.g is not None:
            out["residual_l1"] = self.l1 * torch.stack([s["g_abs"] for s in states]).mean()
        log_ratio = torch.log(states[-1]["theta"]) - torch.log(self.theta_pop)
        out["theta_prior"] = self.prior_weight * (log_ratio**2).mean() / (2 * self.prior_sigma**2)
        return out

    @staticmethod
    def residual_norm(states: list[State]) -> torch.Tensor:
        """Mean |g| over a rollout: the misspecification signal (figure 4)."""
        return torch.stack([s["g_abs"] for s in states]).mean()
