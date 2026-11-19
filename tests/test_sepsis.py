import numpy as np

from grey_wm.data import generate
from grey_wm.policies import EVAL_POLICIES, eval_actions
from grey_wm.sims.sepsis import MAP, SepsisSim


def _setup(n=64, steps=16, seed=0):
    sim = SepsisSim()
    rng = np.random.default_rng(seed)
    params = sim.sample_patients(n, rng)
    x0 = sim.initial_state(params, rng)
    noise = sim.sample_noise(n, steps, rng)
    exog = sim.exog(rng.uniform(0, 24, n), steps, rng)
    return sim, params, x0, noise, exog


def test_norepinephrine_raises_map_monotonically():
    sim, params, x0, noise, exog = _setup()
    finals = []
    for dose in np.linspace(0, 0.5, 5):
        actions = np.zeros((x0.shape[0], 16, 2))
        actions[..., 0] = dose
        states, _ = sim.rollout(params, x0, actions, noise, exog)
        finals.append(states[:, -1, MAP])
    assert np.all(np.diff(np.stack(finals, 1), axis=1) > 0)


def test_replay_with_same_noise_is_deterministic():
    sim, params, x0, noise, exog = _setup()
    actions = np.full((x0.shape[0], 16, 2), 0.1)
    a, _ = sim.rollout(params, x0, actions, noise, exog)
    b, _ = sim.rollout(params, x0, actions, noise, exog)
    np.testing.assert_array_equal(a, b)


def test_confounding_flips_the_observational_correlation():
    def corr(lam):
        d = generate("sepsis", lam, 200, seed=0, steps=64)
        ne, map_ = d["actions"][..., 0].ravel(), d["clean"][:, 1:, 0].ravel()
        return np.corrcoef(ne, map_)[0, 1]

    assert corr(0.0) > 0  # randomised: more drug, higher MAP
    assert corr(1.0) < 0  # by indication: more drug, lower MAP


def test_shared_patients_and_split_across_lambdas():
    a, b = generate("sepsis", 0.0, 50, steps=8), generate("sepsis", 1.0, 50, steps=8)
    np.testing.assert_array_equal(a["params"], b["params"])
    np.testing.assert_array_equal(a["split"], b["split"])
    assert not np.allclose(a["actions"], b["actions"])


def test_eval_policies_respect_bounds():
    rng = np.random.default_rng(0)
    factual = rng.uniform(0, 0.5, (8, 20, 2))
    for name in EVAL_POLICIES:
        out = eval_actions(name, factual, (0.5, 0.5), rng)
        assert out.shape == factual.shape and out.min() >= 0 and out.max() <= 0.5


def test_glucose_port_matches_simglucose():
    import warnings

    from simglucose.patient.t1dpatient import Action, T1DPatient

    from grey_wm.sims.glucose import GlucoseSim

    warnings.filterwarnings("ignore")
    sim = GlucoseSim()
    pt = T1DPatient.withName("adult#004")
    basal = pt._params.u2ss * pt._params.BW / 6000  # U/min
    ref = []
    for t in range(305):
        if t % 5 == 0:
            ref.append(pt.observation.Gsub)
        pt.step(Action(CHO=60 if t == 100 else 0, insulin=basal + (2.0 if t == 100 else 0)))
    params = sim._table[[3]]
    x = sim.initial_state(params, np.random.default_rng(0))
    x[:, :13] = params[:, [sim._col[f"x0_{i + 1}"] for i in range(13)]]
    exog = np.zeros((1, 61, 3))
    exog[0, 20:23, 0] = np.array([25, 25, 10]) / 25
    acts = np.zeros((1, 60, 2))
    acts[..., 0] = basal * 60
    acts[0, 20, 1] = 2.0
    states, _ = sim.rollout(params, x, acts, np.zeros((1, 60, 1)), exog)
    assert np.abs(sim.observe_clean(states)[0, :, 0] - np.array(ref[:61])).max() < 5.0  # mg/dL


def test_glucose_confounding_flips_insulin_association():
    def corr(lam):
        d = generate("glucose", lam, 100, seed=0, steps=144)
        ins = d["actions"][..., 0] / 60 + d["actions"][..., 1] / 5
        cum = np.stack([ins[:, t - 12 : t].sum(1) for t in range(12, ins.shape[1])], 1)
        return np.corrcoef(cum.ravel(), d["clean"][:, 12:-1, 0].ravel())[0, 1]

    assert corr(1.0) > corr(0.0) + 0.3  # by indication, insulin travels with high glucose


def test_additive_greybox_is_monotone_in_dose_whatever_the_weights():
    import torch

    from grey_wm.models.base import filter_rollout
    from grey_wm.models.greybox import GreyBoxModel
    from grey_wm.sims import get_sim

    for env in ("sepsis", "glucose"):
        spec = get_sim(env).spec
        torch.manual_seed(0)
        model = GreyBoxModel(spec).eval()  # with the (monotone) effect residual
        for p in model.parameters():  # random, untrained weights
            torch.nn.init.normal_(p, std=0.5)
        b, steps = 4, 12
        batch = {
            "x": torch.randn(b, steps + 1, spec.obs_dim), "mask": torch.ones(b, steps + 1, spec.obs_dim),
            "dt_since": torch.zeros(b, steps + 1, spec.obs_dim), "exog": torch.zeros(b, steps + 1, spec.exog_dim),
            "actions": torch.zeros(b, steps, spec.action_dim),
        }
        o = spec.outcome_index
        sign = spec.pk_prior[0].signs[o]
        finals = []
        with torch.no_grad():
            for level in torch.linspace(0, 1, 5):
                future = torch.zeros(b, 6, spec.action_dim)
                future[..., 0] = level * spec.action_max[0]
                finals.append(filter_rollout(model, batch, 6, 6, future).mean[:, -1, o])
        steps_ = torch.diff(torch.stack(finals, 1), dim=1) * sign
        assert bool((steps_ >= -1e-6).all()), env


def test_insulin_prior_tracks_the_simulated_time_course():
    """The reduced PK/PD reproduces the simulator's lag: slow start, still building at 6 h."""
    import numpy as np
    import torch

    from grey_wm.models.greybox import GreyBoxModel
    from grey_wm.sims import get_sim

    sim = get_sim("glucose")
    model = GreyBoxModel(sim.spec, residual=False).eval()
    H = 96
    base = torch.zeros(1, 2)
    base[0, 0] = model.base_rate[0]
    raised = base.clone()
    raised[0, 0] += 0.25
    state = model.init_state(1, torch.device("cpu"))
    shifts = []
    with torch.no_grad():
        for _ in range(H):
            state = model.transition(state, raised, torch.zeros(1, sim.spec.exog_dim))
            shifts.append(float(state["shift"][0, 0]) * sim.spec.obs_std[0])
    # Simulator's mean response to +0.25 U/h: -1.9 at 2 h, -9.8 at 4 h, -17.7 at 6 h, -23.5 at 8 h.
    got = np.array([shifts[i] for i in (23, 47, 71, 95)])
    want = np.array([-1.9, -9.8, -17.7, -23.5])
    assert np.all(np.abs(got - want) <= np.maximum(0.3 * np.abs(want), 1.0)), got


def test_calibrated_scale_changes_size_never_direction():
    import torch

    from grey_wm.models.greybox import GreyBoxModel
    from grey_wm.sims import get_sim

    spec = get_sim("sepsis").spec
    model = GreyBoxModel(spec, residual=False).eval()
    model.effect_scale = torch.tensor([0.25, 4.0])
    model.effect_sigma = torch.tensor([0.3, 0.3])
    state = model.init_state(2, torch.device("cpu"))
    action = torch.tensor([[0.3, 0.3], [0.1, 0.2]])
    with torch.no_grad():
        state = model.transition(state, action, torch.zeros(2, spec.exog_dim))
    assert (state["shift"][:, 0] > 0).all()
    assert (state["shift_var"][:, 0] > 0).all()  # a bigger effect is less certain


def test_warm_start_carries_drug_into_a_window():
    import torch

    from grey_wm.models.greybox import GreyBoxModel
    from grey_wm.sims import get_sim

    spec = get_sim("glucose").spec
    model = GreyBoxModel(spec).eval()
    start = model.init_state(1, torch.device("cpu"))
    history = torch.zeros(1, 48, 2)
    history[..., 0] = 2.0
    warm = model.warm_start(start, history)
    assert float(warm["ce"][0, 0]) > float(start["ce"][0, 0])
