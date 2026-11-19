import json

import pytest

import grey_wm.tasks  # noqa: F401
from grey_wm.data import dataset_path, generate, save
from grey_wm.tasks import nested
from simulusion.training.run import run


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    save(generate("sepsis", 0.75, 60, seed=0, steps=48), dataset_path(root, "sepsis", 0.75))
    save(generate("glucose", 0.75, 60, seed=0, steps=72), dataset_path(root, "glucose", 0.75))
    for env, steps in (("sepsis", 48), ("glucose", 72)):  # the randomised set
        save(generate(env, 0.0, 60, seed=0, steps=steps), dataset_path(root, env, 0.0))
    return root


@pytest.mark.parametrize("task", ["gru", "gru-ode", "crn", "transformer", "greybox", "greybox-latent", "greybox-actres", "greybox-res", "greybox-leak"])
def test_task_trains_and_reports_every_metric(task, data_root, tmp_path):
    res = run(task=task, overrides=nested({
        "trainer.max_steps": 3, "trainer.device": "cpu", "trainer.out_dir": str(tmp_path),
        "trainer.run_name": task, "trainer.eval_every": 2, "trainer.console": False,
        "data.root": str(data_root), "data.batch_size": 8,
        "eval.n_context": 12, "eval.horizons": [1, 5, 20], "eval.cf_horizon": 10,
    }))
    m = res.metrics
    for key in ("rmse_h1", "coverage_h5", "cf_rmse", "cf_rmse_flip",
                "norepinephrine_sign_agree", "fluid_sensitivity", "norepinephrine_monotone_violation"):
        assert key in m, key
    assert ("residual_norm" in m) == (task in ("greybox-res", "greybox-actres"))
    for key in ("skill_no_drug", "cfcov_strong", "regret", "rand_rmse", "cal_skill_no_drug", "sd_scale", "selected_step"):
        assert key in m, key
    assert ("norepinephrine_effect_scale" in m) == (task in ("greybox", "greybox-res", "greybox-actres", "greybox-leak"))
    assert (res.model_path / "model.pt").exists()
    assert json.loads((res.model_path / "model.json").read_text())["model"]["kind"] in ("gru", "gru-ode", "crn", "transformer", "greybox")


@pytest.mark.parametrize("task", ["gru", "greybox"])
def test_glucose_env_trains(task, data_root, tmp_path):
    res = run(task=task, overrides=nested({
        "trainer.max_steps": 2, "trainer.device": "cpu", "trainer.out_dir": str(tmp_path),
        "trainer.run_name": task, "trainer.console": False,
        "data.env": "glucose", "data.root": str(data_root), "data.batch_size": 8,
        "optim.context_min": 12, "optim.context_max": 24,
        "eval.n_context": 24, "eval.horizons": [1, 5, 20], "eval.cf_horizon": 10, "eval.effect_window": 12,
    }))
    for key in ("cf_rmse", "basal_sign_agree", "bolus_sensitivity", "basal_monotone_violation"):
        assert key in res.metrics, key
