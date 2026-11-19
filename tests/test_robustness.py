"""Checks for the post hoc robustness tools: the timing-aware effect test and severity-shifted subsets."""

import numpy as np
import pytest

from grey_wm import audit
from grey_wm.data import TrajectorySet, dataset_path, generate, save


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    save(generate("sepsis", 0.0, 400, seed=0, steps=24), dataset_path(root, "sepsis", 0.0))
    return root


def _timed_data(slopes: list[float], n: int = 72, seed: int = 0) -> dict[str, np.ndarray]:
    """Synthetic per-segment rows: residual = slope_j * dose + noise, one drug."""
    rng = np.random.default_rng(seed)
    rows = {"patient": [], "segment": [], "resid": [], "dose": []}
    for _ in range(4):
        dose = rng.uniform(0, 1, n)
        for j, b in enumerate(slopes):
            rows["patient"].append(np.arange(n))
            rows["segment"].append(np.full(n, j))
            rows["resid"].append(b * dose + 0.1 * rng.standard_normal(n))
            rows["dose"].append(dose[:, None])
    return {k: np.concatenate(v) for k, v in rows.items()}


def test_timing_test_ignores_unrelated_errors():
    t = audit.effect_test_timed(_timed_data([0.0, 0.0, 0.0]), boot=300)
    assert t["z"].shape == (3, 1)
    assert not t["flag"]


def test_timing_test_flags_errors_that_cancel_over_the_window():
    # too fast early, too slow late: the window average is about zero, the segments are not
    d = _timed_data([0.5, 0.0, -0.5])
    avg = np.polyfit(d["dose"][:, 0], d["resid"], 1)[0]
    assert abs(avg) < 0.05
    assert audit.effect_test_timed(d, boot=300)["flag"]


def test_shifted_subsets_split_by_severity(data_root):
    from grey_wm import calibration

    calib, _ = calibration.randomised_sets(str(data_root), "sepsis")

    def sickness(c: TrajectorySet) -> float:
        st = c["states"].astype(np.float64)
        return float(np.mean([c.sim.sickness(st[:, t]) for t in range(st.shape[1])]))

    parts = {h: calibration.shifted_subset(calib, h) for h in calibration.SHIFTS}
    assert all(len(p) == len(calib) // 2 for p in parts.values())
    assert sickness(parts["mild50"]) < sickness(parts["first50"]) < sickness(parts["sick50"])
