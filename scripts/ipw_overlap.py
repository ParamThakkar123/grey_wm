"""Share of training windows whose stabilised weight hits the cap, for both IPW schemes.

    python scripts/ipw_overlap.py
"""

from __future__ import annotations

import numpy as np

from grey_wm.data import load_split
from grey_wm.tasks import OptimConfig, stabilised_log_weights
from sweep import ENV_OVERRIDES

CLIP = OptimConfig().ipw_clip


def capped(logw: np.ndarray, cmin: int, cmax: int, hmax: int, window: str, rng) -> tuple[float, float]:
    """Windows drawn as in training after annealing: context ~ U[cmin, cmax], horizon ~ U[1, hmax]."""
    n, t = logw.shape
    context = rng.integers(cmin, cmax + 1, size=n)
    horizon = rng.integers(1, hmax + 1, size=n)
    starts = rng.integers(0, t - (context + horizon - 1))
    first = starts + np.where(window == "horizon", context - 1, 0)
    stop = starts + context + horizon - 1
    s = np.array([logw[i, a:b].sum() for i, (a, b) in enumerate(zip(first, stop))])
    w = np.exp(np.clip(s, -50, 50))
    return float(((w >= CLIP) | (w <= 1 / CLIP)).mean()), float(np.median(np.abs(s)))


def main() -> None:
    rng = np.random.default_rng(0)
    for env in ("sepsis", "glucose"):
        o = ENV_OVERRIDES[env]
        d = OptimConfig()
        win = (o.get("optim.context_min", d.context_min), o.get("optim.context_max", d.context_max),
               o.get("optim.horizon_max", d.horizon_max))
        for lam in (0.25, 0.5, 0.75, 1.0):
            train = load_split("data", env, lam)["train"]
            for oracle in (False, True):
                full = capped(stabilised_log_weights(train, oracle), *win, "full", rng)
                hor = capped(stabilised_log_weights(train, oracle, trunc_pct=1.0), *win, "horizon", rng)
                print(f"{env:8s} lam={lam:.2f} oracle={int(oracle)}  full: capped {full[0]:.2f}  "
                      f"horizon+trunc: capped {hor[0]:.2f}")


if __name__ == "__main__":
    main()
