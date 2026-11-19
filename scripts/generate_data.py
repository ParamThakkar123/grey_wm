"""Generate the confounded datasets: one .npz per (env, lambda).

    python scripts/generate_data.py --env sepsis --lams 0 0.25 0.5 0.75 1 --n 2000
"""

from __future__ import annotations

import argparse
import time

from grey_wm.data import dataset_path, generate, save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="sepsis")
    p.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="data")
    args = p.parse_args()
    for lam in args.lams:
        start = time.perf_counter()
        path = save(generate(args.env, lam, args.n, args.seed), dataset_path(args.root, args.env, lam))
        print(f"{path}  ({time.perf_counter() - start:.1f}s)")


if __name__ == "__main__":
    main()
