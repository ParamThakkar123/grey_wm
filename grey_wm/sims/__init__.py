"""Benchmark simulators with counterfactual ground truth."""

from __future__ import annotations

from grey_wm.sims.base import EnvSpec, PKPrior, Simulator


def get_sim(name: str) -> Simulator:
    if name == "sepsis":
        from grey_wm.sims.sepsis import SepsisSim

        return SepsisSim()
    if name == "glucose":
        from grey_wm.sims.glucose import GlucoseSim

        return GlucoseSim()
    raise KeyError(f"Unknown env '{name}'. Known: sepsis, glucose")


__all__ = ["EnvSpec", "PKPrior", "Simulator", "get_sim"]
