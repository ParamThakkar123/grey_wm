# grey_wm

Code for *Don't Learn Pharmacology from Confounded Data: Grey-Box World Models
for Counterfactual Clinical Simulation* (Param Thakkar, MedWM workshop at AAAI-27).

Trained on clinical records where sicker patients get more drug, sequence models
can learn that a drug makes patients worse while their forecasts of the records
improve. This repository contains:

* a **confounding-stress benchmark**: a sepsis simulator and the UVA/Padova
  glucose simulator, with one knob λ that moves dosing from random (λ = 0) to
  purely by hidden severity (λ = 1), and counterfactual ground truth;
* a **grey-box world model**: the dose reaches the outcome only through a PK/PD
  block with a fixed direction of effect, a neural latent models the rest, and
  one multiplier per drug is set on a small randomised set;
* baselines (GRU, GRU-ODE, CRN, Transformer), their fine-tuning on the
  randomised set, ablations and wrong-prior runs;
* a **simulator-free check**: on held-out randomised patients, does the model's
  error depend on the dose given? (`grey_wm/audit.py`, `effect_test`).

All results are on simulators. Nothing here is validated on real patients, and
the sepsis simulator is not clinically validated. Do not use it for care.

## Setup

Python 3.11+.

```
python -m venv .venv
.venv\Scripts\activate             # Linux/macOS: source .venv/bin/activate
pip install -e .[dev,plots]        # pulls the Simulusion training harness from GitHub
pytest
```

## Reproduce the paper

Every step skips work that is already done, so an interrupted run can be
restarted with the same command. Times are for 2 CPU workers.

```
# 1. data: 2000 trajectories per (simulator, lambda)
python scripts/generate_data.py --env sepsis
python scripts/generate_data.py --env glucose

# 2. main grid, ablations, wrong priors on sepsis (CPU)
sh scripts/run_v2.sh
# Transformer (the paper used a GPU)
python scripts/sweep.py --envs glucose sepsis --models transformer --seeds 0 1 2 \
    --max-steps 3000 --device cuda --out-dir runs_v2 --results results_v2_transformer.jsonl

# 3. fine-tune the baselines on the 100 randomised trajectories
python scripts/finetune.py --lams 0 0.25 0.5 0.75 1 --jobs 2

# 4. simulator-free audit (both drugs' effect skill, randomised effect test)
python scripts/reaudit.py --jobs 2
python scripts/audit_v3_analysis.py     # pre-registered analyses: notes/audit_v3_preregistration.md
python scripts/audit_v3_extra.py        # post hoc extras and paper/tables/audit.tex

# 5. randomised-set size study and wrong priors on glucose
sh scripts/run_weakness.sh
python scripts/reaudit.py --jobs 2 --results results_audit_weak.jsonl --only glucose/lam0.75/greybox_
python scripts/reaudit.py --jobs 2 --results results_audit_weak.jsonl --only _n
python scripts/weakness_tables.py      # paper/tables/size.tex, misspec_glucose.tex
# post hoc calibration variants on the saved grey-box models (Table 7), no retraining
MSYS_NO_PATHCONV=1 python scripts/recalibrate.py --variants base perstep ec50 loose --only /greybox/
python scripts/recalibrate.py --variants base ec50 loose --only greybox_emax greybox_hill greybox_ke
python scripts/recal_tables.py         # paper/tables/recal.tex
python scripts/stats_tests.py           # paper/tables/stats.tex (seed-level permutation tests)
# final-week follow-ups (all post hoc; see the paper's appendix "What Was Decided When")
sh scripts/run_structural.sh           # PK/PD-only, wrong-form priors, seeds 3-4 at lambda=1 (+ fine-tuning)
# seeds 3-4 of the Transformer on a CUDA build (PyTorch 2.12 here):
#   python scripts/sweep.py --envs sepsis glucose --lams 1 --models transformer --seeds 3 4 --max-steps 3000 #     --device cuda --out-dir runs_v2 --results results_seeds_transformer.jsonl
sh scripts/run_final.sh                # IPW GRU, state-dependent elimination, planning, realistic bolus doses
for t in greybox_nodepot greybox_nodelay greybox_adddelay greybox_pkpd; do
  python scripts/reaudit.py --jobs 1 --results results_audit_struct.jsonl --only "$t/"; done
MSYS_NO_PATHCONV=1 python scripts/recalibrate.py --variants fluid_leak   --only sepsis/lam0.00/greybox/ sepsis/lam0.50/greybox/ sepsis/lam1.00/greybox/   # fluid diagnosis
python scripts/effect_test_thresholds.py  # stricter cut-offs; what the extra flags point at
python scripts/planning_table.py       # paper/tables/planning.tex
python scripts/stats_tests.py          # rerun: adds the 5-seed lambda=1 comparison
python scripts/paper_tables.py         # rerun: PK/PD-only and IPW rows in the main table
# confirmatory test, planned in notes/confirmatory_seeds_preregistration.md before the runs
sh scripts/run_resume.sh               # seeds 5-14 at lambda=1: training, fine-tuning, planning (2 CPU workers)
sh scripts/run_resume_gpu.sh           # Transformer seeds 5-14 on CUDA, then the post hoc oracle-IPW GRU
                                       # (stop it after the oracle-IPW runs; the paper ran the rest on the CPU:)
sh scripts/run_extras.sh               # post hoc: fluid-only elimination, linear PD, their effect tests (CPU)
python scripts/confirmatory_test.py    # 40 tests on the fresh seeds; paper/tables/confirm.tex
# second confirmatory test, planned in notes/confirmatory_lam050_preregistration.md before the runs
sh scripts/run_confirm050.sh           # lambda=0.5, seeds 15-24: training, fine-tuning, planning (CPU)
sh scripts/run_confirm050_gpu.sh       # the Transformer's seeds 15-24 (CUDA; PY= a Python with CUDA PyTorch)
python scripts/confirmatory_test.py --lam 0.5 --seeds 15-24   # paper/tables/confirm050.tex
# post hoc robustness (2026-11-16): timing-aware effect test (plan: notes/timing_effect_test_plan.md),
# a randomised set unlike the clinic's (finetune.py --subset, recalibrate.py first50/mild50/sick50),
# stronger fine-tuning (finetune.py --mode head | l2sp)
sh scripts/run_robustness.sh
python scripts/timing_audit.py --summary
python scripts/ft_variants.py
# post hoc (2026-11-18): IPW over the prediction window only, with truncated weights (gru-ipw-h,
# and its oracle control); the shifted randomised sets on seeds 3-9; planning for the new IPW models
sh scripts/run_ipw_h.sh                 # CPU lane; PY_GPU=<CUDA Python> sh scripts/run_ipw_h.sh for the oracle
sh scripts/run_shift_more.sh
sh scripts/run_ipw_plan.sh
python scripts/ipw_overlap.py           # share of training windows whose weight hits the cap
python scripts/ipw_compare.py           # effect skill and regret margin of every IPW variant
python scripts/shift_summary.py         # the shifted-set results (add seeds, e.g. 0 1 2, to restrict)
sh scripts/run_ftvar_more.sh             # output-layer and L2-SP fine-tuning, lambda=1, seeds 3-9
python scripts/ft_variants.py 0-9

# 6. tables and figures
python scripts/paper_tables.py
cd paper && latexmk -pdf main.tex
```

Each run is saved under `runs_v2/<env>/lam<λ>/<model>/s<seed>/` (resolved
config, metrics, checkpoints, `model/`), and its final metrics are appended to
a `results_*.jsonl` file that the table scripts read. Metrics after calibration
on the randomised set carry the `cal_` prefix.

## Layout

| Path | What |
|---|---|
| `grey_wm/sims/sepsis.py` | Sepsis simulator: MAP, heart rate, lactate; norepinephrine and fluids; hidden severity |
| `grey_wm/sims/glucose.py` | UVA/Padova type 1 diabetes model (simglucose equations and 10 adults), vectorised |
| `grey_wm/policies.py` | Confounded dosing policy (knob λ) and the evaluation policies |
| `grey_wm/data.py` | Generation and storage; patients, noise and splits shared across λ |
| `grey_wm/models/` | Grey-box model and ablations, GRU, GRU-ODE, CRN, Transformer |
| `grey_wm/calibration.py` | Randomised sets; effect multipliers and conformal scaling |
| `grey_wm/metrics.py` | Factual and counterfactual error, effect skill, sensitivity, regret, hazard recall |
| `grey_wm/audit.py` | Deployable gates and the randomised effect test |
| `grey_wm/tasks.py` | Each model as a Simulusion `TrainingTask` |
| `scripts/` | Data, training sweeps, fine-tuning, audit, tables and figures |
| `notes/` | Pre-registration of the audit analysis |
| `paper/` | LaTeX source |

## Licence and credits

MIT (see `LICENSE`). The glucose simulator uses the equations and virtual-patient
parameters of [simglucose](https://github.com/jxx123/simglucose) (MIT licence;
parameters vendored in `grey_wm/sims/data/vpatient_params.csv`), which implements
the UVA/Padova model of Dalla Man et al. (2014).

## Citation

```
@inproceedings{thakkar2027greybox,
  title     = {Don't Learn Pharmacology from Confounded Data: Grey-Box World Models for Counterfactual Clinical Simulation},
  author    = {Thakkar, Param},
  booktitle = {First Workshop on Medical World Models (MedWM) at AAAI-27},
  year      = {2027}
}
```
