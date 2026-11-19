# Audit v3: analysis plan, fixed before any v3 numbers were computed

Written 2026-11-10, before the code below was run on any model. Anything decided
after seeing v3 results must be reported as post hoc.

## Why
The v2 audit flagged 30 of 31 seriously broken configurations, but its main check
(randomised gap) caught only 1 of 8 broken fine-tuned models, the thresholds came
from a pilot on the same simulators, flags were computed on seed means, and
"broken" covered the first drug only.

## 1. New deployable check: randomised effect test
Uses only the 72 held-out randomised trajectories and the calibrated model.

* Windows: 4 start times per patient, spaced as in calibration
  (`n_context` to the last start that fits), horizon = `cf_horizon`
  (20 steps sepsis, 72 steps glucose). The model observes everything before the
  start and predicts open loop under the doses actually given.
* Per window: residual = mean over observed outcome steps of
  (observed outcome - predicted mean), divided by the outcome's `obs_std`.
  Windows without an observed outcome are dropped.
* Dose per drug = mean dose over the actions that drive the window, divided by
  the drug's maximum.
* Regress residual on [1, dose_1, dose_2] (least squares).
* Patient-level (cluster) bootstrap, B = 2000, seed 0; 95% percentile interval
  for each drug's coefficient.
* **Flag** if the interval of either drug excludes 0.
* Rationale: doses are random, so for a model whose predicted mean is the
  interventional mean, residuals are unrelated to the doses given. A model that
  reads severity from the dose, or has an effect of the wrong size or sign, leaves
  a dose-dependent residual. Descriptive only: effect ratio = slope of predicted
  outcome on dose / slope of observed outcome on dose, same regression.

## 2. Ground truth, both drugs
* `broken_v1` (as in v2): any first-drug effect skill (no_drug, dose_up, strong)
  < 0, or regret worse than the fixed dose.
* `broken_v2`: `broken_v1`, or second-drug effect skill < 0 under no_drug or
  dose_up (and strong, for drugs that are not occasional pulses; boluses are
  pulses, so glucose bolus gets no_drug and dose_up only).
* Both are reported. `broken_v2` is the main definition from now on.

## 3. Gates
* Old deployable gates unchanged: direction >= 0.99, rand_gap <= 1.25,
  rand_cov >= 0.8.
* New combined gate: any old gate fails, or the effect test flags.

## 4. Analyses
* Unit = one trained model (env, lambda, variant, seed). Seed means reported
  alongside, for comparison with v2.
* Sensitivity and false-alarm rate of each check alone and combined, for
  `broken_v1` and `broken_v2`.
* Threshold-free: AUC of each continuous score for `broken_v2`, per simulator
  and pooled. Scores: rand_gap; 1 - direction_worst; 1 - rand_cov; effect-test
  max |coefficient / bootstrap sd| over drugs.
* Transfer: for each continuous score, choose the threshold on one simulator that
  maximises sensitivity + specificity - 1 (Youden), apply it to the other, both
  directions; report sensitivity and specificity on the held-out simulator.
* Holdout size: effect test on random subsets of 18, 36 and 54 of the 72
  patients (10 draws each, seed 0), sensitivity and false-alarm rate.

## 5. Fine-tuned baselines
Rerun `scripts/finetune.py` unchanged except that it now saves weights, so the
new checks can be computed. The previous results file is archived; any
difference between the two runs is reported.

## Addendum (after results, 2026-11-10)
* Fine-tune rerun: all 120 runs bit-identical to `archive/results_ft_run1.jsonl`.
* Results: `audit_v3_analysis_full.txt`, `audit_v3_summary.json`.
* Post hoc, not planned above: `scripts/audit_v3_extra.py` (which checks catch each
  broken fine-tuned model; AUC within fine-tuned models; "mis-sized" = a drug's true
  sensitivity outside [0.5, 2]; seed-majority configuration counts). Reported as post hoc
  in the paper.
* Also post hoc: the same effect test, unchanged, on the 108 models of the size study and the
  glucose wrong-prior study (`results_audit_weak.jsonl`, summary in `scripts/weakness_tables.py`).
  These are kept out of the 312-model analysis.
* Also post hoc: calibration variants (per-step conformal sd, EC50 multiplier, looser prior on
  kappa) and in-distribution bolus effects, `scripts/recalibrate.py`; the paper's tables keep the
  original calibration.

## Addendum (final week, 2026-11-14 to 2026-11-15), all post hoc
* Effect test, unchanged, on the 39 new grey-box models (wrong-form priors, PK/PD only):
  `results_audit_struct.jsonl`. Kept out of the 312-model analysis.
* Stricter effect-test cut-offs and what the extra flags point at: `scripts/effect_test_thresholds.py`
  (z thresholds approximate the bootstrap interval by its s.d.).
* New baselines and variants (IPW GRU, PK/PD only, wrong-form priors, state-dependent elimination,
  seeds 3-4 at lambda = 1), planning (open and closed loop), realistic bolus doses for every model,
  and the fluid diagnosis (`recalibrate.py --variants fluid_leak`). None changes the main tables'
  numbers, calibration, metrics or the definition of a broken model.
