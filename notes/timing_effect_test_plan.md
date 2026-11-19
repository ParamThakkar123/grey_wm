# Timing-aware effect test: definition fixed before it was computed

Written 2026-11-16, before any model was scored with it. Post hoc relative to the paper's
effect test (notes/audit_v3_preregistration.md), which it complements and does not replace.

## Why
The effect test regresses each window's mean outcome error on each drug's mean dose. A model
whose effect has the right average size but the wrong timing (no insulin depot, no delay, an
extra delay) leaves errors that cancel over the window; it flagged 1 of 9 such models.

## Definition
Same 72 randomised hold-out patients, same four start times and horizon as the effect test.
Each window is cut into 3 equal segments. For segment j, the error is the mean normalised
outcome residual (observed minus predicted) over observed steps in segment j, and the dose of
each drug is its mean dose (fraction of maximum) from the step before the window to the end of
segment j. For each segment separately, regress the error on the doses (OLS with intercept).
Patient-level bootstrap (2000 draws, seed 0) gives each slope's standard error; z = |slope|/se.
Flag the model if any z exceeds 2.394 (two-sided 95% with Bonferroni over the 3 segments).

## Analyses (all reported)
1. Of the 110 seriously broken models (same definition as the paper), how many it flags; of the
   other 202, how many. Same for "effect test or timing test".
2. The 9 wrong-form models (no depot, no delay, added delay) and the 6 linear-PD models: how many
   it flags.
3. The 30 PK/PD-only models and the 15 fluid-only elimination models: how many it flags.
Models: every model in results_audit_v3.jsonl and results_audit_struct.jsonl.
