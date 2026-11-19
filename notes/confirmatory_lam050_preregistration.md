# Second confirmatory test, lambda = 0.5, fresh seeds 15-24: plan fixed before any run

Written 2026-11-16 14:17 IST, before any model with seeds 15-24 was trained (no run directory or
result line for these seeds exists at this time). Nothing below may change after the first result
of these seeds exists; anything else is post hoc and reported as such.

## Why lambda = 0.5
The first confirmatory test (notes/confirmatory_seeds_preregistration.md) covered only lambda = 1.
lambda = 0.5 is the setting where unadjusted baselines start to fail and where the exploratory
3-seed comparisons were weakest (glucose regret against fine-tuned GRU and GRU-ODE were not
significant across lambda >= 0.25). Known before this plan: the 3-seed means at lambda = 0.5
(seeds 0-2), which favour the grey-box model on every metric below. This is a replication on new
seeds, not a test of an untested setting.

## Runs
lambda = 0.5, both simulators, seeds 15 to 24 (never used before), same code, data, configs and
steps as seeds 0-2 at lambda = 0.5: grey-box; GRU, GRU-ODE and CRN (CPU) and the Transformer
(GPU), each fine-tuned on the randomised set with `scripts/finetune.py` (CPU).
Planning (open and closed loop) at lambda = 0.5 with `scripts/planning.py --lam 0.5` and
`scripts/planning_closed_loop.py --lam 0.5`.

## Hypotheses (one-sided, grey-box better), on the 10 fresh seeds only
For each simulator (2) and fine-tuned baseline (4):
1. effect skill (dose up, calibrated) higher;
2. regret lower;
3. CF RMSE lower;
4. open-loop planning regret lower;
5. closed-loop planning regret lower.
40 tests. Exact permutation test of the difference in means, 10 against 10 (184756
permutations), Holm correction over all 40, alpha 0.05. Same script as the first test
(`scripts/confirmatory_test.py`, run with `--lam 0.5 --seeds 15-24`).

## Reporting
Every test, significant or not, in the paper's appendix. A hypothesis that fails is reported as
failed. The two confirmatory tests are reported separately (no pooling, no correction across them).

## Log (added after the runs started; the plan above is unchanged)
- 2026-11-16 ~18:25 IST: the Claude Code session that had started the lane ended and its child
  processes stopped, during the last 4 Transformer fine-tunes (glucose, seeds 21-24).
- 2026-11-17 08:49 IST: resumed (finished runs are skipped); fine-tuning done 09:08, planning done 10:41.
- 2026-11-17 10:4x IST: `python scripts/confirmatory_test.py --lam 0.5 --seeds 15-24`, as planned:
  40 of 40 significant after Holm; largest corrected p 0.045 (glucose regret against fine-tuned GRU-ODE).
