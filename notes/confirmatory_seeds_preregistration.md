# Confirmatory test on fresh seeds: plan fixed before any run

Written 2026-11-15 11:45 IST, before seeds 5-14 were trained. Nothing below may change after the
first result of these seeds exists; anything else is post hoc and reported as such.

## Runs
lambda = 1, both simulators, seeds 5 to 14 (never used before), same code, data,
configs and steps as seeds 0-4: grey-box; GRU, GRU-ODE and CRN (CPU) and the Transformer
(GPU, PyTorch 2.12), each fine-tuned on the randomised set with `scripts/finetune.py`.
Planning (open and closed loop) and the realistic bolus effects with the existing scripts.

## Hypotheses (one-sided, grey-box better), on the 10 fresh seeds only
For each simulator (2) and fine-tuned baseline (4):
1. effect skill (dose up, calibrated) higher;
2. regret lower;
3. CF RMSE lower;
4. open-loop planning regret lower;
5. closed-loop planning regret lower.
40 tests. Exact permutation test of the difference in means, 10 against 10 (184756
permutations), Holm correction over all 40, alpha 0.05. Also reported: the same tests on all
15 seeds (0-14), which mix exploratory and fresh seeds (Monte Carlo, 200000 permutations).

Revision, about 11:45 IST, before any run (the runs started 11:45:18): the first version used 5 fresh seeds, but 5 against 5
gives p >= 0.004, which can never pass Holm over 40 tests (first threshold 0.00125); changed
to 10 fresh seeds.

## Reporting
Every test, significant or not. A hypothesis that fails on the fresh seeds is reported as failed.

## Log (added after the runs started; the plan above is unchanged)
- 2026-11-15 ~13:15 IST: all lanes died (Windows could not start new processes, out of memory),
  after 41 of the 80 CPU training runs and 17 of the 20 Transformer runs.
- 2026-11-16 07:52 IST: resumed with `scripts/run_resume.sh` and `scripts/run_resume_gpu.sh`, which
  skip finished runs; stopped at 08:28 at the user's request and resumed at 09:10. Runs that were
  cut off part-way were restarted from scratch with the same seed; no finished run was repeated or dropped.
- 2026-11-16 ~07:55 IST: `scripts/confirmatory_test.py` written, implementing the tests above as
  written. At that time some training runs of seeds 5-14 had finished, but no fine-tuned model, no
  planning result and no test of the fresh seeds existed.
- The Transformer's fresh seeds were fine-tuned on the CPU, as were seeds 0-4.
