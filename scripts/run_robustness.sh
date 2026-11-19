#!/bin/sh
# Post hoc robustness runs (2026-11-16), after the lambda = 0.5 confirmatory lane. 2 CPU workers.
# 1. timing-aware effect test (notes/timing_effect_test_plan.md) on every audited model;
# 2. a randomised set unlike the clinic's: grey-box recalibrated, baselines fine-tuned, on half of
#    it chosen by hidden severity (first50 = size control, mild50, sick50), lambda = 1, seeds 0-2;
# 3. stronger fine-tuning: output layer only (head) and L2-SP, lambda = 0.5 and 1, seeds 0-2.
PY=${PY:-.venv/Scripts/python.exe}
until grep -q -- "--- end" lane_c050.log 2>/dev/null; do sleep 60; done
echo "--- start $(date) ---"
PYTHONPATH=scripts $PY -u scripts/timing_audit.py --jobs 2
for v in first50 mild50 sick50; do
  $PY -u scripts/recalibrate.py --variants $v --only lam1.00/greybox/s0/ lam1.00/greybox/s1/ lam1.00/greybox/s2/ --results results_shift.jsonl --jobs 2; done
for v in first50 mild50 sick50; do
  $PY -u scripts/finetune.py --lams 1 --models gru gru-ode crn --subset $v --results results_shift_ft.jsonl --jobs 2; done
for m in head l2sp; do
  $PY -u scripts/finetune.py --lams 0.5 1 --models gru gru-ode crn --mode $m --results results_ft_$m.jsonl --jobs 2; done
echo "--- non-Transformer done $(date) ---"
for v in first50 mild50 sick50; do
  $PY -u scripts/finetune.py --lams 1 --models transformer --subset $v --results results_shift_ft.jsonl --jobs 2; done
for m in head l2sp; do
  $PY -u scripts/finetune.py --lams 0.5 1 --models transformer --mode $m --results results_ft_$m.jsonl --jobs 2; done
echo "--- end $(date) ---"
