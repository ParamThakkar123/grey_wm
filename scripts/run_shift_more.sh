#!/bin/sh
# 2026-11-18, post hoc: the shifted-randomised-set check on seeds 3-9 (lambda = 1), so it rests on
# 10 seeds instead of 3. Grey-box recalibrated; GRU, GRU-ODE and CRN fine-tuned. 2 CPU workers,
# after the IPW lane.
PY=${PY:-.venv/Scripts/python.exe}
until grep -q -- "--- end" lane_ipw_h.log 2>/dev/null; do sleep 60; done
echo "--- start $(date) ---"
ONLY=""; for s in 3 4 5 6 7 8 9; do ONLY="$ONLY lam1.00/greybox/s$s/"; done
for v in first50 mild50 sick50; do
  $PY -u scripts/recalibrate.py --variants $v --only $ONLY --results results_shift.jsonl --jobs 2; done
for v in first50 mild50 sick50; do
  $PY -u scripts/finetune.py --lams 1 --models gru gru-ode crn --seeds 3 4 5 6 7 8 9 --subset $v \
      --results results_shift_ft.jsonl --jobs 2; done
echo "--- end $(date) ---"
