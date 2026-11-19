#!/bin/sh
# Two follow-up studies (2026-11-10), 2 CPU workers. Reruns skip finished runs.
# 1. Size of the randomised set: 25 or 50 trajectories instead of 100, used for
#    weight selection, calibration and fine-tuning alike. lambda = 1, both envs.
# 2. Wrong priors on glucose at lambda = 0.75 (Emax, Hill, and k_e timing).
PY=.venv/Scripts/python.exe
COMMON="--seeds 0 1 2 --max-steps 3000 --jobs 2 --out-dir runs_v2"
for n in 25 50; do
  $PY -u scripts/sweep.py --envs sepsis glucose --lams 1 --models greybox gru gru-ode crn $COMMON \
      --results results_size.jsonl --set data.calib_n=$n --tag _n$n
done
$PY -u scripts/finetune.py --lams 1 --models gru_n25 gru-ode_n25 crn_n25 gru_n50 gru-ode_n50 crn_n50 \
    --results results_size_ft.jsonl --jobs 2
$PY -u scripts/sweep.py --envs glucose --misspec-hard --models greybox $COMMON --results results_misspec_glucose.jsonl
for k in 0.5 2; do
  $PY -u scripts/sweep.py --envs glucose --lams 0.75 --models greybox $COMMON --results results_misspec_glucose.jsonl \
      --set model.ke_scale=$k --tag _ke_scale$k
done
