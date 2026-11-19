#!/bin/sh
# The full v2 grid on CPU, most important first. Reruns skip finished runs.
# The Transformer runs on the GPU separately: see scripts/run_v2_gpu.sh
PY=.venv/Scripts/python.exe
COMMON="--seeds 0 1 2 --max-steps 3000 --jobs 2 --out-dir runs_v2 --results results_v2.jsonl"
$PY -u scripts/sweep.py --envs glucose --models greybox gru gru-ode crn $COMMON
$PY -u scripts/sweep.py --envs sepsis --models greybox gru gru-ode crn $COMMON
$PY -u scripts/sweep.py --envs glucose sepsis --lams 0.75 --models greybox-res greybox-actres greybox-leak greybox-latent $COMMON
$PY -u scripts/sweep.py --envs sepsis --misspec-hard --models greybox $COMMON
