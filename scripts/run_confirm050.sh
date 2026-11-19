#!/bin/sh
# Second confirmatory test (notes/confirmatory_lam050_preregistration.md): lambda = 0.5, seeds 15-24. 2 CPU workers.
PY=${PY:-.venv/Scripts/python.exe}
SEEDS="15 16 17 18 19 20 21 22 23 24"
echo "--- start $(date) ---"
$PY -u scripts/sweep.py --envs sepsis glucose --lams 0.5 --models greybox gru gru-ode crn --seeds $SEEDS \
    --max-steps 3000 --jobs 2 --out-dir runs_v2 --results results_confirm050.jsonl
$PY -u scripts/finetune.py --lams 0.5 --models gru gru-ode crn --seeds $SEEDS --results results_confirm050_ft.jsonl --jobs 2
until [ -f runs_v2/glucose/lam0.50/transformer/s24/model/model.pt ] && [ -f runs_v2/sepsis/lam0.50/transformer/s24/model/model.pt ]; do sleep 60; done
$PY -u scripts/finetune.py --lams 0.5 --models transformer --seeds $SEEDS --results results_confirm050_ft.jsonl --jobs 2
echo "--- training done $(date) ---"
$PY -u scripts/planning.py --lam 0.5 results_planning_lam050.jsonl
$PY -u scripts/planning_closed_loop.py --lam 0.5 results_planning_cl_lam050.jsonl
echo "--- end $(date) ---"
