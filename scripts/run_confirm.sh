#!/bin/sh
# Confirmatory fresh seeds 5-14 at lambda = 1 (notes/confirmatory_seeds_preregistration.md), 2 CPU workers.
PY=.venv/Scripts/python.exe
SEEDS="5 6 7 8 9 10 11 12 13 14"
echo "--- start $(date) ---"
$PY -u scripts/sweep.py --envs sepsis glucose --lams 1 --models greybox gru gru-ode crn --seeds $SEEDS \
    --max-steps 3000 --jobs 2 --out-dir runs_v2 --results results_confirm.jsonl
$PY -u scripts/finetune.py --lams 1 --models gru gru-ode crn --seeds $SEEDS --results results_confirm_ft.jsonl --jobs 2
echo "--- end $(date) ---"
