#!/bin/sh
# Last follow-ups (2026-11-14), 2 CPU workers. Reruns skip finished runs.
# 1. Mechanistic-only baseline: the grey-box model with no learned untreated course
#    (dynamics=frozen), every lambda, both envs.
# 2. Structurally wrong priors at lambda = 0.75: glucose without the insulin depot or
#    without the effect delay; sepsis with an effect delay the simulator does not have.
# 3. Two more seeds (3, 4) at lambda = 1 for the grey-box model and the CPU baselines,
#    then fine-tuning of those baselines.
PY=.venv/Scripts/python.exe
COMMON="--max-steps 3000 --jobs 2 --out-dir runs_v2"
echo "--- start $(date) ---"
$PY -u scripts/sweep.py --envs sepsis glucose --models greybox --seeds 0 1 2 $COMMON \
    --results results_pkpd.jsonl --set model.dynamics=frozen --tag _pkpd
$PY -u scripts/sweep.py --envs glucose --lams 0.75 --models greybox --seeds 0 1 2 $COMMON \
    --results results_struct.jsonl --set model.depot=false --tag _nodepot
$PY -u scripts/sweep.py --envs glucose --lams 0.75 --models greybox --seeds 0 1 2 $COMMON \
    --results results_struct.jsonl --set model.delay=none --tag _nodelay
$PY -u scripts/sweep.py --envs sepsis --lams 0.75 --models greybox --seeds 0 1 2 $COMMON \
    --results results_struct.jsonl --set model.delay=add --tag _adddelay
$PY -u scripts/sweep.py --envs sepsis glucose --lams 1 --models greybox gru gru-ode crn --seeds 3 4 $COMMON \
    --results results_seeds.jsonl
$PY -u scripts/finetune.py --lams 1 --models gru gru-ode crn --seeds 3 4 --results results_seeds_ft.jsonl --jobs 2
echo "--- end $(date) ---"
