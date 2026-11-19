#!/bin/sh
# Queued after run_structural.sh (2026-11-14), 2 CPU workers.
# 1. Inverse-probability-weighted GRU (simplified RMSN), every lambda, both envs.
# 2. Grey-box with state-dependent elimination (elim=state), sepsis, every lambda.
# 3. Fine-tuning of the two extra Transformer seeds (trained on the GPU lane).
# 4. Planning (open and closed loop) and realistic bolus doses on every model at lambda = 1.
PY=.venv/Scripts/python.exe
COMMON="--max-steps 3000 --jobs 2 --out-dir runs_v2"
until grep -q -- "--- end" lane_struct.log 2>/dev/null; do sleep 60; done
echo "--- start $(date) ---"
$PY -u scripts/sweep.py --envs sepsis glucose --models gru-ipw --seeds 0 1 2 $COMMON --results results_ipw.jsonl
$PY -u scripts/sweep.py --envs sepsis --models greybox --seeds 0 1 2 $COMMON --results results_elim.jsonl \
    --set model.elim=state --tag _elim
until [ -f runs_v2/glucose/lam1.00/transformer/s4/model/model.pt ] && [ -f runs_v2/sepsis/lam1.00/transformer/s4/model/model.pt ]; do sleep 60; done
$PY -u scripts/finetune.py --lams 1 --models transformer --seeds 3 4 --results results_seeds_ft.jsonl --jobs 2
$PY -u scripts/planning.py
$PY -u scripts/planning_closed_loop.py
$PY -u scripts/bolus_realistic.py
echo "--- end $(date) ---"
