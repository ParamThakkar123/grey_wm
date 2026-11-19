#!/bin/sh
# Queued after run_confirm.sh (2026-11-15), 2 CPU workers. Post hoc unless stated.
# 1. Control: IPW-weighted GRU whose propensity model sees the hidden severity (gru-ipw-oracle).
# 2. Grey-box with state-dependent elimination for fluids only (sepsis).
# 3. Wrong dose-response family: linear instead of Emax, lambda = 0.75, both envs.
# 4. Effect test on the new grey-box models; planning on all seeds (confirmatory for seeds 5-14).
PY=.venv/Scripts/python.exe
COMMON="--max-steps 3000 --jobs 2 --out-dir runs_v2"
until grep -q -- "--- end" lane_confirm.log 2>/dev/null; do sleep 60; done
echo "--- start $(date) ---"
$PY -u scripts/sweep.py --envs sepsis glucose --models gru-ipw-oracle --seeds 0 1 2 $COMMON --results results_ipw_oracle.jsonl
$PY -u scripts/sweep.py --envs sepsis --models greybox --seeds 0 1 2 $COMMON --results results_elim.jsonl \
    --set model.elim=state model.elim_only=1 --tag _elimfluid
$PY -u scripts/sweep.py --envs sepsis glucose --lams 0.75 --models greybox --seeds 0 1 2 $COMMON \
    --results results_struct.jsonl --set model.pd=linear --tag _linear
for t in greybox_linear greybox_elimfluid; do
  $PY -u scripts/reaudit.py --jobs 2 --results results_audit_struct.jsonl --only "$t/"; done
until [ -f runs_v2/glucose/lam1.00/transformer/s14/model/model.pt ] && [ -f runs_v2/sepsis/lam1.00/transformer/s14/model/model.pt ]; do sleep 60; done
$PY -u scripts/finetune.py --lams 1 --models transformer --seeds 5 6 7 8 9 10 11 12 13 14 --results results_confirm_ft.jsonl --jobs 2
$PY -u scripts/planning.py
$PY -u scripts/planning_closed_loop.py
echo "--- end $(date) ---"
