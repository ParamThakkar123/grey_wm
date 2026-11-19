#!/bin/sh
# 2026-11-16: resume the confirmatory lane after the 2026-11-15 13:15 crash (out of memory). 2 CPU workers.
PY=.venv/Scripts/python.exe
SEEDS="5 6 7 8 9 10 11 12 13 14"
echo "--- start $(date) ---"
$PY -u scripts/sweep.py --envs sepsis glucose --lams 1 --models greybox gru gru-ode crn --seeds $SEEDS \
    --max-steps 3000 --jobs 2 --out-dir runs_v2 --results results_confirm.jsonl
$PY -u scripts/finetune.py --lams 1 --models gru gru-ode crn --seeds $SEEDS --results results_confirm_ft.jsonl --jobs 2
until [ -f runs_v2/glucose/lam1.00/transformer/s14/model/model.pt ]; do sleep 60; done
$PY -u scripts/finetune.py --lams 1 --models transformer --seeds $SEEDS --results results_confirm_ft.jsonl --jobs 2
echo "--- confirmatory training done $(date) ---"
$PY -u scripts/planning.py
$PY -u scripts/planning_closed_loop.py
echo "--- planning done $(date) ---"
until grep -q -- "--- end" lane_resume_gpu.log 2>/dev/null; do sleep 60; done
for t in greybox_linear greybox_elimfluid; do
  $PY -u scripts/reaudit.py --jobs 2 --results results_audit_struct.jsonl --only "$t/"; done
echo "--- end $(date) ---"
