#!/bin/sh
# 2026-11-16, post hoc: fluid-only state-dependent elimination (sepsis) and a linear dose-response prior
# (lambda = 0.75, both simulators), then the effect test on these models. 2 CPU workers.
PY=${PY:-.venv/Scripts/python.exe}
COMMON="--max-steps 3000 --jobs 2 --out-dir runs_v2"
echo "--- start $(date) ---"
$PY -u scripts/sweep.py --envs sepsis --models greybox --seeds 0 1 2 $COMMON --results results_elim.jsonl \
    --set model.elim=state model.elim_only=1 --tag _elimfluid
$PY -u scripts/sweep.py --envs sepsis glucose --lams 0.75 --models greybox --seeds 0 1 2 $COMMON \
    --results results_struct.jsonl --set model.pd=linear --tag _linear
for t in greybox_linear greybox_elimfluid; do
  $PY -u scripts/reaudit.py --jobs 2 --results results_audit_struct.jsonl --only "$t/"; done
echo "--- end $(date) ---"
