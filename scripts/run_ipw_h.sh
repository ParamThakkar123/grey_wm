#!/bin/sh
# 2026-11-18, post hoc: IPW over the rollout doses only, per-step weights truncated (RMSN-style).
# Lane 1 (CPU, 2 workers): gru-ipw-h. Lane 2 (GPU, PY_GPU set): the oracle control.
PY=${PY:-.venv/Scripts/python.exe}
echo "--- start $(date) ---"
if [ -n "$PY_GPU" ]; then
  $PY_GPU -u scripts/sweep.py --envs sepsis glucose --models gru-ipw-h-oracle --seeds 0 1 2 --max-steps 3000 \
      --device cuda --out-dir runs_v2 --results results_ipw_h_oracle.jsonl
else
  $PY -u scripts/sweep.py --envs sepsis glucose --models gru-ipw-h --seeds 0 1 2 --max-steps 3000 --jobs 2 \
      --out-dir runs_v2 --results results_ipw_h.jsonl
fi
echo "--- end $(date) ---"
