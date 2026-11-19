#!/bin/sh
# 2026-11-18: planning (lambda = 1) for the new IPW models, after both IPW lanes. Only new models run.
PY=${PY:-.venv/Scripts/python.exe}
until grep -q -- "--- end" lane_ipw_h.log 2>/dev/null && grep -q -- "--- end" lane_ipw_h_gpu.log 2>/dev/null; do sleep 60; done
echo "--- start $(date) ---"
PYTHONPATH=scripts $PY -u scripts/planning.py
PYTHONPATH=scripts $PY -u scripts/planning_closed_loop.py
echo "--- end $(date) ---"
