#!/bin/sh
# 2026-11-18, post hoc: stronger fine-tuning (head, L2-SP) on seeds 3-9 at lambda = 1, so the
# lambda = 1 comparison rests on 10 seeds. GRU, GRU-ODE, CRN; 2 CPU workers.
PY=${PY:-.venv/Scripts/python.exe}
echo "--- start $(date) ---"
for m in head l2sp; do
  $PY -u scripts/finetune.py --lams 1 --models gru gru-ode crn --seeds 3 4 5 6 7 8 9 --mode $m \
      --results results_ft_$m.jsonl --jobs 2; done
echo "--- end $(date) ---"
