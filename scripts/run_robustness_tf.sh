#!/bin/sh
# Transformer part of run_robustness.sh, with 1 worker: with 2, the Transformer fine-tunes ran
# out of memory (2026-11-17 14:30). Finished runs are skipped.
PY=${PY:-.venv/Scripts/python.exe}
echo "--- start $(date) ---"
for v in first50 mild50 sick50; do
  $PY -u scripts/finetune.py --lams 1 --models transformer --subset $v --results results_shift_ft.jsonl --jobs 1; done
for m in head l2sp; do
  $PY -u scripts/finetune.py --lams 0.5 1 --models transformer --mode $m --results results_ft_$m.jsonl --jobs 1; done
echo "--- end $(date) ---"
