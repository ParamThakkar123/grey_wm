#!/bin/sh
# Transformer for the second confirmatory test, on CUDA (PY: a Python with CUDA PyTorch; simulusion importable).
PY=${PY:-python}
echo "--- start $(date) ---"
$PY -u scripts/sweep.py --envs sepsis glucose --lams 0.5 --models transformer --seeds 15 16 17 18 19 20 21 22 23 24 \
    --max-steps 3000 --device cuda --out-dir runs_v2 --results results_confirm050_transformer.jsonl
echo "--- end $(date) ---"
