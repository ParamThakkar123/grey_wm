#!/bin/sh
# 2026-11-16: GPU lane, one run at a time. Transformer seeds 5-14, then the post hoc extras on the GPU.
PY=${PY:-python}   # a Python with CUDA PyTorch
# simulusion must be importable (pip install, or PYTHONPATH)
G="--max-steps 3000 --device cuda --out-dir runs_v2"
echo "--- start $(date) ---"
$PY -u scripts/sweep.py --envs sepsis glucose --lams 1 --models transformer --seeds 5 6 7 8 9 10 11 12 13 14 $G \
    --results results_confirm_transformer.jsonl
$PY -u scripts/sweep.py --envs sepsis glucose --models gru-ipw-oracle --seeds 0 1 2 $G --results results_ipw_oracle.jsonl
$PY -u scripts/sweep.py --envs sepsis --models greybox --seeds 0 1 2 $G --results results_elim.jsonl \
    --set model.elim=state model.elim_only=1 --tag _elimfluid
$PY -u scripts/sweep.py --envs sepsis glucose --lams 0.75 --models greybox --seeds 0 1 2 $G \
    --results results_struct.jsonl --set model.pd=linear --tag _linear
echo "--- end $(date) ---"
