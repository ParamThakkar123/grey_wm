#!/bin/sh
# Transformer on the GPU (the paper used a CUDA build of PyTorch 2.13).
# Use a Python environment with CUDA PyTorch and this package installed; PY overrides the interpreter.
PY=${PY:-python}
$PY -u scripts/sweep.py --envs glucose sepsis --models transformer --seeds 0 1 2 \
  --max-steps 3000 --device cuda --out-dir runs_v2 --results results_v2_transformer.jsonl
