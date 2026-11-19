import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # never share the GPU with a training job
