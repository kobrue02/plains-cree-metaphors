"""Device and mixed-precision detection utilities."""

import torch


def get_device() -> str:
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_precision_kwargs() -> dict:
    """Return fp16/bf16 kwargs for HuggingFace TrainingArguments.

    CUDA  → fp16=True  (fast on all NVIDIA GPUs)
    MPS   → fp16=False (MPS does not support fp16 training)
    CPU   → fp16=False
    """
    if torch.cuda.is_available():
        return {"fp16": True, "bf16": False}
    return {"fp16": False, "bf16": False}
