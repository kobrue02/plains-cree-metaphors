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

    A100/H100 → bf16=True  (native BF16, no gradient scaling needed)
    Other CUDA → fp16=True  (requires FP32 model weights; fails on FP16 models)
    MPS/CPU    → no mixed precision
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return {"fp16": False, "bf16": True}
        return {"fp16": True, "bf16": False}
    return {"fp16": False, "bf16": False}
