"""Experiment configuration for the 3-class figurative language detector."""

from dataclasses import dataclass
import os


@dataclass
class FigurativeConfig:

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder: str = "KonradBRG/xlm-r-plains-cree-en-tlm"

    # ── Training ──────────────────────────────────────────────────────────────
    max_length:    int   = 128
    batch_size:    int   = 32
    grad_accum:    int   = 1
    epochs:        int   = 5
    learning_rate: float = 2e-5
    warmup_ratio:  float = 0.06
    weight_decay:  float = 0.01
    class_weights: bool  = True
    freeze_encoder: bool = False

    # ── Logging & Hub ─────────────────────────────────────────────────────────
    wandb_project: str | None = None
    hub_model_id:  str | None = None

    # ── Output ────────────────────────────────────────────────────────────────
    experiment_name: str = "figurative_3class"
    output_root:     str = "data/figurative"

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.output_root, self.experiment_name)


# ── Named presets ─────────────────────────────────────────────────────────────

def tlm_base() -> FigurativeConfig:
    """TLM-adapted XLM-R base fine-tuned on VUA20+MAGPIE."""
    return FigurativeConfig(
        encoder="KonradBRG/xlm-r-plains-cree-en-tlm",
        experiment_name="tlm_base_figurative",
        wandb_project="fnlp-figurative",
        hub_model_id="KonradBRG/xlm-r-plains-cree-en-tlm-figurative",
    )


def tlm_large() -> FigurativeConfig:
    """TLM-adapted XLM-R large fine-tuned on VUA20+MAGPIE."""
    return FigurativeConfig(
        encoder="KonradBRG/xlm-r-large-plains-cree-en-tlm",
        batch_size=16,
        grad_accum=2,
        experiment_name="tlm_large_figurative",
        wandb_project="fnlp-figurative",
        hub_model_id="KonradBRG/xlm-r-large-plains-cree-en-tlm-figurative",
    )


def baseline() -> FigurativeConfig:
    """Vanilla XLM-R base — control condition, no Cree-specific pre-training."""
    return FigurativeConfig(
        encoder="xlm-roberta-base",
        experiment_name="baseline_figurative",
        wandb_project="fnlp-figurative",
    )
