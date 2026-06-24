"""
ExperimentConfig — single source of truth for every experiment knob.

Swap experiments by changing fields or using the named presets at the bottom.
"""

from dataclasses import dataclass, field
import os


@dataclass
class ExperimentConfig:

    # ── Encoder ───────────────────────────────────────────────────────────────
    # Any HuggingFace model name or local checkpoint path.
    # Key presets:
    #   "xlm-roberta-base"             — baseline, no Cree-specific fine-tuning
    #   "data/awesome_align/model"     — your awesome-align Cree-English checkpoint
    #   "<hf-user>/cree-en-awesome-align"  — HF-hosted version of the above
    encoder: str = "xlm-roberta-base"

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset: str = "CreativeLang/vua20_metaphor"

    # Only compute loss on content-word tokens (VERB, NOUN, ADJ, ADV).
    # Non-content tokens are masked with -100 and don't affect gradients.
    content_words_only: bool = False

    # Explicit POS allowlist — overrides content_words_only if non-empty.
    # e.g. ["VERB", "NOUN", "ADJ", "ADV"]
    pos_filter: list[str] = field(default_factory=list)

    # ── Training ──────────────────────────────────────────────────────────────
    max_length: int = 128
    batch_size: int = 16
    grad_accum: int = 2               # effective batch = batch_size * grad_accum
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    # Up-weight the minority (metaphor) class to counter the ~10% imbalance.
    class_weights: bool = True

    # ── Inference ─────────────────────────────────────────────────────────────
    # Batch size used during predict() — can be larger than train batch_size.
    infer_batch_size: int = 32

    # ── Encoder training ──────────────────────────────────────────────────────
    freeze_encoder: bool = False    # True = probe only (classifier head trains, encoder frozen)

    # ── Encoder hidden-layer selection ────────────────────────────────────────
    # None  → use last_hidden_state (standard).
    # int n → use hidden_states[n] (e.g. 12 for the 12th transformer layer).
    hidden_layer: int | None = None

    # ── Logging & Hub ─────────────────────────────────────────────────────────
    wandb_project:  str | None = None   # e.g. "fnlp-metaphor"; None disables wandb
    hub_model_id:   str | None = None   # e.g. "KonradBRG/model-name"; None skips push

    # ── Output ────────────────────────────────────────────────────────────────
    experiment_name: str = "base_xlmr"
    output_root: str = "data/metaphor"

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.output_root, self.experiment_name)

    @property
    def active_pos_filter(self) -> set[str]:
        if self.pos_filter:
            return set(self.pos_filter)
        if self.content_words_only:
            return {"VERB", "NOUN", "ADJ", "ADV"}
        return set()


# ── Named experiment presets ──────────────────────────────────────────────────
# Use these as starting points and override individual fields as needed.

def baseline() -> ExperimentConfig:
    """Base XLM-R fine-tuned on VUA20 — the control condition."""
    return ExperimentConfig(
        encoder="xlm-roberta-base",
        experiment_name="baseline_xlmr",
    )


def awesome_align_encoder() -> ExperimentConfig:
    """Awesome-align Cree-English checkpoint fine-tuned on VUA20."""
    return ExperimentConfig(
        encoder="data/awesome_align/model",
        experiment_name="awesome_align_encoder",
    )


def content_words_only() -> ExperimentConfig:
    """Base XLM-R, loss only on content words — mirrors MIPVU scope."""
    return ExperimentConfig(
        encoder="xlm-roberta-base",
        content_words_only=True,
        experiment_name="baseline_xlmr_content_only",
    )


def awesome_align_content_words() -> ExperimentConfig:
    """Awesome-align encoder + content-words-only loss."""
    return ExperimentConfig(
        encoder="data/awesome_align/model",
        content_words_only=True,
        experiment_name="awesome_align_content_only",
    )


def tlm_last_layer() -> ExperimentConfig:
    """TLM-adapted XLM-R fine-tuned on VUA20, using the final hidden layer."""
    return ExperimentConfig(
        encoder="KonradBRG/xlm-r-plains-cree-en-tlm",
        hidden_layer=None,
        experiment_name="tlm_last_layer",
        wandb_project="fnlp-metaphor",
        hub_model_id="KonradBRG/xlm-r-plains-cree-en-tlm-metaphor-last",
    )


def tlm_layer_12() -> ExperimentConfig:
    """TLM-adapted XLM-R fine-tuned on VUA20, using hidden_states[12].

    For XLM-R base (12 transformer layers) this equals last_hidden_state.
    For XLM-R large (24 layers) this is the mid-network layer that several
    metaphor-detection papers find more effective.
    """
    return ExperimentConfig(
        encoder="KonradBRG/xlm-r-plains-cree-en-tlm",
        hidden_layer=12,
        experiment_name="tlm_layer_12",
        wandb_project="fnlp-metaphor",
        hub_model_id="KonradBRG/xlm-r-plains-cree-en-tlm-metaphor-layer12",
    )
