"""
Two-head hierarchical figurative classifier: a binary head decides
literal-vs-figurative, and a conditional 3-way head (idiom/metaphor/simile)
is trained only on the figurative subset — so the type head's gradient isn't
diluted by the ~18:1 literal majority the way a single 4-way softmax's is.

Combines both heads into proper joint probabilities (P(literal) = 1 -
P(figurative); P(type_k) = P(figurative) * P(type_k | figurative)) so it's a
drop-in replacement for AutoModelForSequenceClassification wherever
downstream code just consumes `.logits` (predict_sentences, compute_metrics):
softmax(these log-probabilities) recovers the probabilities exactly, and
argmax reproduces the cascade decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# Inlined rather than imported from src.figurative.data: this file is pushed
# to the Hub via register_for_auto_class() below, so it must be import-only
# self-contained for anyone loading it with trust_remote_code=True, without
# our package installed.
LITERAL = 0  # label index — must match src.figurative.data.LITERAL
NUM_TYPE_LABELS = 3  # idiom, metaphor, simile — LABEL_NAMES[1:]


class HierarchicalFigurativeConfig(PretrainedConfig):
    model_type = "hierarchical_figurative"

    def __init__(self, base_checkpoint: str | None = None, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.base_checkpoint = base_checkpoint
        self.dropout = dropout


class HierarchicalFigurativeModel(PreTrainedModel):
    """Binary (literal/figurative) head + conditional 3-way (idiom/metaphor/
    simile) head on a shared mean-pooled encoder representation."""

    config_class = HierarchicalFigurativeConfig
    base_model_prefix = "encoder"

    def __init__(self, config: HierarchicalFigurativeConfig):
        super().__init__(config)
        # Built as an empty shell via from_config, NOT from_pretrained: this
        # ctor also runs when PreTrainedModel.from_pretrained() reconstructs
        # the model under a meta-device context to reload a saved checkpoint
        # (see push_best_silver_sft.py) — calling from_pretrained() again in
        # here trips transformers' "from_pretrained nested under a meta
        # context" anti-pattern check. Real pretrained encoder weights for a
        # *fresh* run are loaded separately, by from_base_checkpoint() below.
        # torch_dtype is pinned to fp32 regardless of path: some base
        # checkpoints (e.g. the raw, non-TLM-adapted FacebookAI/xlm-mlm-100-1280)
        # load natively in fp16, while binary_head/type_head default to fp32 —
        # mixing the two crashes MPS matmul with a dtype-mismatch assertion.
        encoder_config = AutoConfig.from_pretrained(config.base_checkpoint)
        self.encoder = AutoModel.from_config(encoder_config, torch_dtype=torch.float32)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.binary_head = nn.Linear(hidden_size, 1)
        self.type_head = nn.Linear(hidden_size, NUM_TYPE_LABELS)
        self.post_init()

    @classmethod
    def from_base_checkpoint(cls, base_checkpoint: str, dropout: float = 0.1) -> "HierarchicalFigurativeModel":
        """Build a fresh model for training, with the encoder's real pretrained
        weights loaded in — use this (not the bare constructor) whenever
        starting from a base checkpoint rather than reloading an already-saved
        HierarchicalFigurativeModel (for which plain .from_pretrained() is
        correct and sufficient)."""
        config = HierarchicalFigurativeConfig(base_checkpoint=base_checkpoint, dropout=dropout)
        model = cls(config)
        pretrained_encoder = AutoModel.from_pretrained(base_checkpoint, torch_dtype=torch.float32)
        model.encoder.load_state_dict(pretrained_encoder.state_dict())
        return model

    def _init_weights(self, module):
        pass  # encoder arrives pretrained; heads keep nn.Linear's default init

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    @staticmethod
    def _masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _combined_log_probs(self, binary_logit: torch.Tensor, type_logits: torch.Tensor) -> torch.Tensor:
        log_p_literal  = F.logsigmoid(-binary_logit)
        log_p_fig      = F.logsigmoid(binary_logit)
        log_type_probs = F.log_softmax(type_logits, dim=-1)
        log_p_types    = log_p_fig.unsqueeze(-1) + log_type_probs
        return torch.cat([log_p_literal.unsqueeze(-1), log_p_types], dim=-1)

    def _loss(self, binary_logit: torch.Tensor, type_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        is_figurative = (labels != LITERAL).float()
        binary_loss = F.binary_cross_entropy_with_logits(binary_logit, is_figurative)

        fig_mask = labels != LITERAL
        if fig_mask.any():
            type_labels = (labels[fig_mask] - 1).long()  # {idiom=1,metaphor=2,simile=3} -> {0,1,2}
            type_loss = F.cross_entropy(type_logits[fig_mask], type_labels)
        else:
            type_loss = binary_logit.new_zeros(())
        return binary_loss + type_loss

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs) -> SequenceClassifierOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self._masked_mean_pool(outputs.last_hidden_state, attention_mask))

        binary_logit = self.binary_head(pooled).squeeze(-1)
        type_logits  = self.type_head(pooled)

        loss = self._loss(binary_logit, type_logits, labels) if labels is not None else None
        logits = self._combined_log_probs(binary_logit, type_logits)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# Registers config/model with the "custom code on the Hub" mechanism: adds
# an `auto_map` entry to config.json and copies this file into the repo on
# push_to_hub()/save_pretrained(), so anyone can load the checkpoint with
# AutoModel.from_pretrained(repo_id, trust_remote_code=True) — no dependency
# on this package. Registering locally does NOT make that possible on its
# own; it only makes AutoModel work within a process that already imported
# this module (e.g. our own eval scripts).
HierarchicalFigurativeConfig.register_for_auto_class()
HierarchicalFigurativeModel.register_for_auto_class("AutoModelForSequenceClassification")
