"""
Token classifier that can draw from any hidden layer of XLM-R.

When config.hidden_layer is None the behaviour is identical to the standard
XLMRobertaForTokenClassification.  When set to an int n, the n-th entry in
the hidden_states tuple is used instead of last_hidden_state.

Hidden-states indexing for XLM-R base (12 transformer layers):
  hidden_states[0]      → embedding layer output
  hidden_states[1..12]  → transformer layers 1-12
  hidden_states[12]     → == last_hidden_state  (final transformer layer)

For XLM-R large (24 layers) hidden_states[12] is the middle layer — the
setting that several metaphor-detection papers favour.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class XLMRobertaLayerSelectForTokenClassification(XLMRobertaPreTrainedModel):
    """XLM-R token classifier with configurable hidden-layer selection."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta    = XLMRobertaModel(config, add_pooling_layer=False)

        dropout_prob    = getattr(config, "classifier_dropout", None) or config.hidden_dropout_prob
        self.dropout    = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        hidden_layer = getattr(self.config, "hidden_layer", None)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(hidden_layer is not None),
            **kwargs,
        )

        sequence_output = (
            outputs.hidden_states[hidden_layer]
            if hidden_layer is not None
            else outputs.last_hidden_state
        )
        sequence_output = self.dropout(sequence_output)
        logits          = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
