from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
    RobertaIntermediate,
    RobertaOutput,
)
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
)

from .attention import UCADAttention
from .pooling import AttentionPooling, MeanPooling
from transformers.utils import ModelOutput


class UCADLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = UCADAttention(config)

        self.use_cross_attention = config.use_encoder
        self.pool_self = config.pool_self

        if config.pool_method == "mean":
            self.pooler = MeanPooling()

        elif config.pool_method == "attention":
            self.pooler = AttentionPooling(config.hidden_size)

        if self.use_cross_attention:
            self.crossattention_drug = UCADAttention(
                config, position_embedding_type="absolute"
            )
            self.crossattention_lab = UCADAttention(
                config, position_embedding_type="absolute"
            )
            self.crossattention_personal = UCADAttention(
                config, position_embedding_type="absolute"
            )

        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        drug_hidden_states: Optional[torch.FloatTensor] = None,
        lab_hidden_states: Optional[torch.FloatTensor] = None,
        personal_hidden_states: Optional[torch.FloatTensor] = None,
        drug_attention_mask: Optional[torch.FloatTensor] = None,
        lab_attention_mask: Optional[torch.FloatTensor] = None,
        personal_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> ModelOutput:

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        if self.use_cross_attention:
            cross_attention_outputs_drug = self.crossattention_drug(
                attention_output,
                attention_mask,
                head_mask,
                drug_hidden_states,
                drug_attention_mask,
                output_attentions,
            )
            cross_output_drug = cross_attention_outputs_drug[0]

            cross_attention_outputs_lab = self.crossattention_lab(
                attention_output,
                attention_mask,
                head_mask,
                lab_hidden_states,
                lab_attention_mask,
                output_attentions,
            )
            cross_output_lab = cross_attention_outputs_lab[0]

            cross_attention_outputs_personal = self.crossattention_personal(
                attention_output,
                attention_mask,
                head_mask,
                personal_hidden_states,
                personal_attention_mask,
                output_attentions,
            )
            cross_output_personal = cross_attention_outputs_personal[0]

            output_tensors = [
                cross_output_drug,
                cross_output_lab,
                cross_output_personal,
            ]

            if self.pool_self:
                output_tensors.append(attention_output)

            attention_output, pooler_attentions = self.pooler(torch.stack(output_tensors))

        # Feed-forward layers
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

        return ModelOutput(
            last_hidden_state=layer_output,
            attentions=self_attention_outputs[1:],
            pooler_attentions=pooler_attentions,
            personal_cross_attentions=cross_attention_outputs_personal[1:],
            drug_cross_attentions=cross_attention_outputs_drug[1:],
            lab_cross_attentions=cross_attention_outputs_lab[1:],
        )

    def feed_forward_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output