from typing import Optional

import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
)

from transformers.utils import ModelOutput
from src.models.embeddings import MixedTabEmbeddings

from .ucad import UCADLayer


class UCADEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [UCADLayer(config) for _ in range(config.num_hidden_layers)]
        )

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
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> ModelOutput:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_pooler_attentions = () if output_attentions else None
        all_personal_cross_attentions = () if output_attentions else None
        all_drug_cross_attentions = () if output_attentions else None
        all_lab_cross_attentions = () if output_attentions else None


        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Pass all the necessary parameters to the UCADLayer
            layer_output: ModelOutput = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                drug_hidden_states=drug_hidden_states,
                lab_hidden_states=lab_hidden_states,
                personal_hidden_states=personal_hidden_states,
                drug_attention_mask=drug_attention_mask,
                lab_attention_mask=lab_attention_mask,
                personal_attention_mask=personal_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_output[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_output.attentions,)
                all_pooler_attentions = all_pooler_attentions + (layer_output.pooler_attentions,)
                if self.config.add_cross_attention:
                    all_personal_cross_attentions = all_personal_cross_attentions + (layer_output.personal_cross_attentions,)
                    all_drug_cross_attentions = all_drug_cross_attentions + (layer_output.drug_cross_attentions,)
                    all_lab_cross_attentions = all_lab_cross_attentions + (layer_output.lab_cross_attentions,)

        return ModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            pooler_attentions=all_pooler_attentions,
            personal_cross_attentions=all_personal_cross_attentions,
            drug_cross_attentions=all_drug_cross_attentions,
            lab_cross_attentions=all_lab_cross_attentions,
        )


class UCADModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = MixedTabEmbeddings(config)
        self.encoder = UCADEncoder(config)

    def forward(
        self,
        # Main modalities inputs
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        measurement_mask: Optional[torch.FloatTensor] = None,
        year_ids: Optional[torch.LongTensor] = None,
        month_ids: Optional[torch.LongTensor] = None,
        day_ids: Optional[torch.LongTensor] = None,
        # Secondary modalities inputs
        drug_hidden_states: Optional[torch.FloatTensor] = None,
        lab_hidden_states: Optional[torch.FloatTensor] = None,
        personal_hidden_states: Optional[torch.FloatTensor] = None,
        drug_attention_mask: Optional[torch.FloatTensor] = None,
        lab_attention_mask: Optional[torch.FloatTensor] = None,
        personal_attention_mask: Optional[torch.FloatTensor] = None,
        # Other parameters
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,

    ) -> ModelOutput:
        hidden_states = self.embeddings(
            input_ids=input_ids,
            measurement_mask=measurement_mask,
            token_type_ids=token_type_ids,
            year_ids=year_ids,
            month_ids=month_ids,
            day_ids=day_ids,
        )
        input_shape = input_ids.size()

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        extended_drug_attention_mask = self.get_extended_attention_mask(drug_attention_mask, input_shape)
        extended_lab_attention_mask = self.get_extended_attention_mask(lab_attention_mask, input_shape)
        extended_personal_attention_mask = self.get_extended_attention_mask(personal_attention_mask, input_shape)

        result = self.encoder(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
            drug_hidden_states=drug_hidden_states,
            lab_hidden_states=lab_hidden_states,
            personal_hidden_states=personal_hidden_states,
            drug_attention_mask=extended_drug_attention_mask,
            lab_attention_mask=extended_lab_attention_mask,
            personal_attention_mask=extended_personal_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return result
