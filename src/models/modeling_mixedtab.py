from typing import Optional

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import ModelOutput

from src.models.embeddings import MixedTabEmbeddings
from .lm_head import MixedTabLMHead


class MixedTabModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = MixedTabEmbeddings(config)
        self.post_init()  # is this necessary?

    # noinspection PyMethodOverriding
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        measurement_mask: Optional[torch.Tensor] = None,
        year_ids: Optional[torch.Tensor] = None,
        month_ids: Optional[torch.Tensor] = None,
        day_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> ModelOutput:
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=self.device, dtype=torch.long)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            measurement_mask=measurement_mask,
            year_ids=year_ids,
            month_ids=month_ids,
            day_ids=day_ids,

        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]

        return ModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MixedTabForMaskedLM(RobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.decoder = MixedTabModel(config)
        self.lm_head = MixedTabLMHead(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.decoder.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def calculate_loss(self, sequence_output, measurement_mask, labels):
        measurement_mask = measurement_mask.bool()

        prediction_scores, float_preds = self.lm_head(sequence_output)

        float_labels = labels.clone()
        float_mask = measurement_mask & (labels != -100)

        token_labels = labels.clone()
        token_labels[measurement_mask] = -100
        token_labels = token_labels.int()

        # CCE loss for tokens
        token_labels = token_labels.to(prediction_scores.device)
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size),
            token_labels.long().view(-1),
        )

        # MSE loss for floats
        if measurement_mask.any():
            float_labels = float_labels.to(float_preds.device)
            float_loss_fct = MSELoss(reduction="none")
            float_loss = float_loss_fct(float_preds, float_labels) * float_mask
            float_loss = float_loss.sum() / float_mask.sum()
        else:
            float_loss = torch.tensor(0.0, device=float_preds.device)

        return masked_lm_loss, float_loss

    def forward(
        self,
        disease=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> tuple[ModelOutput, dict[str, ModelOutput]]:

        labels = disease.pop("labels")
        float_loss_mask = disease.pop("float_loss_mask")

        decoder_outputs = self.decoder(
            **disease,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = decoder_outputs[0]

        masked_lm_loss, float_loss = self.calculate_loss(sequence_output, float_loss_mask, labels)

        return ModelOutput(
            loss=masked_lm_loss+float_loss,
            cce_loss=masked_lm_loss,
            mse_loss=float_loss,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        ), {}
