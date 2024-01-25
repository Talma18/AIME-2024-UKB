from typing import Optional

import torch
import torch.nn as nn


def create_position_ids_from_input_ids(input_ids: torch.LongTensor, padding_idx: int) -> torch.LongTensor:
    """Create position ids on-the-fly using the input_ids tensor"""
    mask = input_ids.ne(padding_idx).int()
    incremental_ids = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_ids.long() + padding_idx


class DateEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.year_embeddings = nn.Embedding(config.max_year, config.hidden_size)
        self.month_embeddings = nn.Embedding(13, config.hidden_size)
        self.day_embeddings = nn.Embedding(32, config.hidden_size)

    def forward(self, year: torch.LongTensor, month: torch.LongTensor, day: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass of the DateEmbeddings layer
        :param year: Tensor containing the year ids
        :param month: Tensor containing the month ids
        :param day: Tensor containing the day ids
        :return: Tensor containing the embeddings
        """
        year_embeddings = self.year_embeddings(year)
        month_embeddings = self.month_embeddings(month)
        day_embeddings = self.day_embeddings(day)
        return year_embeddings + month_embeddings + day_embeddings


class MixedTabEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id

        # Regular word embeddings for tokens
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)

        # Linear layer to convert floats to embeddings
        self.measurement_embeddings = nn.Linear(1, config.hidden_size)

        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # Date embeddings
        self.use_temporal = config.use_temporal  # Flag to use date embeddings
        if self.use_temporal:
            self.date_embeddings = DateEmbeddings(config)

        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids: torch.LongTensor,
            measurement_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            year_ids: Optional[torch.LongTensor] = None,  # New
            month_ids: Optional[torch.LongTensor] = None,  # New
            day_ids: Optional[torch.LongTensor] = None,  # New
    ) -> torch.FloatTensor:

        input_shape = input_ids.size()

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        if measurement_mask is None:
            measurement_mask = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)

        # Generate embeddings for tokens and floats
        # we need to convert input_ids to int because it is a float tensor
        tokens = (input_ids * ~measurement_mask.bool())
        tokens = tokens.long()
        token_embeds = self.word_embeddings(tokens)
        # unsqueeze to add a dimension for the float embeddings
        measurement_mask = measurement_mask.unsqueeze(-1)
        measurement_embeds = self.measurement_embeddings(input_ids.unsqueeze(-1) * measurement_mask.float())

        # Use the float_mask to combine the two embeddings
        inputs_embeds = ~measurement_mask * token_embeds + measurement_mask * measurement_embeds

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        # Add date embeddings if temporal information is used
        if self.use_temporal:
            if year_ids is None or month_ids is None or day_ids is None:
                raise ValueError("Year, month, and day ids must be provided when using temporal embeddings")

            date_embeddings = self.date_embeddings(year_ids, month_ids, day_ids)
            embeddings += date_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
