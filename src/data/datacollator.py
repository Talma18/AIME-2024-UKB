from dataclasses import dataclass
from typing import List, Union, Dict, Optional, Any, Tuple
from src import Tokenizer

import torch


@dataclass
class DataCollator:
    tokenizer: Tokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_id: bool = False

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling."
            )

    def torch_mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=inputs.dtype)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels, indices_random | indices_replaced

    def collate_batch(self, examples: List[Union[List[int], Dict[str, Any]]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)

        # organize model inputs by modality
        model_inputs = {
            modality: {} for modality in self.tokenizer.modalities
        }

        for modality in self.tokenizer.modalities:
            model_inputs[modality]["input_ids"] = batch[f"{modality}_input_ids"]
            model_inputs[modality]["token_type_ids"] = batch[f"{modality}_token_type_ids"]
            model_inputs[modality]["attention_mask"] = batch[f"{modality}_attention_mask"]
            model_inputs[modality]["measurement_mask"] = batch[f"{modality}_measurement_mask"]
            model_inputs[modality]["float_loss_mask"] = model_inputs[modality]["measurement_mask"].bool()
            model_inputs[modality]["year_ids"] = batch[f"{modality}_year_ids"]
            model_inputs[modality]["month_ids"] = batch[f"{modality}_month_ids"]
            model_inputs[modality]["day_ids"] = batch[f"{modality}_day_ids"]

            if self.mlm:
                model_inputs[modality]["input_ids"], model_inputs[modality]["labels"], modified_mask = self.torch_mask_tokens(
                    model_inputs[modality]["input_ids"],
                )
                model_inputs[modality]["measurement_mask"] = model_inputs[modality]["measurement_mask"] & ~modified_mask

            else:
                labels = model_inputs[modality]["input_ids"].clone()
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                model_inputs[modality]["labels"] = labels

        if self.return_id:
            model_inputs['eid'] = batch["eid"]

        return model_inputs

    def __call__(self, *args, **kwargs):
        return self.collate_batch(*args, **kwargs)

