import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from src import BioBERTaForMaskedLM, BioBertaConfig, Tokenizer, load_dataset
from src.data import DataCollator

model_path = 'saved_models/2024-01-23_13-47-36/checkpoint_157001'
device = 'cuda:0'

tokenizer = Tokenizer.from_pretrained('pretrained_tokenizers/tokenizer/')

print('loading model', model_path)
model = BioBERTaForMaskedLM.from_pretrained(model_path).to(device)

dataset = load_dataset('data/processed/dataset', tokenizer)["train"]

collator = DataCollator(
    tokenizer,
    mlm=False,
    return_id=True,
)

dataloader = DataLoader(
    # dataset=dataset.select(range(1000)),
    dataset=dataset,
    batch_size=32,
    pin_memory=True,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    collate_fn=collator,
)

model = model.eval()

representations = {modality: [] for modality in tokenizer.modalities}
pooler_attentions = {layer: [] for layer in range(model.config.num_hidden_layers)}
eid = []

with torch.no_grad():
    for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
        eid.append(inputs.pop('eid').numpy())

        inputs = {mod: {k: v.to(device) for k, v in mod_input.items()} for mod, mod_input in inputs.items()}

        with torch.cuda.amp.autocast():
            outputs = model(**inputs, output_hidden_states=True, return_encoder_outputs=True, output_attentions=True)

            # outputs is (ModelOutput, dict[str, ModelOutput])
            outputs[1]['disease'] = outputs[0]

            for i, pooler_attention in enumerate(outputs[0].pooler_attentions):
                pooler_attention = pooler_attention.mean(-1).T # [modality, batch_size, seq_len] -> [batch_size, modality]
                pooler_attentions[i].append(pooler_attention.cpu().numpy())

            # save embedding of first token as representation
            for modality in outputs[1]:
                last_hidden_state = outputs[1][modality].hidden_states[-1]  # (batch_size, seq_len, hidden_size)
                representation = last_hidden_state[:, 0, :]
                representations[modality].append(representation.cpu().numpy())


eid = np.concatenate(eid, axis=0)
for modality in representations:
    representations[modality] = np.concatenate(representations[modality], axis=0)

    # save representations as parquet
    df = pd.DataFrame(representations[modality], index=eid, columns=[f'emb_{i}' for i in range(representations[modality].shape[1])])
    df.index.name = 'eid'

    df.to_parquet(f'{model_path}/{modality}_representations.parquet')

for layer in pooler_attentions:
    pooler_attentions[layer] = np.concatenate(pooler_attentions[layer], axis=0)

    # save representations as parquet
    df = pd.DataFrame(pooler_attentions[layer], index=eid, columns=[f'att_{i}' for i in range(pooler_attentions[layer].shape[1])])
    df.index.name = 'eid'

    df.to_parquet(f'{model_path}/pooler_attentions_{layer}.parquet')
