import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from src import Tokenizer, load_dataset
from src.models.modeling_mixedtab import MixedTabForMaskedLM
from src.data import DataCollator

model_path = 'saved_models/2024-01-22_21-17-41/checkpoint_127491'
device = 'cuda:0'

tokenizer = Tokenizer.from_pretrained('pretrained_tokenizers/tokenizer/')

model = MixedTabForMaskedLM.from_pretrained(model_path).to(device)

dataset = load_dataset('data/processed/dataset', tokenizer)["train"]

collator = DataCollator(
    tokenizer,
    mlm=False,
    return_id=True,
)

dataloader = DataLoader(
    # dataset=dataset.select(range(1000)),
    dataset=dataset,
    batch_size=128,
    pin_memory=True,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    collate_fn=collator,
)

model = model.eval()

representations = []
eid = []

with torch.no_grad():
    for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
        eid.append(inputs.pop('eid').numpy())

        inputs = {mod: {k: v.to(device) for k, v in mod_input.items()} for mod, mod_input in inputs.items()}

        with torch.cuda.amp.autocast():
            output, _ = model(**inputs, output_hidden_states=True, return_encoder_outputs=True, output_attentions=True)


            # save embedding of first token as representation
            last_hidden_state = output.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
            representation = last_hidden_state[:, 0, :]
            representations.append(representation.cpu().numpy())


eid = np.concatenate(eid, axis=0)

representations = np.concatenate(representations, axis=0)

# save representations as parquet
df = pd.DataFrame(representations, index=eid, columns=[f'emb_{i}' for i in range(representations.shape[1])])
df.index.name = 'eid'

df.to_parquet(f'{model_path}/representations.parquet')
