from src import Trainer, TrainingArguments,Tokenizer
from src.models.modeling_mixedtab import MixedTabForMaskedLM
from src.models.configuration_bioberta import BioBertaConfig

tokenizer = Tokenizer.from_pretrained('pretrained_tokenizers/tokenizer/')

model_config = BioBertaConfig(
    num_hidden_layers=2,
    hidden_size=312,
    intermediate_size=1200,
    vocab_size=len(tokenizer.encoder),
    type_vocab_size=len(tokenizer.type_encoder),
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    mask_token_id=tokenizer.mask_token_id,
    max_position_embeddings=2048,
    pool_method='mean',
    pool_self=True,
)

model = MixedTabForMaskedLM(model_config)
print('num_params', sum(p.numel() for p in model.parameters()))

args = TrainingArguments(
    tokenizer_path='pretrained_tokenizers/tokenizer',
    dataset_path='data/processed/dataset',
    wandb_mode='online',
    wandb_name='DiseaseOnlyMixedTab',
    batch_size=32,
)

trainer = Trainer(model, tokenizer, args)

trainer.train()
