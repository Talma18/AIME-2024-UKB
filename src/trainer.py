import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data import DataCollator, load_dataset
from src.tokenizer import Tokenizer
from src.utils import get_current_datetime_string, get_parameter_names
from tqdm.auto import tqdm

# from src.eval import get_accuracies, get_binary_metrics, get_joint_accuracies
# from src.models.standard.modeling_standard import HierarchicalStandardModelForMaskedLM

# from src.utils.metrics_tracker import MetricsTracker


@dataclass
class TrainingArguments:
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    learning_rate: float = 3e-4
    logging_steps: int = 500
    warmup_ratio: float = 0.1
    warmup_steps: int = 4000
    num_train_epochs: int = 10
    weight_decay: float = 0.0
    mlm: bool = True
    mlm_probability: float = 0.15
    batch_size: int = 128
    device: str = "cuda:0"
    save_path: str = "saved_models"
    tokenizer_path: str = "tokenizer"
    dataset_path: str = "dataset"
    wandb_mode: str = "online"
    wandb_name: Optional[str] = None
    clip_gradients: bool = True


class Trainer:
    def __init__(self, model, tokenizer, args: TrainingArguments):
        self.model = model.to(args.device)
        self.tokenizer = tokenizer
        self.args = args
        self.scaler = torch.cuda.amp.GradScaler()

        if args.clip_gradients:
            print("Clipping gradients.")
        else:
            print("Not clipping gradients.")

        self.lr_scheduler = None
        self.optimizer = None
        self.test_loader = None
        self.train_loader = None
        self.test_dataset = None
        self.train_dataset = None
        self.display = None
        self.all_steps = None
        self.save_path = None
        self.collator = None
        self.run_path = None
        self.train_pbar = None

        self.post_init()

    def post_init(self):
        """
        Initialize the trainer after the models and tokenizer have been loaded.
        :return:
        """
        self.args.tokenizer_path = self.tokenizer.name_or_path
        dataset = load_dataset(self.args.dataset_path, self.tokenizer)

        self.collator = self.create_data_collator()

        self.train_dataset, self.test_dataset = dataset["train"], dataset["test"]

        max_disease = max(map(len, self.train_dataset['disease_input_ids']))
        max_drug = max(map(len, self.train_dataset['drug_input_ids']))
        max_lab = max(map(len, self.train_dataset['lab_input_ids']))

        print('max_disease', max_disease)
        print('max_drug', max_drug)
        print('max_lab', max_lab)

        print('len', len(self.train_dataset))

        self.train_loader = self.create_data_loader(self.train_dataset, shuffle=True)
        self.test_loader = self.create_data_loader(self.test_dataset)

        self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler()

        self.run_path = self.create_folder()

        full_config = {}
        full_config.update(self.args.__dict__)
        full_config.update(self.model.config.to_dict())
        full_config["save_path"] = self.run_path

        wandb.init(
            project="MyProject",
            entity="MyName",
            config=full_config,
            mode=self.args.wandb_mode,
            name=self.args.wandb_name,
        )

        wandb.watch(self.model, log_freq=100, log_graph=True)

    def create_folder(self):
        """
        Create a folder for saving the models and the training arguments.
        :return:
        """
        self.save_path = Path(self.args.save_path) / get_current_datetime_string()
        self.save_path.mkdir(exist_ok=True, parents=True)
        # (self.save_path / "models.json").write_text(json.dumps(self.model.config.to_dict()))
        (self.save_path / "args.json").write_text(json.dumps(asdict(self.args)))

        # return save path as str
        return str(self.save_path)

    def create_data_loader(self, dataset, shuffle=False) -> DataLoader:
        """
        Create a data loader for training or evaluation.
        :param shuffle:
        :param dataset:
        :return: The data loader
        """
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            pin_memory=self.args.dataloader_pin_memory,
            num_workers=self.args.dataloader_num_workers,
            shuffle=shuffle,
            drop_last=self.args.dataloader_drop_last,
            collate_fn=self.collator,
        )

    def create_data_collator(self) -> DataCollator:
        """
        Create a data collator for training.
        :return: The data collator
        """

        return DataCollator(
            self.tokenizer,
            mlm=self.args.mlm,
            mlm_probability=self.args.mlm_probability,
        )

    def create_optimizer_and_scheduler(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Create the optimizer and learning rate scheduler for training.
        :return: The optimizer and learning rate scheduler
        """
        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, (nn.LayerNorm,))
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optim = torch.optim.AdamW(
            optimizer_grouped_parameters,
            **{
                "lr": self.args.learning_rate,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            },
        )

        self.all_steps = len(self.train_loader) * self.args.num_train_epochs
        # schedule = get_linear_schedule_with_warmup(optim, self.all_steps * self.args.warmup_ratio, self.all_steps)
        schedule = get_linear_schedule_with_warmup(optim, self.args.warmup_steps, self.all_steps)
        return optim, schedule

    def train_step(self, idx: int, inputs: Dict[str, torch.Tensor]):
        """
        Perform a single training step.
        :param idx: The index of the current step
        :param inputs: The inputs to the models
        :return:
        """
        inputs = {mod: {k: v.to(self.args.device) for k, v in mod_input.items()} for mod, mod_input in inputs.items()}

        with torch.cuda.amp.autocast():
            outputs = self.model(**inputs, output_hidden_states=True, return_encoder_outputs=True)
            loss = outputs[0].total_loss
            losses = {
                "total_loss": loss.item(),
                "loss": outputs[0].loss.item(),
                "cce_loss": outputs[0].cce_loss.item(),
                "mse_loss": outputs[0].mse_loss.item(),
            }

            for modality in outputs[1]:
                losses[f'{modality}_loss'] = outputs[1][modality].loss.item()
                losses[f'{modality}_cce_loss'] = outputs[1][modality].cce_loss.item()
                losses[f'{modality}_mse_loss'] = outputs[1][modality].mse_loss.item()
                losses[f'{modality}_std_mean'] = outputs[1][modality].hidden_states[-1].std(dim=1).mean().item()

            wandb.log(losses, step=idx)

        # scaling the loss
        self.scaler.scale(loss).backward()

    def train_epoch(self, epoch: int = 0):
        """
        Train the models for a single epoch.
        :param epoch: The current epoch
        :return:
        """
        self.model.train()
        for idx, inputs in enumerate(self.train_loader):
            self.train_step(epoch * len(self.train_loader) + idx, inputs)

            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            if self.args.clip_gradients:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            scale_before = self.scaler.get_scale()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            scale_after = self.scaler.get_scale()
            optimizer_was_run = scale_before <= scale_after
            if optimizer_was_run:
                self.lr_scheduler.step()

            self.optimizer.zero_grad()
            self.train_pbar.update(1)

    def train(self):
        """
        Train the models for the specified number of epochs.
        :return:
        """
        self.train_pbar = tqdm(total=self.all_steps, desc="Training")

        for epoch in range(self.args.num_train_epochs):
            self.train_epoch(epoch)
            # TODO self.evaluate(epoch)
            self.save((epoch + 1) * len(self.train_loader))

    def save(self, idx: int):
        """
        Save the models and the display.
        :param idx: The index of the current step
        :return:
        """
        self.model.save_pretrained(str(self.save_path / f"checkpoint_{idx + 1}"))
