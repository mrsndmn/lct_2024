import random
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader

from dataclasses import dataclass
from transformers import DefaultDataCollator

from datasets import Dataset

from tqdm.auto import tqdm
import wandb
from wandb import sdk as wandb_sdk

import os

from avm.models.image import get_default_image_model

from scripts.train.metric_learning import clip_loss

@dataclass
class TrainingConfig():
    # Head Training
    freeze_skeleton = True
    batch_size = 10
    learning_rate = 1e-4

    model_checkpoints_path = 'data/models/image/efficient-net-b0'
    save_and_evaluate_model_every_epoch = 1

    num_epochs = 30
    multiply_train_epoch_data = 10

    few_dataset_samples = None


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

    return


def train(config: TrainingConfig, metric_logger: wandb_sdk.wandb_run.Run):
    # Dataloaders
    training_dataset = Dataset.load_from_disk(config.training_dataset_path)

    existing_audio_files = set(os.listdir(config.audio_base_path))

    training_dataset = training_dataset.filter(lambda x: x['file_name'] in existing_audio_files)
    training_dataset = training_dataset.map(lambda x: {"audio": config.audio_base_path + '/' + x['file_name']})
    training_dataset = training_dataset.remove_columns([x for x in training_dataset.column_names if x != 'audio'])

    if config.few_dataset_samples is not None:
        training_dataset = training_dataset.select(range(config.few_dataset_samples))

    print("training_dataset", training_dataset)

    default_data_collator = DefaultDataCollator()
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=default_data_collator,
        num_workers=2,
    )

    # Model and optimizer
    model = get_default_image_model()
    if config.freeze_skeleton:
        freeze_model(model.backbone)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    print("trainable model params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch_i in range(config.num_epochs):
        model.train()

        for dataloader_multiplicator in range(config.multiply_train_epoch_data):
            pbar = tqdm(training_dataloader)
            for batch in pbar:

                model_output = model(batch['input_values'])
                print("model_output", model_output.shape)

                model_output = model_output / (model_output.norm(dim=-1, keepdim=True) + 1e-6)
                if model_output.isnan().any().item():
                    raise Exception("model_output is nan!")

                x_vectors_1, x_vectors_2 = torch.chunk(model_output, 2, dim=0)

                logit_scale = model.logit_scale.exp()
                if logit_scale.isnan().item():
                    raise Exception("logit_scale is nan!")

                logits_per_1 = torch.matmul(x_vectors_1, x_vectors_2.t()) * logit_scale
                if logits_per_1.isnan().any().item():
                    raise Exception("logits_per_1 is nan!")

                loss = clip_loss(logits_per_1)
                pbar.set_description(f"Epoch={epoch_i}({dataloader_multiplicator}); Loss={loss.item():.3f}")

                model.zero_grad()
                loss.backward()
                if loss.isnan().item():
                    raise Exception("loss is nan!")
                
                metric_logger.log({
                    "loss": loss.item(),
                })

                optimizer.step()

        if epoch_i % config.save_and_evaluate_model_every_epoch == 0:
            model_checkpoint_path = os.path.join(config.model_checkpoints_path, metric_logger.name)
            model.save_pretrained(model_checkpoint_path)

            model_checkpoint_path = os.path.join(config.model_checkpoints_path, metric_logger.name)
            print("save model to", model_checkpoint_path)
            model.save_pretrained(model_checkpoint_path)

    return


if __name__ == '__main__':

    config = TrainingConfig()

    from DPF.configs import FilesDatasetConfig
    from DPF.dataset_reader import DatasetReader
    from DPF.pipelines import FilterPipeline
    from DPF.filters.images.info_filter import ImageInfoFilter
    from DPF.filters.images.hash_filters import PHashFilter
    from DPF.transforms import ImageResizeTransforms, Resizer, ResizerModes

    reader = DatasetReader()
    config = FilesDatasetConfig.from_path_and_columns(
        "data/rutube/videos/test_videos",
        video_path_col='file_id',
    )
    processor = reader.read_from_config(config, workers=4)

    transforms = ImageResizeTransforms(Resizer(ResizerModes.MIN_SIZE, size=768))
    processor.apply_transform(transforms)

    with wandb.init(project="lct-avm-image") as metric_logger:
        train(
            config=config,
            metric_logger=metric_logger,
        )
