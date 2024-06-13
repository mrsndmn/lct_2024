import random
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader

from dataclasses import dataclass
from transformers import DefaultDataCollator

from datasets import Dataset, Audio

from tqdm.auto import tqdm
import wandb
from wandb import sdk as wandb_sdk

from qdrant_client import models

from scripts.pipeline.pipeline import PipelineConfig, run_pipeline

import os
from scripts.data.generate_audio_augmentations import AudioAugmentator

from avm.models import get_model

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    # print("caption_loss", caption_loss)
    # print("image_loss", image_loss)
    return (caption_loss + image_loss) / 2.0


@dataclass
class TrainingConfig():
    model_name = 'UniSpeechSatForXVector'
    # from_pretrained = 'data/models/UniSpeechSatForXVector_finetuned/fast-night-88/' # файнтюн на данных рутуба
    from_pretrained = None

    # Head Training
    freeze_skeleton = False
    batch_size = 10
    learning_rate = 1e-4
    interval_step = 0.1

    # freeze_skeleton = True
    # batch_size = 50
    # learning_rate = 3e-4

    model_checkpoints_path = 'data/models/UniSpeechSatForXVector_mini_finetuned'
    save_and_evaluate_model_every_epoch = 1

    num_epochs = 30
    multiply_train_epoch_data = 10

    training_dataset_path = 'data/rutube/compressed_index_audios.dataset'
    audio_base_path = 'data/rutube/compressed_index_audios/'
    few_dataset_samples = None

    normalize_audio = True
    interval_duration_in_seconds = 5
    sampling_rate = 16000


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

    return

def random_subsequence(audio_waveforms, n_frames, start_frame=None):
    if start_frame is None:
        start_frame = random.randint(0, audio_waveforms.shape[-1] - n_frames)
    return audio_waveforms[start_frame:start_frame+n_frames]


class NoopAudioAugmentator:
    def apply_random_augmentation(self, waveform):
        return waveform
    
def min_max_normalize(audio_waveform):
    audio_waveform = (audio_waveform - audio_waveform.min(dim=-1, keepdim=True).values) / (audio_waveform.max(dim=-1, keepdim=True).values - audio_waveform.min(dim=-1, keepdim=True).values + 1e-6) * 2 - 1
    return audio_waveform

def train(config: TrainingConfig, metric_logger: wandb_sdk.wandb_run.Run):
    # Dataloaders
    training_dataset = Dataset.load_from_disk(config.training_dataset_path)

    existing_audio_files = set(os.listdir(config.audio_base_path))

    training_dataset = training_dataset.filter(lambda x: x['file_name'] in existing_audio_files)
    training_dataset = training_dataset.map(lambda x: {"audio": config.audio_base_path + '/' + x['file_name']})
    # training_dataset = training_dataset.cast_column('audio', Audio(sampling_rate=config.sampling_rate))
    # training_dataset = training_dataset.filter(lambda x: x['audio']['array'].shape[-1] >= config.sampling_rate * config.interval_duration_in_seconds)
    training_dataset = training_dataset.remove_columns([x for x in training_dataset.column_names if x != 'audio'])

    if config.few_dataset_samples is not None:
        training_dataset = training_dataset.select(range(config.few_dataset_samples))


    print("training_dataset", training_dataset)
    # print("training_dataset[0]", torch.tensor(training_dataset[0]['audio_waveform']).shape)

    audio_augmentator = AudioAugmentator(expected_sample_rate=config.sampling_rate)
    # audio_augmentator = NoopAudioAugmentator()

    default_data_collator = DefaultDataCollator()
    def collate_fn(samples):

        def read_random_audio_section(filename, total_expected_frames):
            track = sf.SoundFile(filename)

            can_seek = track.seekable() # True
            if not can_seek:
                raise ValueError("Not compatible with seeking")
            
            file_frames = track.frames
            start_frame = random.randint(0, file_frames - total_expected_frames)

            sr = track.samplerate
            assert sr == config.sampling_rate, f"{sr} == {config.sampling_rate}"

            track.seek(start_frame)
            audio_section = track.read(total_expected_frames)
            return audio_section

        total_expected_frames = config.sampling_rate * config.interval_duration_in_seconds
        processed_samples = []
        for sample in samples:
            rand_subsequence = read_random_audio_section(sample['audio'], total_expected_frames)
            processed_samples.append({
                "audio_waveform": torch.from_numpy(rand_subsequence)
            })

        return default_data_collator(processed_samples)


    training_dataloader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Model and optimizer
    model, feature_extractor = get_model(config.model_name, config.from_pretrained)
    if config.freeze_skeleton:
        freeze_model(model.unispeech_sat)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    print("trainable model params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch_i in range(config.num_epochs):
        model.train()

        for dataloader_multiplicator in range(config.multiply_train_epoch_data):
            pbar = tqdm(training_dataloader)
            for batch in pbar:
                with torch.no_grad():
                    audio_waveforms = batch['audio_waveform'].to(torch.float32)

                    # print(audio_waveforms.shape)
                    augmented_audio_waveforms = []
                    # augmented_audio_waveforms_1 = torch.empty_like(audio_waveforms)
                    # augmented_audio_waveforms_2 = torch.empty_like(audio_waveforms)
                    for i in range(audio_waveforms.shape[0]):
                        # augmented_audio_waveforms_1[i, :, :] = audio_augmentator.apply_random_augmentation(audio_waveforms[i, :, :])
                        augmented_audio_waveforms_1 = audio_augmentator.apply_random_augmentation(audio_waveforms[i:i+1, :])
                        if config.normalize_audio:
                            augmented_audio_waveforms_1 = min_max_normalize(augmented_audio_waveforms_1)
                        augmented_audio_waveforms.append(augmented_audio_waveforms_1[0].cpu().numpy())

                    for i in range(audio_waveforms.shape[0]):
                        # augmented_audio_waveforms_2[i, :, :] = audio_augmentator.apply_random_augmentation(audio_waveforms[i, :, :])
                        augmented_audio_waveforms_2 = audio_augmentator.apply_random_augmentation(audio_waveforms[i:i+1, :])
                        if config.normalize_audio:
                            augmented_audio_waveforms_2 = min_max_normalize(augmented_audio_waveforms_2)
                        augmented_audio_waveforms.append(augmented_audio_waveforms_2[0].cpu().numpy())


                    # print("augmented_audio_waveforms_1", augmented_audio_waveforms_1.shape)
                    # print("augmented_audio_waveforms_2", augmented_audio_waveforms_2.shape)

                # augmented_audio_waveforms = torch.cat([augmented_audio_waveforms_1, augmented_audio_waveforms_2], dim=0)

                model_inputs = feature_extractor(
                    augmented_audio_waveforms,
                    return_tensors="pt",
                    sampling_rate=config.sampling_rate
                )
                # print("model_inputs", model_inputs)
                model_output = model(
                    input_values=model_inputs['input_values'].to(device)
                )
                model_output = model_output.embeddings

                model_output = model_output / (model_output.norm(dim=-1, keepdim=True) + 1e-6)
                if model_output.isnan().any().item():
                    raise Exception("model_output is nan!")

                # print("model_output", model_output.shape, "min", model_output.min(), "max", model_output.max())
                x_vectors_1, x_vectors_2 = torch.chunk(model_output, 2, dim=0)
                # print("model_inputs['input_values']", model_inputs['input_values'].shape)
                # print("x_vectors_1.shape", x_vectors_1.shape)
                # print("x_vectors_2.shape", x_vectors_2.shape)

                # todo logit scale like in clip?
                # print("model.logit_scale", model.logit_scale.min().item(), model.logit_scale.max().item())
                logit_scale = model.logit_scale.exp()
                if logit_scale.isnan().item():
                    raise Exception("logit_scale is nan!")

                logits_per_1 = torch.matmul(x_vectors_1, x_vectors_2.t()) * logit_scale
                if logits_per_1.isnan().any().item():
                    raise Exception("logits_per_1 is nan!")

                # print("logits_per_1", logits_per_1.shape, "min", logits_per_1.min().item(), "max", logits_per_1.max().item())

                loss = clip_loss(logits_per_1)
                pbar.set_description(f"Epoch={epoch_i}({dataloader_multiplicator}); Loss={loss.item():.3f}")

                model.zero_grad()
                loss.backward()
                if loss.isnan().item():
                    raise Exception("loss is nan!")
                
                metric_logger.log({
                    "loss": loss.item(),
                    "debug/logit_scale": model.logit_scale.item(),
                    "debug/logit_scale_grad": model.logit_scale.grad.item(),
                    "debug/feature_extractor.weight.grad.norm": model.feature_extractor.weight.grad.norm(2),
                    "debug/feature_extractor.bias.grad.norm": model.feature_extractor.bias.grad.norm(2),
                })

                optimizer.step()

        if epoch_i % config.save_and_evaluate_model_every_epoch == 0:
            model_checkpoint_path = os.path.join(config.model_checkpoints_path, metric_logger.name)
            model.save_pretrained(model_checkpoint_path)

            pipeline_config = PipelineConfig(
                    pipeline_dir = 'data/music_caps/pipeline',
                    sampling_rate = 16000,
                    # Intervals Config
                    interval_step = config.interval_step,
                    interval_duration_in_seconds = config.interval_duration_in_seconds,
                    full_interval_duration_in_seconds = 10,  # максимальная длинна заимствованного интервала для валидации
                    # common data config
                    embeddings_normalization = True,
                    audio_normalization = config.normalize_audio,

                    model_name = config.model_name,
                    model_from_pretrained = model_checkpoint_path,
                    # model_name = 'Wav2Vec2ForXVector'

                    # Validation Data Config
                    validation_audios = 'data/music_caps/augmented_audios',
                    few_validation_samples = 10,

                    # Index Data Config
                    index_audios = 'data/music_caps/audios',
                    index_dataset = 'data/music_caps/audios.dataset',
                    few_index_samples = 10,
            )
            metrics = run_pipeline(pipeline_config)
            print("metrics", metrics)
            metric_logger.log(metrics)

            model_checkpoint_path = os.path.join(config.model_checkpoints_path, metric_logger.name)
            print("save model to", model_checkpoint_path)
            model.save_pretrained(model_checkpoint_path)

    return


if __name__ == '__main__':

    config = TrainingConfig()

    with wandb.init(project="lct-avm") as metric_logger:
        train(
            config=config,
            metric_logger=metric_logger,
        )
