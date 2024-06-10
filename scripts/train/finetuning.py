import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader

from dataclasses import dataclass
from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector, Data2VecAudioForXVector, WavLMForXVector, UniSpeechSatForXVector
from transformers import DefaultDataCollator

from datasets import Dataset, Audio

from tqdm.auto import tqdm
import wandb
from wandb import sdk as wandb_sdk

import os

from scripts.data.generate_audio_augmentations import AudioAugmentator

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


@dataclass
class TrainingConfig():
    model_name = 'UniSpeechSatForXVector'

    batch_size = 10
    learning_rate = 3e-4
    model_checkpoints_path = 'data/models/UniSpeechSatForXVector_finetuned'

    num_epochs = 48

    training_dataset_path = 'data/music_caps/audios.dataset'
    audio_base_path = 'data/music_caps/audios'
    few_dataset_samples = 1000

    normalize_audio = True
    interval_duration_in_seconds = 5
    sampling_rate = 16000


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

    return


def get_model(model_name, from_pretrained=None):
    if model_name == 'Wav2Vec2ForXVector':
        feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
        model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")
    elif model_name == 'Data2VecAudioForXVector':
        feature_extractor = AutoFeatureExtractor.from_pretrained("hf-tiny-model-private/tiny-random-Data2VecAudioForXVector")
        model = Data2VecAudioForXVector.from_pretrained("hf-tiny-model-private/tiny-random-Data2VecAudioForXVector")
    elif model_name == 'WavLMForXVector':
        feature_extractor = AutoFeatureExtractor.from_pretrained("hf-tiny-model-private/tiny-random-WavLMForXVector")
        model = WavLMForXVector.from_pretrained("hf-tiny-model-private/tiny-random-WavLMForXVector")
    elif model_name == 'UniSpeechSatForXVector':
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base-plus-sv")

        if from_pretrained is None:
            from_pretrained = "microsoft/unispeech-sat-base-plus-sv"

        model = UniSpeechSatForXVector.from_pretrained(from_pretrained)
    else:
        raise ValueError(f"unknown model: {model_name}")

    return model, feature_extractor


def random_subsequence(audio_waveforms, n_frames):
    start_frame = random.randint(0, audio_waveforms.shape[-1] - n_frames)
    return audio_waveforms[start_frame:start_frame+n_frames]


def train(config: TrainingConfig, metric_logger: wandb_sdk.wandb_run.Run):
    # Dataloaders
    training_dataset = Dataset.load_from_disk(config.training_dataset_path)

    existing_audio_files = set(os.listdir(config.audio_base_path))

    training_dataset = training_dataset.filter(lambda x: x['file_name'] in existing_audio_files)
    training_dataset = training_dataset.map(lambda x: {"audio": config.audio_base_path + '/' + x['file_name']})
    training_dataset = training_dataset.cast_column('audio', Audio(sampling_rate=config.sampling_rate))
    training_dataset = training_dataset.filter(lambda x: x['audio']['array'].shape[-1] >= config.sampling_rate * config.interval_duration_in_seconds)

    if config.few_dataset_samples is not None:
        training_dataset = training_dataset.select(range(config.few_dataset_samples))

    total_expected_frames = config.sampling_rate * config.interval_duration_in_seconds
    training_dataset = training_dataset.map(lambda x: {"audio_waveform": random_subsequence(x['audio']['array'], total_expected_frames)})
    training_dataset = training_dataset.remove_columns([x for x in training_dataset.column_names if x != 'audio_waveform'])

    print("training_dataset", training_dataset)
    print("training_dataset[0]", torch.tensor(training_dataset[0]['audio_waveform']).shape)

    audio_augmentator = AudioAugmentator(expected_sample_rate=config.sampling_rate)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=DefaultDataCollator(),
    )

    # Model and optimizer
    model, feature_extractor = get_model(config.model_name)
    freeze_model(model.unispeech_sat)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    print("trainable model params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch_i in range(config.num_epochs):
        for batch in tqdm(training_dataloader, desc=f"Epoch {epoch_i}"):

            with torch.no_grad():
                audio_waveforms = batch['audio_waveform'].to(device)

                # todo normalize audio?
                # if config.normalize_audio:
                #     audio_waveforms = (audio_waveforms - audio_waveforms.min(dim=-1, keepdim=True).values) / (audio_waveforms.max(dim=-1, keepdim=True).values - audio_waveforms.max(dim=-1, keepdim=True).values) * 2 - 1

                # print(audio_waveforms.shape)
                augmented_audio_waveforms = []
                # augmented_audio_waveforms_1 = torch.empty_like(audio_waveforms)
                # augmented_audio_waveforms_2 = torch.empty_like(audio_waveforms)
                for i in range(audio_waveforms.shape[0]):
                    # augmented_audio_waveforms_1[i, :, :] = audio_augmentator.apply_random_augmentation(audio_waveforms[i, :, :])
                    augmented_audio_waveforms_1 = audio_augmentator.apply_random_augmentation(audio_waveforms[i:i+1, :])
                    augmented_audio_waveforms.append(augmented_audio_waveforms_1[0].cpu().numpy())

                for i in range(audio_waveforms.shape[0]):
                    # augmented_audio_waveforms_2[i, :, :] = audio_augmentator.apply_random_augmentation(audio_waveforms[i, :, :])
                    augmented_audio_waveforms_2 = audio_augmentator.apply_random_augmentation(audio_waveforms[i:i+1, :])
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

            model_output = model_output / model_output.norm(dim=-1, keepdim=True)
            x_vectors_1, x_vectors_2 = torch.chunk(model_output, 2)

            # todo logit scale like in clip?
            logits_per_1 = torch.matmul(x_vectors_1, x_vectors_2.t())

            loss = clip_loss(logits_per_1)
            metric_logger.log({"loss": loss.item()})

            model.zero_grad()
            loss.backward()
            optimizer.step()

    model_checkpoint_path = os.path.join(config.model_checkpoints_path, metric_logger.name)
    model.save_pretrained(model_checkpoint_path)

    return


if __name__ == '__main__':

    config = TrainingConfig()

    with wandb.init(project="lct-avm") as metric_logger:
        train(
            config=config,
            metric_logger=metric_logger,
        )
