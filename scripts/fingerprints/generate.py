from typing import Optional
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from datasets import Dataset, Audio
from avm.models import get_model
import os
from tqdm.auto import tqdm


class FingerprintAugmentedAudioConfig():
    embeddings_out_dir = 'data/music_caps/augmented_embeddings'
    sampling_rate = 16000
    base_audio_audio_path = 'data/music_caps/augmented_audios'
    dataset_path = 'data/music_caps/augmented_audios.dataset'
    interval_duration_in_seconds = 5
    interval_step = 2.5
    batch_size = 12
    embeddings_normalization = True


class FingerprintAudioConfig():
    embeddings_out_dir = 'data/music_caps/audio_embeddings'
    sampling_rate = 16000
    base_audio_audio_path = 'data/music_caps/audios'
    dataset_path = 'data/music_caps/audios.dataset'
    interval_duration_in_seconds = 5
    interval_step = 2.5
    batch_size = 12
    embeddings_normalization = True


class FingerprintIndexAudios():
    embeddings_out_dir = 'data/rutube/audio_embeddings'
    sampling_rate = 16000
    base_audio_audio_path = 'data/compressed_index_audios/'
    dataset_path = 'data/rutube/audios.dataset'
    interval_duration_in_seconds = 5
    interval_step = 2.5
    batch_size = 12
    embeddings_normalization = True


@dataclass
class FingerprintConfig():
    embeddings_out_dir: str
    sampling_rate: int
    base_audio_audio_path: str
    dataset_path: str
    few_dataset_samples: Optional[int]

    interval_duration_in_seconds: int
    interval_step: float
    batch_size: int
    embeddings_normalization: bool
    audio_normalization: bool

    model_name: str
    model_from_pretrained: Optional[str]


class FingerprintValAudios():
    embeddings_out_dir = 'data/rutube/audio_val_embeddings'
    sampling_rate = 16000
    base_audio_audio_path = 'data/compressed_val_audios/'
    dataset_path = 'data/rutube/audios_val.dataset'
    interval_duration_in_seconds = 5
    interval_step = 2.5
    batch_size = 32


# требования к датасету:
# file_name - должно быть просто название файлика

def generate_fingerprints(config: FingerprintConfig):
    embeddings_normalization = config.embeddings_normalization
    audio_normalization = config.audio_normalization

    embeddings_out_dir = config.embeddings_out_dir
    sampling_rate = config.sampling_rate
    base_audio_audio_path = config.base_audio_audio_path
    dataset_path = config.dataset_path

    interval_duration_in_seconds = config.interval_duration_in_seconds
    interval_step = config.interval_step
    batch_size = config.batch_size

    os.makedirs(embeddings_out_dir, exist_ok=True)
    audios_dataset = Dataset.load_from_disk(dataset_path)

    existing_audio_files = set(os.listdir(base_audio_audio_path))

    audios_dataset = audios_dataset.filter(lambda x: x['file_name'] in existing_audio_files)
    audios_dataset = audios_dataset.map(lambda x: {"audio": base_audio_audio_path + '/' + x['file_name']})
    audios_dataset = audios_dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))

    if config.few_dataset_samples is not None:
        audios_dataset = audios_dataset.select(range(config.few_dataset_samples))

    # print("dataset len", len(audios_dataset))
    # print(audios_dataset[0])

    model, feature_extractor = get_model(config.model_name, from_pretrained=config.model_from_pretrained)

    # print(sum(p.numel() for p in model.parameters()))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print("device", device)
    model.to(device)


    # audio file is decoded on the fly
    for item in tqdm(audios_dataset):

        item_audio = item['audio']['array']

        if audio_normalization:
            item_audio = (item_audio - item_audio.min()) / (item_audio.max() - item_audio.min()) * 2.0 - 1.0

        inputs = feature_extractor(
            [item_audio], sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )

        interval_num_samples = int(interval_duration_in_seconds * sampling_rate)
        interval_step_num_samples = int(interval_step * sampling_rate)

        input_values = inputs['input_values']
        inputs_shifted = []
        for i in range(0, input_values.shape[-1], interval_step_num_samples):
            # inputs['input_values'].shape[-1]
            interval_right_boarder = min(i+interval_num_samples, input_values.shape[-1])
            current_interval = input_values[:, i:interval_right_boarder]
            if current_interval.shape[-1] < interval_num_samples:
                padding_size = interval_num_samples - current_interval.shape[-1]
                # print(f"run padding for {padding_size}")
                current_interval = F.pad(current_interval, [0, padding_size], 'constant', value=0)

            # print("current_interval", current_interval.shape)
            inputs_shifted.append(current_interval)

        inputs_shifted = torch.cat(inputs_shifted, dim=0)
        # print("inputs_shifted", inputs_shifted.shape, "input_values", input_values.shape)

        with torch.no_grad():
            all_embeddings = []
            for i in range(0, inputs_shifted.shape[0], batch_size):

                max_index = min(i+batch_size, inputs_shifted.shape[-1])
                embeddings = model(input_values=inputs_shifted[i:max_index].to(device)).embeddings

                if embeddings_normalization:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                embeddings = embeddings.cpu()
                # print("embeddings", embeddings.shape)
                all_embeddings.append(embeddings)

            all_embeddings = torch.cat(all_embeddings, dim=0)
            # print("all_embeddings", all_embeddings.shape)

            file_name = embeddings_out_dir + '/' + item['file_name'].split('.')[0] + '.pt'
            torch.save(all_embeddings, file_name)


if __name__ == '__main__':

    # config = FingerprintAugmentedAudioConfig()
    config = FingerprintAudioConfig()
    # config = FingerprintIndexAudios()

    generate_fingerprints(config)

