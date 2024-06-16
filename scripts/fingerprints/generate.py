from typing import Optional
from dataclasses import dataclass, field
import torch
from datasets import Dataset, Audio
from avm.models.audio import get_audio_model
import os
from tqdm.auto import tqdm
from avm.fingerprint.audio import AudioFingerPrinter


class FingerprintValAudios():
    embeddings_out_dir = 'data/rutube/embeddings/electric-yogurt-97/audio_val_embeddings'
    sampling_rate = 16000
    base_audio_audio_path = 'data/rutube/compressed_val_audios/'
    dataset_path = 'data/rutube/compressed_val_audios.dataset'
    few_dataset_samples = None

    interval_duration_in_seconds = 5
    interval_step = 1.0
    batch_size = 128
    embeddings_normalization = True
    audio_normalization = True

    model_name = "UniSpeechSatForXVector"
    model_from_pretrained = 'data/models/UniSpeechSatForXVector_mini_finetuned/electric-yogurt-97'

class FingerprintTestAudios():
    embeddings_out_dir = 'data/rutube/embeddings/electric-yogurt-97/audio_test_embeddings'
    sampling_rate = 16000
    base_audio_audio_path = 'data/rutube/test/compressed_test_audios/'
    dataset_path = 'data/rutube/test/compressed_test_audios.dataset'
    few_dataset_samples = None

    interval_duration_in_seconds = 5
    interval_step = 1.0
    batch_size = 128
    embeddings_normalization = True
    audio_normalization = True

    model_name = "UniSpeechSatForXVector"
    model_from_pretrained = 'data/models/UniSpeechSatForXVector_mini_finetuned/electric-yogurt-97'


class FingerprintIndexAudios():
    embeddings_out_dir = 'data/rutube/embeddings/electric-yogurt-97/audio_index_embeddings'
    sampling_rate = 16000
    base_audio_audio_path = 'data/rutube/compressed_index_audios/'
    dataset_path = 'data/rutube/compressed_index_audios.dataset'
    few_dataset_samples = None

    interval_duration_in_seconds = 5.
    interval_step = 1.0
    batch_size = 128
    embeddings_normalization = True
    audio_normalization = True

    model_name = "UniSpeechSatForXVector"
    model_from_pretrained = 'data/models/UniSpeechSatForXVector_mini_finetuned/electric-yogurt-97'


class FingerprintValAudios10s():
    embeddings_out_dir = 'data/rutube/embeddings/electric-yogurt-97/audio_val_embeddings_10s'
    sampling_rate = 16000
    base_audio_audio_path = 'data/rutube/compressed_val_audios/'
    dataset_path = 'data/rutube/compressed_val_audios.dataset'
    few_dataset_samples = None

    interval_duration_in_seconds = 10
    interval_step = 1.0
    batch_size = 64
    embeddings_normalization = True
    audio_normalization = True

    model_name = "UniSpeechSatForXVector"
    model_from_pretrained = 'data/models/UniSpeechSatForXVector_mini_finetuned/electric-yogurt-97'

class FingerprintIndexAudios10s():
    embeddings_out_dir = 'data/rutube/embeddings/electric-yogurt-97/audio_index_embeddings_10s'
    sampling_rate = 16000
    base_audio_audio_path = 'data/rutube/compressed_index_audios/'
    dataset_path = 'data/rutube/compressed_index_audios.dataset'
    few_dataset_samples = None

    interval_duration_in_seconds = 10.
    interval_step = 1.0
    batch_size = 64
    embeddings_normalization = True
    audio_normalization = True

    model_name = "UniSpeechSatForXVector"
    model_from_pretrained = 'data/models/UniSpeechSatForXVector_mini_finetuned/electric-yogurt-97'



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
    # audios_dataset = audios_dataset.filter(lambda x: x['file_name'] in ['ydcrodwtz3mstjq1vhbdflx6kyhj3y0p.wav', 'ded3d179001b3f679a0101be95405d2c.wav'])
    audios_dataset = audios_dataset.map(lambda x: {"audio": base_audio_audio_path + '/' + x['file_name']})
    audios_dataset = audios_dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))

    if config.few_dataset_samples is not None:
        audios_dataset = audios_dataset.select(range(config.few_dataset_samples))

    model, feature_extractor = get_audio_model(config.model_name, from_pretrained=config.model_from_pretrained)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    audio_fingerprinter = AudioFingerPrinter(config, model, feature_extractor)

    # audio file is decoded on the fly
    for item in tqdm(audios_dataset):
        item_audio = item['audio']['array']
        all_embeddings = audio_fingerprinter.fingerprint(item_audio)

        file_name = os.path.join(embeddings_out_dir, item['file_name'].split('.')[0] + '.pt')
        torch.save(all_embeddings, file_name)


if __name__ == '__main__':

    # index_config = FingerprintIndexAudios()
    # generate_fingerprints(index_config)

    # val_config = FingerprintValAudios()
    # generate_fingerprints(val_config)

    test_config = FingerprintTestAudios()
    generate_fingerprints(test_config)

    # embeddings_out_dir = test_config.embeddings_out_dir
    # test_files = os.listdir(embeddings_out_dir)
    
    # for test_file in test_files:
    #     test_file_full_path = os.path.join(embeddings_out_dir, test_file)
    #     emb = torch.load(test_file_full_path)
    #     print(emb.shape)

    # generate_fingerprints(test_config)


    # raise Exception