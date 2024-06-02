import torch
from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector
import os
from tqdm.auto import tqdm


class FingerprintAugmentedAudioConfig():
    embeddings_out_dir = 'data/music_caps/augmented_embeddings'
    sampling_rate = 16000
    base_audio_audio_path = 'data/music_caps/augmented_audios'
    dataset_path = 'data/music_caps/augmented_audios.dataset'

class FingerprintAudioConfig():
    embeddings_out_dir = 'data/music_caps/audio_embeddings'
    sampling_rate = 16000
    base_audio_audio_path = 'data/music_caps/audios'
    dataset_path = 'data/music_caps/audios.dataset'


if __name__ == '__main__':

    # config = FingerprintAugmentedAudioConfig()
    config = FingerprintAudioConfig()

    embeddings_out_dir = config.embeddings_out_dir
    sampling_rate = config.sampling_rate
    base_audio_audio_path = config.base_audio_audio_path
    dataset_path = config.dataset_path

    os.makedirs(embeddings_out_dir, exist_ok=True)
    audios_dataset = Dataset.load_from_disk(dataset_path)
    # audmented_audios_dataset = audmented_audios_dataset.

    existing_audio_files = set(os.listdir(base_audio_audio_path))

    audios_dataset = audios_dataset.filter(lambda x: x['file_name'] in existing_audio_files)
    audios_dataset = audios_dataset.map(lambda x: {"audio": base_audio_audio_path + '/' + x['file_name']})
    audios_dataset = audios_dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))

    print("dataset len", len(audios_dataset))
    print(audios_dataset[0])

    feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
    model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")
    print(sum(p.numel() for p in model.parameters()))

    # audio file is decoded on the fly
    for item in tqdm(audios_dataset):
        inputs = feature_extractor(
            [item['audio']['array']], sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            embeddings = model(**inputs).embeddings

        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

        # print("embeddings", embeddings.shape)

        file_name = embeddings_out_dir + '/' + item['file_name'].split('.')[0] + '.pt'
        torch.save(embeddings, file_name)
