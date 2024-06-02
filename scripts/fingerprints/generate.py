import torch
from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector
import os
from tqdm.auto import tqdm


if __name__ == '__main__':

    embeddings_out_dir = 'data/music_caps/augmented_embeddings'
    os.makedirs(embeddings_out_dir, exist_ok=True)

    sampling_rate = 16000
    base_augmentaed_audio_path = 'data/music_caps/augmented_audios'
    audmented_audios_dataset = Dataset.load_from_disk('data/music_caps/augmented_audios.dataset')

    audmented_audios_dataset = audmented_audios_dataset.map(lambda x: {"audio": base_augmentaed_audio_path + '/' + x['file_name']})
    audmented_audios_dataset = audmented_audios_dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))

    print(audmented_audios_dataset[0])

    feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
    model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")
    print(sum(p.numel() for p in model.parameters()))

    # audio file is decoded on the fly
    for item in tqdm(audmented_audios_dataset):
        inputs = feature_extractor(
            [item['audio']['array']], sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            embeddings = model(**inputs).embeddings

        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

        # print("embeddings", embeddings.shape)

        file_name = embeddings_out_dir + '/' + item['file_name'].split('.')[0] + '.pt'
        torch.save(embeddings, file_name)
