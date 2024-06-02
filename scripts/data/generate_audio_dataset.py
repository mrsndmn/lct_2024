import torch
import torchaudio
import datasets

from tqdm.auto import tqdm

from audio_augmentations.apply import RandomApply
from audio_augmentations.compose import Compose
from audio_augmentations.augmentations.delay import Delay
from audio_augmentations.augmentations.gain import Gain
from audio_augmentations.augmentations.high_low_pass import HighLowPass
from audio_augmentations.augmentations.noise import Noise
from audio_augmentations.augmentations.pitch_shift import PitchShift
from audio_augmentations.augmentations.polarity_inversion import PolarityInversion
from audio_augmentations.augmentations.reverb import Reverb

import os


if __name__ == '__main__':

    augmented_audios_path = 'data/music_caps/augmented_audios'
    os.makedirs(augmented_audios_path, exist_ok=True)

    base_path = "./data/music_caps/audios"
    downloaded_audios = set(os.listdir(base_path))
    augmented_audios = set(os.listdir(augmented_audios_path))

    expected_sample_rate = 16000

    print("downloaded audios len", len(downloaded_audios))

    noise = Noise()
    pitch_shift = PitchShift(n_samples=expected_sample_rate*5, sample_rate=expected_sample_rate)
    gain = Gain()
    delay = Delay(sample_rate=expected_sample_rate)
    reverb = Reverb(sample_rate=expected_sample_rate)
    hilow_pass = HighLowPass(sample_rate=expected_sample_rate)
    polarity_inversion = PolarityInversion()

    waveform_augmentations = [
        {
            "name": 'noise',
            "function": noise,
        },
        {
            "name": 'PitchShift',
            "function": pitch_shift,
        },
        {
            "name": 'Gain',
            "function": gain,
        },
        {
            "name": 'Delay',
            "function": delay,
        },
        {
            "name": 'Reverb',
            "function": reverb,
        },
        {
            "name": 'HighLowPass',
            "function": hilow_pass,
        },
        {
            "name": 'PolarityInversion',
            "function": polarity_inversion,
        },
        {
            "name": 'Compose',
            "function": Compose([
                RandomApply([noise], p=0.5),
                RandomApply([pitch_shift], p=0.5),
                RandomApply([gain], p=0.5),
                RandomApply([delay], p=0.5),
                RandomApply([reverb], p=0.5),
                RandomApply([hilow_pass], p=0.5),
                RandomApply([polarity_inversion], p=0.5),
            ]),
        },
    ]

    result_dataset = []

    for file_name in tqdm(sorted(downloaded_audios)):

        waveform = None
        sample_rate = None
        def get_waveform():
            global waveform, sample_rate
            if waveform is not None:
                return waveform, sample_rate
            waveform, sample_rate = torchaudio.load(base_path + '/' + file_name)
            return waveform, sample_rate

        youtube_id = file_name.split('.')[0]

        for augmentation in waveform_augmentations:
            augmentation_name = augmentation['name']
            augmentation_function = augmentation['function']
            augmented_file_name = youtube_id + "_" + augmentation_name + ".wav"

            # if augmented_file_name not in augmented_audios:
            if True:
                waveform, sample_rate = get_waveform()
                if sample_rate != 16000:
                    print(f'sample rate is not ok, {sample_rate} {file_name}')
                    continue

                augmented_audio = augmentation_function(waveform)
                file_path_with_prefix = augmented_audios_path + '/' + augmented_file_name
                torchaudio.save(file_path_with_prefix, augmented_audio, sample_rate=sample_rate)

            dataset_item = {
                "youtube_id": youtube_id,
                "file_name": augmented_file_name,
                "augmentation": augmentation_name,
            }

            result_dataset.append(dataset_item)

    augmented_dataset = datasets.Dataset.from_list(result_dataset)
    target_dataset_path = "data/music_caps/augmented_audios.dataset"
    print("save to", target_dataset_path)
    augmented_dataset.save_to_disk(target_dataset_path)

