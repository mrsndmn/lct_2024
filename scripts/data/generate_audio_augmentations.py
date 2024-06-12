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

from pathlib import Path

# from torchaudio.transforms import Speed

import os

import random


class SilenceShift(torch.nn.Module):
    def __init__(self, sample_rate, min_shift_ms=0, max_shift_ms=5000):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_shift_ms = min_shift_ms
        self.max_shift_ms = max_shift_ms

        return

    def calc_offset(self, ms):
        return int(ms * (self.sample_rate / 1000))

    def forward(self, audio):
        shift_ms = random.randint(self.min_shift_ms, self.max_shift_ms)

        offset = self.calc_offset(shift_ms)
        delayed_signal = torch.zeros_like(audio)
        delayed_signal[:, :offset] = 0
        delayed_signal[:, offset:] = audio[:, offset:]
        return delayed_signal


class RandomSilence(torch.nn.Module):
    def __init__(self, sample_rate, silence_max_duration=5000):
        super().__init__()
        self.sample_rate = sample_rate
        self.silence_max_duration = silence_max_duration

        return

    def calc_offset(self, ms):
        return int(ms * (self.sample_rate / 1000))

    def forward(self, audio):
        silence_duration = random.randint(0, self.silence_max_duration)
        samples_for_silence = int(silence_duration * self.sample_rate / 1000)
        if samples_for_silence > (audio.shape[1] // 2):
            samples_for_silence = audio.shape[1] // 2

        silence_start_sample = random.randint(0, audio.shape[1] - samples_for_silence)

        audio_clone = torch.clone(audio)
        audio_clone[:, silence_start_sample:silence_start_sample+samples_for_silence] = 0

        return audio_clone


class AudioAugmentator():
    def __init__(self, expected_sample_rate):
        noise = Noise(min_snr=0.1, max_snr=0.2)
        # pitch_shift = PitchShift(n_samples=expected_sample_rate*5, sample_rate=expected_sample_rate)
        gain = Gain()
        delay = Delay(sample_rate=expected_sample_rate)  # симулирует эхо
        silence_shift = SilenceShift(sample_rate=expected_sample_rate)
        random_silence = RandomSilence(sample_rate=expected_sample_rate)
        reverb = Reverb(sample_rate=expected_sample_rate)
        hilow_pass = HighLowPass(sample_rate=expected_sample_rate)
        polarity_inversion = PolarityInversion()
        # Speed()

        self.waveform_augmentations = [
            {
                "name": 'noise',
                "function": noise,
            },
            # {
            #     "name": 'PitchShift',
            #     "function": pitch_shift,
            # },
            {
                "name": 'Gain',
                "function": gain,
            },
            {
                "name": 'Delay',
                "function": delay,
            },
            {
                "name": 'SilenceShift',
                "function": silence_shift,
            },
            {
                "name": 'RandomSilence',
                "function": random_silence,
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
            # {
            #     "name": 'SpeedX1.2',
            #     "function": polarity_inversion,
            # },
            # {
            #     "name": 'Compose',
            #     "function": Compose([
            #         RandomApply([noise], p=0.5),
            #         RandomApply([pitch_shift], p=0.5),
            #         RandomApply([gain], p=0.5),
            #         RandomApply([delay], p=0.5),
            #         RandomApply([reverb], p=0.5),
            #         RandomApply([hilow_pass], p=0.5),
            #         RandomApply([polarity_inversion], p=0.5),
            #     ]),
            # },
        ]
        return

    def apply_random_augmentation(self, audio_waveform):
        augmentation = self.get_random_augmentation()
        return augmentation['function'](audio_waveform)

    def get_random_augmentation(self):
        return random.choice(self.waveform_augmentations)


if __name__ == '__main__':

    augmented_audios_path = Path('data/music_caps/augmented_audios')
    os.makedirs(augmented_audios_path, exist_ok=True)

    base_path = Path("./data/music_caps/audios")
    downloaded_audios = set(os.listdir(base_path))
    augmented_audios = set(os.listdir(augmented_audios_path))

    noise = Noise(min_snr=0.1, max_snr=0.5)

    expected_sample_rate = 16000
    augmented_sample_duration_seconds = 15

    print("downloaded audios len", len(downloaded_audios))

    audio_augmentator = AudioAugmentator(expected_sample_rate=expected_sample_rate)
    result_dataset = []

    background_waveform_zeros = torch.zeros([1, expected_sample_rate * augmented_sample_duration_seconds])

    for file_name in tqdm(sorted(downloaded_audios)):

        background_waveform = background_waveform_zeros.clone()
        waveform = None
        sample_rate = None

        def get_waveform():
            global waveform, sample_rate
            if waveform is not None:
                return waveform, sample_rate
            waveform, sample_rate = torchaudio.load(str(base_path.joinpath(file_name)))
            return waveform, sample_rate

        file_id = file_name.split('.')[0]

        augmentation = audio_augmentator.get_random_augmentation()
        augmentation_name = augmentation['name']
        augmentation_function = augmentation['function']
        augmented_file_name = file_id + "_" + augmentation_name + ".wav"

        # if augmented_file_name not in augmented_audios:
        waveform, sample_rate = get_waveform()
        if sample_rate != expected_sample_rate:
            print(f'sample rate is not ok, {sample_rate} {file_name}')
            continue

        augmented_audio = augmentation_function(waveform)
        augmented_audio_offset = random.randint(0, background_waveform.shape[-1] - waveform.shape[-1])
        background_waveform[:, augmented_audio_offset:(augmented_audio_offset+waveform.shape[-1])] = augmented_audio
        background_waveform = noise(background_waveform)

        file_path_with_prefix = augmented_audios_path.joinpath(augmented_file_name)
        torchaudio.save(str(file_path_with_prefix), background_waveform, sample_rate=sample_rate)

        dataset_item = {
            "file_id": file_id,
            "file_name": augmented_file_name,
            "augmentation": augmentation_name,
            "augmented_audio_offset": augmented_audio_offset,
        }

        result_dataset.append(dataset_item)

    augmented_dataset = datasets.Dataset.from_list(result_dataset)
    target_dataset_path = "data/music_caps/augmented_audios.dataset"
    print("save to", target_dataset_path)
    augmented_dataset.save_to_disk(target_dataset_path)
