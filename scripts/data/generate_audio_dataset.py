import torch
import torchaudio

from audio_augmentations import Noise, PitchShift, Gain, Delay, Reverb

import os



def add_noise():
    pass


def make_audio_mixtures():
    pass


def add_random_silence():
    pass


def pitch_modulation():
    pass


def volume_modulation():
    pass


if __name__ == '__main__':
    downloaded_audios = set(os.listdir("./data/music_caps/audios"))

    print("downloaded audios len", len(downloaded_audios))