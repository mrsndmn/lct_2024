from typing import List
from dataclasses import dataclass, field

import torch


class AudioFingerPrinter():
    def __init__(self):
        pass
        # todo load model

    def fingerprint(self, full_audio_path, file_id) -> torch.Tensor:
        # todo preprocess
        # todo generate fingerprint

        return torch.zeros([100, 256])

