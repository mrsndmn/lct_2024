from typing import List
from dataclasses import dataclass, field

from tqdm.auto import tqdm
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np


@dataclass
class AudioFingerPrinterConfig:
    # важно передавать это поле, тк для индекса и валидации
    # значения моугт различаться
    interval_step: float
    interval_duration_in_seconds: float = field(default=5.0)

    sampling_rate: int = field(default=16000)
    batch_size: int = field(default=128)

    audio_normalization: bool = field(default=True)
    embeddings_normalization: bool = field(default=True)

class AudioFingerPrinter():
    def __init__(self, config: AudioFingerPrinterConfig, model, feature_extractor):
        self.config = config
        self.model = model
        self.feature_extractor = feature_extractor

    def fingerprint_from_file(self, full_audio_path) -> torch.Tensor:
        waveform = torchaudio.load(full_audio_path)

        return self.fingerprint(waveform=waveform)

    def fingerprint(self, waveform: np.ndarray) -> torch.Tensor:

        if self.config.audio_normalization:
            waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-6) * 2.0 - 1.0

        inputs = self.feature_extractor(
            [waveform], sampling_rate=self.config.sampling_rate, return_tensors="pt", padding=True
        )

        interval_num_samples = int(self.config.interval_duration_in_seconds * self.config.sampling_rate)
        interval_step_num_samples = int(self.config.interval_step * self.config.sampling_rate)

        input_values = inputs['input_values']
        inputs_shifted = []
        for i in range(0, input_values.shape[-1], interval_step_num_samples):
            interval_right_boarder = min(i+interval_num_samples, input_values.shape[-1])
            current_interval = input_values[:, i:interval_right_boarder]
            if current_interval.shape[-1] < interval_num_samples:
                padding_size = interval_num_samples - current_interval.shape[-1]
                current_interval = F.pad(current_interval, [0, padding_size], 'constant', value=0)

            inputs_shifted.append(current_interval)

        inputs_shifted = torch.cat(inputs_shifted, dim=0)

        batch_size = self.config.batch_size
        with torch.no_grad():
            all_embeddings = []
            for i in tqdm(range(0, inputs_shifted.shape[0], batch_size), desc="fingerprinting"):

                max_index = min(i+batch_size, inputs_shifted.shape[-1])
                embeddings = self.model(input_values=inputs_shifted[i:max_index].to(self.model.device)).embeddings

                if self.config.embeddings_normalization:
                    embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-6)
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)

            all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_embeddings

