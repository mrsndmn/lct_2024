from typing import List
from dataclasses import dataclass, field

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.transforms import v2


@dataclass
class VideoFingerPrinterConfig:
    batch_size: int = field(default=128)
    frame_size: List[int] = field(default=(224, 224))
    embeddings_normalization: bool = field(default=True)

class VideoFingerPrinter():
    def __init__(self, config: VideoFingerPrinterConfig, model, transforms=None):
        self.config: VideoFingerPrinterConfig = config
        self.model = model

        if transforms is None:
            transforms = self.transforms = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
            ])


        self.transforms = transforms

    def fingerprint_from_file(self, full_path) -> torch.Tensor:
        video_info = torchvision.io.read_video(
            full_path,
            output_format='TCHW',
            pts_unit='sec',
        )
        video_frames = video_info[0]

        # check frame size
        assert video_frames[0, 0].shape == torch.Size(self.config.frame_size)

        return self.fingerprint(video_frames)

    @torch.no_grad()
    def fingerprint(self, video_frames: torch.Tensor) -> torch.Tensor:
        
        # print("video_frames", video_frames.shape)
        video_frames = self.transforms(video_frames)
        # print("video_frames transformed", video_frames.shape)

        batch_size = self.config.batch_size
        all_embeddings = []
        for i in tqdm(range(0, video_frames.shape[0], batch_size), desc="video fingerprinting"):

            max_index = min(i+batch_size, video_frames.shape[0])
            embeddings = self.model(video_frames[i:max_index].to(self.model.logit_scale.device))
            # print("embeddings", embeddings.shape, "i:max_index", i, max_index)

            if self.config.embeddings_normalization:
                embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-6)

            embeddings = embeddings.cpu()
            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        # print("all_embeddings", all_embeddings.shape)

        return all_embeddings

