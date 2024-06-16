from typing import Optional
from dataclasses import dataclass, field
import torch
from datasets import Dataset, Audio
import os
from tqdm.auto import tqdm
from avm.models.image import get_default_image_model_for_x_vector
from avm.fingerprint.video import VideoFingerPrinter, VideoFingerPrinterConfig


@dataclass
class FingerprintConfig():
    base_video_path: str
    embeddings_out_dir: str
    few_dataset_samples: Optional[int]
    model_from_pretrained: str

@dataclass
class FingerprintIndexConfig():
    base_video_path        = "data/rutube/videos/compressed_normalized_index/"
    embeddings_out_dir     = "data/rutube/embeddings/video/index/"
    few_dataset_samples    = None
    model_from_pretrained  = "data/models/image/efficient-net-b0/ruby-moon-17"


@dataclass
class FingerprintValConfig():
    base_video_path        = "data/rutube/videos/compressed_normalized_val/"
    embeddings_out_dir     = "data/rutube/embeddings/video/val/"
    few_dataset_samples    = None
    model_from_pretrained  = "data/models/image/efficient-net-b0/ruby-moon-17"


# требования к датасету:
# file_name - должно быть просто название файлика

def generate_fingerprints(config: FingerprintConfig):

    embeddings_out_dir = config.embeddings_out_dir
    base_video_path = config.base_video_path

    os.makedirs(embeddings_out_dir, exist_ok=True)

    existing_video_files = sorted(os.listdir(base_video_path))

    if config.few_dataset_samples is not None:
        audios_dataset = audios_dataset.select(range(config.few_dataset_samples))

    model = get_default_image_model_for_x_vector(from_pretrained=config.model_from_pretrained)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    vfp_config = VideoFingerPrinterConfig()
    video_fingerprinter = VideoFingerPrinter(vfp_config, model)

    existing_embeddings = set(os.listdir(embeddings_out_dir))

    # audio file is decoded on the fly
    for video_file in tqdm(existing_video_files):
        embedding_file_name = video_file.split('.')[0] + '.pt'
        # if embedding_file_name in existing_embeddings:
        #     continue

        full_video_path = os.path.join(base_video_path, video_file)
        all_embeddings = video_fingerprinter.fingerprint_from_file(full_video_path)

        file_name = os.path.join(embeddings_out_dir, embedding_file_name)
        torch.save(all_embeddings, file_name)


if __name__ == '__main__':

    index_config = FingerprintIndexConfig()
    generate_fingerprints(index_config)

    val_config = FingerprintValConfig()
    generate_fingerprints(val_config)

    test_config = FingerprintValConfig()
    generate_fingerprints(val_config)

    raise Exception