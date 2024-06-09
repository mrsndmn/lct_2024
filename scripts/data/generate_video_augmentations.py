import os
import random
from pathlib import Path

import datasets
import torch
import torchvision.io as tvio
from torchvision.transforms import v2 as tvt
from torchvision.transforms.functional import get_image_size

from tqdm.auto import tqdm

source_path = 'data/video_caps/videos'
result_path = 'data/video_caps/augmented_videos'
target_dataset_path = "data/video_caps/augmented_videos.dataset"


def process_item(src_file_path, target_path):
    frames, _, meta = tvio.read_video(src_file_path, pts_unit='sec', output_format='TCHW')
    
    augmentations = init_augmentations(frames)
    selected_augmentation = random.choice(augmentations)

    augmented = selected_augmentation['func'](frames)

    src_filename = os.path.splitext(os.path.basename(src_file_path))[0]
    src_extension = os.path.splitext(os.path.basename(src_file_path))[1]
    augmented_filename = src_filename + "_" + selected_augmentation['name'] + src_extension
    augmented_file_path = os.path.join(target_path, augmented_filename)

    save(augmented_file_path, augmented, meta)

    return {
        "source_file_name": src_filename + src_extension,
        "augmented_file_name": augmented_filename,
        "augmentation": selected_augmentation['name'],
    }


def save(path, frames_TCHW, meta):
    frames = frames_TCHW.refine_names('T', 'C', 'H', 'W')
    frames = frames.align_to('T', 'H', 'W', 'C')

    tvio.write_video(path, video_array=frames, fps=meta['video_fps'])


def init_augmentations(tensor):
    coefficient = make_width_coefficient_of(tensor)
    return [
        {
            "name": "RandomPerspective",
            "func": tvt.RandomPerspective(distortion_scale=0.2, p=1.0)
        },
        {
            "name": "Resize",
            "func": tvt.RandomResize(min_size=coefficient(0.5), max_size=coefficient(0.8))
        },
        {
            "name": "ColorJitter",
            "func": tvt.ColorJitter(brightness=(0.85, 1), contrast=(0.85, 1), saturation=(0.85, 1), hue=(-0.05, 0.05))
        },
        {
            "name": "GaussianBlur",
            "func": tvt.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
        }
    ]


def make_width_coefficient_of(tensor):
    size = get_image_size(tensor)

    def f(coefficient):
        return int(size[0] * coefficient)
    
    return f


if __name__ == '__main__':
    print("started")
    os.makedirs(result_path, exist_ok=True)

    files = [ f for f in os.listdir(source_path) if f.endswith('.mp4') ]

    print("items to process", len(files))
    raw_dataset = []
    for f in tqdm(sorted(files)):
        result = process_item(os.path.join(source_path,f), result_path)
        raw_dataset.append(result)

    ds = datasets.Dataset.from_list(raw_dataset)
    ds.save_to_disk(target_dataset_path)
