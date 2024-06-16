import torchvision
import random
import torch
from torch.optim import Adam

from typing import List

from torch.utils.data import DataLoader

from dataclasses import dataclass
from transformers import DefaultDataCollator

from datasets import Dataset

from tqdm.auto import tqdm
import wandb
from wandb import sdk as wandb_sdk
from torchvision.transforms import v2
import os

from avm.models.image import get_default_image_model_for_x_vector

from scripts.train.metric_learning import clip_loss

@dataclass
class TrainingConfig():
    # Head Training
    freeze_skeleton = True
    batch_size = 2
    learning_rate = 3e-4

    videos_dir = "data/rutube/videos/compressed_normalized_index/"

    model_checkpoints_path = 'data/models/image/efficient-net-b0'
    save_and_evaluate_model_every_epoch = 1

    num_epochs = 1
    multiply_train_epoch_data = 5

    few_dataset_samples = None

class RandomVideoCoupleFramesDataset():
    def __init__(self, 
                 videos_dir: str,
                 max_frames_distance = 3,
                ):
        
        base_video_path = videos_dir
        video_files = [ os.path.join(base_video_path, file_name) for file_name in os.listdir(base_video_path) ]

        self.video_files = video_files
        self.max_frames_distance    = max_frames_distance

        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomApply([v2.RandomRotation(5)]),
        ])

        return

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, i):
        video_file = self.video_files[i]

        # todo optimize and read only
        # part of video
        video_info = torchvision.io.read_video(
            video_file,
            output_format='TCHW',
            pts_unit='sec',
        )
        video_frames = video_info[0]

        frames_num = video_frames.shape[0]
        frames_distance = random.randint(1, self.max_frames_distance)
        first_frame_i = random.randint(0, frames_num - frames_distance - 1)

        first_frame = video_frames[first_frame_i:first_frame_i+1]
        second_frame_i = first_frame_i+frames_distance
        second_frame = video_frames[second_frame_i:second_frame_i+1]

        first_frame  = self.transforms(first_frame)
        second_frame = self.transforms(second_frame)

        return {
            "frames": torch.cat([first_frame, second_frame], dim=0),
        }


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

    return


def train(config: TrainingConfig, metric_logger: wandb_sdk.wandb_run.Run):
    # Dataloaders
    training_dataset = RandomVideoCoupleFramesDataset(config.videos_dir)

    if config.few_dataset_samples is not None:
        training_dataset = training_dataset.select(range(config.few_dataset_samples))

    print("training_dataset items", len(training_dataset))

    def collate_fn(items):
        frames_pairs = [ x['frames'].unsqueeze(0) for x in items ]
        # for i, fp in enumerate(frames_pairs):
        #     print(i, fp.shape)
        return { "frames": torch.cat(frames_pairs, dim=0) }

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Model and optimizer
    model = get_default_image_model_for_x_vector()
    if config.freeze_skeleton:
        freeze_model(model.backbone)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    print("trainable model params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch_i in range(config.num_epochs):
        model.train()

        for dataloader_multiplicator in range(config.multiply_train_epoch_data):
            pbar = tqdm(training_dataloader)
            for batch in pbar:
                
                frames_batch = batch['frames'].flatten(0, 1) # [ bs * 2, C, H W ]
                frames_batch = frames_batch.to(device)
                model_output = model(frames_batch)
                # print("model_output", model_output.shape)

                model_output = model_output / (model_output.norm(dim=-1, keepdim=True) + 1e-6)
                if model_output.isnan().any().item():
                    raise Exception("model_output is nan!")

                model_output = model_output.unflatten(0, [-1, 2]) # [ bs, 2, C, H, W ]
                x_vectors_1, x_vectors_2 = model_output[:, 0], model_output[:, 1]

                logit_scale = model.logit_scale.exp()
                if logit_scale.isnan().item():
                    raise Exception("logit_scale is nan!")

                logits_per_1 = torch.matmul(x_vectors_1, x_vectors_2.t()) * logit_scale
                if logits_per_1.isnan().any().item():
                    raise Exception("logits_per_1 is nan!")

                loss = clip_loss(logits_per_1)
                pbar.set_description(f"Epoch={epoch_i}({dataloader_multiplicator}); Loss={loss.item():.3f}")

                model.zero_grad()
                loss.backward()
                if loss.isnan().item():
                    raise Exception("loss is nan!")
                
                metric_logger.log({
                    "loss": loss.item(),
                })

                optimizer.step()

        if epoch_i % config.save_and_evaluate_model_every_epoch == 0:
            model_checkpoint_path = os.path.join(config.model_checkpoints_path, metric_logger.name)
            model.save_pretrained(model_checkpoint_path)

            model_checkpoint_path = os.path.join(config.model_checkpoints_path, metric_logger.name)
            print("save model to", model_checkpoint_path)
            model.save_pretrained(model_checkpoint_path)

    return


if __name__ == '__main__':

    config = TrainingConfig()

    with wandb.init(project="lct-avm-image") as metric_logger:
        train(
             config=config,
            metric_logger=metric_logger,
        )
