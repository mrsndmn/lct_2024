
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn

class EfficientNetForXVector(nn.Module):
    def __init__(self, output_embedding_dim=256):
        super().__init__()

        efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone = efficient_net

        self.output_embedding_dim = output_embedding_dim

        self.projection = nn.Sequential(
            nn.Linear(1280, output_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(output_embedding_dim * 4, output_embedding_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)


    def forward(self, inputs):
        # inputs ~ [ bs, 3, 224, 224 ]
        assert inputs.shape[1:] == torch.Size([3, 224, 224])

        activations = self.backbone.extract_features(inputs)
        activations = self.backbone._avg_pooling(activations)
        assert activations.shape[1:] == torch.Size([ 1280 ])
        # print(activations.shape) # torch.Size([1, 1280 ])

        return self.projection(activations) # [ bs, self.output_embedding_dim ]


def get_default_image_model():
    return EfficientNetForXVector()