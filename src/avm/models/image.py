import os
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
        # assert inputs.shape[1:] == torch.Size([3, 224, 224])

        activations = self.backbone.extract_features(inputs)
        activations = self.backbone._avg_pooling(activations) # [ bs, 1280, 1, 1 ]
        activations = activations.flatten(1) # [ bs, 1280 ]
        assert activations.shape[1:] == torch.Size([ 1280 ])
        # print(activations.shape) # torch.Size([1, 1280 ])

        return self.projection(activations) # [ bs, self.output_embedding_dim ]

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        with open("model_state.pt", 'wb') as f:
            torch.save(self.state_dict(), f)

    @classmethod
    def from_pretrained(klass, save_directory):
        
        model = klass()

        os.makedirs(save_directory, exist_ok=True)
        with open("model_state.pt", 'rb') as f:
            model_state_dict = torch.load(f)

        model.load_state_dict(model_state_dict)

        return


class EfficientNetForFrameInFrameDetection(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone = efficient_net

        self.hidden_dim = hidden_dim

        self.projection = nn.Sequential(
            nn.Linear(1280, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        return

    def forward(self, inputs):
        # inputs ~ [ bs, 3, 224, 224 ]
        # assert inputs.shape[1:] == torch.Size([3, 224, 224])

        activations = self.backbone.extract_features(inputs)
        activations = self.backbone._avg_pooling(activations) # [ bs, 1280, 1, 1 ]
        activations = activations.flatten(1) # [ bs, 1280 ]
        assert activations.shape[1:] == torch.Size([ 1280 ])
        # print(activations.shape) # torch.Size([1, 1280 ])

        return self.projection(activations) # [ bs, self.output_embedding_dim ]

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        with open("model_state.pt", 'wb') as f:
            torch.save(self.state_dict(), f)

    @classmethod
    def from_pretrained(klass, save_directory):
        
        model = klass()

        os.makedirs(save_directory, exist_ok=True)
        with open("model_state.pt", 'rb') as f:
            model_state_dict = torch.load(f)

        model.load_state_dict(model_state_dict)

        return




def get_default_image_model_for_x_vector():
    return EfficientNetForXVector()