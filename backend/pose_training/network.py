from torch import nn
from .features import FEATURE_SIZE


class StaticPoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(FEATURE_SIZE, 128), nn.ReLU(), nn.Dropout(.15),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.layers(x)
