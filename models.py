import numpy as np
import torch

from torch import nn
import torch.nn.functional as fn

device = torch.device('cuda')

# CNN Model definition
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.conv_stack(x)
        return output

# DNN Model definition
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 8192),
            nn.ReLU(),
            nn.Linear(8192, 9216),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.neural_stack(x)
        return output
