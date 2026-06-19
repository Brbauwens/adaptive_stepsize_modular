import math
import numpy as np
import torch
from torch import nn, Tensor


def in_out_dim(dataloader):
    ds = dataloader.dataset
    return (np.prod(ds[0][0].shape), len(ds.classes))


class BasicTreeLayerNN(nn.Module):
    def __init__(self, dataloader, hidden1_width, hidden2_width):
        super().__init__()

        input_dim, output_dim = in_out_dim(dataloader)
        self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, hidden1_width),
                nn.ReLU(),
                nn.Linear(hidden1_width, hidden2_width),
                nn.ReLU(),
                nn.Linear(hidden2_width, output_dim)
            )

    def forward(self, x):
        return self.layers(x)


