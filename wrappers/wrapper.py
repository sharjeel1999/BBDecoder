import torch
import torch.nn as nn


class Main_wrapper(nn.Module):
    def __init__(self, layer: nn.Module, name, index):
        super().__init__()

        self.index = index
        self.name = name

        self.main_layer = layer

    def forward(self, x):
        return self.main_layer(x)