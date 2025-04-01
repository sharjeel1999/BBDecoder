import torch
import torch.nn as nn

from typing import Union
import os
import matplotlib.pyplot as plt

from BBDecoder.utilities.similarity import cosine_similarity, kl_divergence

def has_trainable_parameters(module):
    return any(p.requires_grad for p in module.parameters())

class Main_wrapper(nn.Module):
    def __init__(self, layer: Union[nn.Module, nn.Sequential], name, index):
        super().__init__()

        self.index = index
        self.name = name

        self.record_sim = False
        self.sim_method = None
        self.sim_dim = None
        self.sim_scores = []

        self.main_layer = layer
        self.Trainable = has_trainable_parameters(self.main_layer)

    def forward(self, x):
        if self.record_sim == False:
            return self.main_layer(x)
        else:
            out = self.main_layer(x)
            self.inter_channel_div(out, self.sim_dim)
            return out
    
    def inter_channel_div(self, x, dim):
        if self.sim_method == 'cosine':
            sim = cosine_similarity(x, dim)
        elif self.sim_method == 'kl_divergence':
            sim = kl_divergence(x, dim)
        else:
            raise ValueError("Invalid similarity method. Choose 'cosine' or 'kl_divergence'.")
        self.sim_scores.append(sim)


    
        