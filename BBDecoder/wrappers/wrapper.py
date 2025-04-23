import torch
import torch.nn as nn

from typing import Union
import os
import matplotlib.pyplot as plt

from ..utilities import cosine_similarity, kl_divergence

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

        self.record_inter_features = False
        self.inter_features_path = None

        self.main_layer = layer
        self.Trainable = has_trainable_parameters(self.main_layer)

    def forward(self, x, *args, **kwargs):
        out = self.main_layer(x, *args, **kwargs)

        if self.record_inter_features == True:
            if not os.path.exists(self.inter_features_path):
                os.makedirs(self.inter_features_path)
            plt.imsave(os.path.join(self.inter_features_path, f'{self.index}_{self.name}.png'), out[0].cpu().detach().numpy())#, cmap='gray')
            self.record_inter_features = False

        if self.record_sim == True:
            self.inter_channel_div(out.clone(), self.sim_dim)

        return out
    
    def inter_channel_div(self, x, dim):
        if self.sim_method == 'cosine':
            sim = cosine_similarity(x, dim)
        elif self.sim_method == 'kl_divergence':
            sim = kl_divergence(x, dim)
        else:
            raise ValueError("Invalid similarity method. Choose 'cosine' or 'kl_divergence'.")
        self.sim_scores.append(sim)


        