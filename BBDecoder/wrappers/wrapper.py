import torch
import torch.nn as nn

from typing import Union
import os
import matplotlib.pyplot as plt

from BBDecoder.utilities.similarity import cosine_similarity, kl_divergence

class Main_wrapper(nn.Module):
    def __init__(self, layer: Union[nn.Module, nn.Sequential], name, index, track_flag):
        super().__init__()

        self.index = index
        self.name = name
        self.track_flag = track_flag

        self.record_sim = False
        self.sim_method = None
        self.sim_dim = None
        self.sim_scores = []

        self.main_layer = layer

        if hasattr(self.main_layer, 'weight'):
            self.Trainable = True
        else:
            self.Trainable = False

        
        if self.track_flag and self.Trainable:
            self.master_tracker = {}
            self.master_tracker['L1'] = []
            self.master_tracker['L2'] = []
            self.main_layer.weight.register_hook(self.tracker_hook)

    def forward(self, x):
        if self.record_sim == False:
            return self.main_layer(x)
        else:
            out = self.main_layer(x)
            self.inter_channel_div(out, self.sim_dim)
            return out
        
    def tracker_hook(self, grad):
        l1_norm = grad.abs().sum().item()
        l2_norm = torch.sqrt((grad**2).sum()).item()
        self.master_tracker['L1'].append(l1_norm)
        self.master_tracker['L2'].append(l2_norm)
    
    def inter_channel_div(self, x, dim):
        sim = kl_divergence(x, dim)
        