import torch
import torch.nn as nn

from typing import Union
import os
import matplotlib.pyplot as plt
import numpy as np

from ..utilities import cosine_similarity, kl_divergence
from ..analysis import FlowArchive


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
        # self.record_dim = None
        self.f_width, self.f_height = None, None
        self.post_proc_function = None
        self.feats_archive = FlowArchive()

        self.main_layer = layer
        self.Trainable = has_trainable_parameters(self.main_layer)

    def default_processor(self, x):
        return x.cpu().detach().numpy()

    def get_frame_size(self):
        if self.f_width is None or self.f_height is None:
            raise ValueError("Feature dimensions are not set. Ensure to save features before getting frame size.")
        return self.f_width, self.f_height

    def save_recorded_features(self, feats):
        
        layer_path = os.path.join(self.inter_features_path,f'{self.index}_{self.name}')
        if not os.path.exists(layer_path):
            os.makedirs(layer_path)
        
        assert feats.shape[0] == 1, "Batch size must be 1 for feature saving."

        if self.f_width is None and self.f_height is None:
            self.f_width, self.f_height = feats.shape[2], feats.shape[3]

        # if self.record_dim is not None:
        #     feats = feats[0, self.record_dim, :, :]
        

        if self.post_proc_function is None:
            self.post_proc_function = self.default_processor

        print('-- before post proc function --')
        assert callable(self.post_proc_function), \
        f"post_proc_function must be callable, but got type: {type(self.post_proc_function)}"

        feats = self.post_proc_function(feats)

        assert isinstance(feats, np.ndarray), \
        f"output of the post_proc function must be a numpy array, but got type: {type(feats)}"
        

        ind_val = self.feats_archive.max_index
        plt.imsave(os.path.join(layer_path, f'{self.name}_{ind_val}.png'), feats)#, cmap='gray')
        self.feats_archive.add_flow(os.path.join(layer_path, f'{self.name}_{ind_val}.png'))

    def forward(self, x, *args, **kwargs):
        out = self.main_layer(x, *args, **kwargs)

        if self.record_inter_features:
            self.save_recorded_features(out.clone())
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


        