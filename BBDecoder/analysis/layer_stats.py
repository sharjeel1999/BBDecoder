import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

class LayerAnalyzer():
    def __init__(self):
        pass

    def visualize_weight_hist(self, path):
        for name, module in self.model.named_children():
            if module.index in self.layer_inds:
                
                if hasattr(module.main_layer, 'weight'):
                    weights = module.main_layer.weight.detach().cpu().numpy().flatten()

                    plt.figure()
                    plt.hist(weights, bins = 50)
                    plt.title(f"Weight Histogram - {module.name}")
                    plt.xlabel("Weight Value")
                    plt.ylabel("Frequency")
                    plt.grid(True)

                    save_path = os.path.join(path, f'hist_{module.name}.jpg')
                    plt.savefig(save_path)

    def threshold_pruning(self, threshold):
        for name, module in self.model.named_children():
            if hasattr(module.main_layer, 'weight'):
                if module.index in self.layer_inds:
                    weights = module.main_layer.weight.detach()#.cpu().numpy()
                    mask = torch.abs(weights) > threshold
                    module.main_layer.weight = nn.Parameter(module.main_layer.weight * mask)






def test_function_graph(function, function_name, input_shape, y_output, itter_dim = 0):
    diff = len(input_shape) - itter_dim

    x_data = torch.linspace(-3, 3, input_shape[itter_dim + 1] * input_shape[itter_dim + 2], device=y_output.get_device()).view(1, 1, 197, 64)
    x_data = x_data.repeat(1, y_output.shape[1], 1, 1)
    
    y_output = function(x_data)
    
    x_dataf = x_data.reshape(x_data.shape[0]*6, 197*64).detach().cpu().numpy()
    y_outputf = y_output.reshape(y_output.shape[0]*6, 197*64).detach().cpu().numpy()
    
    for i in range(6):
        x_data = x_dataf[i, :]
        y_output = y_outputf[i, :]
        label_size = 24
        legend_size = 18
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_output, 'r-', label=f'{function_name} Function', linewidth=4)

        # Add details for better readability and presentation
        # plt.title(f'Fitting Complex Function to {function_name}', fontsize=16)
        plt.xlabel('x', fontsize=label_size)
        plt.ylabel('y', fontsize=label_size)
        plt.legend(fontsize=label_size)
        plt.grid(True)  # Turn on grid
        plt.tight_layout()  # Adjust layout to not cut off elements
    
        # Increase tick font size
        plt.xticks(fontsize=legend_size)
        plt.yticks(fontsize=legend_size)
    
        #plt.show()
        i = 0
        my_file = Path(f'/home/user/sharjeel/DEIT/saves/activation_saves/sin_rational/{function_name}_{str(i)}_fit.png')
        while my_file.is_file():
            print('=== ', i, ' ===')
            i += 1
            my_file = Path(f'/home/user/sharjeel/DEIT/saves/activation_saves/sin_rational/{function_name}_{str(i)}_fit.png')
            
        plt.savefig(f'/home/user/sharjeel/DEIT/saves/activation_saves/sin_rational/{function_name}_{str(i)}_fit.png')


def check_feature_hist(input_feats, save_path, bins = 50, min = 0, max = 1):
    bins = 50
    x = range(bins)

    hist_after = torch.histc(input_feats, bins=bins, min=min, max=max, out=None)
    plt.bar(x, hist_after.detach().cpu().numpy())
    plt.savefig(save_path)
    plt.clf()


def save_feature_maps(x):
    print('===== attention shape: ', x.shape)
    visual_save_path = "/home/user/sharjeel/DEIT/saves/attention_saves/"
        
    image_name = 'head_0.png'
    cv2.imwrite(os.path.join(visual_save_path, image_name), x[0, 0, :, :].detach().cpu().numpy()*255)
    plt.imshow(x[0, 0, :, :].cpu().detach().numpy())
    plt.show()
    
    image_name = 'head_3.png'
    cv2.imwrite(os.path.join(visual_save_path, image_name), x[0, 3, :, :].detach().cpu().numpy()*255)
    
    image_name = 'head_4.png'
    cv2.imwrite(os.path.join(visual_save_path, image_name), x[0, 4, :, :].detach().cpu().numpy()*255)
    
    image_name = 'head_5.png'
    cv2.imwrite(os.path.join(visual_save_path, image_name), x[0, 5, :, :].detach().cpu().numpy()*255)