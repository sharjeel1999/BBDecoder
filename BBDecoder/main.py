from .visualizer import list_layers, draw_graph
from .analysis import GradAnalyzer, LayerAnalyzer
from .wrappers import Main_wrapper

import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Master_analyzer(nn.Module, GradAnalyzer, LayerAnalyzer):
    def __init__(self,
                 model,
                 save_path,
                 input_size = (1, 3, 32, 32),
                 ):
        """
        Wrappes the entire model and contains functions to visualize and analyze the wrapped model
        Args:
            model: Model to be analyzed.
            input_size: Input size of the model.
        """
        super(Master_analyzer, self).__init__()
        self.model = model
        self.input_size = input_size
        self.save_path = save_path

        self.layer_inds = None

        self.ave_grads = []
        self.max_grads = []
        self.l1_norm = []
        self.l2_norm = []

        self.layer_names = []

        self.wrap_layers()
        # print('----- List of layer and their indices -----')
        # list_layers(self.model)
        self.visualize_architecture()

    def visualize_architecture(self):
        model_graph = draw_graph(self, input_size = self.input_size, expand_nested=True, hide_module_functions=True, directory = self.save_path)
        model_graph.visual_graph.render("Model_architecture", format = "png")


    def forward(self, x):
        # this is just for torchview visualization
        return self.model(x)


    def wrap_layers(self):
        for z, (name, module) in enumerate(self.model.named_children()):
            self.layer_names.append(name)
            
            if isinstance(module, nn.Module) or isinstance(module, nn.Sequential):#and not list(module.children()):  
                setattr(self.model, name, Main_wrapper(module, name, z))
            # else:
                # Recursively wrap layers in submodules (if any)
                # Main_wrapper(module, name, z, False)


    def forward_propagation(self, x):
        return self.model(x)
    
    def backward_propagation(self, loss, collect_grads = False, layers = None):
        """
        Calculates gradients and stores for specified layers.

        Args:
            loss: loss between prediction and ground truth.
            collect_grad: Beggins to collect gradients, L1 and L2 norms if collect_grad = True.
            layers: List of layers to be processed.
        """
        loss.backward()

        if layers is not None:
            if collect_grads:
                self.collect_grads(layers)


    def collect_grads(self, layers):
        self.layer_inds = layers
        
        iter_ave, iter_max, l1_norm, l2_norm, rec_layers = self.check_grads()
        self.rec_layers = rec_layers
        if len(self.ave_grads) == 0:
            self.ave_grads = iter_ave
            self.max_grads = iter_max
            self.l1_norm = l1_norm
            self.l2_norm = l2_norm
        else:
            self.ave_grads = np.vstack((self.ave_grads, iter_ave))
            self.max_grads = np.vstack((self.max_grads, iter_max))
            self.l1_norm = np.vstack((self.l1_norm, l1_norm))
            self.l2_norm = np.vstack((self.l2_norm, l2_norm))


    def save_collected_grads(self, ep, save_folder = None):
        max_grads = np.max(self.max_grads, axis = 0)
        ave_grads = np.mean(self.ave_grads, axis = 0)

        plt.figure(figsize=(15, 6)) # (width, height)

        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), self.rec_layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.tight_layout()

        if save_folder == None:
            save_folder = os.path.join(self.save_path, 'Gradient_saves')
            
        save_path = os.path.join(save_folder, f'Epoch_{ep}_Grad_graph.jpg')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(save_path)
        plt.close()

        self.plot_paired_lines_from_arrays(save_folder, ep)

        self.ave_grads = []
        self.max_grads = []
        self.rec_layers = 0
        self.l1_norm = []
        self.l2_norm = []


    def plot_paired_lines_from_arrays(self, save_dir, ep):
        
        norm_save_path = os.path.join(save_dir, 'norms')
        if not os.path.exists(norm_save_path):
            os.makedirs(norm_save_path)

        a, b = self.l1_norm.shape  # Get the dimensions (both arrays have the same shape)

        for i in range(b):
            plt.figure(figsize=(10, 6))  # Create a new figure for each pair of lines
            plt.title(f"Gradient norm (Layer {self.rec_layers[i]})")  # Unique title for each subplot
            plt.xlabel('Itterations')
            plt.ylabel("Gradient Norm")

            # Plot the two lines on the same graph
            plt.plot(self.l1_norm[:, i], label=f'L1', color='blue')
            plt.plot(self.l2_norm[:, i], label=f'L2', color='orange')

            plt.legend()
            plt.grid(True)
            
            save_path = os.path.join(norm_save_path, f'Epoch_{ep}_{self.rec_layers[i]}.jpg')
            plt.savefig(save_path)

    def save_tracked_data(self, save_folder):
        data = []
        for name, module in self.model.named_children():
            if module.track_flag and module.Trainable:
                tracker = module.master_tracker
                
                # Improve the data saving process
                for i in range(len(tracker['L1'])):
                    data.append({
                        'Layer': module.name,
                        'L1': tracker['L1'][i],
                        'L2': tracker['L2'][i]
                    })

        df = pd.DataFrame(data)
        save_path = os.path.join(save_folder, 'tracked_grads.csv')
        df.to_csv(save_path, index = False)


    def get_sim(self, x, layers, dim, sim_method = 'kl_divergence'):
        """
        Calculates the similarity between the features across dim.

        Args:
            x: A 4D tensor of shape [batch_size, channels, height, width].
            layers: A list containing the layers that need to be processed.
            dim: Dimension to calculate the similarity across.
            sim_method (str, optional): Specifies the method to apply for similarity:
                'cosine' | 'kl_divergence'. Default: 'kl_divergence'.
        """
        for name, module in self.model.named_children():
            if module.index in layers:
                module.record_sim = True
                module.sim_method = sim_method
                module.sim_dim = dim

        with torch.no_grad():
            _ = self.model(x)

        for name, module in self.model.named_children():
            if module.index in layers:
                print(f'Layer Index: {module.index}, Layer Name: {module.name}, Similarity Scores: {module.sim_scores}')
        