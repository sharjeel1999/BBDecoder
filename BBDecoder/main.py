from .visualizer import list_layers
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
                 ):
        """
        model: Model to be analyzed.
        save_folder: Folder to save all results and plots.
        """
        super(Master_analyzer, self).__init__()
        self.model = model
        self.training = False
        # self.save_folder = save_folder
        # self.track_grads = track_grads

        self.layer_inds = None
        self.grad_flag = None
        self.grad_hist_flag = None
        # self.track_grads = None
        self.function_flag = None

        self.ave_grads = []
        self.max_grads = []
        self.l1_norm = []
        self.l2_norm = []

        self.layer_names = []

        self.wrap_layers()
        print('----- List of layer and their indices -----')
        list_layers(self.model)

    # def train(self, mode = True):
    #     self.model.train(mode)

    # def eval(self):
    #     self.model.eval()
    
    # def to(self, device):
    #     self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def initialize_analyser(self, layer_inds, grad_flag, grad_hist_flag, track_grads, function_flag):
        '''
        layer_inds: List of indices of the layers to be analyzed.
        grad_flag: Flag to plot the gradient flow.
        grad_hist_flag: Flag to plot the gradient histograms.
        track_grads: Flag to track the gradients L1 and L2 norms.
        function_flag: Flag to plot the function flow.
        '''

        # self.layer_inds = layer_inds
        # self.grad_flag = grad_flag
        self.grad_hist_flag = grad_hist_flag
        self.track_grads = track_grads
        self.function_flag = function_flag

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


    def save_collected_grads(self, save_folder, ep):
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

        Column_names = ['L1', 'L2']

        for name, module in self.model.named_children():
            print('-- before names;', module.name, module.track_flag, module.Trainable)
            if module.track_flag and module.Trainable:
                tracker = module.master_tracker
                print('tracker name: ', module.name)
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


    def record_sim(self, x, layers, dim, sim_method = 'cosine'):
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
        