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

class Master_analyzer(GradAnalyzer, LayerAnalyzer):
    def __init__(self,
                 model,
                 optimizer,
                 save_folder
                 ):
        """
        model: Model to be analyzed.
        optimizer: Optimizer used to train the model.
        save_folder: Folder to save all results and plots.
        """
        
        self.model = model
        self.optimizer = optimizer
        self.save_folder = save_folder

        self.layer_inds = None
        self.grad_flag = None
        self.grad_hist_flag = None
        self.track_grads = None
        self.function_flag = None

        self.ave_grads = []
        self.max_grads = []

        self.layer_names = []

        self.wrap_layers()
        print('----- List of layer and their indices -----')
        list_layers(self.model)

    def initialize_analyser(self, layer_inds, grad_flag, grad_hist_flag, track_grads, function_flag):
        '''
        layer_inds: List of indices of the layers to be analyzed.
        grad_flag: Flag to plot the gradient flow.
        grad_hist_flag: Flag to plot the gradient histograms.
        track_grads: Flag to track the gradients L1 and L2 norms.
        function_flag: Flag to plot the function flow.
        '''

        self.layer_inds = layer_inds
        # self.grad_flag = grad_flag
        self.grad_hist_flag = grad_hist_flag
        self.track_grads = track_grads
        self.function_flag = function_flag

    def wrap_layers(self):
        for z, (name, module) in enumerate(self.model.named_children()):
            self.layer_names.append(name)
            
            if isinstance(module, nn.Module) or isinstance(module, nn.Sequential):#and not list(module.children()):  
                setattr(self.model, name, Main_wrapper(module, name, z, self.track_grads))
            # else:
                # Recursively wrap layers in submodules (if any)
                # Main_wrapper(module, name, z, False)


    def forward_propagation(self, x):
        return self.model(x)
    
    def backward_propagation(self, loss, collect_grads = False, layers = None):
        self.optimizer.zero_grad()
        loss.backward()

        # print('//////// ', self.layer_inds, self.grad_flag)
        if layers is not None:
            if collect_grads:
                self.collect_grads()

        # self.optimizer.step()

    def collect_grads(self):
        print('--- tick tick ---')
        iter_ave, iter_max, rec_layers = self.check_grads()
        self.rec_layers = rec_layers
        if len(self.ave_grads) == 0:
            print('--- direct equal')
            self.ave_grads = iter_ave
            self.max_grads = iter_max
        else:
            print('before add shape:', self.ave_grads.shape, iter_ave.shape)
            self.ave_grads = np.vstack((self.ave_grads, iter_ave))
            self.max_grads = np.vstack((self.max_grads, iter_max))
        
        print('--- shapes: ', np.array(self.ave_grads).shape, np.array(self.max_grads).shape)


    def save_collected_grads(self, save_folder, ep):
        print('Saving the collected gradients')
        print('grad shapes: ', np.array(self.ave_grads).shape, np.array(self.max_grads).shape)
        max_grads = np.max(self.max_grads, axis = 0)
        ave_grads = np.mean(self.ave_grads, axis = 0)

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
        plt.savefig(save_path)

        self.ave_grads = []
        self.max_grads = []
        self.rec_layers = 0

    def save_tracked_data(self):
        data = []

        Column_names = ['L1', 'L2']

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
        save_path = os.path.join(self.save_folder, 'tracked_grads.csv')
        df.to_csv(save_path, index = False)

    def record_sim(self, x, layers, dim, sim_method = 'cosine', ):
        for name, module in self.model.named_children():
            if module.index in layers:
                module.record_sim = True
                module.sim_method = sim_method
                module.sim_dim = dim

        with torch.no_grad():
            _ = self.model(x)
            
        